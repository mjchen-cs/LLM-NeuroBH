"""Bandwidth prediction & measurement accuracy (paper Tables: prediction / measurement error).

1. trains the GRU-hybrid predictor on HTTP-stable-connection samples of the
   training traces, saves ckpt/predictor.pt
2. relative prediction error of Sliding window / Harmonic mean / EWMA / GRU-hybrid
3. eta = 90th percentile of the over-estimation ratio -> results/eta.json
   (use it to update NETWORK_PROFILES[...]['eta'] in neurobh/config.py)
4. measurement error: client-side chunk measurement vs. HTTP stable connection
"""
import json
import argparse
import numpy as np
from common import load_cfg, traces_for, videos_for, mark_run, C
from neurobh.predictor import (GRUHybrid, SlidingWindow, HarmonicMean, EWMA,
                               stable_samples_from_trace)
from neurobh.env import StreamingEnv
from neurobh.policies import BBA, run_episode
from neurobh.spec import default_spec


def rel_err_stats(pred, true, floor=None):
    """Relative error in %.  The denominator is floored at the lowest ladder bitrate:
    during outages (true bandwidth ~0, frequent in 5G traces) the plain relative error
    explodes to 1e4 %, although any prediction below the lowest bitrate leads to the
    same ABR decision.  Report median / 75th percentile; the mean is outage-dominated."""
    floor = C.BITRATES_KBPS[0] if floor is None else floor
    true = np.maximum(np.asarray(true), floor)
    e = np.abs(np.asarray(pred) - true) / true * 100
    return {"mean": float(e.mean()), "median": float(np.median(e)), "p75": float(np.percentile(e, 75))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-train", type=int, default=3000)
    a = ap.parse_args()
    cfg = load_cfg()
    rng = np.random.default_rng(0)
    train = traces_for(cfg, "train")[:a.max_train]
    seqs = [stable_samples_from_trace(t, C.CHUNK_LEN, rng) for t in train]
    seqs = [s for s in seqs if len(s) > C.HIST_LEN + 2]
    gru = GRUHybrid(C.BITRATES_KBPS)
    gru.fit(seqs, epochs=a.epochs)
    gru.save(cfg["ckpt"]["predictor"])

    rows, eta = [], {}
    for ds in ["slow", "medium", "fast"]:
        test = traces_for(cfg, ds)
        preds = {m.name: [] for m in [SlidingWindow(), HarmonicMean(), EWMA(), gru]}
        true = []
        models = [SlidingWindow(), HarmonicMean(), EWMA(), gru]
        for tr in test:
            s = stable_samples_from_trace(tr, C.CHUNK_LEN, rng)
            for t in range(C.HIST_LEN, len(s)):
                hist = list(s[:t])
                for m in models:
                    preds[m.name].append(m.predict(hist))
                true.append(s[t])
        for name, p in preds.items():
            rows.append({"dataset": ds, "method": name, **rel_err_stats(p, true)})
        over = np.maximum(0, (np.asarray(preds[gru.name]) - np.asarray(true)) / np.asarray(preds[gru.name]))
        eta[ds] = float(np.percentile(over, 90))
    import csv
    with open(f"{cfg['results']}/tab_prediction.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(eta, open(f"{cfg['results']}/eta.json", "w"), indent=2)
    print("prediction error:"); [print(r) for r in rows]
    print("eta (90th pct over-estimation):", eta)

    # measurement error: chunk vs stable
    vids = videos_for(cfg)
    mrows = []
    for mode, label in [("chunk", "client chunk measurement"), ("stable", "HTTP stable connection")]:
        errs = []
        env = StreamingEnv(traces_for(cfg, "medium"), vids, measure=mode, seed=1)
        for k in range(min(100, len(env.traces))):
            log = run_episode(env, BBA(), default_spec(), trace_idx=k)
            errs += [abs(r["sample"] - r["true_bw"]) / r["true_bw"] * 100 for r in log]
        e = np.asarray(errs)
        mrows.append({"method": label, "mean": e.mean(), "median": np.median(e), "p75": np.percentile(e, 75)})
    with open(f"{cfg['results']}/tab_measurement.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mrows[0])); w.writeheader(); w.writerows(mrows)
    print("measurement error:", mrows)
    mark_run(cfg, "exp_predictor")


if __name__ == "__main__":
    main()
