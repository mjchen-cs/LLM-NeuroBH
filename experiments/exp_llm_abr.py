"""Supplementary run for the per-chunk LLM controller baseline.

Adds the variant "LLM-ABR (no delay)": the same zero-shot LLM chooses every
bitrate, but its API latency is NOT charged to the session.  Comparing it with
"LLM-ABR (direct)" separates the two failure causes:
    decision quality  (no delay vs. RobustMPC / NeuroBH)
    API latency        (direct vs. no delay)

The sessions are exactly the first N sessions of experiments/exp_abr.py (same
random draws), so the rows are directly comparable.  Rows are merged into
results/abr_<dataset>.csv (existing rows of the same policy are replaced).

Usage:  python experiments/exp_llm_abr.py --model glm-4-flash --sessions 10
        (add --with-delay to also re-run the latency-charged variant)
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import load_cfg, traces_for, videos_for, C
from neurobh.env import StreamingEnv
from neurobh.llm.client import load_clients
from neurobh.metrics import session_metrics, sample_departures
from neurobh.policies import LLMDirectABR, run_episode
from neurobh.predictor import make as make_pred
from neurobh.spec import default_spec, ServiceSpec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model name in configs/llm.yaml (default: llm_main)")
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--datasets", default="slow,medium,fast",
                    help="must be the same order as used for exp_abr.py (default order)")
    ap.add_argument("--with-delay", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg()
    model = a.model or cfg["llm_main"]
    llm = load_clients("configs/llm.yaml", [model])[model]
    n_all = cfg["sessions_per_dataset"]            # reproduce exp_abr.py's random draws exactly
    vids = videos_for(cfg)
    spec = default_spec()
    rng = np.random.default_rng(0)
    variants = [("LLM-ABR (no delay)", False)] + ([("LLM-ABR (direct)", True)] if a.with_delay else [])
    for ds in a.datasets.split(","):
        traces = traces_for(cfg, ds)
        idx = rng.integers(len(traces), size=n_all)
        starts = rng.integers(0, 10 ** 6, size=n_all)
        vix = rng.integers(len(vids), size=n_all)
        deps = sample_departures(n_all, rng)
        path = f"{cfg['results']}/abr_{ds}.csv"
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        new_rows = []
        for name, charge in variants:
            pol = LLMDirectABR(llm, charge_latency=charge)
            env = StreamingEnv(traces, vids, measure="chunk", predictor=make_pred("harmonic"),
                               seed=1, fixed_max_buffer=60.0)
            for k in range(min(a.sessions, n_all)):
                log = run_episode(env, pol, spec, int(idx[k]), int(vix[k]), int(starts[k]))
                m = session_metrics(log, spec, departure=deps[k])
                row = {"dataset": ds, "policy": name, "session": k, **m}
                for lvl in range(1, 6):          # same phi re-scoring as exp_abr.py
                    sp = ServiceSpec(priorities={**spec.priorities, "data_saving": lvl, "low_latency": 1}).compile()
                    sm = session_metrics(log, sp)
                    row[f"obj_phi{lvl}"] = sm["objective"]; row[f"buf_phi{lvl}"] = sm["buffer"]
                new_rows.append(row)
                print(f"[{ds}] {name} session {k + 1}/{a.sessions}: QoE={m['qoe']:.1f} "
                      f"rebuf={m['rebuf_ratio'] * 100:.1f}% bitrate={m['bitrate']:.0f}kbps")
            print(f"[{ds}] {name}: API failures={pol.failures}, mean latency="
                  f"{np.mean(pol.latencies) if pol.latencies else float('nan'):.2f}s")
        new = pd.DataFrame(new_rows)
        if len(old):
            old = old[~old.policy.isin(new.policy.unique())]
        merged = pd.concat([old, new], ignore_index=True)
        merged.to_csv(path, index=False)
        # compact comparison on the same sessions
        same = merged[merged.session < a.sessions]
        print(f"\n== {ds}: first {a.sessions} sessions ==")
        print(same.groupby("policy")[["qoe", "rebuf_ratio", "bitrate", "buffer"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
