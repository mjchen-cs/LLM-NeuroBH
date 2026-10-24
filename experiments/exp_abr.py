"""Trace-driven comparison with ABR baselines (paper Figs. 5-7 and QoE CDF) and
DRL / neurosymbolic ablations.  All policies use the DEFAULT service spec here
(no LLM) -- this reproduces the original NeuroBH evaluation.

Output: results/abr_<dataset>.csv  (one row per policy x session)
        results/abr_phi_<dataset>.csv  (objective vs. buffer weight phi)
"""
import argparse
import csv
import os
import numpy as np
from common import load_cfg, traces_for, videos_for, mark_run, C
from neurobh.env import StreamingEnv
from neurobh.policies import BBA, BOLA, FESTIVE, Dynamic, RobustMPC, SODA, LLMDirectABR, NeuroBHPolicy, run_episode
from neurobh.predictor import make as make_pred
from neurobh.metrics import session_metrics, sample_departures
from neurobh.spec import default_spec, ServiceSpec


def load_agent(path):
    if not os.path.exists(path):
        return None
    from neurobh.agent import Agent
    return Agent.load(path)


SODA_PARAMS = dict(target=30.0, w_buf=0.05, w_switch=0.005)   # from --tune-soda on the training traces


def tune_soda(cfg, vids, spec, n=40):
    """Grid search of SODA's weights on the TRAINING traces (never on the test sets)."""
    traces = traces_for(cfg, "train")
    rng = np.random.default_rng(123)
    idx = rng.integers(len(traces), size=n); starts = rng.integers(0, 10 ** 6, size=n)
    vix = rng.integers(len(vids), size=n)
    env = StreamingEnv(traces, vids, measure="chunk", predictor=make_pred("harmonic"), seed=1, fixed_max_buffer=60.0)
    res = []
    for bt in (10.0, 20.0, 30.0):
        for wb in (0.01, 0.05, 0.2):
            for ws in (0.005, 0.02, 0.05):
                pol = SODA(target=bt, w_buf=wb, w_switch=ws)
                q = np.mean([session_metrics(run_episode(env, pol, spec, int(idx[k]), int(vix[k]), int(starts[k])), spec)["qoe"]
                             for k in range(n)])
                res.append((q, bt, wb, ws)); print(f"[tune-soda] target={bt:4.0f} w_buf={wb:<5} w_switch={ws:<5} QoE={q:.2f}", flush=True)
    q, bt, wb, ws = max(res)
    print(f"[tune-soda] best: target={bt}, w_buf={wb}, w_switch={ws} (QoE {q:.2f}) -> set SODA_PARAMS")
    return dict(target=bt, w_buf=wb, w_switch=ws)


def policies(cfg, spec, llm_abr=None):
    P = [("BBA", BBA(), "chunk", 60.0), ("BOLA", BOLA(), "chunk", 60.0),
         ("FESTIVE", FESTIVE(), "chunk", 60.0), ("Dynamic", Dynamic(), "chunk", 60.0),
         ("RobustMPC", RobustMPC(spec), "chunk", 60.0), ("SODA", SODA(**SODA_PARAMS), "chunk", 60.0)]
    if llm_abr is not None:
        P.append(("LLM-ABR (direct)", LLMDirectABR(llm_abr), "chunk", 60.0))
    ag = load_agent(cfg["ckpt"]["pensieve"])
    if ag: P.append(("Pensieve", NeuroBHPolicy(ag, spec, use_shield=False), "chunk", 60.0))
    ag = load_agent(cfg["ckpt"]["neurobh"])
    if ag: P.append(("NeuroBH", NeuroBHPolicy(ag, spec), "stable", None))
    ag = load_agent(cfg["ckpt"]["neurobh_noshield"])
    if ag: P.append(("NeuroBH w/o NS", NeuroBHPolicy(ag, spec, use_shield=False), "stable", None))
    ag = load_agent(cfg["ckpt"]["neurobh_a2c"])
    if ag: P.append(("NeuroBH (A2C)", NeuroBHPolicy(ag, spec), "stable", None))
    return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", default="slow,medium,fast")
    ap.add_argument("--sessions", type=int, default=None)
    ap.add_argument("--llm-abr", default=None, help="model in configs/llm.yaml used as a per-chunk LLM controller baseline")
    ap.add_argument("--llm-abr-sessions", type=int, default=10, help="sessions per dataset for the (slow, API-bound) LLM baseline")
    ap.add_argument("--policies", default=None,
                    help="comma-separated subset to run; their rows replace those in the existing results/abr_<ds>.csv")
    ap.add_argument("--tune-soda", action="store_true", help="grid-search SODA's weights on the training traces and exit")
    a = ap.parse_args()
    llm = None
    if a.llm_abr:
        from neurobh.llm.client import load_clients
        llm = load_clients("configs/llm.yaml", [a.llm_abr])[a.llm_abr]
    cfg = load_cfg()
    n_sess = a.sessions or cfg["sessions_per_dataset"]
    vids = videos_for(cfg)
    spec = default_spec()
    if a.tune_soda:
        tune_soda(cfg, vids, spec); return
    only = set(a.policies.split(",")) if a.policies else None
    rng = np.random.default_rng(0)
    for ds in a.datasets.split(","):
        traces = traces_for(cfg, ds)
        idx = rng.integers(len(traces), size=n_sess)
        starts = rng.integers(0, 10 ** 6, size=n_sess)
        vix = rng.integers(len(vids), size=n_sess)
        deps = sample_departures(n_sess, rng)
        rows = []
        for name, pol, meas, fmb in policies(cfg, spec, llm):
            if only and name not in only:
                continue
            pred = make_pred("gru", path=cfg["ckpt"]["predictor"]) if meas == "stable" and \
                os.path.exists(cfg["ckpt"]["predictor"]) else make_pred("robust" if name == "RobustMPC" else "harmonic")
            env = StreamingEnv(traces, vids, measure=meas, predictor=pred, seed=1, fixed_max_buffer=fmb)
            for k in range(n_sess if not name.startswith("LLM-ABR") else min(n_sess, a.llm_abr_sessions)):
                log = run_episode(env, pol, spec, int(idx[k]), int(vix[k]), int(starts[k]))
                m = session_metrics(log, spec, departure=deps[k])
                rows.append({"dataset": ds, "policy": name, "session": k, **m})
                # store per-session phi sweep using the same log (objective under phi)
                for lvl in range(1, 6):
                    sp = ServiceSpec(priorities={**spec.priorities, "data_saving": lvl, "low_latency": 1}).compile()
                    if name == "NeuroBH":     # spec-conditioned: re-run with the new preference
                        log_l = run_episode(env, NeuroBHPolicy(pol.agent, sp), sp, int(idx[k]), int(vix[k]), int(starts[k]))
                    else:                     # intent-unaware baselines: same decisions, re-scored
                        log_l = log
                    rows[-1][f"obj_phi{lvl}"] = session_metrics(log_l, sp)["objective"]
                    rows[-1][f"buf_phi{lvl}"] = session_metrics(log_l, sp)["buffer"]
            if hasattr(pol, "latencies") and pol.latencies:
                with open(f"{cfg['results']}/llm_abr_latency.csv", "a") as f:
                    f.writelines(f"{ds},{x}\n" for x in pol.latencies)
                pol.latencies = []
            print(f"[{ds}] {name:16s} QoE={np.mean([r['qoe'] for r in rows if r['policy'] == name]):.2f} "
                  f"buffer={np.mean([r['buffer'] for r in rows if r['policy'] == name]):.1f}s "
                  f"rebuf={np.mean([r['rebuf_ratio'] for r in rows if r['policy'] == name]) * 100:.2f}% "
                  f"waste={np.mean([r['waste_MB'] for r in rows if r['policy'] == name]):.2f}MB")
        out = f"{cfg['results']}/abr_{ds}.csv"
        if only and os.path.exists(out):                     # merge: keep the other policies' rows
            import pandas as pd
            old = pd.read_csv(out)
            new = pd.concat([old[~old.policy.isin(only)], pd.DataFrame(rows)], ignore_index=True)
            new.to_csv(out, index=False)
        else:
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    mark_run(cfg, "exp_abr")


if __name__ == "__main__":
    main()
