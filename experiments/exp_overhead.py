"""Runtime overhead: per-chunk decision latency (policy + shield) vs. the
session-level LLM latency (taken from results/llm_runs.jsonl).
Output: results/overhead.csv"""
import csv
import os
import time
import numpy as np
from common import load_cfg, traces_for, videos_for, mark_run
from neurobh.agent import Agent
from neurobh.env import StreamingEnv
from neurobh.policies import NeuroBHPolicy, RobustMPC, BBA
from neurobh.spec import default_spec, ServiceSpec


def main():
    cfg = load_cfg()
    vids = videos_for(cfg); traces = traces_for(cfg, "medium")
    spec = ServiceSpec(constraint_texts=["ALWAYS buffer >= 2", "ALWAYS max_buffer <= 20",
                                         "ALWAYS switch <= 1"]).compile()
    ag = Agent.load(cfg["ckpt"]["neurobh"])
    rows = []
    for name, pol in [("NeuroBH policy only", NeuroBHPolicy(ag, spec, use_shield=False)),
                      ("NeuroBH policy + shield", NeuroBHPolicy(ag, spec)),
                      ("RobustMPC", RobustMPC(default_spec())), ("BBA", BBA())]:
        env = StreamingEnv(traces, vids, seed=0)
        ts = []
        for ep in range(20):
            s = env.reset(spec); done = False
            while not done:
                t0 = time.perf_counter(); lv, d, inter = pol.act(env, s); ts.append((time.perf_counter() - t0) * 1000)
                s, _, done, _ = env.step(lv, d, inter)
        rows.append({"component": name, "mean_ms": np.mean(ts), "p95_ms": np.percentile(ts, 95)})
    p = f"{cfg['results']}/llm_runs.jsonl"
    if os.path.exists(p):
        import json, pandas as pd
        df = pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")])
        df = df[df.variant == "full"]
        for m, g in df.groupby("model"):
            rows.append({"component": f"LLM service ({m}), per session", "mean_ms": g.llm_latency.mean() * 1000,
                         "p95_ms": np.percentile(g.llm_latency, 95) * 1000})
        rows.append({"component": "Verifier, per spec", "mean_ms": df.verify_latency.mean() * 1000,
                     "p95_ms": np.percentile(df.verify_latency, 95) * 1000})
    p2 = f"{cfg['results']}/llm_abr_latency.csv"
    if os.path.exists(p2):
        lat = np.array([float(l.split(",")[1]) for l in open(p2) if "," in l])
        rows.append({"component": "LLM-ABR (direct), per chunk", "mean_ms": lat.mean() * 1000,
                     "p95_ms": np.percentile(lat, 95) * 1000})
    with open(f"{cfg['results']}/overhead.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows: print(r)
    mark_run(cfg, "exp_overhead")


if __name__ == "__main__":
    main()
