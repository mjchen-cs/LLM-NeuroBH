"""End-to-end intent-driven streaming (new experiment for the TSC version).

For every (non-conflict) intent, sessions are run on traces of the intent's
network class and every policy is scored under the GROUND-TRUTH spec
(what the service actually asked for):

  BBA, BOLA, RobustMPC, Pensieve        intent-unaware baselines
  NeuroBH (default)                     original NeuroBH, default spec
  LLM-NeuroBH                           spec produced by the LLM service (verified)
  LLM-NeuroBH w/o verifier              spec accepted without verification
  Oracle-NeuroBH                        ground-truth spec (upper bound)

Output: results/e2e.csv
"""
import argparse
import csv
import json
import os
import numpy as np
from common import load_cfg, load_intents, traces_for, videos_for, mark_run, C
from neurobh.agent import Agent
from neurobh.env import StreamingEnv
from neurobh.metrics import session_metrics, sample_departures
from neurobh.policies import BBA, BOLA, FESTIVE, Dynamic, RobustMPC, LLMDirectABR, NeuroBHPolicy, run_episode
from neurobh.predictor import make as make_pred
from neurobh.spec import ServiceSpec, default_spec


def load_llm_specs(path, model, variant):
    specs = {}
    if not os.path.exists(path):
        return specs
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["model"] == model and r["variant"] == variant:
            specs[r["id"]] = ServiceSpec.from_dict(r["spec"])
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--sessions", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm-direct", action="store_true",
                    help="also run the LLM-as-controller baseline (the LLM receives the intent and picks every bitrate)")
    ap.add_argument("--llm-direct-sessions", type=int, default=2)
    a = ap.parse_args()
    cfg = load_cfg()
    model = a.model or cfg["llm_main"]
    n_sess = a.sessions or cfg["sessions_per_intent"]
    runs = f"{cfg['results']}/llm_runs.jsonl"
    llm_full = load_llm_specs(runs, model, "full")
    llm_nover = load_llm_specs(runs, model, "no_verifier")
    if not llm_full:
        raise SystemExit(f"no LLM specs for model '{model}' in {runs}: run exp_llm.py first")
    from neurobh.verifier import Verifier
    V = Verifier()
    intents = [it for it in load_intents() if V.verify_dict(it["gt"]).ok][: a.limit]
    vids = videos_for(cfg)
    ag = Agent.load(cfg["ckpt"]["neurobh"])
    pens = Agent.load(cfg["ckpt"]["pensieve"]) if os.path.exists(cfg["ckpt"]["pensieve"]) else None
    have_pred = os.path.exists(cfg["ckpt"]["predictor"])
    trace_cache = {}
    llm_direct = None
    if a.llm_direct:
        from neurobh.llm.client import load_clients
        llm_direct = load_clients("configs/llm.yaml", [model])[model]
    rng = np.random.default_rng(0)
    rows = []
    dspec = default_spec()
    for j, it in enumerate(intents):
        gt = ServiceSpec.from_dict(it["gt"])
        prof = gt.network_profile
        if prof not in trace_cache:
            trace_cache[prof] = traces_for(cfg, prof)
        traces = trace_cache[prof]
        idx = rng.integers(len(traces), size=n_sess); st = rng.integers(0, 10 ** 6, size=n_sess)
        vix = rng.integers(len(vids), size=n_sess); deps = sample_departures(n_sess, rng)
        P = [("BBA", BBA(), dspec, "chunk", 60.0), ("BOLA", BOLA(), dspec, "chunk", 60.0),
             ("FESTIVE", FESTIVE(), dspec, "chunk", 60.0), ("Dynamic", Dynamic(), dspec, "chunk", 60.0),
             ("RobustMPC", RobustMPC(dspec), dspec, "chunk", 60.0)]
        if llm_direct is not None:
            P.append(("LLM-ABR (direct, intent)", LLMDirectABR(llm_direct, intent_text=it["text"]), dspec, "chunk", 60.0))
        if pens:
            P.append(("Pensieve", NeuroBHPolicy(pens, dspec, use_shield=False), dspec, "chunk", 60.0))
        P += [("NeuroBH (default)", NeuroBHPolicy(ag, dspec), dspec, "stable", None),
              ("LLM-NeuroBH", NeuroBHPolicy(ag, llm_full[it["id"]]), llm_full[it["id"]], "stable", None),
              ("Oracle-NeuroBH", NeuroBHPolicy(ag, gt), gt, "stable", None)]
        if it["id"] in llm_nover:
            sp = llm_nover[it["id"]]
            P.append(("LLM-NeuroBH w/o verifier", NeuroBHPolicy(ag, sp), sp, "stable", None))
        for name, pol, run_spec, meas, fmb in P:
            pred = make_pred("gru", path=cfg["ckpt"]["predictor"]) if meas == "stable" and have_pred \
                else make_pred("robust" if name == "RobustMPC" else "harmonic")
            env = StreamingEnv(traces, vids, measure=meas, predictor=pred, seed=j, fixed_max_buffer=fmb)
            for k in range(n_sess if not name.startswith("LLM-ABR") else min(n_sess, a.llm_direct_sessions)):
                log = run_episode(env, pol, run_spec, int(idx[k]), int(vix[k]), int(st[k]))
                m = session_metrics(log, gt, departure=deps[k])
                rows.append({"id": it["id"], "scenario": it["scenario"], "policy": name, "session": k, **m})
        print(f"[e2e] {j + 1}/{len(intents)} {it['id']}")
    with open(f"{cfg['results']}/e2e.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.groupby("policy")[["objective", "qoe", "violation_rate", "rebuf_ratio", "buffer", "waste_MB"]].mean())
    mark_run(cfg, "exp_e2e", {"llm_model": model})


if __name__ == "__main__":
    main()
