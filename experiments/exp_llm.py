"""LLM Intent-to-Spec quality (new experiments for the TSC version).

For every model x variant x intent:
  first_stage   verifier stage reached by the FIRST reply (passed / json / schema / syntax / range /
                consistency / realizability)
  verified      final spec passed the verifier (after <= K repair rounds)
  prio_pm1      fraction of the 5 priorities within +-1 of the ground truth
  weight_mae    MAE of the induced normalised weight vector s_u
  profile_ok    network profile correct
  sem_agree     semantic agreement of the final constraints with the ground truth (non-conflict intents)
  sem_exact     sem_agree >= 0.999
  deployed_safe the spec that would be deployed passes the verifier (always 1 with the verifier)
  conflict intents: detected (first reply rejected) and resolved (feasible spec after repair)

Variants: full | no_repair | no_verifier | zero_shot
Output: results/llm_runs.jsonl, results/llm_summary.csv
"""
import argparse
import csv
import json
import re
import os
import numpy as np
from common import load_cfg, load_intents, mark_run, C
from neurobh.llm.client import load_clients, LLMError
from neurobh.llm.intent_service import IntentService
from neurobh.spec import ServiceSpec
from neurobh.verifier import Verifier, semantic_agreement

VARIANTS = {"full": dict(), "no_repair": dict(use_repair=False),
            "no_verifier": dict(use_verifier=False), "zero_shot": dict(few_shot=False)}


NET_WORDS = re.compile(r"wifi|wi-fi|3g|4g|5g|lte|fibre|fiber|gigabit|metro|subway|mobile network|"
                       r"光纤|千兆|地铁|弱网|网络|宽带|家用|校园", re.I)


def profile_correct(pred_profile, it):
    """The network profile is only judged when the intent mentions the network;
    otherwise 'unknown' is also an acceptable (honest) answer."""
    gt = it["gt"]["network_profile"]
    if pred_profile == gt:
        return 1
    return int(pred_profile == "unknown" and not NET_WORDS.search(it["text"]))


def score(res, it, V):
    gt = ServiceSpec.from_dict(it["gt"])
    sp = res.spec
    p_gt, p = gt.priorities, sp.priorities
    d = np.array([abs(p[k] - p_gt[k]) for k in C.PRIORITY_KEYS])
    wv = np.array(sp.weights.as_vector()); wg = np.array(gt.weights.as_vector())
    row = {"first_stage": res.first_stage, "verified": int(res.verified), "fallback": int(res.fallback),
           "rounds": res.rounds, "llm_latency": res.llm_latency, "verify_latency": res.verify_latency,
           "json_ok_first": int(res.first_stage not in ("json",)),
           "syntax_ok_first": int(res.first_stage not in ("json", "schema", "syntax")),
           "pass_first": int(res.first_stage == "passed"),
           "prio_exact": float((d == 0).mean()), "prio_pm1": float((d <= 1).mean()),
           "weight_mae": float(np.abs(wv - wg).mean()),
           "profile_ok": profile_correct(sp.network_profile, it),
           "deployed_safe": int(V.verify_spec(sp).ok)}
    if it["conflict"]:
        row.update(detected=int(res.first_stage != "passed"), resolved=int(res.verified and not res.fallback))
        row.update(sem_agree=np.nan, sem_exact=np.nan)
    else:
        a = semantic_agreement(sp, gt, n=3000) if not res.fallback else semantic_agreement(sp, gt, n=3000)
        row.update(sem_agree=a, sem_exact=int(a >= 0.999), detected=np.nan, resolved=np.nan)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma list of model names in configs/llm.yaml")
    ap.add_argument("--variants", default="full,no_repair,no_verifier,zero_shot")
    ap.add_argument("--intents", default="data/intents.jsonl")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true", help="discard previous llm_runs.jsonl")
    a = ap.parse_args()
    cfg = load_cfg()
    names = a.models.split(",") if a.models else cfg["llm_models"]
    clients = load_clients("configs/llm.yaml", names)
    intents = load_intents(a.intents)[: a.limit]
    V = Verifier()
    # the feasibility label depends on the measured eta (results/eta.json) -> recompute it
    for it in intents:
        lab = not V.verify_dict(it["gt"]).ok
        if lab != it["conflict"]:
            print(f"[exp_llm] note: {it['id']} feasibility label changed under current eta -> conflict={lab}")
        it["conflict"] = lab
    # results are APPENDED: (model, variant) pairs that are already complete are skipped,
    # so a crash or an unavailable model never destroys finished runs (use --overwrite to redo)
    runs_path = f"{cfg['results']}/llm_runs.jsonl"
    done = {}
    if os.path.exists(runs_path) and not a.overwrite:
        for line in open(runs_path, encoding="utf-8"):
            try:
                r = json.loads(line); done.setdefault((r["model"], r["variant"]), {})[r["id"]] = r
            except Exception:
                pass
    elif os.path.exists(runs_path):
        os.remove(runs_path)
    out = open(runs_path, "a", encoding="utf-8")
    for mname, llm in clients.items():
        for var in a.variants.split(","):
            have = done.get((mname, var), {})
            if len(have) >= len(intents):
                print(f"[{mname}/{var}] already complete ({len(have)} rows) -> skipped"); continue
            svc = IntentService(llm, V, max_rounds=a.max_rounds, **VARIANTS[var])
            try:
                for j, it in enumerate(intents):
                    if it["id"] in have:
                        continue
                    res = svc.compile(it["text"])
                    row = {"model": mname, "variant": var, "id": it["id"], "scenario": it["scenario"],
                           "lang": it["lang"], "conflict": it["conflict"], **score(res, it, V),
                           "spec": res.spec.to_dict(), "stages": res.stages}
                    have[it["id"]] = row
                    out.write(json.dumps(row, ensure_ascii=False, default=float) + "\n"); out.flush()
                    if (j + 1) % 20 == 0:
                        print(f"[{mname}/{var}] {j + 1}/{len(intents)}")
            except LLMError as e:
                print(f"[exp_llm] model '{mname}' failed and is skipped: {e}")
                break
            done[(mname, var)] = have
    out.close()
    # ---- summary over everything in llm_runs.jsonl
    summary = []
    for (mname, var), have in sorted(done.items()):
        rows = list(have.values())
        if len(rows) < len(intents):
            print(f"[exp_llm] {mname}/{var}: only {len(rows)}/{len(intents)} rows -> not summarised"); continue
        by_id = {it["id"]: it for it in intents}
        for r in rows:                              # re-score with the current rule (also for old rows)
            if r["id"] in by_id:
                r["profile_ok"] = profile_correct(r["spec"]["network_profile"], by_id[r["id"]])
        nc = [r for r in rows if not r["conflict"]]; cf = [r for r in rows if r["conflict"]]
        s = {"model": mname, "variant": var, "n": len(rows),
             "json_ok_first": np.mean([r["json_ok_first"] for r in rows]),
             "syntax_ok_first": np.mean([r["syntax_ok_first"] for r in rows]),
             "pass_first": np.mean([r["pass_first"] for r in nc]),
             "verified_final": np.mean([r["verified"] for r in nc]),
             "deployed_safe": np.mean([r["deployed_safe"] for r in rows]),
             "fallback": np.mean([r["fallback"] for r in rows]),
             "avg_rounds": np.mean([r["rounds"] for r in rows]),
             "prio_pm1": np.mean([r["prio_pm1"] for r in nc]),
             "weight_mae": np.mean([r["weight_mae"] for r in nc]),
             "profile_ok": np.mean([r["profile_ok"] for r in nc]),
             "sem_exact": np.mean([r["sem_exact"] for r in nc]),
             "sem_agree": np.mean([r["sem_agree"] for r in nc]),
             "conflict_detected": np.mean([r["detected"] for r in cf]) if cf else np.nan,
             "conflict_resolved": np.mean([r["resolved"] for r in cf]) if cf else np.nan,
             "llm_latency_mean": np.mean([r["llm_latency"] for r in rows]),
             "llm_latency_p95": np.percentile([r["llm_latency"] for r in rows], 95),
             "verify_latency_mean": np.mean([r["verify_latency"] for r in rows])}
        summary.append(s)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    if summary:
        with open(f"{cfg['results']}/llm_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    mark_run(cfg, "exp_llm", {"mock": "mock" in clients})


if __name__ == "__main__":
    main()
