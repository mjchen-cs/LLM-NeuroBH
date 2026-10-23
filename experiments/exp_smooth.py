"""Intent-driven smoothness (soft weight vs. hard DSL constraint): can the operator fix NeuroBH's bitrate oscillation on slow networks
by stating it in the intent, WITHOUT retraining?  ("on weak networks, smooth playback first")

The SAME trained policy is run with specs that differ only in the smoothness priority
(1..5, default 3; the LLM maps "smooth / few switches" to 4-5).  Every session is scored
  * under the DEFAULT spec (qoe_default)  -> comparable with the baselines of Fig. abr, and
  * under its own spec (qoe_own)          -> what the operator asked for.
Variant "constraint" instead keeps the default weights and adds the DSL constraint below,
which the shield enforces (no retraining).
Sessions = first N sessions of experiments/exp_abr.py (same random draws).

Output: results/smooth_sweep.csv, results/smooth_summary.csv, <figures>/fig_smooth.pdf
Usage:  python experiments/exp_smooth.py            (100 sessions per network class)
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd

from common import load_cfg, traces_for, videos_for, mark_run, C
from neurobh.agent import Agent
from neurobh.env import StreamingEnv
from neurobh.metrics import session_metrics, sample_departures
from neurobh.policies import NeuroBHPolicy, run_episode
from neurobh.predictor import make as make_pred
from neurobh.spec import default_spec, ServiceSpec

LEVELS = [1, 2, 3, 4, 5]
# hard constraint the intent "on weak networks, prefer smooth playback over quick up-switches"
# compiles to: no up-switch while the buffer is below 16 s on a link predicted below 1.5 Mbps
SMOOTH_CONSTRAINT = "ALWAYS switch <= 0 WHEN buffer < 16 AND pred_bw < 1500"


def switch_rate(log):
    return float(np.mean([log[i]["level"] != log[i - 1]["level"] for i in range(1, len(log))])) if len(log) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=100)
    ap.add_argument("--datasets", default="slow,medium,fast", help="same order as exp_abr.py")
    a = ap.parse_args()
    cfg = load_cfg()
    agent = Agent.load(cfg["ckpt"]["neurobh"])
    vids = videos_for(cfg)
    base = default_spec()
    n_all = cfg["sessions_per_dataset"]
    rng = np.random.default_rng(0)
    rows = []
    for ds in a.datasets.split(","):
        traces = traces_for(cfg, ds)
        idx = rng.integers(len(traces), size=n_all)
        starts = rng.integers(0, 10 ** 6, size=n_all)
        vix = rng.integers(len(vids), size=n_all)
        deps = sample_departures(n_all, rng)
        variants = [(f"prio{l}", l, ServiceSpec(priorities={**base.priorities, "smoothness": l}).compile())
                    for l in LEVELS]
        variants.append(("constraint", 3, ServiceSpec(priorities=dict(base.priorities),
                                                      constraint_texts=[SMOOTH_CONSTRAINT]).compile()))
        for name, lvl, spec in variants:
            pred = make_pred("gru", path=cfg["ckpt"]["predictor"]) if os.path.exists(cfg["ckpt"]["predictor"]) \
                else make_pred("harmonic")
            env = StreamingEnv(traces, vids, measure="stable", predictor=pred, seed=1)
            pol = NeuroBHPolicy(agent, spec)
            for k in range(min(a.sessions, n_all)):
                log = run_episode(env, pol, spec, int(idx[k]), int(vix[k]), int(starts[k]))
                md = session_metrics(log, base, departure=deps[k])     # scored with the DEFAULT weights
                mo = session_metrics(log, spec, departure=deps[k])     # scored with its own weights
                rows.append({"dataset": ds, "variant": name, "smoothness": lvl, "epsilon": spec.weights.epsilon, "session": k,
                             "qoe_default": md["qoe"], "qoe_own": mo["qoe"], "vmaf": md["vmaf"],
                             "smooth": md["smooth"], "switch_rate": switch_rate(log),
                             "rebuf_ratio": md["rebuf_ratio"], "buffer": md["buffer"], "waste_MB": md["waste_MB"],
                             "intervention_rate": md["intervention_rate"]})
            x = pd.DataFrame([r for r in rows if r["dataset"] == ds and r["variant"] == name])
            print(f"[{ds}] {name:10s} (eps={spec.weights.epsilon:.1f})  QoE(default)={x.qoe_default.mean():6.2f} "
                  f"|dVMAF|={x.smooth.mean():5.2f} switch={x.switch_rate.mean():.3f} "
                  f"stall={x.rebuf_ratio.mean() * 100:5.2f}% buffer={x.buffer.mean():5.1f}s "
                  f"interv={x.intervention_rate.mean() * 100:4.1f}%", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{cfg['results']}/smooth_sweep.csv", index=False)
    summ = df.groupby(["dataset", "variant"]).mean(numeric_only=True).drop(columns=["session"])
    # baselines of exp_abr.py on the same sessions (default spec)
    refs = []
    for ds in df.dataset.unique():
        p = f"{cfg['results']}/abr_{ds}.csv"
        if os.path.exists(p):
            b = pd.read_csv(p)
            b = b[(b.session < a.sessions) & b.policy.isin(["RobustMPC", "FESTIVE", "Dynamic", "BBA", "Pensieve", "NeuroBH"])]
            r = b.groupby("policy")[["qoe", "smooth", "rebuf_ratio", "buffer"]].mean()
            r.insert(0, "dataset", ds); refs.append(r)
    summ.to_csv(f"{cfg['results']}/smooth_summary.csv")
    print(summ.round(3).to_string())
    if refs:
        print("\nbaselines on the same sessions (exp_abr.py):")
        print(pd.concat(refs).round(3).to_string())
    plot(df, cfg, pd.concat(refs) if refs else None)
    mark_run(cfg, "exp_smooth")


def plot(df, cfg, refs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    colors = {"slow": "#c0392b", "medium": "#2471a3", "fast": "#1e8449"}
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 1.75))
    for ds in df.dataset.unique():
        g = df[(df.dataset == ds) & df.variant.str.startswith("prio")].groupby("smoothness").mean(numeric_only=True)
        cst = df[(df.dataset == ds) & (df.variant == "constraint")].mean(numeric_only=True)
        for j, col in enumerate(["switch_rate", "smooth", "qoe_default"]):
            ax[j].plot([5.6], [cst[col]], marker="*", ms=7, color=colors.get(ds), ls="none")
        ax[0].plot(g.index, g.switch_rate, "-o", ms=2.5, color=colors.get(ds), label=ds.capitalize())
        ax[1].plot(g.index, g.smooth, "-o", ms=2.5, color=colors.get(ds))
        ax[2].plot(g.index, g.qoe_default, "-o", ms=2.5, color=colors.get(ds))
        if refs is not None and "RobustMPC" in refs[refs.dataset == ds].index:
            ax[2].axhline(refs[(refs.dataset == ds)].loc["RobustMPC", "qoe"], color=colors.get(ds), ls=":", lw=0.9)
    for j, lab in enumerate(["Switch rate", r"Avg. $|\Delta$VMAF$|$", "QoE (default weights)"]):
        ax[j].set_xlabel("Smoothness priority in the spec"); ax[j].set_ylabel(lab)
        ax[j].set_xticks(LEVELS + [5.6]); ax[j].set_xticklabels([str(l) for l in LEVELS] + ["C"]); ax[j].set_title(f"({'abc'[j]})", loc="left", fontsize=8)
    ax[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(f"{cfg['figures']}/fig_smooth.pdf", bbox_inches="tight")
    print(f"figure -> {cfg['figures']}/fig_smooth.pdf  (dotted: RobustMPC; star 'C': hard constraint)")


if __name__ == "__main__":
    main()
