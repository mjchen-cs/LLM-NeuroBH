"""Sensitivity of NeuroBH to the shield's worst-case bandwidth bound eta (reviewer question:
"why is NeuroBH conservative on slow networks?").

The SAME trained policy is evaluated with the shield's eta in {0, 0.1, ..., 0.5}, without the
shield, and with the eta of the matching network profile (what the LLM-selected profile would
give).  No retraining: eta is a deployment-time parameter of the symbolic layer, so this is a
pure sensitivity analysis.  Sessions are the first N sessions of experiments/exp_abr.py (same
random draws), so the numbers are directly comparable with Fig. abr.

Output: results/eta_sweep.csv (one row per setting x session), results/eta_summary.csv,
        <figures>/fig_eta.pdf
Usage:  python experiments/exp_eta.py                 (100 sessions per network class)
        python experiments/exp_eta.py --sessions 30   (quick run)
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
from neurobh.spec import default_spec

ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=100)
    ap.add_argument("--datasets", default="slow,medium,fast", help="same order as exp_abr.py")
    a = ap.parse_args()
    cfg = load_cfg()
    agent = Agent.load(cfg["ckpt"]["neurobh"])
    vids = videos_for(cfg)
    spec = default_spec()
    n_all = cfg["sessions_per_dataset"]                  # reproduce exp_abr.py's random draws
    rng = np.random.default_rng(0)
    print(f"default spec: profile={spec.network_profile}, eta={spec.profile['eta']:.2f}; "
          f"profile etas: " + ", ".join(f"{k}={v['eta']:.2f}" for k, v in C.NETWORK_PROFILES.items()))
    rows = []
    for ds in a.datasets.split(","):
        traces = traces_for(cfg, ds)
        idx = rng.integers(len(traces), size=n_all)
        starts = rng.integers(0, 10 ** 6, size=n_all)
        vix = rng.integers(len(vids), size=n_all)
        deps = sample_departures(n_all, rng)
        settings = [(f"{e:.1f}", e, True) for e in ETAS]
        settings += [("profile", C.NETWORK_PROFILES[ds]["eta"], True), ("no shield", None, False)]
        for label, eta, use_shield in settings:
            pred = make_pred("gru", path=cfg["ckpt"]["predictor"]) if os.path.exists(cfg["ckpt"]["predictor"]) \
                else make_pred("harmonic")
            env = StreamingEnv(traces, vids, measure="stable", predictor=pred, seed=1)
            pol = NeuroBHPolicy(agent, spec, use_shield=use_shield, eta=eta)
            for k in range(min(a.sessions, n_all)):
                log = run_episode(env, pol, spec, int(idx[k]), int(vix[k]), int(starts[k]))
                m = session_metrics(log, spec, departure=deps[k])
                sw = np.mean([abs(log[i]["level"] - log[i - 1]["level"]) > 0 for i in range(1, len(log))])
                rows.append({"dataset": ds, "setting": label, "eta": eta if eta is not None else np.nan,
                             "session": k, "switch_rate": sw, **m})
            x = [r for r in rows if r["dataset"] == ds and r["setting"] == label]
            print(f"[{ds}] eta={label:9s} QoE={np.mean([r['qoe'] for r in x]):6.2f} "
                  f"stall={np.mean([r['rebuf_ratio'] for r in x]) * 100:5.2f}% "
                  f"buffer={np.mean([r['buffer'] for r in x]):5.1f}s "
                  f"waste={np.mean([r['waste_MB'] for r in x]):5.2f}MB "
                  f"switch={np.mean([r['switch_rate'] for r in x]):.3f} "
                  f"interv={np.mean([r['intervention_rate'] for r in x]) * 100:5.1f}%", flush=True)
    out = f"{cfg['results']}/eta_sweep.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    df = pd.DataFrame(rows)
    summ = df.groupby(["dataset", "setting"])[["eta", "qoe", "rebuf_ratio", "buffer", "waste_MB", "smooth",
                                               "switch_rate", "intervention_rate", "objective"]].mean()
    summ.to_csv(f"{cfg['results']}/eta_summary.csv")
    print(summ.round(3).to_string())
    plot(df, cfg)
    mark_run(cfg, "exp_eta")


def plot(df, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    colors = {"slow": "#c0392b", "medium": "#2471a3", "fast": "#1e8449"}
    names = {"slow": "Slow", "medium": "Medium", "fast": "Fast"}
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.75))
    panels = [("qoe", "QoE"), ("rebuf_ratio", "Stall ratio (%)"), ("buffer", "Avg. buffer (s)"),
              ("intervention_rate", "Shield interventions (%)")]
    sweep = df[df.setting.isin([f"{e:.1f}" for e in ETAS])]
    for j, (col, lab) in enumerate(panels):
        scale = 100 if col in ("rebuf_ratio", "intervention_rate") else 1
        for ds in df.dataset.unique():
            g = sweep[sweep.dataset == ds].groupby("eta")[col].mean() * scale
            ax[j].plot(g.index, g.values, "-o", ms=2.5, lw=1.2, color=colors.get(ds), label=names.get(ds, ds))
            prof = df[(df.dataset == ds) & (df.setting == "profile")]
            if len(prof):
                ax[j].plot([prof.eta.iloc[0]], [prof[col].mean() * scale], marker="*", ms=7,
                           color=colors.get(ds), ls="none")
            ns = df[(df.dataset == ds) & (df.setting == "no shield")]
            if len(ns) and col != "intervention_rate":
                ax[j].axhline(ns[col].mean() * scale, color=colors.get(ds), ls=":", lw=0.9)
        ax[j].set_xlabel(r"$\eta$"); ax[j].set_ylabel(lab)
        ax[j].set_title(f"({'abcd'[j]})", loc="left", fontsize=8)
    ax[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    os.makedirs(cfg["figures"], exist_ok=True)
    fig.savefig(f"{cfg['figures']}/fig_eta.pdf", bbox_inches="tight")
    print(f"figure -> {cfg['figures']}/fig_eta.pdf  (star: eta of the matching network profile; "
          f"dotted: same policy without shield)")


if __name__ == "__main__":
    main()
