"""Real-world testbed results (paper RQ4): figure + summary tables.

Inputs : data/realworld.csv, data/realworld_chunks.csv  (deploy/abr_server.py)
Outputs: results/rw_summary.csv          per network x policy means
         results/rw_sessions.csv         per session metrics (incl. VMAF-QoE)
         <figures>/fig_rw.pdf            QoE vs. buffer, one panel per network
Per-chunk quality uses the same rate-quality model as the simulator; VMAF-QoE follows Eq. (qoe)
with the default weights (alpha = 1, epsilon = 1, beta = 50 VMAF points per stall second).
"Unplayed data" = data in the buffer when the 180 s session ends (proxy for wasted data).
"""
import os
import numpy as np
import pandas as pd
from common import load_cfg
from plot_all import style, ci95

NAMES = {"neurobh": "NeuroBH", "llm-neurobh": "LLM-NeuroBH", "pensieve": "Pensieve", "bba": "BBA",
         "bola": "BOLA", "festive": "FESTIVE", "dynamic": "Dynamic", "robustmpc": "RobustMPC",
         "neurobh-smooth": "NeuroBH + smooth intent"}
NETS = [("emu-3g", "3G (HSDPA)"), ("emu-fcc", "Broadband (FCC)"), ("emu-oboe", "Oboe, ToS video"),
        ("emu-4g", "4G (Lumos)")]
LADDER = np.array([300, 750, 1200, 1850, 2850, 4300])


def vmaf(b):
    return np.clip(-89.3 + 21.8 * np.log(b), 0, 100)


def sessions(cfg):
    d = pd.read_csv("data/realworld.csv")
    c = pd.read_csv("data/realworld_chunks.csv")
    d = d[d.network.astype(str).str.startswith("emu-")]
    rows = []
    for (net, pol, ses), x in c.sort_values("time").groupby(["network", "policy", "session"]):
        r = d[d.session == ses]
        if not len(r):
            continue
        lv = x["level"].to_numpy(int)
        v = vmaf(LADDER[lv])
        rb = float(r.rebuf_ratio.iloc[0]); stall = rb / max(1 - rb, 1e-6) * 180
        sm = float(np.abs(np.diff(v)).mean()) if len(v) > 1 else 0.0
        k = int(np.ceil(x["buffer"].iloc[-1] / 4))
        rows.append({"network": net, "policy": NAMES.get(pol, pol), "session": ses,
                     "qoe_vmaf": v.mean() - sm - 50 * stall / len(v), "vmaf": v.mean(), "smooth": sm,
                     "stall_s": stall, "rebuf_ratio": rb, "buffer": x["buffer"].mean(),
                     "unplayed_MB": float((LADDER[lv[-k:]] * 4 / 8 / 1000).sum()) if k > 0 else 0.0,
                     "switch_rate": float(np.mean(np.diff(lv) != 0)) if len(lv) > 1 else 0.0,
                     "intervention_rate": x["intervened"].mean(), "bitrate_mbps": float(r.bitrate_mbps.iloc[0]),
                     "intent": r.intent.iloc[0] if "intent" in r else ""})
    return pd.DataFrame(rows)


def main():
    cfg = load_cfg()
    s = sessions(cfg)
    s.to_csv(f"{cfg['results']}/rw_sessions.csv", index=False)
    summ = s.groupby(["network", "policy"])[["qoe_vmaf", "vmaf", "smooth", "stall_s", "buffer", "unplayed_MB",
                                             "switch_rate", "intervention_rate"]].agg(["mean", "count"])
    m = s.groupby(["network", "policy"]).mean(numeric_only=True)
    m["n"] = s.groupby(["network", "policy"]).size()
    # 95% CIs.  Every round replays a different trace to all policies (paired / blocked design), so most
    # variance is between traces.  The error bars are within-round CIs (Cousineau-Morey): each value is
    # centred on its round mean over the 7 default-spec policies before the CI is computed, and the CI is scaled
    # by sqrt(J/(J-1)).  Raw (between-session) CIs are also stored (*_ci95_raw).
    s["t"] = s.session.str.split("-").str[-1].astype(float)
    s = s.sort_values("t")
    s["round"] = s.groupby(["network", "policy"]).cumcount()
    # LLM-NeuroBH is excluded from the round means: its intent (and thus its target) changes every round
    main = [NAMES[k] for k in ["neurobh", "robustmpc", "pensieve", "bba", "bola", "festive", "dynamic"]]
    J = len(main)
    for col in ["qoe_vmaf", "buffer", "stall_s", "unplayed_MB"]:
        base = s[s.policy.isin(main)].groupby(["network", "round"])[col].mean().rename("_rm")
        grand = s[s.policy.isin(main)].groupby("network")[col].mean().rename("_gm")
        x_ = s.join(base, on=["network", "round"]).join(grand, on="network")
        s[col + "_norm"] = x_[col] - x_["_rm"] + x_["_gm"]
        m[col + "_ci95"] = s.groupby(["network", "policy"])[col + "_norm"].agg(ci95) * np.sqrt(J / (J - 1))
        m[col + "_ci95_raw"] = s.groupby(["network", "policy"])[col].agg(ci95)
    m.to_csv(f"{cfg['results']}/rw_summary.csv")
    pd.set_option("display.width", 220)
    print(m.round(2).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.95))
    pols = ["BBA", "BOLA", "FESTIVE", "Dynamic", "RobustMPC", "Pensieve", "NeuroBH", "NeuroBH + smooth intent"]
    for j, (net, title) in enumerate(NETS):
        g = m.loc[net] if net in m.index.get_level_values(0) else None
        if g is None:
            continue
        for p in pols:
            if p not in g.index:
                continue
            col, mk = style("NeuroBH" if p.startswith("NeuroBH") else p)
            hollow = p == "NeuroBH + smooth intent"
            ax[j].errorbar(g.loc[p, "buffer"], g.loc[p, "qoe_vmaf"], xerr=g.loc[p, "buffer_ci95"],
                           yerr=g.loc[p, "qoe_vmaf_ci95"], fmt="none", ecolor=col, elinewidth=0.7, capsize=1.5,
                           zorder=2)
            ax[j].scatter(g.loc[p, "buffer"], g.loc[p, "qoe_vmaf"], s=30 if not hollow else 46,
                          marker="*" if hollow else mk, facecolors="none" if hollow else col,
                          edgecolors=col, linewidths=1.0, zorder=3, label=p if j == 0 else None)
        ax[j].set_title(f"({'abcd'[j]}) {title}", loc="left", fontsize=7.5)
        ax[j].set_xlabel("Avg. buffer (s)")
        if j == 0:
            ax[j].set_ylabel("VMAF-QoE")
    h, l = ax[0].get_legend_handles_labels()
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.86))
    fig.legend(h, l, loc="upper center", ncol=8, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.2, columnspacing=0.8)
    os.makedirs(cfg["figures"], exist_ok=True)
    fig.savefig(f"{cfg['figures']}/fig_rw.pdf", bbox_inches="tight")
    print(f"figure -> {cfg['figures']}/fig_rw.pdf")


if __name__ == "__main__":
    main()
