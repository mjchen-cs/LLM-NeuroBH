"""Generates every figure (PDF) and LaTeX table used in the paper from results/.

Figures -> <cfg.figures>/  (default figs/)
Tables  -> results/*.tex   (copy-paste or \\input into the paper)

If a result was produced from synthetic traces or the mock LLM, the figure is
stamped "SMOKE TEST - NOT REAL DATA".
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_cfg

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.5, "axes.edgecolor": "#52514e",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

# fixed colour per entity (validated categorical palette, fixed order) + marker as 2nd encoding
STYLE = {
    "NeuroBH": ("#2a78d6", "o"), "NeuroBH (default)": ("#2a78d6", "o"),
    "LLM-NeuroBH": ("#eb6834", "*"), "RobustMPC": ("#1baf7a", "s"), "BOLA": ("#eda100", "^"),
    "BBA": ("#e87ba4", "v"), "Pensieve": ("#6250d6", "D"), "NeuroBH w/o NS": ("#008300", "P"),
    "NeuroBH (A2C)": ("#e34948", "X"), "LLM-NeuroBH w/o verifier": ("#e34948", "X"),
    "Oracle-NeuroBH": ("#8a8984", "h"), "Rate-based": ("#8a8984", "<"),
    "FESTIVE": ("#008300", ">"), "Dynamic": ("#e34948", "d"),   # w/o NS & A2C never share a figure with these
    "LLM-ABR (direct)": ("#52514e", "p"), "LLM-ABR (no delay)": ("#8a8984", "p"), "LLM-ABR (direct, intent)": ("#52514e", "p"),
}
INK2 = "#52514e"


# two-sided 95% Student-t quantiles for df = 1..30 (normal 1.96 beyond); avoids a scipy dependency
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
         2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
         2.045, 2.042]


def ci95(x):
    """Half-width of the 95% confidence interval of the mean (Student t)."""
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return 0.0
    t = _T975[n - 2] if n - 1 <= len(_T975) else 1.96
    return float(t * x.std(ddof=1) / np.sqrt(n))


def style(name):
    return STYLE.get(name, ("#8a8984", "o"))


def stamp(fig, cfg, key):
    p = os.path.join(cfg["results"], "RUN_INFO.json")
    info = json.load(open(p)) if os.path.exists(p) else {}
    r = info.get(key, {})
    if r.get("synthetic") or r.get("mock"):
        fig.text(0.5, 0.5, "SMOKE TEST - NOT REAL DATA", ha="center", va="center", fontsize=18,
                 color="#e34948", alpha=0.25, rotation=18, transform=fig.transFigure, zorder=100)


def save(fig, cfg, name):
    path = os.path.join(cfg["figures"], name)
    fig.savefig(path); plt.close(fig); print("figure ->", path)


def order(policies):
    pref = ["BBA", "BOLA", "FESTIVE", "Dynamic", "RobustMPC", "Pensieve", "LLM-ABR (direct)",
            "LLM-ABR (direct, intent)", "NeuroBH w/o NS", "NeuroBH (A2C)", "NeuroBH",
            "NeuroBH (default)", "LLM-NeuroBH w/o verifier", "LLM-NeuroBH", "Oracle-NeuroBH"]
    return [p for p in pref if p in policies] + [p for p in policies if p not in pref]


# ------------------------------------------------------------------ ABR (Figs 5-7)
def fig_abr(cfg):
    for ds in ["slow", "medium", "fast"]:
        p = f"{cfg['results']}/abr_{ds}.csv"
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        # ablations go to fig_ablation; the per-chunk LLM controllers are far off-scale on slow/medium
        # networks (QoE < -30) and are reported in the text instead
        df = df[~df.policy.isin(["NeuroBH w/o NS", "NeuroBH (A2C)", "LLM-ABR (no delay)", "LLM-ABR (direct)"])]
        g = df.groupby("policy").mean(numeric_only=True)
        e = df.groupby("policy")[g.columns.tolist()].agg(ci95)   # 95% CI half-widths over sessions
        fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.75))
        phis = [0.0, 0.1, 0.3, 0.6, 1.0]
        eb = dict(fmt="none", elinewidth=0.7, capsize=1.5, zorder=2)
        for pol in order(g.index):
            c, m = style(pol); r = g.loc[pol]; er = e.loc[pol]
            ax[0].errorbar(r.buffer, r.qoe, xerr=er.buffer, yerr=er.qoe, ecolor=c, **eb)
            ax[0].scatter(r.buffer, r.qoe, color=c, marker=m, s=28, label=pol, edgecolor="white", linewidth=0.6, zorder=3)
            ax[1].errorbar(r.rebuf_ratio * 100, r.bitrate / 1000, xerr=er.rebuf_ratio * 100, yerr=er.bitrate / 1000, ecolor=c, **eb)
            ax[1].scatter(r.rebuf_ratio * 100, r.bitrate / 1000, color=c, marker=m, s=28, edgecolor="white", linewidth=0.6, zorder=3)
            ax[2].errorbar(r.smooth, r.bitrate / 1000, xerr=er.smooth, yerr=er.bitrate / 1000, ecolor=c, **eb)
            ax[2].scatter(r.smooth, r.bitrate / 1000, color=c, marker=m, s=28, edgecolor="white", linewidth=0.6, zorder=3)
            ys = np.array([r[f"obj_phi{l}"] for l in range(1, 6)]); es = np.array([er[f"obj_phi{l}"] for l in range(1, 6)])
            ax[3].fill_between(phis, ys - es, ys + es, color=c, alpha=0.15, lw=0)
            ax[3].plot(phis, ys, color=c, marker=m, ms=3.5, lw=1.5)
        ax[0].set_xlabel("Avg. buffer (s)"); ax[0].set_ylabel("QoE (VMAF-based)")
        ax[1].set_xlabel("Stall ratio (%)"); ax[1].set_ylabel("Avg. bitrate (Mbps)")
        ax[2].set_xlabel("Avg. |ΔVMAF|"); ax[2].set_ylabel("Avg. bitrate (Mbps)")
        ax[3].set_xlabel(r"Buffer weight $\varphi$"); ax[3].set_ylabel("Objective (Eq. 3)")
        for k, a in enumerate(ax):
            a.set_title(f"({'abcd'[k]})", loc="left", fontsize=7.5)
        h, l = ax[0].get_legend_handles_labels()
        fig.legend(h, l, ncol=len(l), loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False, handletextpad=0.2)
        fig.tight_layout(w_pad=0.6)
        stamp(fig, cfg, "exp_abr"); save(fig, cfg, f"fig_abr_{ds}.pdf")

    # QoE CDF (medium) and ablation
    p = f"{cfg['results']}/abr_medium.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.9))
        for pol in order(df.policy.unique()):
            if pol in ("NeuroBH w/o NS", "NeuroBH (A2C)", "LLM-ABR (no delay)"):
                continue
            x = np.sort(df[df.policy == pol].qoe.values)
            c, _ = style(pol)
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=1.5, label=pol)
        ax.set_xlabel("Session QoE"); ax.set_ylabel("CDF"); ax.legend(frameon=False, ncol=2)
        stamp(fig, cfg, "exp_abr"); save(fig, cfg, "fig_qoe_cdf.pdf")

        allds = pd.concat([pd.read_csv(f"{cfg['results']}/abr_{d}.csv") for d in ["slow", "medium", "fast"]
                           if os.path.exists(f"{cfg['results']}/abr_{d}.csv")])
        abl = allds[allds.policy.isin(["NeuroBH", "NeuroBH w/o NS", "NeuroBH (A2C)", "Pensieve"])]
        if len(abl.policy.unique()) > 1:
            g = abl.groupby("policy").mean(numeric_only=True)
            pols = order(g.index)
            fig, ax = plt.subplots(1, 3, figsize=(7.16, 1.6))
            for k, (col, lab, sc) in enumerate([("qoe", "QoE (all networks)", 1), ("rebuf_ratio", "Stall ratio (%)", 100),
                                                 ("buffer", "Avg. buffer (s)", 1)]):
                vals = [g.loc[p_, col] * sc for p_ in pols]
                ax[k].barh(range(len(pols)), vals, color=[style(p_)[0] for p_ in pols], height=0.6)
                ax[k].set_yticks(range(len(pols))); ax[k].set_yticklabels(pols if k == 0 else [])
                ax[k].set_xlabel(lab); ax[k].grid(axis="y", visible=False)
                for i, v in enumerate(vals):
                    ax[k].text(v, i, f" {v:.2f}", va="center", fontsize=6, color=INK2)
            fig.tight_layout(w_pad=0.5)
            stamp(fig, cfg, "exp_abr"); save(fig, cfg, "fig_ablation.pdf")


def fig_train(cfg):
    curves = {"PPO": cfg["ckpt"]["neurobh"], "A2C": cfg["ckpt"]["neurobh_a2c"],
              "PPO w/o NS": cfg["ckpt"]["neurobh_noshield"]}
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 1.7)); found = False
    for k, (lab, ck) in enumerate(curves.items()):
        p = ck.replace(".pt", "_train.csv")
        if not os.path.exists(p):
            continue
        found = True
        d = pd.read_csv(p)
        col = ["#2a78d6", "#e34948", "#008300"][k]
        ax[0].plot(d.step / 1e3, d.episode_qoe, color=col, lw=1.4, label=lab)
        ax[1].plot(d.step / 1e3, d.intervention_rate * 100, color=col, lw=1.4, label=lab)
    if not found:
        plt.close(fig); return
    ax[0].set_xlabel("Training steps (k)"); ax[0].set_ylabel("Episode QoE"); ax[0].legend(frameon=False)
    ax[1].set_xlabel("Training steps (k)"); ax[1].set_ylabel("Shield interventions (%)")
    fig.tight_layout(); stamp(fig, cfg, "exp_abr"); save(fig, cfg, "fig_train.pdf")


# ------------------------------------------------------------------ LLM quality
def fig_llm(cfg):
    p = f"{cfg['results']}/llm_summary.csv"
    if not os.path.exists(p):
        return
    s = pd.read_csv(p)
    full = s[s.variant == "full"].set_index("model")
    metrics = [("pass_first", "Verified@1"), ("verified_final", "Verified@K"), ("prio_pm1", "Priority ±1"),
               ("profile_ok", "Net. profile"), ("sem_exact", "Constraint exact"), ("conflict_detected", "Conflict det.")]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.0), gridspec_kw={"width_ratios": [1.6, 1]})
    models = list(full.index); n = len(metrics); w = 0.8 / n
    for j, (m, lab) in enumerate(metrics):
        ax[0].bar(np.arange(len(models)) + (j - n / 2 + 0.5) * w, full[m].values * 100, w * 0.92,
                  color=cols[j], label=lab)
    ax[0].set_xticks(range(len(models))); ax[0].set_xticklabels(models)
    ax[0].set_ylabel("%"); ax[0].set_ylim(0, 105); ax[0].grid(axis="x", visible=False)
    ax[0].legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    ax[0].set_xlabel("(a) Spec quality of the full pipeline")
    # first-reply failure stages
    runs = f"{cfg['results']}/llm_runs.jsonl"
    if os.path.exists(runs):
        df = pd.DataFrame([json.loads(l) for l in open(runs, encoding="utf-8")])
        df = df[(df.variant == "full") & (~df.conflict)]
        stages = ["passed", "json", "schema", "syntax", "range", "consistency", "realizability"]
        sc = ["#1baf7a", "#e34948", "#eda100", "#eb6834", "#e87ba4", "#6250d6", "#2a78d6"]
        bottom = np.zeros(len(models))
        for st, c in zip(stages, sc):
            v = np.array([(df[df.model == m].first_stage == st).mean() * 100 for m in models])
            ax[1].bar(range(len(models)), v, 0.6, bottom=bottom, color=c, label=st, edgecolor="white", linewidth=0.8)
            bottom += v
        ax[1].set_xticks(range(len(models))); ax[1].set_xticklabels(models)
        ax[1].set_ylabel("First reply (%)"); ax[1].grid(axis="x", visible=False)
        ax[1].legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28), fontsize=6)
        ax[1].set_xlabel("(b) Verifier stage of the first reply")
    fig.tight_layout()
    stamp(fig, cfg, "exp_llm"); save(fig, cfg, "fig_llm_quality.pdf")

    # ablation table
    lab = {"full": "Full (few-shot + verifier + repair)", "no_repair": "w/o repair loop",
           "no_verifier": "w/o verifier", "zero_shot": "w/o few-shot examples"}
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Model & Variant & Verified@K & Deployed safe & Priority $\pm1$ & Constr. exact & Conflict res. & Rounds \\",
             r"\midrule"]
    for _, r in s.iterrows():
        lines.append(f"{r.model} & {lab.get(r.variant, r.variant)} & {r.verified_final * 100:.1f} & "
                     f"{r.deployed_safe * 100:.1f} & {r.prio_pm1 * 100:.1f} & {r.sem_exact * 100:.1f} & "
                     f"{r.conflict_resolved * 100:.1f} & {r.avg_rounds:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{cfg['results']}/tab_llm_ablation.tex", "w").write("\n".join(lines))


# ------------------------------------------------------------------ end-to-end
def fig_e2e(cfg):
    p = f"{cfg['results']}/e2e.csv"
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    # objective normalised by the oracle per intent
    o = df[df.policy == "Oracle-NeuroBH"].groupby("id").objective.mean()
    df["obj_norm"] = df.apply(lambda r: r.objective / o[r.id] if o[r.id] > 0 else np.nan, axis=1)
    g = df.groupby("policy").mean(numeric_only=True)
    # normalised objective = mean objective / mean oracle objective (robust to single bad sessions)
    g["obj_norm"] = df.groupby("policy").objective.mean() / df[df.policy == "Oracle-NeuroBH"].objective.mean()
    pols = order(g.index)
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 1.9))
    for k, (col, lab, sc) in enumerate([("obj_norm", "Intent-aligned objective\n(norm. to oracle)", 1),
                                         ("violation_rate", "Constraint violation (%)", 100),
                                         ("waste_MB", "Wasted data per session (MB)", 1)]):
        vals = [g.loc[p_, col] * sc for p_ in pols]
        # intent-unaware baselines share one neutral colour; NeuroBH variants keep their identity colour
        cols = [style(p_)[0] if "NeuroBH" in p_ else "#b4b3ad" for p_ in pols]
        ax[k].barh(range(len(pols)), [max(v, 0) for v in vals], color=cols, height=0.62)
        ax[k].set_yticks(range(len(pols))); ax[k].set_yticklabels(pols if k == 0 else [])
        ax[k].set_xlabel(lab); ax[k].grid(axis="y", visible=False); ax[k].invert_yaxis()
        if k == 0:
            ax[k].set_xlim(0, 1.25)
        for i, v in enumerate(vals):
            ax[k].text(max(v, 0), i, f" {v:.2f}" + (" (<0)" if v < 0 else ""), va="center", fontsize=5.8, color=INK2)
        ax[k].set_title(f"({'abc'[k]})", loc="left")
    fig.tight_layout(w_pad=0.4)
    stamp(fig, cfg, "exp_e2e"); save(fig, cfg, "fig_e2e.pdf")

    # per-scenario objective of LLM-NeuroBH vs default NeuroBH vs best baseline
    sc = df[df.policy.isin(["NeuroBH (default)", "LLM-NeuroBH", "RobustMPC"])]
    t = sc.groupby(["scenario", "policy"]).obj_norm.mean().unstack()
    fig, ax = plt.subplots(1, 1, figsize=(7.16, 1.7))
    x = np.arange(len(t.index)); w = 0.26
    for j, pol in enumerate(["RobustMPC", "NeuroBH (default)", "LLM-NeuroBH"]):
        if pol in t:
            ax.bar(x + (j - 1) * w, t[pol].values, w * 0.92, color=style(pol)[0], label=pol)
    ax.set_xticks(x); ax.set_xticklabels([s.replace("_", "\n") for s in t.index], fontsize=6)
    ax.set_ylabel("Norm. objective"); ax.legend(frameon=False, ncol=3); ax.grid(axis="x", visible=False)
    stamp(fig, cfg, "exp_e2e"); save(fig, cfg, "fig_e2e_scenario.pdf")


# ------------------------------------------------------------------ tables
def tables(cfg):
    p = f"{cfg['results']}/tab_prediction.csv"
    if os.path.exists(p):
        d = pd.read_csv(p)
        lines = [r"\begin{tabular}{llrrr}", r"\toprule", r"Dataset & Method & Mean (\%) $\downarrow$ & Median (\%) $\downarrow$ & 75th (\%) $\downarrow$ \\", r"\midrule"]
        for _, r in d.iterrows():
            lines.append(f"{r.dataset} & {r.method} & {r['mean']:.2f} & {r['median']:.2f} & {r.p75:.2f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(f"{cfg['results']}/tab_prediction.tex", "w").write("\n".join(lines))
    p = f"{cfg['results']}/overhead.csv"
    if os.path.exists(p):
        d = pd.read_csv(p)
        lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Component & Mean (ms) & 95th (ms) \\", r"\midrule"]
        for _, r in d.iterrows():
            lines.append(f"{r.component} & {r.mean_ms:.2f} & {r.p95_ms:.2f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(f"{cfg['results']}/tab_overhead.tex", "w").write("\n".join(lines))


if __name__ == "__main__":
    cfg = load_cfg()
    fig_abr(cfg); fig_train(cfg); fig_llm(cfg); fig_e2e(cfg); tables(cfg)
