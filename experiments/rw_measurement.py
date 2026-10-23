"""Real-world (testbed) bandwidth measurement and prediction error -> paper Tables 1 and 2.

Inputs
  data/realworld_chunks.csv   written by deploy/abr_server.py (one row per received segment)
  data/server_log.csv         copy of the media server's --log file (has gt_kbps when the
                              server ran with tools/tc_replay.py + --gt-log)
Joined on the request id X-NeuroBH-Req.

Outputs (results/)
  tab_rw_measurement.csv   per policy: error of the sample the policy actually used
                           (client chunk throughput for baselines, server sample for NeuroBH),
                           plus client-vs-server for all segments
  tab_rw_prediction.csv    prediction error of Sliding window / Harmonic mean / EWMA / GRU hybrid,
                           all fed with the server samples, target = ground truth of the next segment
  tab_rw_measurement.tex / tab_rw_prediction.tex
Usage:  python experiments/rw_measurement.py [--chunks data/realworld_chunks.csv] [--server data/server_log.csv]
"""
import argparse
import numpy as np
import pandas as pd
from common import load_cfg, C
from neurobh.predictor import make as make_pred

NAMES = {"neurobh": "NeuroBH", "llm-neurobh": "LLM-NeuroBH", "pensieve": "Pensieve", "bba": "BBA",
         "bola": "BOLA", "festive": "FESTIVE", "dynamic": "Dynamic", "robustmpc": "RobustMPC"}


def err(pred, true):
    true = np.maximum(np.asarray(true, float), C.BITRATES_KBPS[0])
    e = np.abs(np.asarray(pred, float) - true) / true * 100
    return {"n": len(e), "mean": e.mean(), "median": np.median(e), "p75": np.percentile(e, 75)}


def to_tex(df, cols, path):
    with open(path, "w") as f:
        f.write("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\nMethod & Mean (\\%) & Median (\\%) & 75th (\\%)\\\\\n\\midrule\n")
        for _, r in df.iterrows():
            f.write(f"{r['method']} & " + " & ".join(f"{r[c]:.2f}" for c in cols) + "\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/realworld_chunks.csv")
    ap.add_argument("--server", default="data/server_log.csv")
    ap.add_argument("--predictor", default="ckpt/predictor.pt")
    a = ap.parse_args()
    cfg = load_cfg()
    ch = pd.read_csv(a.chunks)
    # the log may mix rows of older media_server.py versions (13 columns) and newer ones (14 columns)
    cols = ["req", "client", "path", "level", "bytes", "t_start", "t_end", "server_kbps", "goodput_kbps",
            "gt_kbps", "cc", "min_rtt_ms", "retrans", "delivery_rate_kbps"]
    sv = pd.read_csv(a.server, header=None, names=cols, skiprows=1, engine="python")
    sv = sv[sv.req != "req"]
    for k in ["server_kbps", "goodput_kbps", "gt_kbps"]:
        sv[k] = pd.to_numeric(sv[k], errors="coerce")
    ch = ch[ch.network.astype(str).str.startswith("emu-")] if ch.network.astype(str).str.startswith("emu-").any() else ch
    sv = sv.rename(columns={"server_kbps": "server_kbps_log"})
    d = ch.merge(sv[["req", "gt_kbps", "server_kbps_log", "goodput_kbps", "cc"]], on="req", how="left")
    d["server_kbps"] = d["server_kbps"].fillna(d["server_kbps_log"])     # baselines did not query /bw
    g = d.dropna(subset=["gt_kbps"])
    print(f"{len(d)} segments, {len(g)} with ground truth "
          f"({d.req.isna().sum()} without request id: is the player using media_server.py?)")
    if not len(g):
        raise SystemExit("no ground truth: run the media server with --gt-log and tools/tc_replay.py")

    import os
    models = [make_pred("sliding"), make_pred("harmonic"), make_pred("ewma")]
    if os.path.exists(a.predictor):
        models.append(make_pred("gru", path=a.predictor))

    def measurement(g):
        rows = []
        for pol, x in g.groupby("policy"):
            rows.append({"method": NAMES.get(pol, pol), "kind": "sample used by the policy",
                         **err(x["sample_used_kbps"], x.gt_kbps)})
        rows.append({"method": "client chunk throughput (all)", "kind": "all segments", **err(g.client_kbps, g.gt_kbps)})
        gs = g.dropna(subset=["server_kbps"])
        rows.append({"method": "server stable connection (all)", "kind": "all segments", **err(gs.server_kbps, gs.gt_kbps)})
        return pd.DataFrame(rows)

    def prediction(d):
        # per session: server samples -> predict the ground truth of the next segment
        preds = {mm.name: [] for mm in models}
        true = []
        for _, x in d.sort_values("time").groupby("session"):
            s_ = x["server_kbps"].fillna(x["client_kbps"]).to_numpy(float)
            gt = x["gt_kbps"].to_numpy(float)
            for t in range(3, len(s_)):
                if np.isnan(gt[t]):
                    continue
                for mm in models:
                    preds[mm.name].append(mm.predict(list(s_[:t])))
                true.append(gt[t])
        return pd.DataFrame([{"method": k, **err(v, true)} for k, v in preds.items()])

    m = measurement(g); p = prediction(d)
    m.to_csv(f"{cfg['results']}/tab_rw_measurement.csv", index=False)
    to_tex(m, ["mean", "median", "p75"], f"{cfg['results']}/tab_rw_measurement.tex")
    p.to_csv(f"{cfg['results']}/tab_rw_prediction.csv", index=False)
    to_tex(p, ["mean", "median", "p75"], f"{cfg['results']}/tab_rw_prediction.tex")
    print("\nmeasurement error (%), all networks:\n", m.round(2).to_string(index=False))
    print("\nprediction error (%), all networks:\n", p.round(2).to_string(index=False))
    per = []
    for net in sorted(d.network.dropna().unique()):
        mm_ = measurement(g[g.network == net]); mm_.insert(0, "network", net)
        pp_ = prediction(d[d.network == net]); pp_.insert(0, "network", net); pp_["kind"] = "prediction"
        per += [mm_, pp_]
    per = pd.concat(per, ignore_index=True)
    per.to_csv(f"{cfg['results']}/tab_rw_per_network.csv", index=False)
    print("\nper network (median / 75th %):")
    print(per.pivot_table(index=["kind", "method"], columns="network", values="median").round(2).to_string())
    gs = g.dropna(subset=["server_kbps"])

    if "cc" in d and d.cc.notna().any():          # Table: congestion control (run sessions per --cc)
        print("\nper congestion control (server sample error):")
        for cc, x in gs.groupby("cc"):
            print(f"  {cc}: {err(x.server_kbps, x.gt_kbps)}")


if __name__ == "__main__":
    main()
