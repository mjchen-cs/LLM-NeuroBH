"""Convert Puffer raw logs (video_sent + video_acked CSVs) into Pensieve-style traces.

For every chunk:  throughput = size / (t_acked - t_sent).  Each streaming
session becomes one trace "<time_s> <Mbps>" (time = send time relative to the
first chunk).  Output: data/traces/puffer.zip (read directly by the code).

Usage (PowerShell, from the repository root):
    python tools/convert_puffer.py --raw data/puffer_raw --max-sessions 3000
where data/puffer_raw contains one or more day folders / files such as
    video_sent_2019-01-26T11_2019-01-27T11.csv
    video_acked_2019-01-26T11_2019-01-27T11.csv
Files are streamed in chunks, so multi-GB CSVs are fine.
"""
import argparse
import glob
import re
import io
import os
import zipfile

import numpy as np
import pandas as pd


def _time_to_seconds(x):
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmedian(x)
    if m > 1e17: return x / 1e9          # ns
    if m > 1e14: return x / 1e6          # us
    if m > 1e11: return x / 1e3          # ms
    return x                             # s


def _norm(c):
    """'time (ns GMT)' -> 'time', strip spaces."""
    c = c.strip()
    return "time" if c.lower().startswith("time") else c


def _read(path, cols, chunksize=2_000_000):
    head = pd.read_csv(path, nrows=5)
    mapping = {c: _norm(c) for c in head.columns}
    use = [c for c in head.columns if mapping[c] in cols]
    missing = set(cols) - {mapping[c] for c in use}
    if missing:
        raise SystemExit(f"[puffer] {os.path.basename(path)}: columns not found {sorted(missing)}; "
                         f"available {list(head.columns)}  -> please send this line to the author")
    for part in pd.read_csv(path, usecols=use, chunksize=chunksize):
        yield part.rename(columns=mapping)


def convert_pair(sent_path, acked_path, keys):
    print(f"[puffer] reading {os.path.basename(acked_path)}")
    acked = pd.concat(list(_read(acked_path, ["time"] + keys)), ignore_index=True)
    acked = acked.rename(columns={"time": "t_ack"}).drop_duplicates(keys)
    out = []
    print(f"[puffer] reading {os.path.basename(sent_path)} (streamed)")
    for part in _read(sent_path, ["time", "size"] + keys):
        part = part.rename(columns={"time": "t_sent"})
        out.append(part.merge(acked, on=keys, how="inner"))
    df = pd.concat(out, ignore_index=True)
    df["t_sent"] = _time_to_seconds(df["t_sent"]); df["t_ack"] = _time_to_seconds(df["t_ack"])
    dt = df["t_ack"] - df["t_sent"]
    df = df[(dt > 1e-3) & (df["size"] > 0)].copy()
    df["mbps"] = df["size"] * 8 / 1e6 / (df["t_ack"] - df["t_sent"])
    df = df[(df["mbps"] > 0.01) & (df["mbps"] < 2000)]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/puffer_raw")
    ap.add_argument("--out", default="data/traces/puffer.zip")
    ap.add_argument("--min-chunks", type=int, default=60, help="drop sessions shorter than this (~2 min)")
    ap.add_argument("--max-sessions", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sents = sorted(glob.glob(os.path.join(a.raw, "**", "*video_sent*.csv"), recursive=True))
    if not sents:
        raise SystemExit(f"no *video_sent*.csv under {a.raw}")
    rng = np.random.default_rng(a.seed)
    traces = []
    for sp in sents:
        d, b = os.path.split(sp)
        ap_ = os.path.join(d, b.replace("video_sent", "video_acked"))
        if not os.path.exists(ap_):
            print(f"[puffer] skip {sp}: matching video_acked file not found"); continue
        head = [_norm(c) for c in pd.read_csv(sp, nrows=5).columns]
        print(f"[puffer] columns: {head}")
        keys = [k for k in ["session_id", "index", "expt_id", "channel", "video_ts"] if k in head]
        df = convert_pair(sp, ap_, keys)
        sess_keys = [k for k in ["session_id", "index", "expt_id"] if k in df.columns]
        for sid, g in df.groupby(sess_keys):
            if len(g) < a.min_chunks:
                continue
            g = g.sort_values("t_sent")
            t = g["t_sent"].values - g["t_sent"].values[0]
            keep = np.concatenate([[True], np.diff(t) > 0])
            day = re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(sp))
            traces.append((f"puffer_{day.group(0) if day else 'x'}_{len(traces)}", t[keep], g["mbps"].values[keep]))
        print(f"[puffer] {len(traces)} sessions so far")
    if len(traces) > a.max_sessions:
        idx = rng.choice(len(traces), a.max_sessions, replace=False)
        traces = [traces[i] for i in sorted(idx)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, t, b in traces:
            buf = io.StringIO()
            for x, y in zip(t, b):
                buf.write(f"{x:.3f}\t{y:.6f}\n")
            z.writestr(name, buf.getvalue())
    m = np.array([b.mean() for _, _, b in traces])
    print(f"[puffer] wrote {len(traces)} traces to {a.out}; mean Mbps p5/p50/p95 = {np.percentile(m, [5, 50, 95]).round(2)}")


if __name__ == "__main__":
    main()
