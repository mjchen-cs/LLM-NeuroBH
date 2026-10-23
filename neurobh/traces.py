"""Network trace loading.

Supported format (Pensieve / Puffer-converted / FCC / HSDPA / Lumos5G after
conversion): one text file per trace, each line ``<time_s> <bandwidth_Mbps>``.
Put each dataset into its own folder, e.g. data/traces/hsdpa/*.txt.

If you do not have the datasets locally, ``synthetic_traces`` produces Markov
traces for smoke tests ONLY -- never report numbers obtained from them.
"""
import os
import glob
import numpy as np


class Trace:
    def __init__(self, name, times, bw_kbps):
        self.name = name
        self.times = np.asarray(times, dtype=np.float64)
        self.bw = np.asarray(bw_kbps, dtype=np.float64)
        # make times start at 0 and strictly increasing
        self.times = self.times - self.times[0]
        if len(self.times) > 1:
            dt = np.diff(self.times)
            if np.any(dt <= 0):
                self.times = np.arange(len(self.bw), dtype=np.float64)

    @property
    def duration(self):
        return float(self.times[-1]) if len(self.times) > 1 else 1.0

    def mean_kbps(self):
        return float(self.bw.mean())


def load_trace_file(path, unit="mbps"):
    t, b = [], []
    with open(path) as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                t.append(float(parts[0])); b.append(float(parts[1]))
            except ValueError:
                continue
    if len(b) < 5:
        return None
    b = np.asarray(b)
    if unit == "mbps":
        b = b * 1000.0
    b = np.maximum(b, 10.0)  # avoid zero bandwidth (stall forever)
    return Trace(os.path.basename(path), t, b)


def _load_trace_zip(zpath, unit="mbps", max_traces=None):
    """Read traces directly from <folder>.zip (no need to unzip)."""
    import io, zipfile
    traces = []
    with zipfile.ZipFile(zpath) as z:
        for name in sorted(n for n in z.namelist() if not n.endswith("/")):
            t, b = [], []
            for line in io.TextIOWrapper(z.open(name), encoding="utf-8", errors="ignore"):
                parts = line.replace(",", " ").split()
                if len(parts) >= 2:
                    try:
                        t.append(float(parts[0])); b.append(float(parts[1]))
                    except ValueError:
                        pass
            if len(b) >= 5:
                b = np.maximum(np.asarray(b) * (1000.0 if unit == "mbps" else 1.0), 10.0)
                traces.append(Trace(os.path.basename(name), t, b))
            if max_traces and len(traces) >= max_traces:
                break
    return traces


def trace_source_exists(folder):
    return bool(folder) and (os.path.isdir(folder) or os.path.exists(folder.rstrip("/\\") + ".zip"))


def load_trace_dir(folder, unit="mbps", max_traces=None):
    if not os.path.isdir(folder) and os.path.exists(folder.rstrip("/\\") + ".zip"):
        traces = _load_trace_zip(folder.rstrip("/\\") + ".zip", unit, max_traces)
        if not traces:
            raise FileNotFoundError(f"no traces found in {folder}.zip")
        return traces
    files = sorted(glob.glob(os.path.join(folder, "*")))
    traces = []
    for fp in files:
        if os.path.isdir(fp):
            continue
        tr = load_trace_file(fp, unit)
        if tr is not None:
            traces.append(tr)
        if max_traces and len(traces) >= max_traces:
            break
    if not traces:
        raise FileNotFoundError(f"no traces found in {folder}")
    return traces


def synthetic_traces(n=50, profile="medium", duration=400, seed=0):
    """Markov-modulated bandwidth traces.  FOR SMOKE TESTS ONLY."""
    rng = np.random.default_rng(seed)
    levels = {"slow": [400, 900, 1600, 2500, 4000],
              "medium": [1500, 3000, 6000, 12000, 25000],
              "fast": [20000, 60000, 120000, 250000, 400000]}[profile]
    traces = []
    for i in range(n):
        s = rng.integers(len(levels))
        t, b = [], []
        for k in range(duration):
            if rng.random() < 0.08:
                s = int(np.clip(s + rng.choice([-1, 1]), 0, len(levels) - 1))
            b.append(max(50.0, levels[s] * (1 + 0.25 * rng.standard_normal())))
            t.append(float(k))
        traces.append(Trace(f"syn_{profile}_{i}", t, b))
    return traces


def get_traces(folder=None, profile="medium", n_syn=50, seed=0):
    """folder may be a comma-separated list; missing entries are skipped."""
    parts = [f.strip() for f in (folder or "").split(",") if f.strip()]
    found = [f for f in parts if trace_source_exists(f)]
    if found:
        out = []
        for f in found:
            tr = load_trace_dir(f); out += tr
            print(f"[traces] {f}: {len(tr)} traces")
        return out
    print(f"[traces] WARNING: '{folder}' not found -> using SYNTHETIC {profile} traces (smoke test only)")
    return synthetic_traces(n_syn, profile, seed=seed)
