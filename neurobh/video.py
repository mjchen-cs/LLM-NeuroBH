"""Video model: chunk sizes, perceptual quality (VMAF) and static/dynamic flags.

Real data (recommended):
  * sizes:  Pensieve format, folder with files video_size_0 ... video_size_{K-1},
            one chunk size in bytes per line (level 0 = lowest bitrate).
  * vmaf:   optional CSV ``chunk,level,vmaf`` measured with ffmpeg/libvmaf.
            Without it, a logarithmic rate-quality model is used.
  * dynamic flag: a chunk is "dynamic" (high motion / scene change) if its
            size at the top level is > 1.1 x the median -> weighted by delta.
"""
import os
import numpy as np
from . import config as C


def vmaf_model(bitrate_kbps, complexity=1.0):
    """Log rate-quality model: 300 kbps -> ~35, 4300 kbps -> ~93 (complexity=1)."""
    eff = np.asarray(bitrate_kbps, dtype=np.float64) / complexity
    v = -89.3 + 21.8 * np.log(np.maximum(eff, 1.0))
    return np.clip(v, 0.0, 100.0)


class Video:
    def __init__(self, sizes, vmaf, bitrates, name="video"):
        self.sizes = np.asarray(sizes, dtype=np.float64)   # [K levels, N chunks]
        self.vmaf = np.asarray(vmaf, dtype=np.float64)     # [K, N]
        self.bitrates = list(bitrates)
        self.name = name
        top = self.sizes[-1]
        self.dynamic = top > 1.1 * np.median(top)          # [N] bool
        self.n_chunks = self.sizes.shape[1]
        self.n_levels = self.sizes.shape[0]

    @staticmethod
    def synthetic(bitrates=C.BITRATES_KBPS, n_chunks=C.TOTAL_CHUNKS, L=C.CHUNK_LEN, seed=0, name=None):
        rng = np.random.default_rng(seed)
        complexity = np.clip(1.0 + 0.25 * rng.standard_normal(n_chunks), 0.6, 1.8)
        sizes = np.stack([b * 1000 / 8 * L * complexity * (1 + 0.05 * rng.standard_normal(n_chunks))
                          for b in bitrates])
        vmaf = np.stack([vmaf_model(b, complexity) for b in bitrates])
        return Video(sizes, vmaf, bitrates, name or f"syn_video_{seed}")

    @staticmethod
    def from_pensieve(folder, bitrates, vmaf_csv=None, L=C.CHUNK_LEN):
        sizes = []
        for k in range(len(bitrates)):
            with open(os.path.join(folder, f"video_size_{k}")) as f:
                sizes.append([float(x) for x in f.read().split()])
        n = min(len(s) for s in sizes)
        sizes = np.array([s[:n] for s in sizes])
        if vmaf_csv and os.path.exists(vmaf_csv):
            vmaf = np.zeros_like(sizes)
            import csv
            with open(vmaf_csv) as f:
                for row in csv.DictReader(f):
                    c, l = int(row["chunk"]), int(row["level"])
                    if c < n:
                        vmaf[l, c] = float(row["vmaf"])
        else:
            # complexity estimated from realised bitrate of each chunk vs nominal
            real = sizes * 8 / 1000 / L
            complexity = np.clip(real[-1] / bitrates[-1], 0.5, 2.0)
            vmaf = np.stack([vmaf_model(b, complexity) for b in bitrates])
        return Video(sizes, vmaf, bitrates, os.path.basename(folder.rstrip("/\\")))


def get_videos(folder=None, bitrates=C.BITRATES_KBPS, n_syn=5, seed=0):
    """Returns a list of Video objects.  folder may contain sub-folders, each in
    Pensieve format (optionally with vmaf.csv)."""
    if folder and os.path.isdir(folder):
        vids = []
        subs = [os.path.join(folder, d) for d in sorted(os.listdir(folder))]
        subs = [d for d in subs if os.path.isdir(d)] or [folder]
        for d in subs:
            if os.path.exists(os.path.join(d, "video_size_0")):
                vids.append(Video.from_pensieve(d, bitrates, os.path.join(d, "vmaf.csv")))
        if vids:
            return vids
    print(f"[video] WARNING: '{folder}' not found -> using SYNTHETIC videos (smoke test only)")
    return [Video.synthetic(bitrates, seed=seed + i) for i in range(n_syn)]
