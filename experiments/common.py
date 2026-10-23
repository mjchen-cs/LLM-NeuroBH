"""Shared helpers for all experiment scripts (run from the repository root:
   python experiments/exp_xxx.py)."""
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from neurobh import config as C                     # noqa: E402
from neurobh.traces import load_trace_dir, synthetic_traces, trace_source_exists   # noqa: E402
from neurobh.video import get_videos               # noqa: E402

SYNTHETIC = {"flag": False}


def load_cfg(path="configs/exp.yaml"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    C.BITRATES_KBPS[:] = cfg.get("ladder_kbps", C.BITRATES_KBPS)
    C.CHUNK_LEN = cfg.get("chunk_len", C.CHUNK_LEN)
    os.makedirs(cfg["results"], exist_ok=True)
    os.makedirs(cfg["figures"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["ckpt"]["neurobh"]) or ".", exist_ok=True)
    return cfg


def traces_for(cfg, key, seed=0, n_syn=40):
    folders = cfg["traces"][key]
    folders = folders if isinstance(folders, list) else [folders]
    out = []
    for fd in folders:
        if trace_source_exists(fd):
            out += load_trace_dir(fd)
    if not out:
        prof = {"train": "medium", "unknown": "medium"}.get(key, key)
        print(f"[data] WARNING: no traces for '{key}' -> SYNTHETIC {prof} traces (smoke test only)")
        SYNTHETIC["flag"] = True
        out = synthetic_traces(n_syn, prof, seed=seed + hash(key) % 100)
    return out


def videos_for(cfg, key="test"):
    if not os.path.isdir(cfg["videos"][key]):
        SYNTHETIC["flag"] = True
    return get_videos(cfg["videos"][key])


def mark_run(cfg, name, extra=None):
    """Records whether a result file was produced from synthetic data/mock LLM."""
    p = os.path.join(cfg["results"], "RUN_INFO.json")
    info = {}
    if os.path.exists(p):
        info = json.load(open(p))
    info[name] = {"synthetic": SYNTHETIC["flag"], **(extra or {})}
    json.dump(info, open(p, "w"), indent=2)


def load_intents(path="data/intents.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
