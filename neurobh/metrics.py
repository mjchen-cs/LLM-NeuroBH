"""Session metrics computed from env logs under an arbitrary (ground-truth) spec."""
import numpy as np
from . import config as C


def session_metrics(log, eval_spec, L=C.CHUNK_LEN, departure=None):
    """eval_spec: spec whose weights/constraints define 'what the service wanted'.
    departure: viewing fraction f in (0,1] for the waste metric (None -> full view)."""
    w = eval_spec.weights
    q, viol, inter = [], [], []
    for i, r in enumerate(log):
        wq = w.delta if r["dynamic"] else w.alpha
        q.append(wq * r["vmaf"] - w.epsilon * r["smooth"] - w.beta * r["rebuf"])
        inter.append(r["intervened"])
        if i == 0:
            continue                       # startup chunk excluded from safety statistics
        sw = max(r["level"] - r["prev_level"], 0)
        cond = {"buffer": r["buffer_before"], "max_buffer": r["mb_old"], "bitrate": r["bitrate"],
                "switch": sw, "pred_bw": r["pred_bw"]}
        cons = {"buffer": r["trough"], "max_buffer": r["mb_new"], "bitrate": r["bitrate"],
                "switch": sw, "pred_bw": r["pred_bw"]}
        viol.append(any(not c.holds(cond, cons) for c in eval_spec.constraints))
    n = len(log)
    total_rebuf = sum(r["rebuf"] for r in log)
    play = n * L
    buf = np.mean([r["buffer_after"] for r in log])
    out = {
        "qoe": float(np.mean(q)),
        "objective": float(np.mean(q) - w.phi * buf),
        "bitrate": float(np.mean([r["bitrate"] for r in log])),
        "vmaf": float(np.mean([r["vmaf"] for r in log])),
        "rebuf_ratio": float(total_rebuf / (play + total_rebuf)),
        "stall_s": float(total_rebuf),
        "startup_s": float(log[0]["startup"]),
        "smooth": float(np.mean([r["smooth"] for r in log[1:]])) if n > 1 else 0.0,
        "buffer": float(buf),
        "max_buffer": float(np.mean([r["mb_new"] for r in log])),
        "violation_rate": float(np.mean(viol)) if viol else 0.0,
        "any_violation": float(any(viol)),
        "intervention_rate": float(np.mean(inter)),
        "data_MB": float(sum(r["size"] for r in log) / 1e6),
    }
    out["waste_MB"] = waste_mb(log, L, departure) if departure is not None else 0.0
    return out


def waste_mb(log, L, f):
    """Downloaded-but-never-played data when the viewer leaves at fraction f."""
    n = len(log)
    P = f * n * L                                   # departure playback position (s)
    for i, r in enumerate(log):
        played = (i + 1) * L - r["buffer_after"]
        if played >= P or i == n - 1:
            waste = 0.0
            for j in range(i + 1):
                start = j * L
                if start + L <= P:
                    continue
                frac = 1.0 if start >= P else (start + L - P) / L
                waste += log[j]["size"] * frac
            return waste / 1e6
    return 0.0


def sample_departures(n, rng, p_complete=0.35, a=0.8, b=1.5):
    """Viewing fraction model fitted to the short-video measurement in Fig. 2
    (most viewers leave early; ~1/3 finish)."""
    f = rng.beta(a, b, size=n)
    f[rng.random(n) < p_complete] = 1.0
    return np.clip(f, 0.02, 1.0)


def aggregate(rows, keys=None):
    keys = keys or [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
