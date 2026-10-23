"""Global constants for LLM-NeuroBH.

All units: bitrate in kbps, bandwidth in kbps, time in seconds, size in bytes.
Values below are defaults and can be overridden from experiment scripts.
"""
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------- video ladder
# Default training/test ladder = Envivo-Dash3 (same as Pensieve).
BITRATES_KBPS: List[int] = [300, 750, 1200, 1850, 2850, 4300]
CHUNK_LEN = 4.0                      # L, seconds per chunk
TOTAL_CHUNKS = 48                    # chunks per video (~3.2 min)

# ------------------------------------------------------------ buffer control
BUF_DELTAS = [-8, -6, -4, -2, 0, 2, 4, 6, 8]   # a_b, seconds (paper Sec. 5)
MAX_BUFFER_MIN = 2.0                 # lower bound for B^max
MAX_BUFFER_MAX = 60.0                # upper bound for B^max
MAX_BUFFER_INIT = 20.0               # default B^max at session start

# --------------------------------------------------------------- network sim
RTT = 0.08                           # seconds, per request
PACKET_PAYLOAD = 0.95                # TCP/IP payload efficiency
STABLE_NOISE = 0.03                  # relative noise of HTTP stable-connection samples
HIST_LEN = 8                         # k past chunks in the state (paper: k=8)

# ----------------------------------------------------- network profiles
# Bandwidth domain used by the verifier (kbps) and default prediction error
# bound eta (P^0 worst case: bw_low = pred_bw * (1 - eta)).  eta should be
# replaced by the empirical 90th percentile measured in exp_predictor.py.
NETWORK_PROFILES = {
    "slow":    {"bw_min": 300.0,  "bw_max": 6000.0,   "eta": 0.35, "predictor": "hybrid"},
    "medium":  {"bw_min": 500.0,  "bw_max": 100000.0, "eta": 0.30, "predictor": "hybrid"},
    "fast":    {"bw_min": 2000.0, "bw_max": 1000000.0, "eta": 0.25, "predictor": "gru"},
    "unknown": {"bw_min": 300.0,  "bw_max": 100000.0, "eta": 0.35, "predictor": "hybrid"},
}


def _load_eta():
    """Use the empirical prediction-error bound measured by experiments/exp_predictor.py
    (results/eta.json) when available; clipped to [0.1, 0.5]."""
    import json, os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "eta.json")
    if os.path.exists(p):
        try:
            eta = json.load(open(p))
            for k, v in eta.items():
                if k in NETWORK_PROFILES:
                    NETWORK_PROFILES[k]["eta"] = float(min(max(v, 0.1), 0.5))
            NETWORK_PROFILES["unknown"]["eta"] = max(NETWORK_PROFILES[k]["eta"] for k in ("slow", "medium", "fast"))
        except Exception:
            pass


_load_eta()

# ------------------------------------------------ preference level -> weights
# The LLM outputs 1..5 priority levels; the mapping to the numeric weights of
# Eq.(2)/(3) is deterministic and therefore auditable.
LEVEL_QUALITY = {1: 0.4, 2: 0.7, 3: 1.0, 4: 1.3, 5: 1.6}        # alpha
DYNAMIC_BONUS = 1.25                                             # delta = alpha * bonus (delta > alpha)
LEVEL_SMOOTH = {1: 0.2, 2: 0.5, 3: 1.0, 4: 1.4, 5: 1.8}         # epsilon
LEVEL_STALL = {1: 10.0, 2: 25.0, 3: 50.0, 4: 80.0, 5: 120.0}    # beta  (VMAF pts per stall second)
LEVEL_BUFFER = {1: 0.0, 2: 0.1, 3: 0.3, 4: 0.6, 5: 1.0}         # phi   (per buffered second)
LEVEL_INIT_MAXBUF = {1: 30.0, 2: 24.0, 3: 20.0, 4: 12.0, 5: 6.0}  # B^max_0 from max(latency, data_saving)

PRIORITY_KEYS = ["quality", "smoothness", "stall_avoidance", "data_saving", "low_latency"]
DEFAULT_PRIORITIES = {"quality": 3, "smoothness": 3, "stall_avoidance": 3,
                      "data_saving": 2, "low_latency": 2}

REWARD_SCALE = 100.0                 # reward / REWARD_SCALE for PPO


@dataclass
class Weights:
    delta: float
    alpha: float
    epsilon: float
    beta: float
    phi: float
    init_max_buffer: float = MAX_BUFFER_INIT

    def as_vector(self) -> List[float]:
        # normalised user-preference state s_u (paper Sec. 5.1)
        return [self.delta / 2.0, self.alpha / 2.0, self.epsilon / 2.0,
                self.beta / 120.0, self.phi]


def priorities_to_weights(p: dict) -> Weights:
    q = int(p.get("quality", 3)); s = int(p.get("smoothness", 3))
    r = int(p.get("stall_avoidance", 3)); d = int(p.get("data_saving", 2))
    l = int(p.get("low_latency", 2))
    alpha = LEVEL_QUALITY[q]
    return Weights(delta=alpha * DYNAMIC_BONUS, alpha=alpha, epsilon=LEVEL_SMOOTH[s],
                   beta=LEVEL_STALL[r], phi=LEVEL_BUFFER[max(d, l)],
                   init_max_buffer=LEVEL_INIT_MAXBUF[max(d, l)])
