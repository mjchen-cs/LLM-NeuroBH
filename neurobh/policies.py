"""Policies with a common interface:  act(env, state) -> (level, delta_idx, intervened)

  NeuroBHPolicy  : spec-conditioned PPO + neurosymbolic shield (LLM-NeuroBH when the
                   spec comes from the LLM Intent service)
  BBA, BOLA, FESTIVE, Dynamic, RobustMPC, RateBased : classic baselines (intent-unaware)
  SODA           : recent baseline (SIGCOMM'24), re-implemented from the paper
  Pensieve       : NeuroBHPolicy trained with --no-shield --no-buffer-action --fixed-spec
  LLMDirectABR   : an LLM queried on EVERY chunk as the ABR controller (zero-shot,
                   optionally given the service intent) -- the "LLM on the fast path"
                   alternative to LLM-NeuroBH.  Its API latency is charged to the
                   session as decision delay (policy.last_delay).
"""
import itertools
import numpy as np
from . import config as C
from .shield import Shield

D0 = C.BUF_DELTAS.index(0)


class NeuroBHPolicy:
    def __init__(self, agent, spec, use_shield=True, eta=None, deterministic=True, name="NeuroBH"):
        self.agent, self.spec, self.name = agent, spec, name
        self.shield = Shield(spec, eta=eta) if use_shield else None
        self.det = deterministic

    def reset(self):
        pass

    def act(self, env, s):
        obs = env.observe(s, self.spec)
        lv, d, _, _ = self.agent.act(obs, self.det)
        if self.shield is None:
            return lv, d, False
        dec = self.shield(s, lv, d)
        return dec.level, dec.delta_idx, dec.intervened


class RateBased:
    name = "Rate-based"
    def reset(self): pass
    def act(self, env, s):
        bw = s["pred_bw"] * 0.9
        lv = max([i for i, b in enumerate(env.bitrates) if b <= bw] or [0])
        return lv, D0, False


class BBA:
    name = "BBA"
    def __init__(self, reservoir=5.0, cushion=10.0):
        self.r, self.c = reservoir, cushion
    def reset(self): pass
    def act(self, env, s):
        b, K = s["buffer"], env.K
        if b < self.r: return 0, D0, False
        if b >= self.r + self.c: return K - 1, D0, False
        return int((K - 1) * (b - self.r) / self.c), D0, False


class BOLA:
    name = "BOLA"
    def __init__(self, gamma_p=5.0, buffer_max=60.0):
        self.gp, self.bmax = gamma_p, buffer_max
    def reset(self): pass
    def act(self, env, s):
        S = np.asarray(s["next_sizes"]); v = np.log(S / S[0])
        Q, Qmax = s["buffer"] / env.L, self.bmax / env.L
        V = (Qmax - 1) / (v[-1] + self.gp)
        score = (V * (v + self.gp) - Q) / S
        return int(np.argmax(score)), D0, False


class RobustMPC:
    """RobustMPC (Yin et al.) with horizon h, optimising the VMAF QoE of Eq. (2)
    with the DEFAULT weights (it is intent-unaware)."""
    name = "RobustMPC"
    def __init__(self, spec, horizon=5):
        self.h = horizon; self.w = spec.weights; self.combos = None
    def reset(self): pass
    def act(self, env, s):
        K, L = env.K, env.L
        h = min(self.h, s["remaining"])
        if self.combos is None or self.combos.shape[1] != h:
            self.combos = np.array(list(itertools.product(range(K), repeat=h)))
        bw = max(s["pred_bw"], 1.0)
        i0 = s["chunk"]; vid = env.video
        idx = np.minimum(np.arange(i0, i0 + h), vid.n_chunks - 1)
        sizes = vid.sizes[:, idx]; vm = vid.vmaf[:, idx]; dyn = vid.dynamic[idx]
        n = len(self.combos)
        buf = np.full(n, s["buffer"]); qoe = np.zeros(n)
        prev_v = np.full(n, vid.vmaf[s["prev_level"], max(i0 - 1, 0)])
        for t in range(h):
            lv = self.combos[:, t]
            dt = sizes[lv, t] * 8 / 1000 / bw
            reb = np.maximum(dt - buf, 0)
            buf = np.maximum(buf - dt, 0) + L
            v = vm[lv, t]
            wq = self.w.delta if dyn[t] else self.w.alpha
            qoe += wq * v - self.w.epsilon * np.abs(v - prev_v) - self.w.beta * reb
            prev_v = v
        return int(self.combos[np.argmax(qoe), 0]), D0, False


class SODA:
    """SODA (Chen et al., SIGCOMM'24), re-implemented from the paper's description:
    smoothed online optimisation over a short prediction horizon.  Every step minimises
        sum_k  (100 - v_k)  +  w_b (B_k - B_target)^2  +  w_s (v_k - v_{k-1})^2  +  beta * rebuf_k
    over the next h chunks, where v is VMAF, B the buffer after chunk k, and the throughput is the
    predicted bandwidth.  As in SODA, the search is restricted to monotone bitrate sequences
    (non-decreasing or non-increasing), which keeps the solver cheap, and only the first decision is
    applied (receding horizon).  The quality term, the buffer cost around a target level and the
    quadratic switching cost are SODA's three cost components; the explicit rebuffering term makes the
    discrete version stall-averse.  Hyper-parameters were tuned on the training traces
    (experiments/exp_abr.py --tune-soda)."""
    name = "SODA"
    def __init__(self, horizon=5, target=20.0, w_buf=0.05, w_switch=0.02, beta=50.0):
        self.h, self.bt, self.wb, self.ws, self.beta = horizon, target, w_buf, w_switch, beta
        self.combos = {}
    def reset(self): pass
    def _monotone(self, K, h):
        if (K, h) not in self.combos:
            up = list(itertools.combinations_with_replacement(range(K), h))
            seq = set(up) | {tuple(reversed(c)) for c in up}
            self.combos[(K, h)] = np.array(sorted(seq))
        return self.combos[(K, h)]
    def act(self, env, s):
        K, L = env.K, env.L
        h = max(1, min(self.h, s["remaining"]))
        combos = self._monotone(K, h)
        bw = max(s["pred_bw"], 1.0)
        i0 = s["chunk"]; vid = env.video
        idx = np.minimum(np.arange(i0, i0 + h), vid.n_chunks - 1)
        sizes = vid.sizes[:, idx]; vm = vid.vmaf[:, idx]
        n = len(combos)
        buf = np.full(n, s["buffer"]); cost = np.zeros(n)
        prev_v = np.full(n, vid.vmaf[s["prev_level"], max(i0 - 1, 0)]) if i0 > 0 else None
        for t in range(h):
            lv = combos[:, t]
            dt = sizes[lv, t] * 8 / 1000 / bw
            reb = np.maximum(dt - buf, 0)
            buf = np.maximum(buf - dt, 0) + L
            v = vm[lv, t]
            cost += (100 - v) + self.wb * (buf - self.bt) ** 2 + self.beta * reb
            if prev_v is not None:
                cost += self.ws * (v - prev_v) ** 2
            prev_v = v
        return int(combos[np.argmin(cost), 0]), D0, False


class FESTIVE:
    """FESTIVE (Jiang et al., CoNEXT'12): harmonic mean of the last 20 samples,
    target 0.85 x estimate, gradual up-switch (one level, only after the current
    level has been kept for (level+1) chunks), immediate one-level down-switch."""
    name = "FESTIVE"
    def __init__(self, window=20, alpha=0.85):
        self.k, self.a = window, alpha
    def reset(self):
        self.stay = 0
    def act(self, env, s):
        hist = [x for x in s["hist_tput"] if x > 0]
        if not hist:
            self.stay = 0; return 0, D0, False
        x = np.asarray((env.samples or hist)[-self.k:])
        est = len(x) / np.sum(1.0 / x) * self.a
        target = max([i for i, b in enumerate(env.bitrates) if b <= est] or [0])
        cur = s["prev_level"]; self.stay += 1
        if target > cur and self.stay >= cur + 1:
            self.stay = 0; return cur + 1, D0, False
        if target < cur:
            self.stay = 0; return cur - 1, D0, False
        return cur, D0, False


class Dynamic:
    """DYNAMIC (Spiteri et al., MMSys'19; dash.js default): throughput rule at
    low buffer, BOLA at high buffer, with hysteresis at 10 s."""
    name = "Dynamic"
    def __init__(self, switch_buffer=10.0):
        self.th = switch_buffer; self.bola = BOLA()
    def reset(self):
        self.use_bola = False
    def act(self, env, s):
        bw = s["pred_bw"] * 0.9
        lv_t = max([i for i, b in enumerate(env.bitrates) if b <= bw] or [0])
        lv_b = self.bola.act(env, s)[0]
        if not self.use_bola and s["buffer"] >= self.th and lv_b >= lv_t:
            self.use_bola = True
        elif self.use_bola and s["buffer"] < self.th and lv_b < lv_t:
            self.use_bola = False
        return (lv_b if self.use_bola else lv_t), D0, False


LLM_ABR_PROMPT = """You are the adaptive-bitrate (ABR) controller of a DASH video player.
Choose the bitrate level of the NEXT chunk to maximise quality of experience
(high bitrate, few quality switches, and above all no rebuffering).
{intent}
Current state (JSON):
{state}
Reply with ONLY a JSON object: {{"level": <integer 0..{kmax}>}}"""


class LLMDirectABR:
    """Zero-shot LLM queried on every chunk (no training)."""
    name = "LLM-ABR (direct)"
    def __init__(self, llm, intent_text=None, charge_latency=True):
        self.llm, self.intent = llm, intent_text
        self.charge = charge_latency
        self.last_delay = 0.0
        self.latencies, self.failures = [], 0
    def reset(self):
        self.last_delay = 0.0
    def act(self, env, s):
        import json, re
        st = {"buffer_s": round(s["buffer"], 2),
              "last_throughputs_kbps": [round(x) for x in s["hist_tput"][-5:]],
              "last_download_times_s": [round(x, 2) for x in s["hist_dl"][-5:]],
              "predicted_bandwidth_kbps": round(s["pred_bw"]),
              "bitrate_ladder_kbps": list(env.bitrates),
              "next_chunk_sizes_KB": [round(x / 1000) for x in s["next_sizes"]],
              "chunk_duration_s": env.L, "previous_level": s["prev_level"],
              "remaining_chunks": s["remaining"]}
        intent = f"Service requirement: {self.intent}" if self.intent else ""
        msg = LLM_ABR_PROMPT.format(intent=intent, state=json.dumps(st), kmax=env.K - 1)
        try:
            r = self.llm.chat([{"role": "user", "content": msg}])
            txt = re.sub(r"<think>.*?</think>", "", r["text"], flags=re.S)
            m = re.search(r'"level"\s*:\s*(\d+)', txt) or re.search(r"\b(\d)\b", txt)
            lv = int(m.group(1)); assert 0 <= lv < env.K
            self.last_delay = r["latency"] if self.charge else 0.0
            self.latencies.append(r["latency"])
        except Exception:
            self.failures += 1
            lv, self.last_delay = s["prev_level"], 0.0
        return lv, D0, False


def run_episode(env, policy, spec, trace_idx=None, video_idx=None, start=None):
    """Runs one session; returns env.log (list of per-chunk dicts)."""
    s = env.reset(spec, trace_idx, video_idx, start)
    if hasattr(policy, "reset"):
        policy.reset()
    done = False
    while not done:
        lv, d, inter = policy.act(env, s)
        delay = getattr(policy, "last_delay", 0.0)
        s, _, done, _ = env.step(lv, d, inter, extra_delay=delay)
    return env.log
