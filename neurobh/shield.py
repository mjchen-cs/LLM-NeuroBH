"""Neurosymbolic safety layer.

psi(s) = if P0(s, Theta(s)) subset xi then Theta(s) else Gamma(s)        (paper Eq. 7)

* P0  : worst-case one-step dynamics.  The next chunk is downloaded at the
        pessimistic bandwidth bw_low = pred_bw * (1 - eta); the buffer trough is
        buffer - size_kbit / bw_low.
* xi  : inductive invariant = all DSL constraints of the ServiceSpec hold on
        the worst-case successor AND B^max_new >= b_req + L (room to keep the
        invariant).
* Gamma: deterministic piecewise-linear safe policy: the highest ladder level
        (and the B^max delta closest to the proposal) whose worst-case
        successor stays in xi.  Each branch condition is linear in
        (buffer * bw_low), i.e. the linear predicates chi_i of Eq. 8.
"""
from dataclasses import dataclass
import numpy as np
from . import config as C


@dataclass
class Decision:
    level: int
    delta_idx: int
    intervened: bool
    feasible: bool          # False -> no safe action exists (best effort)


class Shield:
    def __init__(self, spec, bitrates=C.BITRATES_KBPS, L=C.CHUNK_LEN, eta=None, no_stall=True):
        self.spec = spec
        self.no_stall = no_stall          # implicit invariant: worst-case trough >= b_req (>= 0)
        self.bitrates = list(bitrates)
        self.L = L
        self.eta = spec.profile["eta"] if eta is None else eta
        self.b_req = spec.buffer_floor()

    # ---------------------------------------------------------------- core
    def successor(self, buf, mb_old, prev_level, pred_bw, level, delta_idx, size_bytes, bw=None):
        """Values of the DSL variables for (state, action).  If bw is None the
        worst-case bandwidth of P0 is used; otherwise bw is the realised one."""
        bw_eff = pred_bw * (1 - self.eta) if bw is None else bw
        bw_eff = max(bw_eff, 1.0)
        trough = buf - size_bytes * 8 / 1000.0 / bw_eff
        mb_new = float(np.clip(mb_old + C.BUF_DELTAS[delta_idx], C.MAX_BUFFER_MIN, C.MAX_BUFFER_MAX))
        r = self.bitrates[level]
        sw = max(level - prev_level, 0)        # up-switch steps (down-switches always allowed)
        cond = {"buffer": buf, "max_buffer": mb_old, "bitrate": r, "switch": sw, "pred_bw": pred_bw}
        cons = {"buffer": trough, "max_buffer": mb_new, "bitrate": r, "switch": sw, "pred_bw": pred_bw}
        return cond, cons, mb_new

    def violated(self, cond, cons, mb_new, check_room=True):
        v = [c for c in self.spec.constraints if not c.holds(cond, cons)]
        if check_room:
            if mb_new < min(self.b_req + self.L, C.MAX_BUFFER_MAX) - 1e-9:
                v.append("room")
            if self.no_stall and cons["buffer"] < self.b_req - 1e-9:
                v.append("stall")
        return v

    def is_safe(self, s, level, delta_idx):
        cond, cons, mb_new = self.successor(s["buffer"], s["max_buffer"], s["prev_level"], s["pred_bw"],
                                            level, delta_idx, s["next_sizes"][level])
        return len(self.violated(cond, cons, mb_new)) == 0

    def gamma(self, s, prefer_delta=4):
        """Safe fallback policy Gamma(s)."""
        deltas = sorted(range(len(C.BUF_DELTAS)), key=lambda d: (abs(d - prefer_delta), -d))
        for lv in reversed(range(len(self.bitrates))):
            for d in deltas:
                if self.is_safe(s, lv, d):
                    return lv, d, True
        # no safe action: best effort = minimise #violations, prefer low level
        best, best_v = (0, prefer_delta), 1e9
        for lv in range(len(self.bitrates)):
            for d in deltas:
                cond, cons, mb_new = self.successor(s["buffer"], s["max_buffer"], s["prev_level"],
                                                    s["pred_bw"], lv, d, s["next_sizes"][lv])
                nv = len(self.violated(cond, cons, mb_new))
                if nv < best_v:
                    best, best_v = (lv, d), nv
        return best[0], best[1], False

    def __call__(self, s, level, delta_idx) -> Decision:
        if self.is_safe(s, level, delta_idx):
            return Decision(level, delta_idx, False, True)
        lv, d, feas = self.gamma(s, prefer_delta=delta_idx)
        return Decision(lv, d, True, feas)

    def safe_mask(self, s):
        """[n_levels, n_deltas] boolean mask of actions in xi (used for analysis)."""
        m = np.zeros((len(self.bitrates), len(C.BUF_DELTAS)), dtype=bool)
        for lv in range(len(self.bitrates)):
            for d in range(len(C.BUF_DELTAS)):
                m[lv, d] = self.is_safe(s, lv, d)
        return m
