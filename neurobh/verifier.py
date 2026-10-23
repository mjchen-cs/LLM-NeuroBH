"""Symbolic verification service for LLM-generated ServiceSpecs.

Stage 1  schema     : JSON fields / priority ranges / network profile
Stage 2  syntax     : DSL grammar and units
Stage 3  range      : physically meaningful thresholds (ladder, buffer range)
Stage 4  consistency: SMT satisfiability of the constraint set (Z3), with an
                      unsat core naming the conflicting constraints
Stage 5  realizability under worst-case dynamics P0:
            forall s in xi  exists a :  P0(s, a) subset xi
         checked by searching a counter-example state with Z3 (QF_NRA):
            exists (pred_bw, buffer, prev_level, max_buffer_old) in xi
                 s.t. every action a violates the spec.
         A counter-example is turned into natural-language feedback for the
         LLM repair loop.

Assumption (stated in the paper): the verifier quantifies over the
"serviceable" bandwidth region pred_bw*(1-eta) >= lowest bitrate; below it no
ABR policy can avoid stalls regardless of the spec.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import z3

from . import config as C
from .dsl import VAR_RANGE, DSLError
from .spec import ServiceSpec, SpecError, extract_json


@dataclass
class VerifyResult:
    ok: bool
    stage: str                       # passed | json | schema | syntax | range | consistency | realizability
    messages: List[str] = field(default_factory=list)
    counterexample: Optional[dict] = None
    spec: Optional[ServiceSpec] = None
    seconds: float = 0.0

    def feedback(self) -> str:
        if self.ok:
            return "OK"
        s = f"Verification failed at stage '{self.stage}'.\n" + "\n".join(f"- {m}" for m in self.messages)
        if self.counterexample:
            s += "\nCounter-example state: " + ", ".join(f"{k}={v}" for k, v in self.counterexample.items())
        return s


class Verifier:
    def __init__(self, bitrates=C.BITRATES_KBPS, L=C.CHUNK_LEN, timeout_ms=8000):
        self.bitrates = list(bitrates)
        self.L = L
        self.K = len(bitrates)
        self.timeout_ms = timeout_ms

    # ======================================================== entry point
    def verify_text(self, reply_text: str) -> VerifyResult:
        t0 = time.time()
        try:
            d = extract_json(reply_text)
        except SpecError as e:
            return VerifyResult(False, "json", [str(e)], seconds=time.time() - t0)
        r = self.verify_dict(d)
        r.seconds = time.time() - t0
        return r

    def verify_dict(self, d: dict) -> VerifyResult:
        t0 = time.time()
        try:
            spec = ServiceSpec.from_dict(d, compile_now=False)
        except SpecError as e:
            return VerifyResult(False, "schema", [str(e)], seconds=time.time() - t0)
        try:
            spec.compile()
        except SpecError as e:
            return VerifyResult(False, "syntax", [str(e)], spec=spec, seconds=time.time() - t0)
        r = self.verify_spec(spec)
        r.seconds = time.time() - t0
        return r

    def verify_spec(self, spec: ServiceSpec) -> VerifyResult:
        msgs = self._range_check(spec)
        if msgs:
            return VerifyResult(False, "range", msgs, spec=spec)
        msgs = self._consistency(spec)
        if msgs:
            return VerifyResult(False, "consistency", msgs, spec=spec)
        ok, msgs, cex = self._realizability(spec)
        if not ok:
            return VerifyResult(False, "realizability", msgs, cex, spec=spec)
        return VerifyResult(True, "passed", [], spec=spec)

    # ======================================================== stage 3
    def _range_check(self, spec):
        msgs = []
        lo_b, hi_b = self.bitrates[0], self.bitrates[-1]
        for c in spec.constraints:
            for a in c.then + c.when:
                lo, hi = VAR_RANGE[a.var]
                if not lo <= a.value <= hi:
                    msgs.append(f"'{c.canonical()}': value {a.value} of '{a.var}' outside physical range [{lo}, {hi}]")
                if a.var == "switch" and not float(a.value).is_integer():
                    msgs.append(f"'{c.canonical()}': switch counts ladder steps and must be an integer")
            for a in c.then:
                if a.var == "bitrate" and a.op in ("<=", "<") and a.value < lo_b:
                    msgs.append(f"'{c.canonical()}': cap {a.value} kbps is below the lowest available bitrate {lo_b} kbps")
                if a.var == "bitrate" and a.op in (">=", ">") and a.value > hi_b:
                    msgs.append(f"'{c.canonical()}': floor {a.value} kbps is above the highest available bitrate {hi_b} kbps")
        return msgs

    # ======================================================== z3 helpers
    def _ok_expr(self, spec, st, level, delta, eta, b_req, nominal=True):
        """Z3 boolean: action (level, delta) is safe in symbolic state st."""
        bw, buf, prev, mbo = st["pred_bw"], st["buffer"], st["prev"], st["mb_old"]
        r = float(self.bitrates[level])
        kbit = r * self.L
        bwl = bw * (1 - eta)
        raw = mbo + C.BUF_DELTAS[delta]
        mbn = z3.If(raw < C.MAX_BUFFER_MIN, z3.RealVal(C.MAX_BUFFER_MIN),
                    z3.If(raw > C.MAX_BUFFER_MAX, z3.RealVal(C.MAX_BUFFER_MAX), raw))
        sw = z3.If(prev >= level, z3.IntVal(0), level - prev)   # up-switch steps
        terms = []
        for c in spec.constraints:
            then = []
            for a in c.then:
                if a.var == "buffer":
                    # trough = buf - kbit / bwl  ;  multiply by bwl > 0
                    lhs = (buf - a.value) * bwl
                    then.append({"<=": lhs <= kbit, ">=": lhs >= kbit, "<": lhs < kbit, ">": lhs > kbit}[a.op])
                elif a.var == "max_buffer":
                    then.append(a.z3(mbn))
                elif a.var == "bitrate":
                    then.append(a.z3(z3.RealVal(r)))
                elif a.var == "switch":
                    then.append(a.z3(z3.ToReal(sw)))
                elif a.var == "pred_bw":
                    then.append(a.z3(bw))
            when = []
            for a in c.when:
                v = {"buffer": buf, "max_buffer": mbo, "bitrate": z3.RealVal(r),
                     "switch": z3.ToReal(sw), "pred_bw": bw}[a.var]
                when.append(a.z3(v))
            body = z3.And(*then)
            terms.append(z3.Implies(z3.And(*when), body) if when else body)
        terms.append(mbn >= min(b_req + self.L, C.MAX_BUFFER_MAX))
        # implicit invariant: worst-case trough never below b_req (>= 0: no stall)
        terms.append((buf - b_req) * bwl >= kbit)
        return z3.And(*terms)

    def _symbols(self):
        return {"pred_bw": z3.Real("pred_bw"), "buffer": z3.Real("buffer"),
                "prev": z3.Int("prev_level"), "mb_old": z3.Real("max_buffer_old")}

    # ======================================================== stage 4
    def _consistency(self, spec):
        """Is there ANY state/action satisfying all constraints simultaneously?"""
        s = z3.Solver(); s.set("timeout", self.timeout_ms)
        bw, buf, mbn, r, sw = z3.Reals("pred_bw buffer max_buffer bitrate switch")
        prof = spec.profile
        s.add(bw >= prof["bw_min"], bw <= prof["bw_max"], buf >= 0, buf <= 60,
              mbn >= C.MAX_BUFFER_MIN, mbn <= C.MAX_BUFFER_MAX, sw >= 0, sw <= self.K - 1,
              z3.Or(*[r == b for b in self.bitrates]))
        env = {"pred_bw": bw, "buffer": buf, "max_buffer": mbn, "bitrate": r, "switch": sw}
        tracked = {}
        for i, c in enumerate(spec.constraints):
            then = z3.And(*[a.z3(env[a.var]) for a in c.then])
            e = z3.Implies(z3.And(*[a.z3(env[a.var]) for a in c.when]), then) if c.when else then
            p = z3.Bool(f"c{i}"); tracked[str(p)] = c
            s.assert_and_track(e, p)
        res = s.check()
        if res == z3.unsat:
            core = [tracked[str(x)].canonical() for x in s.unsat_core()]
            return [f"constraints are mutually contradictory: {core}. Remove or relax one of them."]
        return []

    # ======================================================== stage 5
    def _realizability(self, spec):
        prof = spec.profile
        eta = prof["eta"]
        b_req = spec.buffer_floor()
        st = self._symbols()
        bw_lo = max(prof["bw_min"], self.bitrates[0] / (1 - eta))
        mb_cap = spec.max_buffer_cap()
        if mb_cap < b_req + self.L - 1e-9:
            return False, [f"max_buffer cap {mb_cap:g} s is smaller than buffer floor + chunk length "
                           f"= {b_req:g} + {self.L:g} s: the player cannot hold enough video to keep the buffer floor. "
                           f"Lower the buffer floor or raise the max_buffer cap (latency vs. stall-protection conflict)."], \
                   {"max_buffer_cap_s": mb_cap, "buffer_floor_s": b_req}
        dom = [st["pred_bw"] >= bw_lo, st["pred_bw"] <= prof["bw_max"],
               st["prev"] >= 0, st["prev"] <= self.K - 1,
               st["mb_old"] >= max(C.MAX_BUFFER_MIN, min(b_req + self.L, C.MAX_BUFFER_MAX)),
               st["mb_old"] <= min(C.MAX_BUFFER_MAX, mb_cap),
               st["buffer"] >= min(b_req + self.L, C.MAX_BUFFER_MAX), st["buffer"] <= st["mb_old"]]
        all_bad = [z3.Not(self._ok_expr(spec, st, lv, d, eta, b_req))
                   for lv in range(self.K) for d in range(len(C.BUF_DELTAS))]
        s = z3.SolverFor("QF_NIRA") if hasattr(z3, "SolverFor") else z3.Solver()
        s.set("timeout", self.timeout_ms)
        s.add(*dom); s.add(*all_bad)
        res = s.check()
        if res == z3.unsat:
            return True, [], None
        if res == z3.unknown:
            return self._realizability_grid(spec, eta, b_req, bw_lo)
        m = s.model()
        def val(x):
            v = m.eval(x, model_completion=True)
            try:
                return round(float(v.as_fraction()), 2) if hasattr(v, "as_fraction") else int(v.as_long())
            except Exception:
                return str(v)
        cex = {"pred_bw_kbps": val(st["pred_bw"]), "buffer_s": val(st["buffer"]),
               "prev_level": val(st["prev"]), "max_buffer_s": val(st["mb_old"])}
        return False, self._explain(spec, cex, eta, b_req), cex

    def _realizability_grid(self, spec, eta, b_req, bw_lo):
        """Fallback when Z3 returns unknown: dense grid search."""
        from .shield import Shield
        sh = Shield(spec, self.bitrates, self.L, eta)
        for bw in np.geomspace(bw_lo, spec.profile["bw_max"], 40):
            for mbo in np.linspace(max(C.MAX_BUFFER_MIN, min(b_req + self.L, 60)), min(60, spec.max_buffer_cap()), 12):
                for buf in np.linspace(min(b_req + self.L, 60), mbo, 8):
                    for prev in range(self.K):
                        s = {"buffer": buf, "max_buffer": mbo, "prev_level": prev, "pred_bw": bw,
                             "next_sizes": [b * self.L * 1000 / 8 for b in self.bitrates]}
                        if not sh.safe_mask(s).any():
                            cex = {"pred_bw_kbps": round(bw, 1), "buffer_s": round(buf, 2),
                                   "prev_level": prev, "max_buffer_s": round(mbo, 2)}
                            return False, self._explain(spec, cex, eta, b_req), cex
        return True, [], None

    def _explain(self, spec, cex, eta, b_req):
        """Natural-language diagnosis used as repair feedback."""
        from .shield import Shield
        sh = Shield(spec, self.bitrates, self.L, eta)
        s = {"buffer": cex["buffer_s"], "max_buffer": cex["max_buffer_s"], "prev_level": int(cex["prev_level"]),
             "pred_bw": cex["pred_bw_kbps"], "next_sizes": [b * self.L * 1000 / 8 for b in self.bitrates]}
        blame = {}
        for lv in range(self.K):
            for d in range(len(C.BUF_DELTAS)):
                cond, cons, mbn = sh.successor(s["buffer"], s["max_buffer"], s["prev_level"], s["pred_bw"],
                                               lv, d, s["next_sizes"][lv])
                for v in sh.violated(cond, cons, mbn):
                    key = v if isinstance(v, str) else v.canonical()
                    blame[key] = blame.get(key, 0) + 1
        top = sorted(blame.items(), key=lambda kv: -kv[1])[:3]
        msgs = [f"In a reachable state (predicted bandwidth {cex['pred_bw_kbps']} kbps, buffer {cex['buffer_s']} s, "
                f"previous level {cex['prev_level']}, max_buffer {cex['max_buffer_s']} s) NO bitrate/buffer action "
                f"satisfies all constraints under the worst-case bandwidth ({int(eta * 100)}% below prediction)."]
        for k, n in top:
            if k == "room":
                msgs.append(f"max_buffer must stay >= buffer floor + chunk length = {b_req + self.L:.1f} s, "
                            f"but a max_buffer cap forbids it (latency cap vs. stall-protection conflict).")
            else:
                msgs.append(f"'{k}' blocks {n} of {self.K * len(C.BUF_DELTAS)} actions.")
        msgs.append("Relax the blocking constraint(s), lower the buffer floor, or add a WHEN condition on pred_bw.")
        return msgs


# ================================================================ evaluation
def semantic_agreement(spec_a, spec_b, bitrates=C.BITRATES_KBPS, L=C.CHUNK_LEN, n=4000, seed=0):
    """Fraction of random (state, action) pairs on which the two specs agree
    about safety (1.0 = semantically equivalent on the sampled domain)."""
    from .shield import Shield
    rng = np.random.default_rng(seed)
    eta = 0.3
    sa, sb = Shield(spec_a, bitrates, L, eta), Shield(spec_b, bitrates, L, eta)
    agree = 0
    for _ in range(n):
        s = {"buffer": rng.uniform(0, 40), "max_buffer": rng.uniform(2, 60),
             "prev_level": int(rng.integers(len(bitrates))),
             "pred_bw": float(np.exp(rng.uniform(np.log(200), np.log(200000)))),
             "next_sizes": [b * L * 1000 / 8 for b in bitrates]}
        lv, d = int(rng.integers(len(bitrates))), int(rng.integers(len(C.BUF_DELTAS)))
        ca, cb = sa.successor(s["buffer"], s["max_buffer"], s["prev_level"], s["pred_bw"], lv, d, s["next_sizes"][lv]), \
                 sb.successor(s["buffer"], s["max_buffer"], s["prev_level"], s["pred_bw"], lv, d, s["next_sizes"][lv])
        va = not sa.violated(*ca, check_room=False)
        vb = not sb.violated(*cb, check_room=False)
        agree += (va == vb)
    return agree / n
