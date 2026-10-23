"""Constraint DSL for streaming-service safety specifications.

Grammar (case-insensitive keywords):

    constraint := "ALWAYS" conj [ "WHEN" conj ]
    conj       := atom { "AND" atom }
    atom       := VAR OP NUMBER [UNIT]
    VAR        := buffer | max_buffer | bitrate | switch | pred_bw
    OP         := <= | >= | < | >
    UNIT       := s | sec | ms | kbps | mbps        (converted to s / kbps)

Semantics (P^0 = worst-case one-step dynamics, see shield.py):
    buffer      buffer level (s).  In the ALWAYS part it is the *minimum* level
                reached while the next chunk downloads (the trough); in the WHEN
                part it is the level at decision time.
    max_buffer  maximum buffer B^max (s) after the action.
    bitrate     bitrate of the next chunk (kbps).
    switch      up-switch size max(level(next) - level(prev), 0) in ladder steps
                (down-switches are always allowed so that safety is never blocked).
    pred_bw     predicted bandwidth for the next chunk (kbps).

A constraint holds iff (WHEN-part is false) or (ALWAYS-part is true).
"""
import re
from dataclasses import dataclass
from typing import List, Optional

VARS = ("buffer", "max_buffer", "bitrate", "switch", "pred_bw")
OPS = ("<=", ">=", "<", ">")
UNITS = {"s": ("time", 1.0), "sec": ("time", 1.0), "secs": ("time", 1.0), "seconds": ("time", 1.0),
         "ms": ("time", 1e-3), "kbps": ("rate", 1.0), "mbps": ("rate", 1000.0)}
VAR_KIND = {"buffer": "time", "max_buffer": "time", "bitrate": "rate", "switch": "count",
            "pred_bw": "rate"}
# physically meaningful ranges, used by the type/range check of the verifier
VAR_RANGE = {"buffer": (0.0, 60.0), "max_buffer": (2.0, 60.0), "bitrate": (0.0, 100000.0),
             "switch": (0.0, 10.0), "pred_bw": (0.0, 1e7)}


class DSLError(Exception):
    pass


@dataclass
class Atom:
    var: str
    op: str
    value: float

    def holds(self, x: float) -> bool:
        if self.op == "<=": return x <= self.value + 1e-9
        if self.op == ">=": return x >= self.value - 1e-9
        if self.op == "<": return x < self.value
        return x > self.value

    def z3(self, v):
        if self.op == "<=": return v <= self.value
        if self.op == ">=": return v >= self.value
        if self.op == "<": return v < self.value
        return v > self.value

    def __str__(self):
        val = int(self.value) if float(self.value).is_integer() else round(self.value, 3)
        return f"{self.var} {self.op} {val}"


@dataclass
class Constraint:
    then: List[Atom]
    when: List[Atom]
    text: str = ""

    def holds(self, cond_vals: dict, cons_vals: dict) -> bool:
        if self.when and not all(a.holds(cond_vals[a.var]) for a in self.when):
            return True
        return all(a.holds(cons_vals[a.var]) for a in self.then)

    def canonical(self) -> str:
        s = "ALWAYS " + " AND ".join(str(a) for a in self.then)
        if self.when:
            s += " WHEN " + " AND ".join(str(a) for a in self.when)
        return s

    def __str__(self):
        return self.canonical()


_ATOM_RE = re.compile(r"^\s*([a-zA-Z_]+)\s*(<=|>=|<|>|=<|=>|≤|≥)\s*([-+]?\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$")


def _parse_atom(s: str) -> Atom:
    m = _ATOM_RE.match(s)
    if not m:
        raise DSLError(f"cannot parse atom '{s.strip()}' (expected: VAR OP NUMBER, e.g. 'buffer >= 2')")
    var, op, num, unit = m.group(1).lower(), m.group(2), float(m.group(3)), m.group(4).lower()
    op = {"=<": "<=", "=>": ">=", "≤": "<=", "≥": ">="}.get(op, op)
    if var not in VARS:
        raise DSLError(f"unknown variable '{var}'; allowed variables: {', '.join(VARS)}")
    if unit:
        if unit not in UNITS:
            raise DSLError(f"unknown unit '{unit}' in '{s.strip()}'")
        kind, factor = UNITS[unit]
        if kind != VAR_KIND[var]:
            raise DSLError(f"unit '{unit}' is not valid for variable '{var}'")
        num *= factor
    return Atom(var, op, num)


def _parse_conj(s: str) -> List[Atom]:
    parts = re.split(r"\s+AND\s+|\s*&&\s*|\s*∧\s*", s.strip(), flags=re.IGNORECASE)
    if any(re.search(r"\s+OR\s+|\|\|", p, flags=re.IGNORECASE) for p in parts):
        raise DSLError("disjunction (OR) is not supported; split into several constraints")
    return [_parse_atom(p) for p in parts if p.strip()]


def parse_constraint(text: str) -> Constraint:
    t = text.strip().rstrip(";.")
    m = re.match(r"^\s*ALWAYS\s+(.*?)(?:\s+WHEN\s+(.*))?$", t, flags=re.IGNORECASE)
    if not m:
        raise DSLError(f"constraint must start with ALWAYS: '{text}'")
    then = _parse_conj(m.group(1))
    when = _parse_conj(m.group(2)) if m.group(2) else []
    if not then:
        raise DSLError(f"empty ALWAYS part in '{text}'")
    return Constraint(then, when, text)


def parse_constraints(texts) -> List[Constraint]:
    out, errs = [], []
    for t in texts or []:
        try:
            out.append(parse_constraint(t))
        except DSLError as e:
            errs.append(str(e))
    if errs:
        raise DSLError("; ".join(errs))
    return out


# --------------------------------------------------------------- features
N_CONSTRAINT_FEATS = 6


def constraint_features(cons: List[Constraint], top_bitrate: float) -> list:
    """Fixed-length summary of a constraint set, fed to the policy network so
    that a single spec-conditioned policy serves every service intent."""
    buf_floor, mb_cap, br_cap, br_floor, sw_cap, n_cond = 0.0, 60.0, top_bitrate, 0.0, 5.0, 0
    for c in cons:
        if c.when:
            n_cond += 1
        for a in c.then:
            if a.var == "buffer" and a.op in (">=", ">") and not c.when:
                buf_floor = max(buf_floor, a.value)
            elif a.var == "max_buffer" and a.op in ("<=", "<") and not c.when:
                mb_cap = min(mb_cap, a.value)
            elif a.var == "bitrate" and a.op in ("<=", "<") and not c.when:
                br_cap = min(br_cap, a.value)
            elif a.var == "bitrate" and a.op in (">=", ">"):
                br_floor = max(br_floor, a.value)
            elif a.var == "switch" and a.op in ("<=", "<"):
                sw_cap = min(sw_cap, a.value)
    return [buf_floor / 20.0, mb_cap / 60.0, br_cap / top_bitrate, br_floor / top_bitrate,
            sw_cap / 5.0, min(n_cond, 3) / 3.0]
