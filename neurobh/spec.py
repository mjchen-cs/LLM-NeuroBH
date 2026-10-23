"""ServiceSpec: the machine-readable contract produced by the LLM Intent service.

JSON schema (what the LLM must output):
{
  "scenario": "<short label>",
  "network_profile": "slow" | "medium" | "fast" | "unknown",
  "priorities": {"quality":1-5, "smoothness":1-5, "stall_avoidance":1-5,
                 "data_saving":1-5, "low_latency":1-5},
  "constraints": ["ALWAYS buffer >= 2", "ALWAYS max_buffer <= 8", ...],
  "explanation": "<one sentence>"
}
"""
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from . import config as C
from .dsl import parse_constraints, Constraint, DSLError, constraint_features


class SpecError(Exception):
    pass


@dataclass
class ServiceSpec:
    scenario: str = "default"
    network_profile: str = "unknown"
    priorities: dict = field(default_factory=lambda: dict(C.DEFAULT_PRIORITIES))
    constraint_texts: List[str] = field(default_factory=list)
    explanation: str = ""
    # compiled
    constraints: List[Constraint] = field(default_factory=list)
    weights: Optional[C.Weights] = None

    # ------------------------------------------------------------ building
    @staticmethod
    def from_dict(d: dict, compile_now=True) -> "ServiceSpec":
        if not isinstance(d, dict):
            raise SpecError("top-level JSON must be an object")
        pr = d.get("priorities", {})
        if not isinstance(pr, dict):
            raise SpecError("'priorities' must be an object")
        prio = {}
        for k in C.PRIORITY_KEYS:
            v = pr.get(k, C.DEFAULT_PRIORITIES[k])
            try:
                v = int(round(float(v)))
            except (TypeError, ValueError):
                raise SpecError(f"priority '{k}' must be an integer 1-5, got {v!r}")
            if not 1 <= v <= 5:
                raise SpecError(f"priority '{k}' must be in 1..5, got {v}")
            prio[k] = v
        unknown = set(pr) - set(C.PRIORITY_KEYS)
        if unknown:
            raise SpecError(f"unknown priority keys {sorted(unknown)}; allowed: {C.PRIORITY_KEYS}")
        prof = str(d.get("network_profile", "unknown")).lower()
        if prof not in C.NETWORK_PROFILES:
            raise SpecError(f"network_profile must be one of {list(C.NETWORK_PROFILES)}, got '{prof}'")
        cons = d.get("constraints", []) or []
        if isinstance(cons, str):
            cons = [cons]
        if not isinstance(cons, list) or not all(isinstance(c, str) for c in cons):
            raise SpecError("'constraints' must be a list of DSL strings")
        s = ServiceSpec(str(d.get("scenario", "custom")), prof, prio, list(cons),
                        str(d.get("explanation", "")))
        if compile_now:
            s.compile()
        return s

    def compile(self):
        try:
            self.constraints = parse_constraints(self.constraint_texts)
        except DSLError as e:
            raise SpecError(f"DSL syntax error: {e}")
        self.weights = C.priorities_to_weights(self.priorities)
        return self

    def to_dict(self):
        return {"scenario": self.scenario, "network_profile": self.network_profile,
                "priorities": dict(self.priorities),
                "constraints": [c.canonical() for c in self.constraints] if self.constraints else list(self.constraint_texts),
                "explanation": self.explanation}

    # ------------------------------------------------------------ helpers
    @property
    def profile(self):
        return C.NETWORK_PROFILES[self.network_profile]

    def buffer_floor(self) -> float:
        """Unconditional lower bound on the buffer (safety invariant threshold b_req)."""
        b = 0.0
        for c in self.constraints:
            if not c.when:
                for a in c.then:
                    if a.var == "buffer" and a.op in (">=", ">"):
                        b = max(b, a.value)
        return b

    def max_buffer_cap(self) -> float:
        """Unconditional upper bound on B^max."""
        m = C.MAX_BUFFER_MAX
        for c in self.constraints:
            if not c.when:
                for a in c.then:
                    if a.var == "max_buffer" and a.op in ("<=", "<"):
                        m = min(m, a.value)
        return m

    def features(self, top_bitrate) -> list:
        return self.weights.as_vector() + constraint_features(self.constraints, top_bitrate)


def default_spec() -> ServiceSpec:
    return ServiceSpec().compile()


def extract_json(text: str) -> dict:
    """Robustly extract the first JSON object from an LLM reply."""
    if text is None:
        raise SpecError("empty LLM reply")
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)       # reasoning models
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.S)
    cand = m.group(1) if m else None
    if cand is None:
        start = t.find("{")
        if start < 0:
            raise SpecError("no JSON object found in reply")
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{": depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    cand = t[start:i + 1]; break
        if cand is None:
            raise SpecError("unbalanced braces in reply")
    try:
        return json.loads(cand)
    except json.JSONDecodeError as e:
        raise SpecError(f"invalid JSON: {e}")


# --------------------------------------------------------- training sampler
def random_spec(rng: np.random.Generator, bitrates=C.BITRATES_KBPS) -> ServiceSpec:
    """Random (but realistic) specs used to train the spec-conditioned policy, so
    that any spec produced later by the LLM is in-distribution."""
    prio = {k: int(rng.integers(1, 6)) for k in C.PRIORITY_KEYS}
    cons = []
    if rng.random() < 0.5:
        cons.append(f"ALWAYS buffer >= {rng.choice([1, 2, 3, 4, 5])}")
    if rng.random() < 0.4:
        cons.append(f"ALWAYS max_buffer <= {rng.choice([6, 8, 10, 15, 20, 30])}")
    if rng.random() < 0.3:
        cons.append(f"ALWAYS bitrate <= {rng.choice(bitrates[2:])}")
    if rng.random() < 0.3:
        cons.append(f"ALWAYS switch <= {rng.choice([1, 2])}")
    if rng.random() < 0.3:
        lv = int(rng.integers(2, len(bitrates) - 1))
        cons.append(f"ALWAYS bitrate >= {bitrates[lv]} WHEN pred_bw >= {int(bitrates[lv] * rng.choice([1.5, 2.0, 3.0]))}")
    prof = str(rng.choice(list(C.NETWORK_PROFILES)))
    return ServiceSpec("random", prof, prio, cons).compile()
