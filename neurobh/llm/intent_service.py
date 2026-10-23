"""LLM Intent-to-Spec service with verifier-in-the-loop repair.

    intent --LLM--> JSON ServiceSpec --Verifier--> ok ?  -> deploy to ABR service
                          ^                          | no
                          +----- feedback (<=K) -----+  -> fallback to the default safe spec

Ablation switches: use_verifier=False (accept any parsable spec),
use_repair=False (no feedback round), few_shot=False.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .. import config as C
from ..spec import ServiceSpec, default_spec, extract_json, SpecError
from ..verifier import Verifier
from .prompts import build_messages, repair_message


@dataclass
class IntentResult:
    spec: ServiceSpec
    verified: bool                  # final spec passed the verifier
    fallback: bool                  # default spec was used
    rounds: int                     # number of LLM calls
    first_stage: str                # verifier stage of the FIRST reply ("passed" if correct at once)
    stages: List[str] = field(default_factory=list)
    llm_latency: float = 0.0        # seconds, sum over rounds
    verify_latency: float = 0.0
    replies: List[str] = field(default_factory=list)


class IntentService:
    def __init__(self, llm, verifier: Optional[Verifier] = None, max_rounds=3, use_verifier=True,
                 use_repair=True, few_shot=True, bitrates=C.BITRATES_KBPS, L=C.CHUNK_LEN):
        self.llm = llm
        self.verifier = verifier or Verifier(bitrates, L)
        self.max_rounds = max_rounds if use_repair else 1
        self.use_verifier = use_verifier
        self.few_shot = few_shot
        self.bitrates, self.L = bitrates, L

    def compile(self, intent: str) -> IntentResult:
        msgs = build_messages(intent, self.bitrates, self.L, self.few_shot)
        res = IntentResult(default_spec(), False, True, 0, "", [])
        for k in range(self.max_rounds):
            r = self.llm.chat(msgs)
            res.rounds += 1
            res.llm_latency += r["latency"]
            res.replies.append(r["text"])
            t0 = time.time()
            v = self.verifier.verify_text(r["text"])
            res.verify_latency += time.time() - t0
            res.stages.append(v.stage)
            if k == 0:
                res.first_stage = v.stage
            if not self.use_verifier:
                # ablation: accept anything that parses (json + DSL syntax)
                if v.spec is not None and v.stage not in ("json", "schema", "syntax"):
                    res.spec, res.fallback, res.verified = v.spec, False, v.ok
                return res
            if v.ok:
                res.spec, res.verified, res.fallback = v.spec, True, False
                return res
            msgs = msgs + [{"role": "assistant", "content": r["text"]}, repair_message(v.feedback())]
        return res
