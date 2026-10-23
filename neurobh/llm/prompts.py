"""Prompts of the LLM Intent-to-Spec service (zero-training, few-shot)."""
import json
from .. import config as C

SYSTEM_PROMPT = """You are the service-design assistant of an HTTP adaptive video streaming platform.
You translate a natural-language service intent (from an operator, a content provider or an SLA)
into a machine-checkable ServiceSpec that configures the ABR (adaptive bitrate) service.

Output ONLY one JSON object, no prose, with exactly these fields:
{
  "scenario": "<short snake_case label>",
  "network_profile": "slow" | "medium" | "fast" | "unknown",
  "priorities": {"quality": 1-5, "smoothness": 1-5, "stall_avoidance": 1-5, "data_saving": 1-5, "low_latency": 1-5},
  "constraints": ["<DSL constraint>", ...],
  "explanation": "<one short sentence>"
}

network_profile: slow = <= 6 Mbps (3G/HSDPA, crowded metro, rural), medium = 6-100 Mbps (4G, home/public WiFi),
fast = >= 100 Mbps (5G, fibre), unknown = not stated.
priorities: 1 = not important, 3 = normal, 5 = critical. Use 3 when the intent says nothing.

Constraint DSL (one constraint per string):
  ALWAYS <atom> [AND <atom>...] [WHEN <atom> [AND <atom>...]]
  atom := <var> <op> <number>       op in  <=  >=  <  >      (no OR, no '=')
  variables and units:
    buffer      seconds of video in the playback buffer (ALWAYS buffer >= x means it never drops below x s)
    max_buffer  maximum buffer the player may keep, seconds (bounds latency and wasted data)
    bitrate     bitrate of the next chunk, kbps
    switch      size of an up-switch in ladder steps (down-switches are always allowed)
    pred_bw     predicted bandwidth, kbps (use in WHEN for conditional rules)
  Convert units: 1 Mbps = 1000 kbps; ms -> s.
  Available bitrate ladder (kbps): LADDER.  Chunk length: CHUNK s.

Rules:
- Encode every explicit numeric requirement of the intent as a constraint; do NOT invent constraints that the
  intent does not imply.  Soft wishes ("prefer", "try to") go into priorities, not constraints.
- A latency target of T seconds means ALWAYS max_buffer <= T.
- "No stalls / no rebuffering" without a number -> priorities.stall_avoidance = 5 (no constraint).
- Session-level metrics (rebuffer ratio, average bitrate) are NOT variables; express them through
  priorities or through buffer/bitrate constraints.
- A feasible spec needs max_buffer cap >= buffer floor + CHUNK s.
"""

FEW_SHOT = [
    ("A 4G live stream of a cooking class; viewers ask questions in real time so keep latency under 10 seconds; "
     "never let the buffer fall below 2 s.",
     {"scenario": "live_class", "network_profile": "medium",
      "priorities": {"quality": 3, "smoothness": 3, "stall_avoidance": 4, "data_saving": 2, "low_latency": 5},
      "constraints": ["ALWAYS max_buffer <= 10", "ALWAYS buffer >= 2"],
      "explanation": "Interactive live stream: latency cap and buffer floor."}),
    ("用户套餐流量有限，单个视频块码率不能超过1.2Mbps，画质一般即可，但别频繁跳码率。",
     {"scenario": "data_plan", "network_profile": "unknown",
      "priorities": {"quality": 2, "smoothness": 4, "stall_avoidance": 3, "data_saving": 5, "low_latency": 2},
      "constraints": ["ALWAYS bitrate <= 1200"],
      "explanation": "Limited data plan: bitrate cap, prefer smooth playback."}),
]

REPAIR_PROMPT = """The ServiceSpec you returned was rejected by the symbolic verifier:

{feedback}

Return a corrected JSON ServiceSpec (same format, JSON only).  Keep every requirement that the user stated
explicitly unless it is the cause of the conflict; when two requirements conflict, keep the one that protects
playback safety (no stall) first, then latency, then quality, then data saving, and state the relaxation
in "explanation"."""


def system_prompt(bitrates=C.BITRATES_KBPS, L=C.CHUNK_LEN):
    return SYSTEM_PROMPT.replace("LADDER", str(list(bitrates))).replace("CHUNK", f"{L:g}")


def build_messages(intent, bitrates=C.BITRATES_KBPS, L=C.CHUNK_LEN, few_shot=True):
    msgs = [{"role": "system", "content": system_prompt(bitrates, L)}]
    if few_shot:
        for q, a in FEW_SHOT:
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": json.dumps(a, ensure_ascii=False)})
    msgs.append({"role": "user", "content": intent})
    return msgs


def repair_message(feedback):
    return {"role": "user", "content": REPAIR_PROMPT.format(feedback=feedback)}


PARAPHRASE_PROMPT = """Rewrite the following streaming-service requirement in a different wording, as a
{role} would say it.  Keep the same language ({lang}), keep EVERY number and unit and every requirement, do not add
new requirements.  Output only the rewritten sentence(s).

Requirement: {text}"""
