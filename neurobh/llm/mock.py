"""MockLLM -- OFFLINE PIPELINE TEST ONLY.  Never report its numbers.

It looks up the benchmark ground truth of the intent and corrupts it with a
configurable probability (wrong number, unit error, invented variable,
dropped constraint, broken JSON, wrong priority) to exercise the verifier and
the repair loop.  On a repair request it fixes the error with p = 0.7.
"""
import json
import random
import time


class MockLLM:
    def __init__(self, name="mock", error_rate=0.2, seed=0, intents_path="data/intents.jsonl"):
        self.name, self.model = name, "mock"
        self.error_rate = error_rate
        self.rng = random.Random(seed)
        self.gt = {}
        try:
            with open(intents_path, encoding="utf-8") as f:
                for line in f:
                    it = json.loads(line); self.gt[it["text"]] = it["gt"]
        except FileNotFoundError:
            pass

    def _corrupt(self, spec):
        s = json.loads(json.dumps(spec))
        kind = self.rng.choice(["number", "unit", "var", "drop", "json", "prio", "conflict"])
        if kind == "json":
            return json.dumps(s)[:-3]
        if kind == "number" and s["constraints"]:
            c = s["constraints"][0]; toks = c.split()
            for i, t in enumerate(toks):
                if t.replace(".", "").isdigit():
                    toks[i] = str(int(float(t) * self.rng.choice([0.5, 2, 10]))); break
            s["constraints"][0] = " ".join(toks)
        elif kind == "unit" and s["constraints"]:
            s["constraints"][0] = s["constraints"][0] + " ms"
        elif kind == "var":
            s["constraints"].append("ALWAYS rebuffer_ratio <= 0.01")
        elif kind == "drop" and s["constraints"]:
            s["constraints"].pop()
        elif kind == "prio":
            k = self.rng.choice(list(s["priorities"])); s["priorities"][k] = self.rng.choice([1, 5])
        elif kind == "conflict":
            s["constraints"].append("ALWAYS max_buffer <= 3")
        return json.dumps(s, ensure_ascii=False)

    def chat(self, messages, use_cache=True):
        t0 = time.time()
        if "ABR) controller" in messages[-1]["content"]:       # LLMDirectABR prompt
            import re
            st = json.loads(messages[-1]["content"].split("Current state (JSON):")[1].strip().splitlines()[0])
            lad = st["bitrate_ladder_kbps"]; bw = st["predicted_bandwidth_kbps"] * self.rng.uniform(0.6, 1.3)
            lv = max([i for i, b in enumerate(lad) if b <= bw] or [0])
            return {"text": json.dumps({"level": lv}), "latency": 0.5 + abs(self.rng.gauss(0.8, 0.4)),
                    "cached": False, "usage": None}
        user_turns = [m for m in messages if m["role"] == "user"]
        known = [m["content"] for m in user_turns if m["content"] in self.gt]
        intent = known[-1] if known else user_turns[-1]["content"]
        is_repair = user_turns[-1]["content"] != intent
        gt = self.gt.get(intent, {"scenario": "unknown", "network_profile": "unknown",
                                  "priorities": {"quality": 3}, "constraints": []})
        spec = dict(gt)
        if gt["scenario"].startswith("conflict"):
            # emulate the LLM keeping the conflict first, relaxing it on repair
            if is_repair and self.rng.random() < 0.8:
                spec = dict(gt); spec["constraints"] = gt["constraints"][1:]
            text = json.dumps(spec, ensure_ascii=False)
        elif is_repair:
            text = json.dumps(spec, ensure_ascii=False) if self.rng.random() < 0.7 else self._corrupt(spec)
        else:
            text = self._corrupt(spec) if self.rng.random() < self.error_rate else json.dumps(spec, ensure_ascii=False)
        lat = 0.6 + abs(self.rng.gauss(0.6, 0.3))   # emulated latency (not slept)
        return {"text": text, "latency": lat, "cached": False, "usage": None}
