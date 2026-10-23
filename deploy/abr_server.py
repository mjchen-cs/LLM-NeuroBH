"""ABR decision service for real-world experiments (dash.js + HTTP/3 server).

All learned AND classic policies run behind the same endpoint, so every
algorithm sees exactly the same player integration:

  POST /intent  {"session": "s1", "text": "<natural-language intent>"}
        -> LLM Intent service + verifier; binds the verified spec to the session
  POST /decide  {"session": "s1", "policy": "neurobh", "buffer": 7.2, "max_buffer": 20,
                 "prev_level": 2, "samples_kbps": [...], "dl_times": [...], "levels": [...],
                 "next_sizes": [bytes per level], "remaining": 30, "chunk": 18}
        -> {"level": 3, "max_buffer": 22.0, "intervened": false, "pred_bw": 5321.0}
        policy in: neurobh | llm-neurobh (uses the session spec) | neurobh-smooth | pensieve | bba | bola |
                   festive | dynamic | robustmpc
  POST /report  {"network": "wifi", "policy": "...", "session": "...", "qoe": ..., "bitrate_mbps": ...,
                 "rebuf_ratio": ..., "latency_s": ..., "fluctuation": ..., "startup_s": ...}
        -> appends one row to data/realworld.csv  (plotted by experiments/plot_testbed.py)

Run:  python deploy/abr_server.py --ckpt ckpt/neurobh.pt --pensieve ckpt/pensieve.pt --llm glm-4-flash
Then open deploy/player.html?policy=neurobh&network=wifi  (served by any static web server).
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from neurobh import config as C                               # noqa: E402
from neurobh.agent import Agent                               # noqa: E402
from neurobh.env import StreamingEnv                          # noqa: E402
from neurobh.policies import BBA, BOLA, FESTIVE, Dynamic, RobustMPC, NeuroBHPolicy   # noqa: E402
from neurobh.predictor import make as make_pred               # noqa: E402
from neurobh.shield import Shield                             # noqa: E402
from neurobh.spec import default_spec                         # noqa: E402
from neurobh.video import Video, vmaf_model                   # noqa: E402

REPORT_FIELDS = ["network", "video", "policy", "session", "qoe", "qoe_vmaf", "vmaf", "bitrate_mbps", "rebuf_ratio",
                 "latency_s", "fluctuation", "startup_s", "intent"]
CHUNK_FIELDS = ["time", "network", "policy", "session", "chunk", "req", "bytes", "dl_time", "client_kbps",
                "server_kbps", "sample_used_kbps", "pred_bw", "buffer", "level", "max_buffer", "intervened"]


class ShimEnv:
    """Just enough of StreamingEnv for policies to act on a live request."""
    def __init__(self, req, samples, horizon=5):
        self.bitrates, self.K, self.L = list(C.BITRATES_KBPS), len(C.BITRATES_KBPS), C.CHUNK_LEN
        self.samples = samples
        sizes = np.tile(np.asarray(req["next_sizes"], dtype=float)[:, None], (1, horizon))
        vm = np.tile(np.asarray(req.get("next_vmaf") or vmaf_model(self.bitrates))[:, None], (1, horizon))
        self.video = Video(sizes, vm, self.bitrates, "live")
        self.video.n_chunks = int(req.get("chunk", 0)) + int(req.get("remaining", 1))
        self.spec = None

    def observe(self, s, spec):
        return StreamingEnv.observe(self, s, spec)


SMOOTH_CONSTRAINT = "ALWAYS switch <= 0 WHEN buffer < 16 AND pred_bw < 1500"


def smooth_spec():
    """Default spec + the hysteresis constraint an operator intent such as "on weak networks,
    prefer smooth playback over quick up-switches" compiles to (see experiments/exp_smooth.py)."""
    from neurobh.spec import ServiceSpec
    return ServiceSpec(priorities=dict(default_spec().priorities), constraint_texts=[SMOOTH_CONSTRAINT]).compile()


def build_handler(agents, predictor, llm_service, report_path):
    sessions, lock = {}, threading.Lock()
    chunk_path = os.path.join(os.path.dirname(report_path) or ".", "realworld_chunks.csv")

    def log_chunk(row):
        with lock:
            new = not os.path.exists(chunk_path)
            with open(chunk_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CHUNK_FIELDS, extrasaction="ignore")
                if new: w.writeheader()
                w.writerow(row)
    classic = {"bba": BBA, "bola": BOLA, "festive": FESTIVE, "dynamic": Dynamic}

    class H(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)

        def do_OPTIONS(self):
            self.send_response(204); self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type"); self.end_headers()

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            sid = req.get("session", "default")
            if self.path == "/intent":
                if llm_service is None:
                    return self._send({"error": "server started without --llm"}, 400)
                res = llm_service.compile(req["text"])
                sessions.setdefault(sid, {})["spec"] = res.spec
                return self._send({"spec": res.spec.to_dict(), "verified": res.verified,
                                   "fallback": res.fallback, "rounds": res.rounds})
            if self.path == "/report":
                with lock:
                    new = not os.path.exists(report_path)
                    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
                    if not new:                       # file written by an older version: migrate header
                        with open(report_path, newline="", encoding="utf-8") as f:
                            rd = csv.DictReader(f); old_rows = list(rd); old_fields = rd.fieldnames or []
                        if old_fields != REPORT_FIELDS:
                            with open(report_path, "w", newline="", encoding="utf-8") as f:
                                w = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction="ignore")
                                w.writeheader(); w.writerows(old_rows)
                    with open(report_path, "a", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction="ignore")
                        if new: w.writeheader()
                        w.writerow(req)
                return self._send({"ok": True})
            if self.path != "/decide":
                return self._send({"error": "unknown endpoint"}, 404)

            pol_name = req.get("policy", "neurobh").lower()
            ss = sessions.setdefault(sid, {})
            if pol_name == "llm-neurobh":
                spec = ss.get("spec", default_spec())
            elif pol_name == "neurobh-smooth":            # fixed intent "weak network: smooth playback first"
                spec = smooth_spec()
            else:
                spec = default_spec()
            samples = [float(x) for x in req.get("samples_kbps", [])] or [1000.0]
            pred = predictor.predict(samples) if pol_name in ("neurobh", "llm-neurobh", "neurobh-smooth") else \
                make_pred("robust" if pol_name == "robustmpc" else "harmonic").predict(samples)
            hist = lambda k: ([0.0] * C.HIST_LEN + [float(x) for x in req.get(k, [])])[-C.HIST_LEN:]
            s = {"buffer": float(req["buffer"]), "max_buffer": float(req.get("max_buffer", C.MAX_BUFFER_INIT)),
                 "prev_level": int(req.get("prev_level", 0)), "pred_bw": pred,
                 "next_sizes": [float(x) for x in req["next_sizes"]],
                 "next_vmaf": req.get("next_vmaf") or list(vmaf_model(C.BITRATES_KBPS)),
                 "is_dynamic": bool(req.get("is_dynamic", False)), "hist_tput": hist("samples_kbps"),
                 "hist_dl": hist("dl_times"), "hist_lv": hist("levels"),
                 "remaining": int(req.get("remaining", 1)), "chunk": 0}
            env = ShimEnv(req, samples)
            key = (sid, pol_name)
            if key not in ss:                                 # one stateful policy object per session
                if pol_name in ("neurobh", "llm-neurobh", "neurobh-smooth"):
                    ss[key] = NeuroBHPolicy(agents["neurobh"], spec)
                elif pol_name == "pensieve":
                    ss[key] = NeuroBHPolicy(agents["pensieve"], default_spec(), use_shield=False)
                elif pol_name == "robustmpc":
                    ss[key] = RobustMPC(default_spec())
                else:
                    ss[key] = classic[pol_name]()
                if hasattr(ss[key], "reset"): ss[key].reset()
            # the maximum buffer is SERVER state (as in the simulator): it starts at the spec's initial
            # value clipped to its max_buffer cap, e.g. 10 s for "ALWAYS max_buffer <= 10"; the value
            # sent by the player is ignored, otherwise a too large start value violates the spec forever
            mb_key = key + ("max_buffer",)
            if pol_name in ("neurobh", "llm-neurobh", "neurobh-smooth"):
                if mb_key not in ss:
                    ss[mb_key] = float(min(spec.weights.init_max_buffer, spec.max_buffer_cap()))
                s["max_buffer"] = ss[mb_key]
            lv, d, inter = ss[key].act(env, s)
            if pol_name in ("neurobh", "llm-neurobh", "neurobh-smooth"):
                mb = float(np.clip(s["max_buffer"] + C.BUF_DELTAS[d], C.MAX_BUFFER_MIN, C.MAX_BUFFER_MAX))
                ss[mb_key] = mb
            else:
                mb = 60.0
            last = req.get("last") or {}
            if last:                     # measurement of the segment just received + this decision
                log_chunk({"time": f"{time.time():.3f}", "network": req.get("network", ""), "policy": pol_name,
                           "session": sid, "chunk": req.get("chunk"), "req": last.get("req"),
                           "bytes": last.get("bytes"), "dl_time": last.get("dl_time"),
                           "client_kbps": last.get("client_kbps"), "server_kbps": last.get("server_kbps"),
                           "sample_used_kbps": samples[-1], "pred_bw": pred, "buffer": s["buffer"],
                           "level": int(lv), "max_buffer": mb, "intervened": int(bool(inter))})
            return self._send({"level": int(lv), "max_buffer": mb, "intervened": bool(inter), "pred_bw": pred})

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/neurobh.pt")
    ap.add_argument("--pensieve", default="ckpt/pensieve.pt")
    ap.add_argument("--predictor", default="ckpt/predictor.pt")
    ap.add_argument("--llm", default=None, help="model name in configs/llm.yaml (enables /intent)")
    ap.add_argument("--ladder", default=None, help="comma list of kbps of YOUR manifest (must have as many levels as training)")
    ap.add_argument("--report", default="data/realworld.csv")
    ap.add_argument("--port", type=int, default=8333)
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.ladder:
        C.BITRATES_KBPS[:] = [int(x) for x in a.ladder.split(",")]
    agents = {"neurobh": Agent.load(a.ckpt)}
    if os.path.exists(a.pensieve):
        agents["pensieve"] = Agent.load(a.pensieve)
    pred = make_pred("gru", path=a.predictor) if os.path.exists(a.predictor) else make_pred("harmonic")
    svc = None
    if a.llm:
        from neurobh.llm.client import load_clients
        from neurobh.llm.intent_service import IntentService
        svc = IntentService(load_clients("configs/llm.yaml", [a.llm])[a.llm])
    print(f"ABR decision service on :{a.port}  (policies: neurobh, llm-neurobh, pensieve, bba, bola, festive, dynamic, robustmpc)")
    ThreadingHTTPServer(("0.0.0.0", a.port), build_handler(agents, pred, svc, a.report)).serve_forever()


if __name__ == "__main__":
    main()
