"""NeuroBH media server: serves the DASH files and measures bandwidth at the transport layer.

Runs on a LINUX machine (cloud VM or lab server); it reads the kernel's TCP_INFO of each
persistent ("stable") HTTP connection while a segment is being sent:
    * the ACK progress (tcpi_bytes_acked) is followed until the segment is fully acknowledged;
      the ACK rate between 50% and 95% of the segment -> stable-connection bandwidth sample
      (excludes the request round trip and the initial burst; the connection is persistent)
    * fallback: bytes acknowledged / transfer time
The sample of segment i is returned by   GET /neurobh/bw?req=<id>   (the player calls it right
after segment i has been received; <id> comes from the response header X-NeuroBH-Req).  The
header X-NeuroBH-Bandwidth of segment i carries the latest sample known when it started.

Testbed mode (--gt-log, written by tools/tc_replay.py on the same machine): the shaped
bandwidth is averaged over each transfer and logged as ground truth.

Per-segment log (CSV, --log):  req, client, path, level, bytes, t_start, t_end, server_kbps,
                               goodput_kbps, gt_kbps, cc, min_rtt_ms, retrans
Usage
    sudo modprobe tcp_bbr                     # once, for --cc bbr
    python3 media_server.py --root dash/ --port 8080 --cc bbr --log server_log.csv
    python3 media_server.py --root dash/ --port 8080 --cc cubic --gt-log gt.csv --log server_log.csv
Without tc (kernel lacks sch_htb/netem, e.g. Jetson):
    python3 media_server.py --root dash/ --port 8080 --cc cubic --pace-zip hsdpa.zip --log server_log.csv
    (per-socket pacing cap follows the traces; ground truth -> gt_pace.csv)
Optional TLS: --cert fullchain.pem --key privkey.pem (needed only if the player page is https).
Only the Python standard library is required.
"""
import argparse
import bisect
import csv
import json
import mimetypes
import os
import re
import socket
import ssl
import statistics
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

TCP_INFO_LEN = 232
SEG_RE = re.compile(r"chunk-(\d+)-(\d+)\.(m4s|webm)$")


def tcp_info(sock):
    """Subset of Linux struct tcp_info (include/uapi/linux/tcp.h)."""
    b = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, TCP_INFO_LEN)
    if len(b) < 168:
        return None
    return {"app_limited": b[7] & 1,
            "retrans": struct.unpack_from("I", b, 100)[0],          # tcpi_total_retrans
            "rtt_us": struct.unpack_from("I", b, 68)[0],
            "cwnd": struct.unpack_from("I", b, 80)[0],
            "bytes_acked": struct.unpack_from("Q", b, 120)[0],
            "notsent": struct.unpack_from("I", b, 144)[0],
            "min_rtt_us": struct.unpack_from("I", b, 148)[0],
            "unacked": struct.unpack_from("I", b, 24)[0],
            "delivery_rate": struct.unpack_from("Q", b, 160)[0]}    # bytes/s


class GroundTruth:
    """Shaped rate written by tools/tc_replay.py: CSV 'epoch,kbps' (one row per change)."""
    def __init__(self, path):
        self.path, self.t, self.v, self.pos, self.lock = path, [], [], 0, threading.Lock()

    def _refresh(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            f.seek(self.pos)
            for line in f:
                if not line.endswith("\n"):
                    break
                self.pos += len(line)
                try:
                    t, v = line.split(",")[:2]
                    self.t.append(float(t)); self.v.append(float(v))
                except ValueError:
                    pass

    def mean(self, t0, t1):
        with self.lock:
            self._refresh()
            if not self.t or t1 <= t0:
                return None
            tot, i = 0.0, max(bisect.bisect_right(self.t, t0) - 1, 0)
            a = t0
            while a < t1:
                b = min(t1, self.t[i + 1]) if i + 1 < len(self.t) else t1
                if b > a:
                    tot += self.v[i] * (b - a)
                a, i = b, i + 1
                if i >= len(self.t):
                    i = len(self.t) - 1
                    tot += self.v[i] * (t1 - a); break
            return tot / (t1 - t0)


SO_MAX_PACING_RATE = getattr(socket, "SO_MAX_PACING_RATE", 47)


class Pacer:
    """Trace replay WITHOUT tc: every open connection gets a per-socket pacing cap
    (SO_MAX_PACING_RATE, enforced by TCP's internal pacing; no qdisc kernel module needed).
    The cap follows a bandwidth trace and is logged as ground truth.  Use it when the kernel
    has no sch_htb / sch_netem (e.g. Jetson L4T).  The real path must be faster than the cap."""
    def __init__(self, path, zip_path, gt_path, scale=1.0, cap_mbps=50.0, seed=0):
        self.path, self.zip_path, self.scale, self.cap = path, zip_path, scale, cap_mbps
        self.socks, self.lock, self.kbps = set(), threading.Lock(), 5000.0
        self.gt = open(gt_path, "a")
        self.traces = self._load(seed)                  # fixed, seeded order -> reproducible
        self.idx, self.restart = 0, threading.Event()
        print(f"[pace] {len(self.traces)} traces loaded")
        threading.Thread(target=self._run, daemon=True).start()

    def _load(self, seed):
        import random
        import zipfile
        if self.path:
            items = [(os.path.basename(self.path), open(self.path).read())]
        else:
            z = zipfile.ZipFile(self.zip_path)
            items = [(n, z.read(n).decode(errors="ignore")) for n in sorted(z.namelist()) if not n.endswith("/")]
        out = []
        for name, text in items:
            pts = []
            for line in text.splitlines():
                p_ = line.split()
                try:
                    pts.append((float(p_[0]), float(p_[1])))
                except (ValueError, IndexError):
                    pass
            if len(pts) >= 10:
                out.append((name, pts))
        random.Random(seed).shuffle(out)
        return out

    def select(self, k):
        """Restart the replay at the beginning of trace k (GET /neurobh/pace?round=k).  The session
        runner calls this before every session, so all policies of round k see the SAME bandwidth."""
        with self.lock:
            self.idx = int(k) % len(self.traces)
        self.restart.set()
        return self.traces[self.idx][0]

    def apply(self, sock):
        try:
            sock.setsockopt(socket.SOL_SOCKET, SO_MAX_PACING_RATE, int(self.kbps * 1000 / 8))
        except OSError as e:
            print(f"[pace] setsockopt failed: {e}")

    def register(self, sock):
        with self.lock:
            self.socks.add(sock); self.apply(sock)

    def unregister(self, sock):
        with self.lock:
            self.socks.discard(sock)

    def _set(self, kbps):
        with self.lock:
            self.kbps = kbps
            for s_ in list(self.socks):
                if s_.fileno() < 0:
                    self.socks.discard(s_)
                else:
                    self.apply(s_)
        self.gt.write(f"{time.time():.3f},{kbps:.1f}\n"); self.gt.flush()

    def _run(self):
        while True:
            self.restart.clear()
            with self.lock:
                name, pts = self.traces[self.idx]
            print(f"[pace] trace #{self.idx} {name} ({len(pts)} samples)")
            t_base, w0 = pts[0][0], time.time()
            for t, mbps in pts:
                while True:                              # sleep in small steps so select() takes effect at once
                    wait = w0 + (t - t_base) - time.time()
                    if wait <= 0 or self.restart.is_set():
                        break
                    time.sleep(min(wait, 0.05))
                if self.restart.is_set():
                    break
                self._set(max(min(mbps * self.scale, self.cap) * 1000, 100.0))
            else:                                        # trace finished -> continue with the next one
                with self.lock:
                    self.idx = (self.idx + 1) % len(self.traces)


def build(root, cc, gt, log_path, video_levels=6, pacer=None):
    results, events, last_by_client = {}, {}, {}
    lock = threading.Lock()
    counter = [0]
    log_new = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="")
    logw = csv.writer(logf)
    if log_new:
        logw.writerow(["req", "client", "path", "level", "bytes", "t_start", "t_end", "server_kbps",
                       "goodput_kbps", "gt_kbps", "cc", "min_rtt_ms", "retrans", "delivery_rate_kbps"])

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"                       # persistent connections

        def setup(self):
            super().setup()
            if cc:
                try:
                    self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, cc.encode())
                except OSError as e:
                    print(f"[media] cannot set congestion control '{cc}': {e} (sudo modprobe tcp_{cc}?)")
            if pacer:
                pacer.register(self.connection)

        def finish(self):
            if pacer:
                pacer.unregister(self.connection)
            super().finish()

        def cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "X-NeuroBH-Req, X-NeuroBH-Bandwidth")
            self.send_header("Cache-Control", "no-store")

        def do_OPTIONS(self):
            self.send_response(204); self.cors()
            self.send_header("Access-Control-Allow-Headers", "*"); self.send_header("Content-Length", "0")
            self.end_headers()

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code); self.cors()
            self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            client = self.client_address[0]
            if u.path == "/neurobh/pace":
                if pacer is None:
                    return self._json({"pacing": False})
                k = parse_qs(u.query).get("round", ["0"])[0]
                return self._json({"pacing": True, "round": k, "trace": pacer.select(k)})
            if u.path == "/neurobh/bw":
                rid = parse_qs(u.query).get("req", [""])[0]
                ev = events.get(rid)
                if ev is not None:
                    ev.wait(3.0)                            # until the segment has been fully acknowledged
                with lock:
                    r = results.get(rid) or last_by_client.get(client)
                return self._json(r or {"kbps": None})
            path = os.path.normpath(os.path.join(root, unquote(u.path).lstrip("/")))
            if not path.startswith(os.path.abspath(root)) or not os.path.isfile(path):
                return self._json({"error": "not found"}, 404)
            with open(path, "rb") as f:
                data = f.read()
            m = SEG_RE.search(path)
            if m and int(m.group(1)) >= video_levels:      # audio representation: not measured
                m = None
            with lock:
                counter[0] += 1
                rid = f"{int(time.time())}-{counter[0]}"
                last = last_by_client.get(client, {}).get("kbps")
                if m:
                    events[rid] = threading.Event()
            self.send_response(200); self.cors()
            self.send_header("Content-Type", mimetypes.guess_type(path)[0] or
                             ("video/iso.segment" if path.endswith(".m4s") else "video/webm" if path.endswith(".webm") else "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            if m:
                self.send_header("X-NeuroBH-Req", rid)
                if last:
                    self.send_header("X-NeuroBH-Bandwidth", f"{last:.1f}")
            self.end_headers()
            if not m:                                       # manifest / init segment / audio: no measurement
                self.wfile.write(data); return
            sock = self.connection
            raw = sock if not isinstance(sock, ssl.SSLSocket) else sock
            info0 = tcp_info(raw)
            t0 = time.time()
            acked0 = info0["bytes_acked"] if info0 else 0
            timeline = []                                   # (time, bytes acknowledged in this transfer)
            step = 64 * 1024
            for off in range(0, len(data), step):
                self.wfile.write(data[off:off + step])
                ti = tcp_info(raw)
                if ti:
                    timeline.append((time.time(), ti["bytes_acked"] - acked0))
            self.wfile.flush()
            deadline = time.time() + 10.0                   # follow the ACKs until everything is acknowledged
            ti = tcp_info(raw)
            while ti and (ti["notsent"] > 0 or ti["unacked"] > 0) and time.time() < deadline:
                time.sleep(0.002)
                ti = tcp_info(raw)
                if ti:
                    timeline.append((time.time(), ti["bytes_acked"] - acked0))
            t1 = time.time()
            acked = (ti["bytes_acked"] - acked0) if (ti and info0) else len(data)
            goodput = acked * 8 / 1000 / max(t1 - t0, 1e-3)
            # stable-connection sample: ACK rate over the second half of the segment (bytes), which
            # excludes request latency and the initial window burst of the persistent connection
            # (between 50% and 95% of the bytes: the last ACK can be held back by delayed ACKs)
            server = goodput
            if acked > 0:
                a_ = next(((t, b_) for t, b_ in timeline if b_ >= 0.50 * acked), None)
                z_ = next(((t, b_) for t, b_ in timeline if b_ >= 0.95 * acked), None)
                if a_ and z_ and z_[0] - a_[0] > 0.02 and z_[1] > a_[1]:
                    server = (z_[1] - a_[1]) * 8 / 1000 / (z_[0] - a_[0])
            delivery = (ti or {}).get("delivery_rate", 0) * 8 / 1000
            g = gt.mean(t0, t1) if gt else None
            row = {"req": rid, "kbps": round(server, 1), "goodput_kbps": round(goodput, 1),
                   "gt_kbps": None if g is None else round(g, 1), "t_start": t0, "t_end": t1}
            with lock:
                results[rid] = row; last_by_client[client] = row
                if len(results) > 20000:                    # bound memory
                    for k in list(results)[:10000]:
                        results.pop(k, None); events.pop(k, None)
                logw.writerow([rid, client, u.path, m.group(1), len(data), f"{t0:.3f}", f"{t1:.3f}",
                               f"{server:.1f}", f"{goodput:.1f}", "" if g is None else f"{g:.1f}", cc or "",
                               f"{(ti or {}).get('min_rtt_us', 0) / 1000:.1f}", (ti or {}).get("retrans", ""),
                               f"{delivery:.1f}"])
                logf.flush()
                events[rid].set()

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder that contains the DASH folders (e.g. dash/)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cc", default="bbr", help="reno | cubic | bbr  (per-connection congestion control)")
    ap.add_argument("--gt-log", default=None, help="ground-truth file written by tools/tc_replay.py")
    ap.add_argument("--log", default="server_log.csv")
    ap.add_argument("--video-levels", type=int, default=6,
                    help="representations 0..N-1 are video (tools/encode_dash.py: 6); higher ids = audio")
    ap.add_argument("--cert"); ap.add_argument("--key")
    ap.add_argument("--pace-trace", help="replay this trace with per-socket pacing (no tc needed)")
    ap.add_argument("--pace-zip", help="replay random traces from this zip with per-socket pacing")
    ap.add_argument("--pace-gt-log", default="gt_pace.csv")
    ap.add_argument("--pace-scale", type=float, default=1.0)
    ap.add_argument("--pace-cap-mbps", type=float, default=50.0)
    a = ap.parse_args()
    if not hasattr(socket, "TCP_INFO"):
        raise SystemExit("TCP_INFO is not available: run this server on Linux")
    root = os.path.abspath(a.root)
    pacer = None
    if a.pace_trace or a.pace_zip:
        pacer = Pacer(a.pace_trace, a.pace_zip, a.pace_gt_log, a.pace_scale, a.pace_cap_mbps)
        a.gt_log = a.pace_gt_log                     # ground truth = the pacing cap
        time.sleep(0.5)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), build(root, a.cc, GroundTruth(a.gt_log) if a.gt_log else None, a.log,
                                                a.video_levels, pacer))
    if a.cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(a.cert, a.key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    print(f"[media] serving {root} on :{a.port}  cc={a.cc}  gt={a.gt_log}  log={a.log}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
