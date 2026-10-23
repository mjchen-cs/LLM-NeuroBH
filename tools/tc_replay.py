"""Replay a bandwidth trace on the media server's egress with Linux tc (testbed with ground truth).

Only traffic FROM the media-server port (video downloads) is shaped, so SSH etc. are unaffected.
Every rate change is appended to --gt-log as "epoch_seconds,kbps"; deploy/media_server.py reads
this file to attach the ground-truth bandwidth to every segment.

Requires root.  Run it on the SAME machine as media_server.py.
    sudo python3 tc_replay.py --dev eth0 --port 8080 --trace traces/hsdpa/xxx.txt --gt-log gt.csv
    sudo python3 tc_replay.py --dev eth0 --port 8080 --traces-zip data/traces/oboe.zip \
         --gt-log gt.csv --minutes 120 --scale 1.0 --delay 20
    sudo python3 tc_replay.py --dev eth0 --clear            # remove the shaping
Trace format (Pensieve): one line per sample "<time_s> <bandwidth_Mbps>".
The real path between server and client must be FASTER than the shaped rate, otherwise the
ground truth is wrong: use traces with rates below the measured capacity (iperf3) of your path.
"""
import argparse
import io
import os
import random
import subprocess
import sys
import time
import zipfile


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"failed: {cmd}\n{r.stderr}")
    return r


def setup(dev, port, kbps, delay_ms):
    sh(f"tc qdisc del dev {dev} root", check=False)
    sh(f"tc qdisc add dev {dev} root handle 1: htb default 10")
    sh(f"tc class add dev {dev} parent 1: classid 1:10 htb rate 10gbit")              # everything else
    sh(f"tc class add dev {dev} parent 1: classid 1:20 htb rate {kbps}kbit ceil {kbps}kbit")
    leaf = f"netem delay {delay_ms}ms limit 10000" if delay_ms > 0 else "fq_codel"
    sh(f"tc qdisc add dev {dev} parent 1:20 handle 20: {leaf}")
    sh(f"tc filter add dev {dev} protocol ip parent 1: prio 1 u32 match ip sport {port} 0xffff flowid 1:20")
    sh(f"tc filter add dev {dev} protocol ipv6 parent 1: prio 2 u32 match ip6 sport {port} 0xffff flowid 1:20",
       check=False)


def set_rate(dev, kbps):
    kbps = max(int(kbps), 64)
    sh(f"tc class change dev {dev} parent 1: classid 1:20 htb rate {kbps}kbit ceil {kbps}kbit")
    return kbps


def load_one(text):
    pts = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 2:
            try:
                pts.append((float(p[0]), float(p[1])))
            except ValueError:
                pass
    return pts


def iter_traces(a):
    if a.trace:
        yield os.path.basename(a.trace), load_one(open(a.trace).read())
        return
    z = zipfile.ZipFile(a.traces_zip)
    names = [n for n in z.namelist() if not n.endswith("/")]
    random.Random(a.seed).shuffle(names)
    for n in names:
        yield n, load_one(z.read(n).decode(errors="ignore"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True, help="egress interface (ip route get <client ip>)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--trace"); ap.add_argument("--traces-zip")
    ap.add_argument("--gt-log", default="gt.csv")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply trace bandwidth")
    ap.add_argument("--cap-mbps", type=float, default=50.0, help="clip to this rate (path capacity)")
    ap.add_argument("--delay", type=int, default=0, help="extra one-way delay in ms (netem)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clear", action="store_true")
    a = ap.parse_args()
    if os.geteuid() != 0:
        sys.exit("run as root (sudo)")
    if a.clear:
        sh(f"tc qdisc del dev {a.dev} root", check=False); print("shaping removed"); return
    if not (a.trace or a.traces_zip):
        sys.exit("give --trace or --traces-zip")
    setup(a.dev, a.port, 5000, a.delay)
    end = time.time() + a.minutes * 60
    gt = open(a.gt_log, "a")
    try:
        while time.time() < end:
            for name, pts in iter_traces(a):
                if len(pts) < 10:
                    continue
                print(f"[tc] trace {name} ({len(pts)} samples)")
                t_base, w0 = pts[0][0], time.time()
                for t, mbps in pts:
                    wait = w0 + (t - t_base) - time.time()
                    if wait > 0:
                        time.sleep(wait)
                    kbps = set_rate(a.dev, min(mbps * a.scale, a.cap_mbps) * 1000)
                    gt.write(f"{time.time():.3f},{kbps}\n"); gt.flush()
                    if time.time() >= end:
                        break
                if time.time() >= end:
                    break
            if a.trace is None and a.traces_zip is None:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sh(f"tc qdisc del dev {a.dev} root", check=False)
        print("shaping removed")


if __name__ == "__main__":
    main()
