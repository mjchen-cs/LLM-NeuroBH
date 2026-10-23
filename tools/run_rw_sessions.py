"""Run real-world streaming sessions automatically (round-robin over policies).

Opens deploy/player.html in YOUR Chrome via Playwright, one session at a time, cycling
through the policies (A B C ... A B C ...) so that all policies see the same network period.
Each session reports one row to data/realworld.csv (and per-segment rows to
data/realworld_chunks.csv) through deploy/abr_server.py, which must already be running.

Setup (once):   pip install playwright        (no browser download needed: uses installed Chrome)
Before running: 1) media server on the Linux machine   2) python deploy/abr_server.py ...
                3) in deploy/:  python -m http.server 8080 --bind 127.0.0.1
Example
    python tools/run_rw_sessions.py --mpd http://<server ip>:8080/bbb/manifest.mpd --network wifi --rounds 10
    python tools/run_rw_sessions.py --mpd ... --network 4g --rounds 10 --policies neurobh,robustmpc,bba
Ctrl+C stops; re-running the same command RESUMES (sessions already in data/realworld.csv are skipped).
If Chrome closes or crashes, it is restarted and the session is retried (up to 3 times).
"""
import argparse
import os
import random
import time
import urllib.parse

POLICIES = ["neurobh", "llm-neurobh", "robustmpc", "pensieve", "bba", "bola", "festive", "dynamic"]
INTENTS = [   # used by llm-neurobh, rotated over the rounds (feasible intents from data/intents.jsonl style)
    "直播带货频道，端到端延迟不超过10秒，尽量不要卡顿。",
    "Users on a limited data plan: keep the buffer under 15 seconds and avoid wasting data.",
    "远程医疗问诊，画质要清晰，任何时候码率不低于1200kbps，卡顿尽量少。",
    "Movie night at home on fibre: maximise picture quality, smooth playback matters.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpd", required=True)
    ap.add_argument("--network", required=True, help="label, e.g. wifi, 4g, 5g, wifi-cubic")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--secs", type=int, default=180)
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--page", default="http://127.0.0.1:8080/player.html")
    ap.add_argument("--channel", default="chrome", help="chrome | msedge | chromium")
    ap.add_argument("--gap", type=float, default=5.0, help="pause between sessions (s)")
    ap.add_argument("--shuffle", action="store_true", help="random policy order inside each round")
    ap.add_argument("--report", default="data/realworld.csv", help="used to resume: finished sessions are skipped")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    pols = a.policies.split(",")
    video = urllib.parse.urlparse(a.mpd).path.rstrip("/").split("/")[-2]
    done = count_done(a.report, a.network, video)
    if any(done.values()):
        print(f"[resume] sessions already in {a.report} for network={a.network}, video={video}: {dict(done)}")

    preflight(a)

    def launch(p):
        return p.chromium.launch(channel=None if a.channel == "chromium" else a.channel, headless=False,
                                 args=["--autoplay-policy=no-user-gesture-required"])

    def one_session(browser, url):
        ctx = browser.new_context(viewport={"width": 1000, "height": 700})   # fresh cache per session
        try:
            page = ctx.new_page()
            page.goto(url)
            t0 = time.time()
            while time.time() - t0 < a.secs + 60:
                time.sleep(2)
                txt = page.inner_text("#log")
                if "reported" in txt or "report failed" in txt:
                    if "report failed" in txt:
                        print("    page log (last lines):\n      " + "\n      ".join(txt.strip().splitlines()[-4:]))
                    return "reported" in txt, time.time() - t0
            print("    page log (last lines):\n      " + "\n      ".join(page.inner_text("#log").strip().splitlines()[-4:]))
            return False, time.time() - t0
        finally:
            try:
                ctx.close()
            except Exception:
                pass

    with sync_playwright() as p:
        browser = launch(p)
        fails = 0
        for r in range(a.rounds):
            order = pols[:]
            if a.shuffle:
                random.Random(r).shuffle(order)
            for pol in order:
                if done.get(pol, 0) > r:              # this round of this policy is already in the CSV
                    continue
                q = {"policy": pol, "network": a.network, "mpd": a.mpd, "secs": a.secs}
                if pol == "llm-neurobh":
                    q["intent"] = INTENTS[r % len(INTENTS)]
                url = a.page + "?" + urllib.parse.urlencode(q)
                for attempt in range(3):
                    print(f"[round {r + 1}/{a.rounds}] {pol} ..." + (f" (retry {attempt})" if attempt else ""),
                          flush=True)
                    if not pace_round(a.mpd, r):          # restart trace r (also on a retry)
                        print("    trace could not be reset -> session not started, retrying", flush=True)
                        time.sleep(10)
                        continue
                    try:
                        ok, dt = one_session(browser, url)
                    except Exception as e:            # page / browser closed or crashed -> restart Chrome
                        print(f"    browser error: {type(e).__name__}: {str(e).splitlines()[0]} -> restarting Chrome",
                              flush=True)
                        try:
                            browser.close()
                        except Exception:
                            pass
                        time.sleep(3)
                        browser = launch(p)
                        continue
                    print(f"    {'ok' if ok else 'NO REPORT (check abr_server / console)'} ({dt:.0f}s)", flush=True)
                    fails = 0 if ok else fails + 1
                    if fails >= 2:
                        browser.close()
                        raise SystemExit("two sessions in a row failed -> stopped. Check that abr_server.py is "
                                         "running in its own terminal, then re-run the same command (it resumes).")
                    break
                time.sleep(a.gap)
        browser.close()


def pace_round(mpd, r, tries=3):
    """If the media server replays traces (--pace-zip), restart trace r: every policy of round r
    then streams over exactly the same bandwidth trace (paired comparison, as in simulation).
    Returns False only if a pacing server could not be reached after `tries` attempts; a server
    without pacing (real network) returns True."""
    import json
    import urllib.request
    u = urllib.parse.urlparse(mpd)
    for i in range(tries):
        try:
            with urllib.request.urlopen(f"{u.scheme}://{u.netloc}/neurobh/pace?round={r}", timeout=10) as resp:
                d = json.loads(resp.read())
            if d.get("pacing"):
                print(f"    [pace] trace for round {r + 1}: {d.get('trace')}", flush=True)
            return True
        except Exception as e:
            print(f"    [pace] could not select trace (try {i + 1}/{tries}: {e})", flush=True)
            time.sleep(3)
    return False


def preflight(a):
    """Stop immediately (instead of after hours) if one of the three services is not reachable."""
    import json
    import urllib.request
    checks = [("player page (python -m http.server in deploy/)", a.page, None),
              ("media server on the Linux machine (manifest)", a.mpd, None),
              ("ABR decision service (python deploy/abr_server.py)", "http://localhost:8333/decide",
               json.dumps({"session": "preflight", "policy": "bba", "buffer": 0, "next_sizes": [1, 2, 3, 4, 5, 6],
                           "samples_kbps": [1000]}).encode())]
    bad = []
    for name, url, body in checks:
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"} if body else {})
            with urllib.request.urlopen(req, timeout=10) as r:
                ok = r.status == 200
        except Exception as e:
            ok = False
            print(f"[preflight] {name}: NOT reachable ({e})")
        if ok:
            print(f"[preflight] {name}: ok")
        else:
            bad.append(name)
    if bad:
        raise SystemExit("start the missing service(s) first -- each one in its OWN terminal: " + "; ".join(bad))


def count_done(path, network, video):
    """Completed sessions per policy for this network and video (rows without a video column,
    written by an older version, are not counted)."""
    import csv
    import collections
    c = collections.Counter()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("network") == network and row.get("video") == video:
                    c[row.get("policy")] += 1
    return c


if __name__ == "__main__":
    main()
