# Real-World Testbed (RQ4)

This guide describes how to build and run the real-world testbed. Following it step by step reproduces Fig. 9, Table 2 (smoothness intent) and Table 4 (measurement error) of the paper.

If you only want to regenerate the figures and tables from the logs of the paper's runs, skip to [Section 7](#7-analysis). Those logs are included in `data/`.

---

## 0. Overview

```
 Client PC (repository root)                                         Media server: NVIDIA Jetson Orin Nano (Ubuntu 22.04)
 ┌───────────────────────────────────────────┐                      ┌──────────────────────────────────────────┐
 │ W1: deploy/abr_server.py        (:8333)   │                      │ J: tmux session "media"                  │
 │     8 ABR policies + LLM intent service   │                      │    deploy/media_server.py     (:8080)    │
 │ W2: python -m http.server       (:8080)   │  ── video chunks ──→ │    · serves the DASH videos              │
 │     serves player.html                    │  ←── /neurobh/bw ─── │    · per-chunk bandwidth from TCP_INFO   │
 │ W3: tools/run_rw_sessions.py              │  ── /neurobh/pace ─→ │    · trace-driven pacing                 │
 │     drives Chrome through all policies    │                      │      (SO_MAX_PACING_RATE)                │
 │                                           │                      │    · restarts the round's trace for      │
 │                                           │                      │      every session                       │
 └───────────────────────────────────────────┘                      └──────────────────────────────────────────┘
   Output: data/realworld.csv (one row per session), data/realworld_chunks.csv (one row per chunk)
           ~/neurobh/server_log.csv on the server (server-side measurements and ground-truth bandwidth)
```

| Terminal | Where | Role | Keep open |
|---|---|---|---|
| **W1** | client, repository root | ABR decision service `abr_server.py` | yes |
| **W2** | client, `deploy/` | static page server for the player | yes |
| **W3** | client, repository root | runs experiments, uploads files, analysis | while running experiments |
| **J** | `ssh <user>@<JETSON_IP>` | media server (inside tmux) | yes (tmux, in the background) |

In the commands below, replace `<JETSON_IP>` with your server's IP address and `<user>` with your login on the server.

**Important**

1. W1, W2 and W3 must be **three separate terminals**, all running at the same time. If the ABR decision service in W1 is stopped, every session of a run fails.
2. While an experiment is running, do not open the scripts or data files it uses in an IDE. Auto-save can write stale content back to disk.
3. Keep the client PC awake and keep the Chrome window open (not minimized). Avoid other heavy network traffic on the client during runs.

---

## 1. Hardware and network

| Item | Setup used in the paper |
|---|---|
| Media server | NVIDIA Jetson Orin Nano Super, Ubuntu 22.04 (L4T, kernel 5.15), Python 3.10 |
| Client | Windows laptop, Google Chrome, Python 3.10 (conda) |
| Link | Server and client on the same Wi-Fi router |
| Congestion control | TCP Cubic. The Jetson kernel ships only `reno` and `cubic`, without `tcp_bbr`. |
| Bandwidth emulation | Inside `media_server.py`, by per-connection `SO_MAX_PACING_RATE` that follows a trace. |

Notes:

- `tools/tc_replay.py` is a `tc`-based alternative for Linux servers whose kernel has the `sch_htb`, `sch_netem` and `cls_u32` modules. The Jetson kernel lacks them.
- The paced rate is at most 30 Mbps, far below the Wi-Fi throughput (100–200 Mbps), so the Wi-Fi link is not the bottleneck.
- Wired Ethernet works as well, as long as the client can reach the server's address.

---

## 2. One-time setup

### 2.1 Client software (W3)

```powershell
conda activate neurobh
pip install -r requirements.txt   # includes playwright; it drives the installed Chrome, so no browser download is needed
ffmpeg -version                   # full build with libx264, its bin folder on PATH (e.g. the gyan.dev "full" build on Windows)
```

`.env` must contain `ZHIPU_API_KEY`, which the `llm-neurobh` policy uses (see the main README). Without a key, intents already in `data/llm_cache.jsonl` still work.

### 2.2 Download the test videos (into `video/`)

The Blender download links serve `.zip` archives, so unzip them after downloading:

```powershell
New-Item -ItemType Directory -Force video | Out-Null
curl.exe -L -o video\bbb.zip https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_h264.mov.zip
curl.exe -L -o video\tos.zip https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip
Get-ChildItem video\*.zip | ForEach-Object { Expand-Archive $_.FullName -DestinationPath video -Force }
Get-ChildItem video | Select-Object Name, @{n='MB';e={[int]($_.Length/1MB)}}
# expected: big_buck_bunny_1080p_h264.mov (about 690 MB) and tears_of_steel_1080p.mov (about 560 MB)
```

If a downloaded file is only a few tens of KB, it is an HTML page rather than the video. Delete it and download the file with a browser.

### 2.3 Encode DASH (W3, 10–20 minutes per video)

```powershell
python tools/encode_dash.py --src video/big_buck_bunny_1080p_h264.mov --out dash/bbb --duration 300
python tools/encode_dash.py --src video/tears_of_steel_1080p.mov --out dash/tos --start 60 --duration 300
```

Check the output:

- Each video reports `75 segments x 6 levels`.
- The packaging command ends with `dash/bbb/manifest.mpd`, with forward slashes only.
- The ToS levels range from `(432,180)` to `(1920,800)`. To verify, run `ffprobe -v error -show_entries stream=width,height -of csv=p=0 dash/tos/_levels/v5.mp4`; it should print `1920,800`.
- `dash/bbb` contains about 526–532 `chunk-*.m4s` files.

Add `--vmaf` to compute per-chunk VMAF. This is slow. Without it, the analysis uses the logarithmic rate-quality model of the paper.

### 2.4 Prepare the server (J)

```bash
ssh <user>@<JETSON_IP>
mkdir -p ~/neurobh/dash
python3 --version                                          # 3.10 or newer
cat /proc/sys/net/ipv4/tcp_available_congestion_control    # must list cubic
sudo apt install -y tmux iperf3
```

### 2.5 Upload to the server (W3)

```powershell
scp deploy/media_server.py data/traces/hsdpa.zip data/traces/fcc.zip data/traces/oboe.zip data/traces/lumos4g.zip <user>@<JETSON_IP>:~/neurobh/
cmd /c "tar -cf - -C dash --exclude=_levels bbb | ssh <user>@<JETSON_IP> tar -xf - -C neurobh/dash"
cmd /c "tar -cf - -C dash --exclude=_levels tos | ssh <user>@<JETSON_IP> tar -xf - -C neurobh/dash"
```

On Windows, wrap the `tar | ssh` pipe in `cmd /c`, because PowerShell pipes alter binary data. On Linux/macOS, run the pipe directly.

After uploading, check on J that `ls ~/neurobh/dash/bbb | wc -l` and `ls ~/neurobh/dash/tos | wc -l` each print about 541.

**After every change to `media_server.py`, upload it again with the first `scp` command and restart the server on J.**

---

## 3. Starting the services (each time)

### J: media server (runs inside tmux, so it survives SSH disconnects)

```bash
ssh <user>@<JETSON_IP>
sudo nvpmodel -m 0 && sudo jetson_clocks        # fixed max-performance mode (Jetson only)
tmux new -s media                               # if the session already exists: tmux attach -t media
cd ~/neurobh
python3 media_server.py --root ~/neurobh/dash --port 8080 --cc cubic \
        --pace-zip hsdpa.zip --pace-cap-mbps 20 --log server_log.csv
```

Wait for `[pace] 142 traces loaded` and `[media] serving ...`. Then detach from tmux with **Ctrl+B, release, then d**. The server keeps running in the background.

tmux basics:

| Action | Keys / command |
|---|---|
| Reattach to the session | `tmux attach -t media` |
| Stop the server | Ctrl+C inside the session |
| Detach and leave it running | Ctrl+B, then d |
| Close the session list if it opens by mistake | q or Esc |

### W1: ABR decision service

```powershell
conda activate neurobh
python deploy/abr_server.py --ckpt ckpt/neurobh.pt --pensieve ckpt/pensieve.pt --llm glm-4-flash
```

The service is up when it prints `ABR decision service on :8333 ...`. Keep this window open.

### W2: player page

```powershell
cd deploy
python -m http.server 8080 --bind 127.0.0.1
```

It is ready when it prints `Serving HTTP on 127.0.0.1 port 8080`.

### W3: connectivity check

```powershell
curl.exe -s -o NUL -w "%{http_code}`n" http://<JETSON_IP>:8080/bbb/manifest.mpd     # 200
curl.exe -s http://<JETSON_IP>:8080/bbb/sizes.json | Select-Object -First 1          # {"seg_dur": 4.0, ...
```

Do not use `curl -I`: the media server does not implement HEAD and returns 501.

---

## 4. Trial run (first setup or after code changes)

Open the two URLs below in Chrome and let each play for 60 seconds. At the end, the page should show `sizes.json: 75 segments` and `reported`.

```
http://127.0.0.1:8080/player.html?policy=neurobh&network=test&secs=60&mpd=http://<JETSON_IP>:8080/bbb/manifest.mpd
http://127.0.0.1:8080/player.html?policy=llm-neurobh&network=test&secs=60&mpd=http://<JETSON_IP>:8080/bbb/manifest.mpd&intent=Live%20shopping%20stream%2C%20latency%20under%2010%20seconds
```

Notes:

- Put the `mpd` parameter before `intent`, and URL-encode the intent. Otherwise the URL may be truncated and the page reports `ERROR: missing &mpd=`.
- The report should contain `"video":"bbb"`.

Delete the trial data afterwards (W3):

```powershell
Remove-Item data\realworld.csv, data\realworld_chunks.csv
```

This deletes the logs of the paper's runs as well. Back them up first if you want to keep them.

---

## 5. Experiments (Fig. 9 and Table 2)

Each group takes two steps:

1. **J:** run `tmux attach -t media`, stop the server with Ctrl+C, start it with the command from the table below, and detach with Ctrl+B, d.
2. **W3:** run the experiment command from the table. Leave W1 and W2 running.

| # | Network | J: media-server command (in `~/neurobh`) | W3: experiment command | Time |
|---|---|---|---|---|
| 1 | 3G (HSDPA) | `python3 media_server.py --root ~/neurobh/dash --port 8080 --cc cubic --pace-zip hsdpa.zip --pace-cap-mbps 20 --log server_log.csv` | `python tools/run_rw_sessions.py --mpd http://<JETSON_IP>:8080/bbb/manifest.mpd --network emu-3g --rounds 6` | about 3 h |
| 2 | Broadband (FCC) | as #1, with `fcc.zip` instead of `hsdpa.zip` | as #1, with `--network emu-fcc` | about 3 h |
| 3 | 4G (Lumos) | as #1, with `lumos4g.zip --pace-cap-mbps 30` | as #1, with `--network emu-4g` | about 3 h |
| 4 | Oboe (ToS video) | as #1, with `oboe.zip --pace-cap-mbps 30` | `python tools/run_rw_sessions.py --mpd http://<JETSON_IP>:8080/tos/manifest.mpd --network emu-oboe --rounds 6` | about 3 h |
| 5 | 3G + smoothness intent | same as #1 (`hsdpa.zip`) | `python tools/run_rw_sessions.py --mpd http://<JETSON_IP>:8080/bbb/manifest.mpd --network emu-3g --rounds 6 --policies neurobh-smooth` | about 20 min |
| 6 | FCC + smoothness intent | same as #2 (`fcc.zip`) | as #5, with `--network emu-fcc` | about 20 min |

Expected output in W3:

```
[preflight] player page ...: ok
[preflight] media server ...: ok
[preflight] ABR decision service ...: ok
    [pace] trace for round 1: norway_car_1
[round 1/6] neurobh ...
    ok (181s)
```

How the runs are organized:

- **Paired design.** All policies in a round must show the **same trace name**. Before every session, the script asks the server to restart that round's trace from the beginning.
- **Fixed trace order.** The trace order is fixed by a random seed. The `neurobh-smooth` sessions of groups 5–6 therefore replay exactly the traces of groups 1–2 and are paired round by round.
- **Policies.** The eight policies run in this order: `neurobh`, `llm-neurobh`, `robustmpc`, `pensieve`, `bba`, `bola`, `festive`, `dynamic`.
- **Intents.** `llm-neurobh` rotates four intents over the rounds: live commerce, limited data plan, telemedicine and movie night. They are defined in `tools/run_rw_sessions.py`.
- **Smoothness policy.** `neurobh-smooth` is the default spec plus the hard constraint `ALWAYS switch <= 0 WHEN buffer < 16 AND pred_bw < 1500`, defined in `deploy/abr_server.py`.

### Interruptions and re-runs

- **Run stopped** (Ctrl+C, power loss, Chrome crash). Run the same W3 command again. Completed sessions are skipped, and the script prints `[resume] ...`.
- **Two consecutive sessions failed.** The script stops itself. The usual cause is that W1 is not running. Start W1 and run the command again.
- **Sessions out of alignment.** Re-run any session that started although the trace could not be selected. Such a session is not aligned with the rest of its round. Remove it from the logs and run the original W3 command again; only the missing sessions are re-run. For example, to remove the last `llm-neurobh` and `robustmpc` sessions of `emu-oboe`:

```powershell
python -c "import pandas as pd; d=pd.read_csv('data/realworld.csv'); c=pd.read_csv('data/realworld_chunks.csv'); m=d.network=='emu-oboe'; s=[d[m&(d.policy==p)].session.iloc[-1] for p in ['llm-neurobh','robustmpc']]; d[~d.session.isin(s)].to_csv('data/realworld.csv',index=False); c[~c.session.isin(s)].to_csv('data/realworld_chunks.csv',index=False); print('removed', s)"
```

### Progress (W3)

```powershell
python -c "import pandas as pd; d=pd.read_csv('data/realworld.csv'); print(d.groupby(['network','video','policy']).size().unstack(fill_value=0))"
```

---

## 6. Simulation supplements (no server needed, a few minutes)

```powershell
python experiments/exp_eta.py       # η sensitivity -> results/eta_sweep.csv, eta_summary.csv
python experiments/exp_smooth.py    # smoothness: soft weight vs. hard constraint -> results/smooth_sweep.csv, smooth_summary.csv
```

---

## 7. Analysis

```powershell
scp <user>@<JETSON_IP>:~/neurobh/server_log.csv data/server_log.csv
python experiments/plot_testbed.py     # -> results/rw_summary.csv, rw_sessions.csv, figs/fig_rw.pdf (Fig. 9)
python experiments/rw_measurement.py   # -> results/tab_rw_measurement.*, tab_rw_prediction.*, tab_rw_per_network.csv (Table 4)
```

How the analysis works:

- **`plot_testbed.py`** computes:
  - the VMAF-QoE of each session, using the paper's rate-quality model and the default weights (β = 50 VMAF points per second of stall);
  - the data left unplayed in the buffer when the session ends;
  - 95% within-round (Cousineau–Morey) confidence intervals over the 6 paired rounds.
- **`rw_measurement.py`** aligns the client chunk log with the server log by request ID (`X-NeuroBH-Req`) and uses the paced rate as ground truth.

---

## 8. Shutdown

1. W1 and W2: press Ctrl+C.
2. J: run `tmux attach -t media`, press Ctrl+C, then type `exit` to close the tmux session.
3. Optional (Jetson): restore a power-saving mode with `sudo nvpmodel -m <mode>`. List the available modes with `nvpmodel -q`.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `hostname contains invalid characters` | The placeholder `<JETSON_IP>` was not replaced | Use your server's IP address |
| `curl -I` returns 501 | The media server does not implement HEAD | Use `curl.exe -s -o NUL -w "%{http_code}"` |
| Port 8080 on the server refuses connections while `tmux ls` lists the session | The server process inside tmux exited | `tmux attach -t media` and restart it |
| SSH drops after disabling Wi-Fi on the server | The SSH session itself runs over Wi-Fi | Re-enable Wi-Fi locally on the server (`sudo nmcli radio wifi on`) |
| Every session ends with `NO REPORT` after 181 s, or every chunk is at level 0 | The ABR decision service (W1) is not running | Start W1 in its own terminal, then re-run (missing sessions are resumed) |
| Report shows `"video":"EnvivioDash3"` | The `mpd` parameter was lost from a hand-written URL | Put `mpd` first and the URL-encoded `intent` last |
| `TargetClosedError` | The Chrome page was closed | The script restarts Chrome and retries automatically |
| Encoding reports `0 segments` and the chunks land in `dash/` | Mixed path separators passed to ffmpeg | Use the current `encode_dash.py`; delete `dash/chunk-*` and `dash/init-*`, then re-encode |
| `Stream 1 is not mapped to an AdaptationSet` | The `.mov` file has a timecode track | Use the current `encode_dash.py`; it maps only `v:0` and `a:0` |
| Edited scripts revert to an older version | An IDE tab with the file open auto-saved stale content | Close the tab and reopen the file from disk |

---

## 10. Files

| File | Purpose |
|---|---|
| `tools/encode_dash.py` | Encodes the 6-level DASH ladder with ffmpeg. Writes `sizes.json`, Pensieve-format `video_size_*` and an optional `vmaf.csv`. |
| `deploy/media_server.py` | Media server. Serves the DASH content, measures each chunk via TCP_INFO and provides `/neurobh/bw` and `/neurobh/pace`. Options: `--pace-zip` (trace pacing) and `--cc` (congestion control). |
| `deploy/abr_server.py` | ABR decision service with `/decide`, `/intent` and `/report`. Writes `data/realworld*.csv`. Policies include `neurobh-smooth`. |
| `deploy/player.html`, `deploy/neurobh_rule.js`, `deploy/dash.all.min.js` | Player page, custom dash.js ABR rule, local copy of dash.js 4.7.4 |
| `tools/run_rw_sessions.py` | Session automation: preflight checks, paired traces, resume, automatic Chrome recovery |
| `tools/tc_replay.py` | `tc`-based trace shaping for Linux servers with htb/netem |
| `experiments/plot_testbed.py` | Fig. 9 and summary tables |
| `experiments/rw_measurement.py` | Table 4 (measurement and prediction error) |
| `experiments/exp_eta.py`, `experiments/exp_smooth.py` | η sensitivity and smoothness intent (simulation) |
