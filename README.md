# LLM-NeuroBH

Code and data for the paper

> **LLM-NeuroBH: Verifiable Intent-Driven Adaptive Video Streaming Services with Large Language Models**
> Miaojiang Chen, Wenjing Xiao, Haiqiang Chen, Min Chen, Ahmed Farouk, Houbing Herbert Song, Athanasios V. Vasilakos
> Submitted to *IEEE Transactions on Services Computing*.

LLM-NeuroBH turns a natural-language service intent (e.g. *"live-commerce stream, keep latency under 10 s and avoid stalls"*) into a verified service specification and executes it with a neurosymbolic ABR policy:

1. **LLM intent service.** An LLM compiles the intent into a `ServiceSpec`: QoE priorities plus hard constraints in a small DSL (`ALWAYS <atoms> [WHEN <atoms>]`).
2. **Verifier with repair loop.** A five-stage verifier checks the spec: schema, syntax, ranges, consistency (Z3) and worst-case realizability. Violations are fed back to the LLM for repair.
3. **Spec-conditioned PPO policy with a neurosymbolic shield.** The shield enforces the verified constraints under a worst-case bandwidth bound (η).

The LLMs are used through **free OpenAI-compatible APIs**. They are not trained or fine-tuned. Only the PPO policy and the GRU bandwidth predictor are trained.

The repository contains:

- **Simulation experiments**: RQ1–RQ3, RQ5, the ablations, and the η-sensitivity and smoothness experiments.
- **Real-world testbed**: RQ4. A Jetson media server, a dash.js player in Chrome and an ABR decision service. See [`deploy/README_realworld.md`](deploy/README_realworld.md).

---

## Repository layout

```
.
├── neurobh/                    core package
│   ├── config.py               bitrate ladder, priority→weight mapping, network profiles (η from results/eta.json)
│   ├── env.py                  trace-driven streaming simulator (buffer dynamics, QoE, wasted data)
│   ├── traces.py / video.py    trace loading (folder or .zip), video sizes and rate-quality model
│   ├── predictor.py            bandwidth predictors: sliding window / harmonic / EWMA / GRU-hybrid
│   ├── dsl.py                  constraint DSL: ALWAYS <atoms> [WHEN <atoms>]
│   ├── spec.py                 ServiceSpec (the service specification produced by the LLM)
│   ├── verifier.py             five-stage verifier: schema / syntax / range / consistency (Z3) / worst-case realizability
│   ├── shield.py               neurosymbolic shield ψ = if P0(s,Θ(s)) ⊆ ξ then Θ(s) else Γ(s)
│   ├── agent.py / train.py     spec-conditioned PPO (adaptive KL) with projection term; training entry point
│   ├── policies.py             NeuroBH, BBA, BOLA, FESTIVE, DYNAMIC, RobustMPC, Pensieve, LLM-ABR
│   ├── metrics.py              QoE, objective, constraint-violation rate, wasted data
│   └── llm/                    client.py (API, cache, rate limit), prompts.py, intent_service.py (verifier-in-the-loop repair),
│                               benchmark.py (bilingual Chinese/English intent benchmark), mock.py (offline pipeline test only)
├── experiments/                simulation experiments and plotting scripts
├── deploy/                     real-world testbed: media server, ABR decision service, player page, dash.js rule
├── tools/                      Puffer conversion, DASH encoding, testbed session automation, tc-based shaping
├── configs/exp.yaml            data paths, checkpoints, number of sessions, LLMs used
├── configs/llm.yaml            LLM endpoint configuration
├── data/                       traces/*.zip, videos/, intents.jsonl, llm_cache.jsonl, testbed logs
├── ckpt/                       trained models used in the paper
└── results/                    experiment results used in the paper (csv / tex)
```

---

## 1. Installation

```bash
conda create -n neurobh python=3.10 -y
conda activate neurobh
pip install -r requirements.txt
```

- For GPU training, install the PyTorch build that matches your CUDA version (see pytorch.org). The paper used an RTX 5090 with CUDA 12.8. Everything also runs on CPU, only slower.
- The real-world testbed additionally needs ffmpeg, Google Chrome and a Linux media server. See [`deploy/README_realworld.md`](deploy/README_realworld.md).

---

## 2. LLM endpoints and API keys

Copy `.env.example` to `.env` and fill in your own keys. `.env` is git-ignored, so **never commit it**.

```
ZHIPU_API_KEY=...
SILICONFLOW_API_KEY=...
```

Environment variables that are already set take precedence over `.env`.

The four free models compared in the paper (`llm_models` in `configs/exp.yaml`):

| Name in `configs/llm.yaml` | Provider | Key variable | Used for |
|---|---|---|---|
| `glm-4-flash` | Zhipu BigModel (open.bigmodel.cn) | `ZHIPU_API_KEY` | main model (RQ3, testbed) |
| `qwen2.5-7b` | SiliconFlow (siliconflow.cn) | `SILICONFLOW_API_KEY` | model comparison (RQ2) |
| `glm-4-9b-0414` | SiliconFlow | `SILICONFLOW_API_KEY` | model comparison (RQ2) |
| `qwen3-8b` | SiliconFlow (thinking disabled) | `SILICONFLOW_API_KEY` | model comparison (RQ2) |

Notes:

- `configs/llm.yaml` also defines OpenRouter, Groq and a local Ollama model (`qwen2.5-7b-local`). Model names and free quotas change over time, so check the provider console.
- **Reproducing without API keys.** Every LLM reply used in the paper is cached in `data/llm_cache.jsonl`, keyed by a hash of the request. Re-running the experiments hits the cache, so the LLM results can be reproduced without an API key and without spending quota.

Check that a key works:

```bash
python -c "from neurobh.llm.client import load_clients; c=load_clients('configs/llm.yaml',['glm-4-flash']); print(c['glm-4-flash'].chat([{'role':'user','content':'hi'}]))"
```

---

## 3. Data

### Traces

Traces live in `data/traces/`. They use the Pensieve format: one file per trace, and each line is `<time_s> <bandwidth_Mbps>`. The code reads `.zip` archives directly.

| File | Content | Use |
|---|---|---|
| `train.zip` | Pensieve FCC+HSDPA training set (127 traces) plus 70% of Oboe and Lumos5G 4G/5G | training |
| `hsdpa.zip`, `fcc.zip` | HSDPA test set (142), FCC test set (290) | test: slow |
| `oboe.zip`, `lumos4g.zip` | Oboe (129), Lumos5G-4G (53), held-out 30% | test: medium |
| `5g.zip` | Lumos5G-5G (37), held-out 30% | test: fast |
| `puffer.zip` *(not included)* | 3,000 sessions converted from 3 days of Puffer data | training |

`puffer.zip` (about 40 MB) is not included. It is only needed to retrain the policy from scratch; training runs without it, using `train.zip` only. To build it, put the raw Puffer `video_sent` and `video_acked` CSV files in `data/puffer_raw/` and run:

```bash
python tools/convert_puffer.py --raw data/puffer_raw --out data/traces/puffer.zip --max-sessions 3000
```

### Videos

Each simulated video lives in `data/videos/{train,test}/<name>/`. It needs the files `video_size_0 … video_size_5` (Pensieve format) and can optionally have a `vmaf.csv` file with columns `chunk,level,vmaf`. Without `vmaf.csv`, the logarithmic rate-quality model of the paper is used.

### Intents

`data/intents.jsonl` holds the benchmark of 155 bilingual (Chinese/English) intents with ground-truth specs. Regenerate it with `python -m neurobh.llm.benchmark`.

### Synthetic fallback

If a data folder is missing, the code falls back to synthetic traces and puts a "SMOKE TEST - NOT REAL DATA" watermark on the figures. Such results must not be reported.

---

## 4. Running the simulation experiments

Run all commands from the repository root. The trained checkpoints and all results used in the paper are included, so you can go directly to step (8) to regenerate the figures, or re-run any single step.

```bash
# (1) Bandwidth predictor and η (run first; η is written to results/eta.json and read by later steps)
python experiments/exp_predictor.py --epochs 10

# (2) Train the four policies (can run in parallel; about 2.8 h each on an RTX 5090)
python -m neurobh.train --steps 2000000 --out ckpt/neurobh.pt
python -m neurobh.train --steps 2000000 --no-shield --out ckpt/neurobh_noshield.pt
python -m neurobh.train --steps 2000000 --algo a2c --out ckpt/neurobh_a2c.pt
python -m neurobh.train --steps 2000000 --no-shield --no-buffer-action --fixed-spec --measure chunk --predictor harmonic --out ckpt/pensieve.pt

# (3) LLM spec quality and pipeline ablation (RQ2; 4 models x 4 variants; resumable)
python experiments/exp_llm.py

# (4) ABR comparison (RQ1) and the per-chunk LLM baseline
python experiments/exp_abr.py --llm-abr glm-4-flash
python experiments/exp_llm_abr.py --model glm-4-flash --sessions 10      # LLM-ABR (no delay)

# (5) End-to-end intent-driven streaming (RQ3)
python experiments/exp_e2e.py --model glm-4-flash --llm-direct

# (6) Overhead (RQ5)
python experiments/exp_overhead.py

# (7) η sensitivity and intent-driven smoothness (a few minutes each)
python experiments/exp_eta.py
python experiments/exp_smooth.py

# (8) Figures and tables
python experiments/plot_all.py
```

The scripts `run_all.sh` (Linux/macOS) and `run_all.ps1` (Windows) run steps (1)–(8) in order. Steps (1)–(2) overwrite the checkpoints in `ckpt/`.

`run_smoke.sh` is a pipeline check of about 5 minutes. It uses synthetic data, the mock LLM and a few training steps. It backs up `ckpt/` and `results/` first, then overwrites them, and prints how to restore the backups when it finishes.

Figures are written to `figs/` (the `figures` key in `configs/exp.yaml`).

---

## 5. Mapping to the paper

| Paper | Script | Output |
|---|---|---|
| Fig. 6: NeuroBH vs. baselines (slow / medium / fast) | `exp_abr.py`, `exp_llm_abr.py` → `plot_all.py` | `results/abr_*.csv` → `figs/fig_abr_*.pdf` |
| Fig. 7: spec quality of the four LLMs | `exp_llm.py` → `plot_all.py` | `results/llm_runs.jsonl`, `llm_summary.csv` → `figs/fig_llm_quality.pdf` |
| Table 1: ablation of the intent-compilation pipeline | `exp_llm.py` → `plot_all.py` | `results/tab_llm_ablation.tex` |
| Fig. 8: end-to-end intent-driven streaming | `exp_e2e.py` → `plot_all.py` | `results/e2e.csv` → `figs/fig_e2e.pdf` |
| Fig. 9: real-world testbed | testbed → `plot_testbed.py` | `data/realworld*.csv` → `results/rw_summary.csv`, `figs/fig_rw.pdf` |
| Table 2: smoothness intent on slow networks (simulation + testbed) | `exp_smooth.py`, testbed | `results/smooth_summary.csv`, `results/rw_summary.csv` |
| Table 3: measurement / prediction error, original HTTP/3 testbed | original testbed | numbers reported in the paper; raw data not included |
| Table 4: measurement error, new testbed | `rw_measurement.py` | `results/tab_rw_per_network.csv` |
| Text: held-out prediction error and η | `exp_predictor.py` | `results/tab_prediction.csv`, `results/eta.json` |
| Text: ablation (w/o NS, A2C, Pensieve) | `exp_abr.py` | `results/abr_*.csv` |
| Text: η sensitivity | `exp_eta.py` | `results/eta_summary.csv` |
| Table 5: runtime overhead | `exp_overhead.py` → `plot_all.py` | `results/overhead.csv`, `tab_overhead.tex` |

Error bars in Fig. 6 are 95% Student-t confidence intervals over the 200 sessions. Error bars in Fig. 9 are 95% within-round (Cousineau–Morey) confidence intervals over the 6 paired testbed rounds.

---

## 6. Baselines

| Type | Baselines |
|---|---|
| Buffer-based | BBA, BOLA |
| Rate-based | FESTIVE |
| Hybrid | DYNAMIC (dash.js default rule), RobustMPC |
| Learning-based | Pensieve (same traces and network architecture, no shield) |
| LLM-based | **LLM-ABR (direct):** the LLM picks the bitrate for every chunk, and the API latency counts as decision delay. **LLM-ABR (no delay):** the same, but the latency is ignored. **LLM-ABR (direct, intent):** the service intent is added to the prompt (RQ3). |

LLM-ABR calls the API once per chunk, so only 10 sessions per network class are run (`--llm-abr-sessions`).

To add a baseline:

1. Implement a class with `act(env, state) -> (level, delta_idx, intervened)`.
2. Register it in `policies()` in `experiments/exp_abr.py`.
3. For the testbed, also add it to `deploy/abr_server.py`.

---

## 7. Real-world testbed (RQ4)

The testbed uses four components:

- a Jetson media server (`deploy/media_server.py`) with trace-driven pacing;
- the ABR decision service (`deploy/abr_server.py`);
- a static page server for the dash.js player;
- the session automation script (`tools/run_rw_sessions.py`), which drives Chrome through Playwright.

The raw logs of the paper's runs are included (`data/realworld.csv`, `data/realworld_chunks.csv`, `data/server_log.csv`), so Fig. 9 and Table 4 can be regenerated directly:

```bash
python experiments/plot_testbed.py
python experiments/rw_measurement.py
```

The full setup and run instructions are in **[`deploy/README_realworld.md`](deploy/README_realworld.md)**.

---

## 8. Notes

- **η (worst-case bandwidth discount).** `exp_predictor.py` sets η to the 90th percentile of the over-estimation error on the validation set, clipped to [0.1, 0.5]. The default spec uses the "unknown" network profile, whose η is 0.5.
- **Resuming.** `exp_llm.py` and `tools/run_rw_sessions.py` skip completed work when re-run with the same command. Add `--overwrite` to `exp_llm.py` to start over.

---

## Third-party components and data

- `deploy/dash.all.min.js`: dash.js v4.7.4 (BSD-3-Clause, DASH Industry Forum).
- Trace datasets: Pensieve (FCC, HSDPA/Norway), Oboe, Lumos5G and Puffer. Please cite the original datasets when you use them.
- Test videos for the testbed: *Big Buck Bunny* and *Tears of Steel* (© Blender Foundation, CC BY 3.0). They are downloaded and encoded locally, not redistributed here.

## Citation

```bibtex
@article{chen2026llmneurobh,
  title   = {{LLM-NeuroBH}: Verifiable Intent-Driven Adaptive Video Streaming Services with Large Language Models},
  author  = {Chen, Miaojiang and Xiao, Wenjing and Chen, Haiqiang and Chen, Min and Farouk, Ahmed and Song, Houbing Herbert and Vasilakos, Athanasios V.},
  journal = {IEEE Transactions on Services Computing},
  note    = {under review},
  year    = {2026}
}
```

## License

Released under the MIT License (see [LICENSE](LICENSE)). Third-party files keep their own licenses.
