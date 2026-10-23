#!/usr/bin/env bash
# Offline pipeline check: synthetic data + Mock LLM + tiny training (about 5 minutes).
# NOT for reporting results.  It overwrites ckpt/ and results/, so both are backed up first.
set -e
TS=$(date +%Y%m%d-%H%M%S)
if [ -d ckpt ]; then cp -r ckpt "ckpt.bak.$TS"; fi
if [ -d results ]; then cp -r results "results.bak.$TS"; fi
python experiments/exp_predictor.py --epochs 3 --max-train 40
python -m neurobh.train --steps 4096 --rollout 2048 --pool 40 --out ckpt/neurobh.pt
python -m neurobh.train --steps 4096 --rollout 2048 --pool 40 --no-shield --out ckpt/neurobh_noshield.pt
python -m neurobh.train --steps 4096 --rollout 2048 --pool 40 --algo a2c --out ckpt/neurobh_a2c.pt
python -m neurobh.train --steps 4096 --rollout 2048 --no-shield --no-buffer-action --fixed-spec --measure chunk --predictor harmonic --out ckpt/pensieve.pt
python experiments/exp_abr.py --sessions 8 --llm-abr mock --llm-abr-sessions 3
python experiments/exp_llm.py --models mock --limit 60
python experiments/exp_e2e.py --model mock --sessions 2 --limit 20 --llm-direct --llm-direct-sessions 1
python experiments/exp_overhead.py
python experiments/plot_all.py
echo "Smoke test finished. Restore the released checkpoints/results with:"
echo "  rm -rf ckpt results && mv ckpt.bak.$TS ckpt && mv results.bak.$TS results"
