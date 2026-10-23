#!/usr/bin/env bash
# Full simulation pipeline (real data + free LLM APIs).  Run from the repository root.
# The LLM models compared in exp_llm.py are listed in configs/exp.yaml (llm_models).
set -e
MAIN=${MAIN:-glm-4-flash}
STEPS=${STEPS:-2000000}
[ -f data/intents.jsonl ] || python -m neurobh.llm.benchmark
python experiments/exp_predictor.py --epochs 10
python -m neurobh.train --steps $STEPS --out ckpt/neurobh.pt &
python -m neurobh.train --steps $STEPS --no-shield --out ckpt/neurobh_noshield.pt &
python -m neurobh.train --steps $STEPS --algo a2c --out ckpt/neurobh_a2c.pt &
python -m neurobh.train --steps $STEPS --no-shield --no-buffer-action --fixed-spec --measure chunk --predictor harmonic --out ckpt/pensieve.pt &
python experiments/exp_llm.py                # runs while the policies train
wait
python experiments/exp_abr.py --llm-abr $MAIN
python experiments/exp_llm_abr.py --model $MAIN --sessions 10
python experiments/exp_e2e.py --model $MAIN --llm-direct
python experiments/exp_overhead.py
python experiments/exp_eta.py
python experiments/exp_smooth.py
python experiments/plot_all.py
