# Full simulation pipeline (Windows PowerShell).  Run from the repository root.
# The LLM models compared in exp_llm.py are listed in configs/exp.yaml (llm_models).
$MAIN = "glm-4-flash"; $STEPS = 2000000
if (-not (Test-Path data/intents.jsonl)) { python -m neurobh.llm.benchmark }
python experiments/exp_predictor.py --epochs 10
$jobs = @(
  (Start-Process python "-m neurobh.train --steps $STEPS --out ckpt/neurobh.pt" -PassThru -NoNewWindow),
  (Start-Process python "-m neurobh.train --steps $STEPS --no-shield --out ckpt/neurobh_noshield.pt" -PassThru -NoNewWindow),
  (Start-Process python "-m neurobh.train --steps $STEPS --algo a2c --out ckpt/neurobh_a2c.pt" -PassThru -NoNewWindow),
  (Start-Process python "-m neurobh.train --steps $STEPS --no-shield --no-buffer-action --fixed-spec --measure chunk --predictor harmonic --out ckpt/pensieve.pt" -PassThru -NoNewWindow)
)
python experiments/exp_llm.py
$jobs | Wait-Process
python experiments/exp_abr.py --llm-abr $MAIN
python experiments/exp_llm_abr.py --model $MAIN --sessions 10
python experiments/exp_e2e.py --model $MAIN --llm-direct
python experiments/exp_overhead.py
python experiments/exp_eta.py
python experiments/exp_smooth.py
python experiments/plot_all.py
