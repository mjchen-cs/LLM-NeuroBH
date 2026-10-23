"""Train the spec-conditioned NeuroBH policy (or the Pensieve-style baseline).

Examples
  # NeuroBH (shield + buffer action + random verified specs)  -- main model
  python -m neurobh.train --traces data/traces/puffer --videos data/videos/train --out ckpt/neurobh.pt
  # w/o neurosymbolic shield (ablation, Fig. ablation)
  python -m neurobh.train ... --no-shield --out ckpt/neurobh_noshield.pt
  # Pensieve-style baseline: no shield, fixed max buffer, fixed default spec, chunk measurement
  python -m neurobh.train ... --no-shield --no-buffer-action --fixed-spec --measure chunk --out ckpt/pensieve.pt
  # different DRL algorithm (ablation)
  python -m neurobh.train ... --algo a2c --out ckpt/neurobh_a2c.pt
"""
import argparse
import csv
import os
import time
import numpy as np
import torch

from . import config as C
from .agent import Agent
from .env import StreamingEnv
from .predictor import make as make_predictor
from .shield import Shield
from .spec import random_spec, default_spec
from .traces import get_traces
from .verifier import Verifier
from .video import get_videos


def build_spec_pool(n, seed, log=print):
    rng = np.random.default_rng(seed)
    V = Verifier(timeout_ms=3000)
    pool, tried = [], 0
    while len(pool) < n and tried < 5 * n:
        tried += 1
        sp = random_spec(rng)
        if V.verify_spec(sp).ok:
            pool.append(sp)
    log(f"[train] spec pool: {len(pool)} verified specs out of {tried} sampled")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/traces/train,data/traces/puffer",
                    help="comma-separated trace folders/zips; missing ones are skipped")
    ap.add_argument("--videos", default="data/videos/train")
    ap.add_argument("--out", default="ckpt/neurobh.pt")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--rollout", type=int, default=4096)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "a2c"])
    ap.add_argument("--no-shield", action="store_true")
    ap.add_argument("--no-buffer-action", action="store_true")
    ap.add_argument("--fixed-spec", action="store_true")
    ap.add_argument("--measure", default="stable", choices=["stable", "chunk"])
    ap.add_argument("--predictor", default="gru", help="gru | harmonic | robust | ewma | sliding")
    ap.add_argument("--predictor-ckpt", default="ckpt/predictor.pt")
    ap.add_argument("--intervene-penalty", type=float, default=0.1)
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    traces = get_traces(a.traces, "medium", seed=a.seed)
    videos = get_videos(a.videos)
    pred = None
    if a.predictor == "gru":
        if os.path.exists(a.predictor_ckpt):
            pred = make_predictor("gru", path=a.predictor_ckpt)
        else:
            print(f"[train] {a.predictor_ckpt} missing -> run experiments/exp_predictor.py first; using harmonic")
            pred = make_predictor("harmonic")
    else:
        pred = make_predictor(a.predictor)
    env = StreamingEnv(traces, videos, measure=a.measure, predictor=pred, seed=a.seed,
                       fixed_max_buffer=60.0 if a.no_buffer_action else None)
    agent = Agent(env.K, buffer_action=not a.no_buffer_action)
    pool = [default_spec()] if a.fixed_spec else build_spec_pool(a.pool, a.seed)
    rng = np.random.default_rng(a.seed)

    log_path = a.out.replace(".pt", "_train.csv")
    fcsv = open(log_path, "w", newline=""); wr = csv.writer(fcsv)
    wr.writerow(["step", "episode_reward", "episode_qoe", "intervention_rate", "kl_coef", "time_s"])
    t0 = time.time(); step = 0; ep_rewards, ep_qoes, ep_inter = [], [], []

    def new_episode():
        sp = pool[rng.integers(len(pool))]
        sh = None if a.no_shield else Shield(sp)
        return env.reset(sp), sp, sh

    s, spec, shield = new_episode(); ep_r = 0.0
    while step < a.steps:
        B = {k: [] for k in ["obs", "level", "delta", "logp", "value", "reward", "done",
                             "intervened", "shield_level", "shield_delta"]}
        for _ in range(a.rollout):
            obs = env.observe(s, spec)
            lv, d, lp, v = agent.act(obs)
            if shield is not None:
                dec = shield(s, lv, d)
                exe_l, exe_d, inter = dec.level, dec.delta_idx, dec.intervened
            else:
                exe_l, exe_d, inter = lv, d, False
            s2, r, done, info = env.step(exe_l, exe_d, inter)
            if inter:
                r -= a.intervene_penalty
            for k, x in zip(B, [obs, lv, d, lp, v, r, float(done), float(inter), exe_l, exe_d]):
                B[k].append(x)
            ep_r += r; step += 1
            if done:
                ep_rewards.append(ep_r)
                ep_qoes.append(np.mean([x["qoe"] for x in env.log]))
                ep_inter.append(np.mean([x["intervened"] for x in env.log]))
                s, spec, shield = new_episode(); ep_r = 0.0
            else:
                s = s2
        B["last_value"] = agent.act(env.observe(s, spec))[3]
        st = agent.update(B, algo=a.algo)
        if ep_rewards:
            row = [step, np.mean(ep_rewards[-50:]), np.mean(ep_qoes[-50:]), np.mean(ep_inter[-50:]),
                   st["kl_coef"], round(time.time() - t0, 1)]
            wr.writerow(row); fcsv.flush()
            print(f"[train] step {step:>9d}  R={row[1]:.3f}  QoE={row[2]:.2f}  shield={row[3]:.3f}  "
                  f"kl_coef={row[4]:.4f}  {row[5]}s")
        if step % (a.rollout * 25) < a.rollout:
            agent.save(a.out)
    agent.save(a.out)
    print(f"[train] saved {a.out}; curve -> {log_path}")


if __name__ == "__main__":
    main()
