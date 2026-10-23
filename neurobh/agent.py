"""Spec-conditioned actor-critic (PPO with adaptive KL penalty, paper Sec. 5.2).

Inputs (flat vector built by StreamingEnv.observe):
  [hist_tput(8) | hist_dl(8) | hist_level(8) | next_sizes(K) | next_vmaf(K) |
   scalars(6) | spec features: s_u(5) + constraint features(6)]
Vector inputs go through 1-D CNN branches (128 filters, kernel 4), scalars and
spec features through FC layers (128); two categorical heads: bitrate level
(K) and max-buffer delta (9).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .dsl import N_CONSTRAINT_FEATS

N_PREF = 5


def obs_dim(K):
    return 3 * C.HIST_LEN + 2 * K + 6 + N_PREF + N_CONSTRAINT_FEATS


class Trunk(nn.Module):
    def __init__(self, K, hidden=128, filters=128):
        super().__init__()
        self.K = K
        H = C.HIST_LEN
        self.conv_h = nn.ModuleList([nn.Conv1d(1, filters, 4) for _ in range(3)])
        self.conv_n = nn.ModuleList([nn.Conv1d(1, filters, min(4, K)) for _ in range(2)])
        self.fc_s = nn.Linear(6, hidden)
        self.fc_u = nn.Linear(N_PREF + N_CONSTRAINT_FEATS, hidden)
        n_in = 3 * filters * (H - 3) + 2 * filters * (K - min(4, K) + 1) + 2 * hidden
        self.fc = nn.Linear(n_in, hidden)

    def forward(self, x):
        H, K = C.HIST_LEN, self.K
        i = 0; outs = []
        for conv in self.conv_h:
            outs.append(F.relu(conv(x[:, i:i + H].unsqueeze(1))).flatten(1)); i += H
        for conv in self.conv_n:
            outs.append(F.relu(conv(x[:, i:i + K].unsqueeze(1))).flatten(1)); i += K
        outs.append(F.relu(self.fc_s(x[:, i:i + 6]))); i += 6
        outs.append(F.relu(self.fc_u(x[:, i:])))
        return F.relu(self.fc(torch.cat(outs, 1)))


class ActorCritic(nn.Module):
    def __init__(self, K, n_delta=len(C.BUF_DELTAS), hidden=128):
        super().__init__()
        self.K, self.n_delta = K, n_delta
        self.actor = Trunk(K, hidden)
        self.critic = Trunk(K, hidden)
        self.head_level = nn.Linear(hidden, K)
        self.head_delta = nn.Linear(hidden, n_delta)
        self.head_v = nn.Linear(hidden, 1)

    def dist(self, x):
        z = self.actor(x)
        return self.head_level(z), self.head_delta(z)

    def value(self, x):
        return self.head_v(self.critic(x)).squeeze(-1)


class Agent:
    def __init__(self, K, lr_actor=1e-4, lr_critic=1e-3, buffer_action=True, device="cpu"):
        # paper used 1e-5 / 1e-4 with RMSProp; with Adam 1e-4 / 1e-3 converges faster
        self.net = ActorCritic(K).to(device)
        self.K = K
        self.device = device
        self.buffer_action = buffer_action
        a_params = list(self.net.actor.parameters()) + list(self.net.head_level.parameters()) + \
            list(self.net.head_delta.parameters())
        c_params = list(self.net.critic.parameters()) + list(self.net.head_v.parameters())
        self.opt_a = torch.optim.Adam(a_params, lr=lr_actor)
        self.opt_c = torch.optim.Adam(c_params, lr=lr_critic)
        self.kl_coef = 0.2
        self.kl_target = 0.01

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        ll, ld = self.net.dist(x)
        if not self.buffer_action:
            ld = torch.full_like(ld, -1e9); ld[0, C.BUF_DELTAS.index(0)] = 0.0
        if deterministic:
            lv, d = int(ll.argmax()), int(ld.argmax())
        else:
            lv = int(torch.distributions.Categorical(logits=ll).sample())
            d = int(torch.distributions.Categorical(logits=ld).sample())
        lp = float(F.log_softmax(ll, -1)[0, lv] + (F.log_softmax(ld, -1)[0, d] if self.buffer_action else 0.0))
        v = float(self.net.value(x))
        return lv, d, lp, v

    def update(self, batch, epochs=4, mb=256, gamma=0.995, lam=0.95, clip=0.2, ent_coef=0.01,
               imit_coef=0.5, algo="ppo"):
        obs = torch.as_tensor(np.asarray(batch["obs"]), dtype=torch.float32)
        lv = torch.as_tensor(batch["level"]); dl = torch.as_tensor(batch["delta"])
        old_lp = torch.as_tensor(batch["logp"], dtype=torch.float32)
        rew = np.asarray(batch["reward"], dtype=np.float32)
        val = np.asarray(batch["value"] + [batch["last_value"]], dtype=np.float32)
        done = np.asarray(batch["done"], dtype=np.float32)
        inter = torch.as_tensor(batch["intervened"], dtype=torch.float32)
        s_lv = torch.as_tensor(batch["shield_level"]); s_dl = torch.as_tensor(batch["shield_delta"])
        # GAE
        adv = np.zeros_like(rew); g = 0.0
        for t in reversed(range(len(rew))):
            nv = val[t + 1] * (1 - done[t])
            delta = rew[t] + gamma * nv - val[t]
            g = delta + gamma * lam * (1 - done[t]) * g
            adv[t] = g
        ret = torch.as_tensor(adv + val[:-1]); adv = torch.as_tensor(adv)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(rew)
        if algo == "a2c":
            epochs, mb = 1, n
        stats = {}
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, mb):
                idx = perm[i:i + mb]
                ll, ld = self.net.dist(obs[idx])
                if not self.buffer_action:
                    ld = ld.detach() * 0
                lpl, lpd = F.log_softmax(ll, -1), F.log_softmax(ld, -1)
                new_lp = lpl.gather(1, lv[idx, None]).squeeze(1)
                if self.buffer_action:
                    new_lp = new_lp + lpd.gather(1, dl[idx, None]).squeeze(1)
                ratio = torch.exp(new_lp - old_lp[idx])
                if algo == "a2c":
                    pg = -(new_lp * adv[idx]).mean(); kl = torch.tensor(0.0)
                else:
                    pg = -torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[idx]).mean()
                    kl = (old_lp[idx] - new_lp).mean()
                ent = -(lpl.exp() * lpl).sum(-1).mean()
                if self.buffer_action:
                    ent = ent - (lpd.exp() * lpd).sum(-1).mean()
                # projection step of the neurosymbolic mirror descent: distil the
                # safe policy Gamma on states where the shield intervened
                m = inter[idx]
                imit = torch.tensor(0.0)
                if m.sum() > 0:
                    ce = F.nll_loss(lpl, s_lv[idx], reduction="none")
                    if self.buffer_action:
                        ce = ce + F.nll_loss(lpd, s_dl[idx], reduction="none")
                    imit = (ce * m).sum() / m.sum()
                loss_a = pg + self.kl_coef * kl - ent_coef * ent + imit_coef * imit
                self.opt_a.zero_grad(); loss_a.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5); self.opt_a.step()
                v = self.net.value(obs[idx])
                loss_c = F.mse_loss(v, ret[idx])
                self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()
                stats = {"pg": float(pg), "kl": float(kl), "ent": float(ent), "imit": float(imit),
                         "vloss": float(loss_c)}
        # adaptive KL coefficient (paper: d < d_targ/1.5 -> halve, else double)
        if algo == "ppo":
            d = stats.get("kl", 0.0)
            if d < self.kl_target / 1.5:
                self.kl_coef /= 2
            elif d > self.kl_target * 1.5:
                self.kl_coef *= 2
            self.kl_coef = float(np.clip(self.kl_coef, 1e-4, 10))
        stats["kl_coef"] = self.kl_coef
        return stats

    def save(self, path):
        torch.save({"net": self.net.state_dict(), "K": self.K, "buffer_action": self.buffer_action}, path)

    @staticmethod
    def load(path):
        ck = torch.load(path, map_location="cpu")
        a = Agent(ck["K"], buffer_action=ck["buffer_action"])
        a.net.load_state_dict(ck["net"]); a.net.eval()
        return a
