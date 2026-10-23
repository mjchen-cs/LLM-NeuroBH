"""Short-term bandwidth predictors.

Classic: SlidingWindow, HarmonicMean (MPC), RobustHarmonic (RobustMPC), EWMA (dash.js)
Learned: GRUHybrid = GRU regression for stable periods + interval classifier
         for volatile periods (paper Sec. 4.2), switched by the coefficient of
         variation of the last samples.
"""
import numpy as np
from . import config as C


class SlidingWindow:
    name = "Sliding window"
    def __init__(self, k=5): self.k = k
    def predict(self, s): return float(np.mean(s[-self.k:])) if s else 1000.0


class HarmonicMean:
    name = "Harmonic mean"
    def __init__(self, k=5): self.k = k
    def predict(self, s):
        if not s: return 1000.0
        x = np.asarray(s[-self.k:]); return float(len(x) / np.sum(1.0 / x))


class RobustHarmonic(HarmonicMean):
    """RobustMPC: harmonic mean discounted by the max past relative error."""
    name = "Robust HM"
    def __init__(self, k=5):
        super().__init__(k); self.preds = []
    def reset(self): self.preds = []
    def predict(self, s):
        p = super().predict(s)
        # preds[j] was issued after sample j and predicted sample j+1
        errs = [abs(self.preds[j] - s[j + 1]) / s[j + 1]
                for j in range(max(0, len(self.preds) - self.k), len(self.preds)) if j + 1 < len(s)] or [0.0]
        self.preds.append(p)
        return p / (1 + max(errs))


class EWMA:
    name = "EWMA"
    def __init__(self, a=0.3): self.a = a
    def predict(self, s):
        if not s: return 1000.0
        v = s[0]
        for x in s[1:]: v = self.a * x + (1 - self.a) * v
        return float(v)


RATIO_EDGES = np.array([0.0, 0.25, 0.5, 0.7, 0.85, 0.95, 1.05, 1.2, 1.5, 2.0, 3.0, np.inf])


def _hm(x):
    x = np.asarray(x[-5:], dtype=np.float64)
    return float(len(x) / np.sum(1.0 / x))


class GRUHybrid:
    """Scale-invariant GRU-hybrid predictor.  Inputs are log-ratios of the last
    k samples to their harmonic mean (HM), so one model serves 3G to 5G.
      * regression head : log(next / HM)               -> used when CV < cv_thresh
      * interval head   : bin of next / HM (RATIO_EDGES) -> used for volatile periods,
                          prediction = HM x geometric centre of the most likely bin."""
    name = "NeuroBH (GRU hybrid)"

    def __init__(self, bitrates=C.BITRATES_KBPS, cv_thresh=0.2, path=None, hidden=64):
        import torch, torch.nn as nn
        self.torch = torch
        self.edges = RATIO_EDGES
        self.cv_thresh = cv_thresh
        nb = len(self.edges) - 1

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.gru = nn.GRU(2, hidden, batch_first=True)
                s.reg = nn.Linear(hidden, 1)
                s.cls = nn.Linear(hidden, nb)
            def forward(s, x):
                h, _ = s.gru(x)
                z = h[:, -1]
                return s.reg(z).squeeze(-1), s.cls(z)
        self.net = Net()
        if path:
            self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()

    @staticmethod
    def _x(seq):
        seq = np.asarray(seq, dtype=np.float64)
        hm = _hm(seq)
        r = np.log(seq / hm)
        d = np.concatenate([[0.0], np.diff(np.log(seq))])
        return np.stack([r, d], -1).astype(np.float32), hm

    def predict(self, s):
        if len(s) < 2: return float(s[-1]) if s else 1000.0
        seq = list(s[-C.HIST_LEN:])
        seq = [seq[0]] * (C.HIST_LEN - len(seq)) + seq
        x, hm = self._x(seq)
        with self.torch.no_grad():
            reg, logits = self.net(self.torch.tensor(x).unsqueeze(0))
        last = np.asarray(s[-5:])
        cv = last.std() / max(last.mean(), 1e-6)
        if cv < self.cv_thresh:
            return float(hm * np.exp(reg.item()))
        k = int(logits.argmax(-1).item())
        lo, hi = self.edges[k], self.edges[k + 1]
        mid = lo * 1.5 if np.isinf(hi) else (np.sqrt(lo * hi) if lo > 0 else hi / 2)
        return float(hm * mid)

    # ---------------------------------------------------------- training
    def fit(self, sequences, epochs=10, lr=1e-3, batch=256, log=print):
        """sequences: list of 1-D arrays of bandwidth samples (kbps)."""
        torch = self.torch
        X, R = [], []
        for seq in sequences:
            for t in range(C.HIST_LEN, len(seq)):
                x, hm = self._x(seq[t - C.HIST_LEN:t]); X.append(x); R.append(seq[t] / hm)
        X = torch.tensor(np.asarray(X))
        R = np.asarray(R, dtype=np.float64)
        yr = torch.tensor(np.log(R), dtype=torch.float32)
        yc = torch.tensor(np.clip(np.searchsorted(self.edges, R, side="right") - 1, 0, len(self.edges) - 2))
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        n = len(X)
        for ep in range(epochs):
            perm = torch.randperm(n); tot = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                r, c = self.net(X[idx])
                loss = torch.nn.functional.smooth_l1_loss(r, yr[idx]) + 0.5 * torch.nn.functional.cross_entropy(c, yc[idx])
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
            log(f"[predictor] epoch {ep + 1}/{epochs} loss={tot / n:.4f}")
        self.net.eval()

    def save(self, path):
        self.torch.save(self.net.state_dict(), path)


def stable_samples_from_trace(trace, window=C.CHUNK_LEN, rng=None, noise=C.STABLE_NOISE):
    """Emulates the HTTP stable-connection samples: average bottleneck rate per
    chunk-duration window plus small measurement noise."""
    rng = rng or np.random.default_rng(0)
    t, b = trace.times, trace.bw
    out, cur = [], 0.0
    while cur + window <= t[-1]:
        m = (t >= cur) & (t < cur + window)
        if m.any():
            out.append(float(b[m].mean()) * (1 + noise * rng.standard_normal()))
        cur += window
    return np.maximum(np.asarray(out), 1.0)


def make(name, bitrates=C.BITRATES_KBPS, path=None):
    name = name.lower()
    if name in ("gru", "hybrid", "neurobh"):
        return GRUHybrid(bitrates, path=path)
    return {"sliding": SlidingWindow, "harmonic": HarmonicMean, "robust": RobustHarmonic,
            "ewma": EWMA}[name]()
