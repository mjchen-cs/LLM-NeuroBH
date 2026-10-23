"""Trace-driven HTTP adaptive streaming simulator with buffer-aware control.

Action a_t = (level, delta_idx): bitrate of the next chunk and the change of
the maximum buffer B^max (paper Sec. 5.1).  Buffer dynamics follow Eq. (1):

    B_{n+1} = min( (B_n - D_n/C_n)^+ + L - dt_n , B^max_n )

Throughput measurement modes:
  * "stable": HTTP/3 stable connection (server-side sample of the bottleneck
              rate during the push, small noise)               -> NeuroBH
  * "chunk" : client-side size / download_time (includes RTT and idle gaps,
              biased for short chunks)                         -> baselines
"""
import numpy as np
from . import config as C


class StreamingEnv:
    def __init__(self, traces, videos, bitrates=C.BITRATES_KBPS, L=C.CHUNK_LEN,
                 measure="stable", predictor=None, seed=0, fixed_max_buffer=None):
        self.traces, self.videos = traces, videos
        self.bitrates, self.L = list(bitrates), L
        self.K = len(bitrates)
        self.measure = measure
        self.predictor = predictor
        self.rng = np.random.default_rng(seed)
        self.fixed_max_buffer = fixed_max_buffer
        self.spec = None

    # ----------------------------------------------------------- episode
    def reset(self, spec, trace_idx=None, video_idx=None, start=None):
        self.spec = spec
        self.trace = self.traces[self.rng.integers(len(self.traces)) if trace_idx is None else trace_idx]
        self.video = self.videos[self.rng.integers(len(self.videos)) if video_idx is None else video_idx]
        n = len(self.trace.bw)
        self.ptr = int(self.rng.integers(n)) if start is None else int(start) % n
        self.offset = 0.0
        self.wall = 0.0
        self.chunk = 0
        self.buffer = 0.0
        self.max_buffer = (self.fixed_max_buffer if self.fixed_max_buffer is not None
                           else min(spec.weights.init_max_buffer, spec.max_buffer_cap()))
        self.prev_level = 0
        self.prev_vmaf = None
        self.hist_tput = [0.0] * C.HIST_LEN     # kbps samples
        self.hist_dl = [0.0] * C.HIST_LEN
        self.hist_lv = [0.0] * C.HIST_LEN
        self.samples = []
        self.pred_bw = float(self.bitrates[0]) * 1.5
        if self.predictor is not None and hasattr(self.predictor, "reset"):
            self.predictor.reset()
        self.log = []
        return self.state()

    def _seg_dur(self, i):
        t = self.trace.times
        return float(t[i + 1] - t[i]) if i + 1 < len(t) else 1.0

    def _advance(self, dt):
        """Advance trace clock by dt seconds without downloading."""
        while dt > 1e-12:
            rem = self._seg_dur(self.ptr) - self.offset
            if dt < rem:
                self.offset += dt; return
            dt -= rem; self.offset = 0.0; self.ptr = (self.ptr + 1) % len(self.trace.bw)

    def _download(self, size_bytes):
        """Returns (download_time, true average bandwidth in kbps)."""
        left = float(size_bytes); t = 0.0
        while left > 1e-9:
            bw = self.trace.bw[self.ptr]                      # kbps
            rate = bw * 1000 / 8 * C.PACKET_PAYLOAD           # bytes/s
            rem = self._seg_dur(self.ptr) - self.offset
            if rate * rem >= left:
                dt = left / rate
                self.offset += dt; t += dt; left = 0.0
            else:
                left -= rate * rem; t += rem
                self.offset = 0.0; self.ptr = (self.ptr + 1) % len(self.trace.bw)
        true_bw = size_bytes * 8 / 1000 / t / C.PACKET_PAYLOAD
        return t, true_bw

    # ----------------------------------------------------------- state
    def state(self):
        i = min(self.chunk, self.video.n_chunks - 1)
        return {
            "buffer": self.buffer, "max_buffer": self.max_buffer, "prev_level": self.prev_level,
            "pred_bw": self.pred_bw, "next_sizes": self.video.sizes[:, i].copy(),
            "next_vmaf": self.video.vmaf[:, i].copy(), "is_dynamic": bool(self.video.dynamic[i]),
            "hist_tput": list(self.hist_tput), "hist_dl": list(self.hist_dl), "hist_lv": list(self.hist_lv),
            "remaining": self.video.n_chunks - self.chunk, "chunk": self.chunk,
        }

    def observe(self, s=None, spec=None):
        """Flat observation vector for the policy network."""
        s = s or self.state(); spec = spec or self.spec
        K = self.K
        h_t = [np.log1p(x / 1000.0) / 6.0 for x in s["hist_tput"]]
        h_d = [min(x, 20.0) / 10.0 for x in s["hist_dl"]]
        h_l = [x / (K - 1) for x in s["hist_lv"]]
        nxt_s = [x / 1e6 / 3.0 for x in s["next_sizes"]]
        nxt_v = [x / 100.0 for x in s["next_vmaf"]]
        scal = [s["buffer"] / 20.0, s["max_buffer"] / 60.0, s["prev_level"] / (K - 1),
                np.log1p(s["pred_bw"] / 1000.0) / 6.0, s["remaining"] / self.video.n_chunks,
                float(s["is_dynamic"])]
        feats = spec.features(self.bitrates[-1])
        return np.asarray(h_t + h_d + h_l + nxt_s + nxt_v + scal + feats, dtype=np.float32)

    # ----------------------------------------------------------- step
    def step(self, level, delta_idx, intervened=False, extra_delay=0.0):
        """extra_delay: decision latency (s) before the request is sent, e.g. an LLM call."""
        s0 = self.state()
        w = self.spec.weights
        i = self.chunk
        size = self.video.sizes[level, i]
        vmaf = self.video.vmaf[level, i]
        dyn = bool(self.video.dynamic[i])
        if self.fixed_max_buffer is None:
            mb_new = float(np.clip(self.max_buffer + C.BUF_DELTAS[delta_idx], C.MAX_BUFFER_MIN, C.MAX_BUFFER_MAX))
        else:
            mb_new = self.fixed_max_buffer
        dl, true_bw = self._download(size)
        dl += C.RTT + extra_delay
        self._advance(C.RTT + extra_delay)
        rebuf = max(dl - self.buffer, 0.0)
        trough = self.buffer - dl
        buf = max(self.buffer - dl, 0.0) + self.L
        sleep = 0.0
        if buf > mb_new:
            sleep = buf - mb_new
            self._advance(sleep)
            buf = mb_new
        self.wall += dl + sleep

        # measurement sample
        if self.measure == "stable":
            sample = true_bw * (1 + C.STABLE_NOISE * self.rng.standard_normal())
        else:
            sample = size * 8 / 1000 / dl
        self.samples.append(max(sample, 1.0))

        # QoE (Eq. 2) and reward (Eq. 3)
        wq = w.delta if dyn else w.alpha
        smooth = 0.0 if self.prev_vmaf is None else abs(vmaf - self.prev_vmaf)
        stall = rebuf if i > 0 else 0.0                  # startup delay reported separately
        qoe = wq * vmaf - w.epsilon * smooth - w.beta * stall
        reward = (qoe - w.phi * buf) / C.REWARD_SCALE

        self.log.append(dict(chunk=i, level=level, bitrate=self.bitrates[level], vmaf=vmaf, dynamic=dyn,
                             size=size, dl=dl, rebuf=stall, startup=rebuf if i == 0 else 0.0,
                             buffer_before=s0["buffer"], trough=trough, buffer_after=buf,
                             mb_old=s0["max_buffer"], mb_new=mb_new, sleep=sleep, wall=self.wall,
                             prev_level=s0["prev_level"], pred_bw=s0["pred_bw"], true_bw=true_bw,
                             sample=sample, intervened=intervened, qoe=qoe, smooth=smooth))

        # bookkeeping
        self.hist_tput = self.hist_tput[1:] + [sample]
        self.hist_dl = self.hist_dl[1:] + [dl]
        self.hist_lv = self.hist_lv[1:] + [level]
        self.buffer, self.max_buffer = buf, mb_new
        self.prev_level, self.prev_vmaf = level, vmaf
        self.chunk += 1
        if self.predictor is not None:
            self.pred_bw = float(self.predictor.predict(self.samples))
        else:
            self.pred_bw = float(sample)
        done = self.chunk >= self.video.n_chunks
        return self.state(), reward, done, self.log[-1]
