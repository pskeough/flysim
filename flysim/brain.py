"""Whole-brain leaky integrate-and-fire model on the FlyWire v783 connectome.

A re-implementation of the Shiu et al. (2024, Nature, doi:10.1038/s41586-024-07763-9)
Brian2 model (github.com/philshiu/Drosophila_brain_model, model.py) so it can be
stepped in small chunks inside a body-simulation loop and perturbed live.

Same equations and constants as the reference `default_params`:
    dv/dt = (v_0 - v + g + I_ext) / t_mbr      (frozen while refractory)
    dg/dt = -g / tau                            (frozen while refractory)
    spike: v > v_th  ->  v = v_rst, g = 0, refractory t_rfc
    synapse: g_post += w_syn * (sign x synapse count), after delay t_dly
    activation ("optogenetic"): Poisson input at rate r, each event adds
        w_syn * f_poi = 68.75 mV to v; activated neurons have no refractory period
    silencing: outgoing weights of the neuron set to zero (as model.silence does)

Integration is the exact solution of the linear system over dt (Brian2 'linear').
I_ext is an addition for this project (graded sensory drive, in mV of steady-state
depolarisation); it is zero unless a stimulus sets it.

Backends with identical semantics:
    'numba' - fused CPU kernel over the active set (default)
    'warp'  - NVIDIA Warp CUDA kernels, 10 ms chunks captured as CUDA graphs
    'torch' - vectorised reference, launch-bound (kept for cross-checking only)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import numba as nb
except ImportError:  # pragma: no cover
    nb = None

DATA_DIR = Path(__file__).resolve().parents[1] / "third_party" / "Drosophila_brain_model"


@dataclass(frozen=True)
class LIFParams:
    dt_ms: float = 0.1  # Brian2 default clock
    v_0: float = -52.0  # mV, resting potential
    v_rst: float = -52.0  # mV
    v_th: float = -45.0  # mV
    t_mbr: float = 20.0  # ms
    tau: float = 5.0  # ms
    t_rfc: float = 2.2  # ms
    t_dly: float = 1.8  # ms
    w_syn: float = 0.275  # mV per synapse
    f_poi: float = 250.0  # Poisson weight multiplier


def load_connectome(version: int = 783, data_dir: Path = DATA_DIR):
    """Returns (flywire_ids int64[N], pre int32[E], post int32[E], count_signed float32[E])."""
    comp = pd.read_csv(data_dir / f"Completeness_{version}.csv", index_col=0)
    con = pd.read_parquet(data_dir / f"Connectivity_{version}.parquet")
    ids = comp.index.to_numpy(dtype=np.int64)
    pre = con["Presynaptic_Index"].to_numpy(dtype=np.int32)
    post = con["Postsynaptic_Index"].to_numpy(dtype=np.int32)
    w = con["Excitatory x Connectivity"].to_numpy(dtype=np.float32)
    return ids, pre, post, w


def _to_csr(n, pre, post, w):
    order = np.argsort(pre, kind="stable")
    pre_s, post_s, w_s = pre[order], post[order], w[order]
    row_ptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(row_ptr, pre_s + 1, 1)
    row_ptr = np.cumsum(row_ptr)
    return row_ptr, post_s.astype(np.int32), w_s.astype(np.float32)


if nb is not None:

    @nb.njit(cache=True, fastmath=True, nogil=True)
    def _run_numba(n_steps, t0, v, g, refr, rest, buf, row_ptr, col, wts,
                   out_mask, rfc_steps, poi_idx, poi_p, a, b, c, v_th, v_rst, poi_w,
                   counts, spk_t, spk_i, spk_cap, act, act64, pending_until, pinned,
                   eps, stats):
        """Fused step loop over the active set.

        The model has no spontaneous activity: a neuron at rest (v == rest, g == 0,
        not refractory, no synaptic input in flight, not Poisson-driven) is a fixed
        point. `act` (1 byte per neuron, scanned 8 at a time through the uint64 view
        `act64`) marks the neurons that are not; they are integrated in index order.
        A neuron is marked when a spike is sent to it (pending_until = arrival step),
        when it is Poisson-driven (pinned) or when its I_ext changes, and unmarked
        (snapped to rest) once |v - rest| and |g| < eps mV with nothing in flight.
        The delay ring has L + 1 slots so the slot written this step is never the
        slot being read."""
        L = buf.shape[0] - 1
        nw = act64.shape[0]
        n_spk = 0
        n_act = 0
        for s in range(n_steps):
            t = t0 + s
            row = buf[t % (L + 1)]
            wrow = buf[(t + L) % (L + 1)]
            for k in range(poi_idx.shape[0]):
                if np.random.random() < poi_p[k]:
                    v[poi_idx[k]] += poi_w
            n_act = 0
            for w in range(nw):
                if act64[w] == 0:
                    continue
                for i in range(w * 8, w * 8 + 8):
                    if act[i] == 0:
                        continue
                    n_act += 1
                    ri = rest[i]
                    vi = v[i]
                    gi = g[i]
                    r = refr[i]
                    if r > 0:
                        r -= 1
                    else:
                        vi = ri + (vi - ri) * a + gi * b
                        gi = gi * c
                    gin = row[i]
                    if gin != 0.0:
                        gi += gin
                        row[i] = 0.0
                    if r == 0 and vi > v_th:
                        vi = v_rst
                        gi = 0.0
                        r = rfc_steps[i]
                        counts[i] += 1
                        if n_spk < spk_cap:
                            spk_t[n_spk] = t
                            spk_i[n_spk] = i
                        n_spk += 1
                        if out_mask[i]:
                            arr = t + L
                            for e in range(row_ptr[i], row_ptr[i + 1]):
                                j = col[e]
                                wrow[j] += wts[e]
                                pending_until[j] = arr
                                act[j] = 1
                    elif (r == 0 and pinned[i] == 0 and t >= pending_until[i]
                          and abs(gi) < eps and abs(vi - ri) < eps):
                        vi = ri
                        gi = 0.0
                        act[i] = 0
                    v[i] = vi
                    g[i] = gi
                    refr[i] = r
        stats[0] = n_act
        return n_spk


class Brain:
    """Stateful whole-brain LIF simulator.

    Perturbation API (all by index; see `index_of` for FlyWire root IDs):
        set_activation(idx, rate_hz)   Poisson drive (0 removes it)
        set_silenced(idx, bool)        zero outgoing synapses
        set_current(idx, mV)           graded steady drive I_ext
        run(ms)                        advance; returns spike counts for the window
    """

    def __init__(self, backend: str = "numba", device: str | None = None,
                 params: LIFParams = LIFParams(), version: int = 783, seed: int = 0):
        self.p = params
        self.ids, pre, post, w = load_connectome(version)
        self.n = len(self.ids)
        self.n_syn = len(pre)
        self._id2idx = {int(f): i for i, f in enumerate(self.ids)}
        self.row_ptr, self.col, cnt = _to_csr(self.n, pre, post, w)
        self.wts = (cnt * params.w_syn).astype(np.float32)

        p = params
        self.dt = p.dt_ms
        self.a = math.exp(-p.dt_ms / p.t_mbr)
        self.c = math.exp(-p.dt_ms / p.tau)
        self.b = p.tau / (p.tau - p.t_mbr) * (self.c - self.a)
        self.delay_steps = int(round(p.t_dly / p.dt_ms))  # 18
        self.rfc_default = int(round(p.t_rfc / p.dt_ms))  # 22
        self.poi_w = p.w_syn * p.f_poi  # 68.75 mV

        self.backend = backend
        self.t_step = 0
        self.poi_rate = np.zeros(self.n, dtype=np.float64)
        self.silenced = np.zeros(self.n, dtype=np.bool_)
        self.i_ext_np = np.zeros(self.n, dtype=np.float32)
        np.random.seed(seed)

        if backend == "numba":
            if nb is None:
                raise RuntimeError("numba not installed")
            self._init_numba()
        elif backend == "torch":
            self._init_torch(device)
        elif backend == "warp":
            from .brain_warp import WarpState
            self._warp = WarpState(self, device=device or "cuda:0", seed=seed)
            self._sync_inputs()
        else:
            raise ValueError(backend)

    def clone(self, seed: int):
        """A second brain on the same connectome (warp only): its own membrane state, inputs and noise
        seed; the connectivity on the GPU is shared with this one, not copied."""
        if self.backend != "warp":
            raise NotImplementedError("clone: warp backend only")
        from .brain_warp import WarpState
        b = object.__new__(Brain)
        skip = ("_warp", "poi_rate", "silenced", "i_ext_np", "_pending", "t_step")
        b.__dict__.update({k: v for k, v in self.__dict__.items() if k not in skip})
        b.t_step = 0
        b.poi_rate = np.zeros(self.n, dtype=np.float64)
        b.silenced = np.zeros(self.n, dtype=np.bool_)
        b.i_ext_np = np.zeros(self.n, dtype=np.float32)
        b._warp = WarpState(b, device=self._warp.dev, seed=seed, share=self._warp)
        b._sync_inputs()
        return b

    # ---------------------------------------------------------------- lookup
    def index_of(self, flywire_ids):
        """FlyWire root IDs -> model indices; unknown IDs are dropped (and reported)."""
        out, missing = [], []
        for f in flywire_ids:
            i = self._id2idx.get(int(f))
            (out if i is not None else missing).append(i if i is not None else int(f))
        self.last_missing = missing
        return np.array(out, dtype=np.int64)

    # ---------------------------------------------------------- perturbations
    def set_activation(self, idx, rate_hz: float):
        self.poi_rate[np.asarray(idx)] = rate_hz
        self._sync_inputs()

    def set_silenced(self, idx, on: bool = True):
        self.silenced[np.asarray(idx)] = on
        self._sync_inputs()

    def set_current(self, idx, mv: float):
        self.i_ext_np[np.asarray(idx)] = mv
        self._sync_inputs()

    def set_inputs(self, poi_rate, silenced, i_ext):
        """Replace all exogenous inputs at once (used by the app's input mixer)."""
        self.poi_rate[:] = poi_rate
        self.silenced[:] = silenced
        self.i_ext_np[:] = i_ext
        self._sync_inputs()

    def launch(self, ms: float):
        """Start a window; on warp this returns immediately (GPU runs while the CPU works)."""
        if self.backend == "warp":
            n_steps = int(round(ms / self.dt))
            self.t_step += n_steps
            self._warp.launch(n_steps)
            self._pending = None
        else:
            self._pending = self.run(ms)

    def collect(self):
        """Spike counts of the window started by `launch` (blocks until done)."""
        return self._warp.collect() if self.backend == "warp" else self._pending

    @property
    def last_gpu_ms(self):
        return self._warp.last_gpu_ms if self.backend == "warp" else None

    def set_background(self, rate_hz, w_mv):
        """Background synaptic noise (warp only): per-neuron Poisson rate (Hz) of events that add
        w_mv to the synaptic conductance, like one ordinary input connection. Unlike
        set_activation it does not force spikes or remove the refractory period."""
        if self.backend != "warp":
            raise NotImplementedError("set_background: warp backend only")
        r = np.broadcast_to(np.asarray(rate_hz, float), (self.n,))
        w = np.broadcast_to(np.asarray(w_mv, float), (self.n,))
        self._warp.set_background(r, w)

    def set_syn_gain(self, pre, post):
        """Drug lever (warp only): the weight of synapse i->j becomes w_ij * pre[i] * post[j].
        pre models release or the transmitter's receptors as a whole (by presynaptic transmitter);
        post models the receiving cells (by region or cell class). Ones = no drug."""
        if self.backend != "warp":
            raise NotImplementedError("set_syn_gain: warp backend only")
        self._warp.set_syn_gain(pre, post)

    def reset_state(self):
        """Membrane state back to rest, for repeated trials on one instance (and the runaway watchdog)."""
        if self.backend == "numba":          # added 2026-10-08: the watchdog and revive called this and raised on numba
            self.v[:] = self.p.v_0
            self.g[:] = 0.0
            self.refr[:] = 0
            self.buf[:] = 0.0
            self._act[:] = 0
            self._pending_until[:] = 0
            return
        if self.backend != "warp":
            raise NotImplementedError("reset_state: warp backend only; build a new Brain instead")
        self._warp.reset_state()

    def run_chunked(self, ms: float, chunk_ms: float = 10.0):
        """run() in 10 ms pieces so the warp backend replays its captured CUDA graph."""
        tot = np.zeros(self.n, np.int64)
        for _ in range(int(round(ms / chunk_ms))):
            tot += self.run(chunk_ms)
        return tot

    def clear_perturbations(self):
        self.poi_rate[:] = 0
        self.silenced[:] = False
        self.i_ext_np[:] = 0
        self._sync_inputs()

    def _sync_inputs(self):
        if self.backend == "warp":
            self._warp.sync_inputs(self.poi_rate, self.silenced, self.i_ext_np)
            return
        poi_idx = np.nonzero(self.poi_rate > 0)[0]
        poi_p = self.poi_rate[poi_idx] * self.dt * 1e-3
        rfc = np.full(self.n, self.rfc_default, dtype=np.int32)
        rfc[poi_idx] = 0  # as model.poi: activated neurons have no refractory period
        if self.backend == "numba":
            self._poi_idx = poi_idx.astype(np.int64)
            self._poi_p = poi_p.astype(np.float64)
            self._rfc = rfc
            self._out_mask = ~self.silenced
            new_rest = self._pad((self.p.v_0 + self.i_ext_np).astype(np.float32),
                                 np.float32(self.p.v_0))
            changed = np.nonzero(new_rest != self._rest)[0]
            self._rest = new_rest
            self._pinned[:] = 0
            self._pinned[poi_idx] = 1
            self._act[poi_idx] = 1
            self._act[changed] = 1
        else:
            T = self._torch
            d = self._dev
            self._poi_idx = T.as_tensor(poi_idx, device=d)
            self._poi_p = T.as_tensor(poi_p, device=d, dtype=T.float32)
            self._rfc = T.as_tensor(rfc, device=d)
            self._out_mask = T.as_tensor(~self.silenced, device=d)
            self._i_ext = T.as_tensor(self.i_ext_np, device=d)

    # ----------------------------------------------------------------- numba
    def _init_numba(self):
        n, L = self.n, self.delay_steps
        n_pad = (n + 7) // 8 * 8
        self.v = np.full(n_pad, self.p.v_0, dtype=np.float32)
        self.g = np.zeros(n_pad, dtype=np.float32)
        self.refr = np.zeros(n_pad, dtype=np.int32)
        self.buf = np.zeros((L + 1, n_pad), dtype=np.float32)
        self._rest = np.full(n_pad, self.p.v_0, dtype=np.float32)
        self._act = np.zeros(n_pad, dtype=np.uint8)
        self._act64 = self._act.view(np.uint64)
        self._pending_until = np.zeros(n_pad, dtype=np.int64)
        self._pinned = np.zeros(n_pad, dtype=np.uint8)
        self._spk_cap = 2_000_000
        self._spk_t = np.zeros(self._spk_cap, dtype=np.int64)
        self._spk_i = np.zeros(self._spk_cap, dtype=np.int32)
        self._stats = np.zeros(4, dtype=np.int64)
        self.eps = 1e-6  # mV; below this a relaxing neuron is snapped to rest
        self._sync_inputs()

    def _pad(self, x, fill):
        out = np.full(len(self.v), fill, dtype=x.dtype)
        out[: self.n] = x
        return out

    @property
    def n_active(self):
        """Neurons integrated on the last step; all others are exactly at rest."""
        return int(self._stats[0]) if self.backend == "numba" else self.n

    def _run_numba(self, n_steps):
        counts = np.zeros(len(self.v), dtype=np.int32)
        n_spk = _run_numba(
            n_steps, self.t_step, self.v, self.g, self.refr, self._rest, self.buf,
            self.row_ptr, self.col, self.wts, self._pad(self._out_mask, False),
            self._pad(self._rfc, 0), self._poi_idx, self._poi_p, self.a, self.b,
            self.c, self.p.v_th, self.p.v_rst, self.poi_w, counts, self._spk_t,
            self._spk_i, self._spk_cap, self._act, self._act64, self._pending_until,
            self._pinned, self.eps, self._stats)
        self.t_step += n_steps
        return counts[: self.n], min(n_spk, self._spk_cap)

    # ----------------------------------------------------------------- torch
    def _init_torch(self, device):
        import torch
        self._torch = torch
        d = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._dev = d
        n, L = self.n, self.delay_steps
        self.v = torch.full((n,), self.p.v_0, device=d)
        self.g = torch.zeros(n, device=d)
        self.refr = torch.zeros(n, dtype=torch.int32, device=d)
        self.buf = torch.zeros((L, n), device=d)
        self.t_row_ptr = torch.as_tensor(self.row_ptr, device=d)
        self.t_deg = self.t_row_ptr[1:] - self.t_row_ptr[:-1]
        self.t_col = torch.as_tensor(self.col, device=d, dtype=torch.int64)
        self.t_wts = torch.as_tensor(self.wts, device=d)
        self._sync_inputs()

    def _run_torch(self, n_steps):
        T = self._torch
        counts = T.zeros(self.n, dtype=T.int32, device=self._dev)
        v, g, refr, buf = self.v, self.g, self.refr, self.buf
        L = self.delay_steps
        rest = self.p.v_0 + self._i_ext
        n_spk = 0
        for s in range(n_steps):
            slot = (self.t_step + s) % L
            active = refr == 0
            refr.sub_((~active).int())
            v_new = rest + (v - rest) * self.a + g * self.b
            v.copy_(T.where(active, v_new, v))
            g.copy_(T.where(active, g * self.c, g))
            g.add_(buf[slot])
            buf[slot].zero_()
            if self._poi_idx.numel():
                hit = T.rand(self._poi_idx.numel(), device=self._dev) < self._poi_p
                v.index_add_(0, self._poi_idx, hit.float() * self.poi_w)
            spk = (refr == 0) & (v > self.p.v_th)
            idx = spk.nonzero().squeeze(1)
            if idx.numel():
                v[idx] = self.p.v_rst
                g[idx] = 0.0
                refr[idx] = self._rfc[idx]
                counts[idx] += 1
                n_spk += idx.numel()
                src = idx[self._out_mask[idx]]
                if src.numel():
                    deg = self.t_deg[src]
                    starts = self.t_row_ptr[:-1][src]
                    rep = T.repeat_interleave(T.arange(src.numel(), device=self._dev), deg)
                    offs = T.arange(rep.numel(), device=self._dev) - T.repeat_interleave(
                        T.cumsum(deg, 0) - deg, deg)
                    e = starts[rep] + offs
                    buf[slot].index_add_(0, self.t_col[e], self.t_wts[e])
        self.t_step += n_steps
        return counts.cpu().numpy(), n_spk

    # ------------------------------------------------------------------- run
    def run(self, ms: float):
        n_steps = int(round(ms / self.dt))
        if self.backend == "numba":
            return self._run_numba(n_steps)[0]
        if self.backend == "warp":
            self.t_step += n_steps
            return self._warp.run(n_steps)
        return self._run_torch(n_steps)[0]

    def last_spikes(self, n):
        """(step, index) arrays of the spikes from the last numba `run` (capped)."""
        return self._spk_t[:n], self._spk_i[:n]

    @property
    def t_ms(self):
        return self.t_step * self.dt
