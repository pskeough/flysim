"""NVIDIA Warp (CUDA) backend for flysim.brain.Brain.

Same equations as the numba kernel. Per 0.1 ms step: one launch updates every neuron
(Poisson drive, exact integration, delayed arrivals, threshold/reset) and one launch
propagates spikes (each spiking neuron walks its CSR row with atomic adds into the
delay ring). A chunk of steps is captured once as a CUDA graph and replayed, so a
10 ms chunk costs one graph launch and one device->host copy of spike counts.
"""
from __future__ import annotations

import numpy as np
import warp as wp

wp.config.quiet = True


@wp.kernel
def _step_neurons(t_arr: wp.array(dtype=wp.int32), v: wp.array(dtype=wp.float32),
                  g: wp.array(dtype=wp.float32), refr: wp.array(dtype=wp.int32),
                  rest: wp.array(dtype=wp.float32), buf: wp.array2d(dtype=wp.float32),
                  poi_p: wp.array(dtype=wp.float32), rfc: wp.array(dtype=wp.int32),
                  spiked: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
                  bg_p: wp.array(dtype=wp.float32), bg_w: wp.array(dtype=wp.float32),
                  a: float, b: float, c: float, v_th: float, v_rst: float, poi_w: float,
                  L: int, seed: int):
    i = wp.tid()
    t = t_arr[0]
    slot = t % (L + 1)
    vi = v[i]
    gi = g[i]
    r = refr[i]
    ri = rest[i]
    p = poi_p[i]
    if p > 0.0:
        st = wp.rand_init(seed * 7919 + t, i)
        if wp.randf(st) < p:
            vi = vi + poi_w
    if r > 0:
        r = r - 1
    else:
        vi = ri + (vi - ri) * a + gi * b
        gi = gi * c
    # background: random synaptic events (conductance, like an ordinary synapse; keeps refractoriness)
    q = bg_p[i]
    if q > 0.0:
        st2 = wp.rand_init(seed * 7919 + t, i + 1048576)  # offset past n: separate stream
        if wp.randf(st2) < q:
            gi = gi + bg_w[i]
    gin = buf[slot, i]
    if gin != 0.0:
        gi = gi + gin
        buf[slot, i] = 0.0
    s = 0
    if r == 0 and vi > v_th:
        vi = v_rst
        gi = 0.0
        r = rfc[i]
        s = 1
        counts[i] = counts[i] + 1
    spiked[i] = s
    v[i] = vi
    g[i] = gi
    refr[i] = r


@wp.kernel
def _propagate(t_arr: wp.array(dtype=wp.int32), spiked: wp.array(dtype=wp.int32),
               out_mask: wp.array(dtype=wp.int32), row_ptr: wp.array(dtype=wp.int32),
               col: wp.array(dtype=wp.int32), wts: wp.array(dtype=wp.float32),
               buf: wp.array2d(dtype=wp.float32), pre_gain: wp.array(dtype=wp.float32),
               post_gain: wp.array(dtype=wp.float32), use_post: int, L: int):
    i = wp.tid()
    if spiked[i] == 0 or out_mask[i] == 0:
        return
    gp = pre_gain[i]
    if gp == 0.0:
        return
    wslot = (t_arr[0] + L) % (L + 1)
    if use_post == 0:
        for e in range(row_ptr[i], row_ptr[i + 1]):
            wp.atomic_add(buf, wslot, col[e], wts[e] * gp)
    else:
        for e in range(row_ptr[i], row_ptr[i + 1]):
            j = col[e]
            wp.atomic_add(buf, wslot, j, wts[e] * gp * post_gain[j])


@wp.kernel
def _advance(t_arr: wp.array(dtype=wp.int32)):
    t_arr[0] = t_arr[0] + 1


class WarpState:
    def __init__(self, brain, device="cuda:0", chunk_steps=100, seed=0, share=None):
        """share: another WarpState on the same device whose connectivity arrays (read-only) this one reuses."""
        wp.init()
        self.dev = wp.get_device(device)
        self.br = brain
        n, L = brain.n, brain.delay_steps
        self.n, self.L = n, L
        d = self.dev
        self.v = wp.full(n, brain.p.v_0, dtype=wp.float32, device=d)
        self.g = wp.zeros(n, dtype=wp.float32, device=d)
        self.refr = wp.zeros(n, dtype=wp.int32, device=d)
        self.buf = wp.zeros((L + 1, n), dtype=wp.float32, device=d)
        self.spiked = wp.zeros(n, dtype=wp.int32, device=d)
        self.counts = wp.zeros(n, dtype=wp.int32, device=d)
        self.t_arr = wp.zeros(1, dtype=wp.int32, device=d)
        if share is not None:
            self.row_ptr, self.col, self.wts = share.row_ptr, share.col, share.wts
        else:
            self.row_ptr = wp.array(brain.row_ptr.astype(np.int32), dtype=wp.int32, device=d)
            self.col = wp.array(brain.col, dtype=wp.int32, device=d)
            self.wts = wp.array(brain.wts, dtype=wp.float32, device=d)
        self.rest = wp.full(n, brain.p.v_0, dtype=wp.float32, device=d)
        self.poi_p = wp.zeros(n, dtype=wp.float32, device=d)
        self.rfc = wp.full(n, brain.rfc_default, dtype=wp.int32, device=d)
        self.out_mask = wp.ones(n, dtype=wp.int32, device=d)
        self.bg_p = wp.zeros(n, dtype=wp.float32, device=d)
        self.bg_w = wp.zeros(n, dtype=wp.float32, device=d)
        # drugs (pharma.py): per-neuron multipliers on every synapse a neuron sends (pre) and receives (post).
        # Per brain, never shared, so a drugged fly and a sober fly can share one connectome on the GPU.
        self.pre_gain = wp.ones(n, dtype=wp.float32, device=d)
        self.post_gain = wp.ones(n, dtype=wp.float32, device=d)
        self.use_post = 0
        self.seed = seed
        self.chunk_steps = chunk_steps
        self._graph = None
        with wp.ScopedDevice(d):
            self._ev0 = wp.Event(enable_timing=True)
            self._ev1 = wp.Event(enable_timing=True)
        self.last_gpu_ms = 0.0

    def reset_state(self):
        """Back to rest (v = v_0, no conductance, empty delay buffer); keeps inputs and the RNG clock."""
        self.v.fill_(self.br.p.v_0)
        self.g.zero_()
        self.refr.zero_()
        self.buf.zero_()

    def set_syn_gain(self, pre, post):
        """Synaptic gain per neuron: the weight of synapse i->j is scaled by pre[i] * post[j].
        Assigned in place, so a captured CUDA graph keeps working; the post multiply is compiled
        out of the graph while every post gain is 1 (switching re-captures once)."""
        self.pre_gain.assign(np.asarray(pre, dtype=np.float32))
        self.post_gain.assign(np.asarray(post, dtype=np.float32))
        use = int(not np.all(np.asarray(post) == 1.0))
        if use != self.use_post:
            self.use_post = use
            self._graph = None

    def set_background(self, rate_hz, w_mv):
        self.bg_p.assign((np.asarray(rate_hz) * self.br.dt * 1e-3).astype(np.float32))
        self.bg_w.assign(np.asarray(w_mv, dtype=np.float32))

    def sync_inputs(self, poi_rate, silenced, i_ext):
        br = self.br
        self.poi_p.assign((poi_rate * br.dt * 1e-3).astype(np.float32))
        rfc = np.full(self.n, br.rfc_default, dtype=np.int32)
        rfc[poi_rate > 0] = 0
        self.rfc.assign(rfc)
        self.out_mask.assign((~silenced).astype(np.int32))
        self.rest.assign((br.p.v_0 + i_ext).astype(np.float32))

    def _launch_steps(self, n_steps):
        br = self.br
        for _ in range(n_steps):
            wp.launch(_step_neurons, dim=self.n, device=self.dev, inputs=[
                self.t_arr, self.v, self.g, self.refr, self.rest, self.buf, self.poi_p,
                self.rfc, self.spiked, self.counts, self.bg_p, self.bg_w, br.a, br.b, br.c, br.p.v_th,
                br.p.v_rst, br.poi_w, self.L, self.seed])
            wp.launch(_propagate, dim=self.n, device=self.dev, inputs=[
                self.t_arr, self.spiked, self.out_mask, self.row_ptr, self.col,
                self.wts, self.buf, self.pre_gain, self.post_gain, self.use_post, self.L])
            wp.launch(_advance, dim=1, device=self.dev, inputs=[self.t_arr])

    def launch(self, n_steps):
        """Queue a window on the GPU stream and return without waiting."""
        self.counts.zero_()
        stream = self.dev.stream
        stream.record_event(self._ev0)
        if n_steps == self.chunk_steps:
            if self._graph is None:
                with wp.ScopedCapture(device=self.dev) as cap:
                    self._launch_steps(n_steps)
                self._graph = cap.graph
            wp.capture_launch(self._graph)
        else:
            self._launch_steps(n_steps)
        stream.record_event(self._ev1)

    def collect(self):
        out = self.counts.numpy()  # synchronises
        self.last_gpu_ms = wp.get_event_elapsed_time(self._ev0, self._ev1)
        return out

    def run(self, n_steps):
        self.launch(n_steps)
        return self.collect()
