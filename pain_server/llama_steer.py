"""Pain steering on the quantised GGUF through llama.cpp, with the dose set per request (prototype, 2026-10-08).

Why: server.py de-quantises the Q4_K_M weights into bf16 in transformers (about 8 GB). llama.cpp runs the Q4_K_M
weights as they are. Stock llama-server only takes a control vector at start-up (--control-vector-scaled,
fixed scale), but llama.dll exports llama_set_adapter_cvec(ctx, data, len, n_embd, il_start, il_end), which copies
a new vector into the context in a few microseconds, so the dose can change on every call. This module drives
llama.dll from the installed build through ctypes (no llama-cpp-python, which has no CUDA 13 / sm_120 wheel here).

Layer mapping: llama.cpp adds the control vector to the output of block il (0-based, after the FFN residual), the
same place the transformers forward hook on model.model.layers[il] adds it. data is n_embd x n_layer floats and
starts at layer 1, so layer il lives at offset (il - 1) * n_embd.

Struct layouts are those of llama.h at tag b10435 (the build in LLAMA_DIR; install.bat downloads the official
ggml-org/llama.cpp release b10435 Windows build into llama/). Another llama.cpp tag can move these offsets: keep
b10435 unless the offsets below are re-checked against that tag's llama.h. The param structs are only touched
through llama_*_default_params plus a few fixed offsets, and are handled as oversized byte buffers (Windows x64
passes and returns structs larger than 8 bytes through a pointer, so the spare tail is harmless).

    from pain_server.llama_steer import LlamaSteer
    m = LlamaSteer(gguf, vecs_npz, gpu=0); text, stats = m.generate(prompt_text, dose=3.0, which="pain_s2")
"""
from __future__ import annotations

import ctypes as C
import os
import time
from pathlib import Path

import numpy as np

LLAMA_DIR = Path(os.environ.get("LLAMA_DIR") or Path(__file__).resolve().parents[1] / "llama")

# offsets in llama.h b10435
MP_N_GPU_LAYERS = 16
CP_N_CTX, CP_N_BATCH, CP_N_UBATCH, CP_N_THREADS, CP_N_THREADS_BATCH = 0, 4, 8, 28, 32


class _Blob(C.Structure):
    _fields_ = [("b", C.c_ubyte * 512)]


class _Batch(C.Structure):
    _fields_ = [("n_tokens", C.c_int32), ("token", C.POINTER(C.c_int32)), ("embd", C.POINTER(C.c_float)),
                ("pos", C.POINTER(C.c_int32)), ("n_seq_id", C.POINTER(C.c_int32)),
                ("seq_id", C.POINTER(C.POINTER(C.c_int32))), ("logits", C.POINTER(C.c_int8))]


class _LogitBias(C.Structure):
    _fields_ = [("token", C.c_int32), ("bias", C.c_float)]


class _ChainParams(C.Structure):
    _fields_ = [("no_perf", C.c_bool)]


def _put(blob, off, ctype, val):
    ctype.from_buffer(blob.b, off).value = val


def _lib():
    if not (LLAMA_DIR / "llama.dll").exists():
        raise SystemExit(f"llama.dll not found in {LLAMA_DIR}: run install.bat, or set LLAMA_DIR to a llama.cpp "
                         f"b10435 Windows build")
    os.add_dll_directory(str(LLAMA_DIR))
    L = C.CDLL(str(LLAMA_DIR / "llama.dll"))
    P, i32, f32, b = C.c_void_p, C.c_int32, C.c_float, C.c_bool
    sig = {
        "llama_backend_init": (None, []),
        "llama_model_default_params": (_Blob, []),
        "llama_context_default_params": (_Blob, []),
        "llama_model_load_from_file": (P, [C.c_char_p, _Blob]),
        "llama_init_from_model": (P, [P, _Blob]),
        "llama_model_get_vocab": (P, [P]),
        "llama_model_n_embd": (i32, [P]),
        "llama_model_n_layer": (i32, [P]),
        "llama_set_adapter_cvec": (i32, [P, C.POINTER(f32), C.c_size_t, i32, i32, i32]),
        "llama_tokenize": (i32, [P, C.c_char_p, i32, C.POINTER(i32), i32, b, b]),
        "llama_token_to_piece": (i32, [P, i32, C.c_char_p, i32, i32, b]),
        "llama_vocab_is_eog": (b, [P, i32]),
        "llama_batch_get_one": (_Batch, [C.POINTER(i32), i32]),
        "llama_decode": (i32, [P, _Batch]),
        "llama_get_memory": (P, [P]),
        "llama_memory_clear": (None, [P, b]),
        "llama_sampler_chain_default_params": (_ChainParams, []),
        "llama_sampler_chain_init": (P, [_ChainParams]),
        "llama_sampler_chain_add": (None, [P, P]),
        "llama_sampler_init_temp": (P, [f32]),
        "llama_sampler_init_top_p": (P, [f32, C.c_size_t]),
        "llama_sampler_init_dist": (P, [C.c_uint32]),
        "llama_sampler_init_greedy": (P, []),
        "llama_sampler_sample": (i32, [P, P, i32]),
        "llama_sampler_free": (None, [P]),
        "llama_get_logits_ith": (C.POINTER(f32), [P, i32]),
        "llama_vocab_n_tokens": (i32, [P]),
        "llama_free": (None, [P]),
        "llama_synchronize": (None, [P]),
        "llama_model_free": (None, [P]),
        "llama_sampler_init_grammar": (P, [P, C.c_char_p, C.c_char_p]),
        "llama_sampler_init_logit_bias": (P, [i32, i32, C.c_void_p]),
    }
    for name, (res, args) in sig.items():
        f = getattr(L, name)
        f.restype, f.argtypes = res, args
    # ggml loads ggml-cuda.dll by full path, and Python's DLL search mode does not look in that DLL's folder for its
    # CUDA dependencies, so the CUDA backend failed silently (CPU only, ~2 tok/s). Pre-load them (CUDA 13 build, or
    # the CUDA 12 build install.bat picks for older drivers).
    for dep in ("cudart64_13.dll", "cublasLt64_13.dll", "cublas64_13.dll",
                "cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll"):
        if (LLAMA_DIR / dep).exists():
            C.CDLL(str(LLAMA_DIR / dep))
    G = C.CDLL(str(LLAMA_DIR / "ggml.dll"))          # the CUDA and CPU backends are separate DLLs, loaded here
    G.ggml_backend_load_all_from_path.argtypes = [C.c_char_p]
    G.ggml_backend_load_all_from_path(str(LLAMA_DIR).encode())
    G.ggml_backend_dev_by_name.restype, G.ggml_backend_dev_by_name.argtypes = C.c_void_p, [C.c_char_p]
    # activation read-out (eval callback): tensor name and a device -> host copy, from ggml-base
    B = C.CDLL(str(LLAMA_DIR / "ggml-base.dll"))
    B.ggml_get_name.restype, B.ggml_get_name.argtypes = C.c_char_p, [C.c_void_p]
    B.ggml_backend_tensor_get.restype = None
    B.ggml_backend_tensor_get.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_size_t]
    B.ggml_backend_tensor_set.restype = None          # perturbation write-back (Perturb); exported by ggml-base.dll
    B.ggml_backend_tensor_set.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_size_t]
    L.ggml = G
    L.base = B
    return L


import re  # noqa: E402

_NOW_RE = re.compile(r"(?:^|\n)[ \t*]*NOW[ \t*]*: ?$")

# ggml_backend_sched_eval_callback: bool (*)(struct ggml_tensor * t, bool ask, void * user_data)
EVAL_CB = C.CFUNCTYPE(C.c_bool, C.c_void_p, C.c_bool, C.c_void_p)
CP_CB_EVAL = 88                       # llama_context_params.cb_eval at b10435 (after 22 4-byte fields)
T_TYPE, T_NE = 0, 16                  # ggml_tensor: enum ggml_type type at 0; int64 ne[4] at 16
GGML_TYPE_F32 = 0


class Readout:
    """Residual-stream read-out during generation (direction two, 2026-10-08).

    dirs: {name: {layer: unit vector}}. While on, the eval callback copies l_out-<layer> (block output, after the
    control vector) for each requested layer at every decode and projects it on the directions. The control vector's
    own contribution at that layer (known: it is in the cvec buffer) is subtracted, so the read-out is what the model
    carries, not what was added. Prompt decodes: mean over the prompt tokens (mode "prompt", for extraction) or skipped
    (mode "gen"). Per generated token: the projection averaged over the layers -> self.tokens [ {name: value} ]."""

    def __init__(self, dirs):
        self.names = sorted(dirs)
        self.layers = sorted({L for d in dirs.values() for L in d})
        self.R = {L: np.stack([dirs[n][L] for n in self.names]).astype(np.float32) for L in self.layers}
        self.want = {f"l_out-{L}".encode(): L for L in self.layers}
        self.on, self.mode = False, "gen"
        self.cur, self.tokens, self.prompt_means = {}, [], {}
        self.cvec = None                       # {layer: added vector} for the correction
        self.capture_full = None               # {layer: (n_tok, d)} when set to {} (extraction)


class Perturb:
    """Pharmacology-style perturbation of the residual stream, written back from the eval callback (drugs, 2026-10-24).

    noise  {layer: sigma}  gaussian noise added to l_out-<layer>: sigma x the row's rms x N(0, I). Ethanol as a degraded
                           signal. Row 0 of a multi-row (prompt) decode, the attention sink with its massive activations,
                           is left alone.
    gain   (lo, hi, g)     residual-update gain on layers lo..hi: h'_L = h'_(L-1) + g (h_L - h'_(L-1)). g = 1 leaves the
                           layers alone, 0 skips them. Sedation as a damped forward pass.
    prefill                also perturb the prompt rows (default off: the prompt is read cleanly, the thought is not).
    The state is plain Python, so it can change between tokens for free (a control vector re-set costs ~45 ms).
    Checked on CPU (results/drug_mech_scout_2026-10-24/cb_write_probe*.py): tensor_set from the callback changes
    the logits, a get -> set round trip changes nothing, gain 1.0 is the identity."""

    POOL = 2048

    def __init__(self, n_embd, seed=0):
        self.noise, self.gain, self.prefill = {}, None, False
        self.rng = np.random.default_rng(seed)
        self.pool = self.rng.standard_normal((self.POOL, n_embd)).astype(np.float32)
        self.want, self.prev, self.on, self.n_writes = {}, None, False, 0

    def set(self, noise=None, gain=None, prefill=False):
        self.noise = {int(k): float(v) for k, v in (noise or {}).items() if float(v) > 0}
        self.gain = (int(gain[0]), int(gain[1]), float(gain[2])) if gain and float(gain[2]) < 1.0 else None
        self.prefill = bool(prefill)
        ls = set(self.noise)
        if self.gain:
            ls |= set(range(max(self.gain[0] - 1, 0), self.gain[1] + 1))   # layer lo-1 supplies the layer-lo input
        self.want = {f"l_out-{L}".encode(): L for L in ls}
        self.prev = None

    def apply(self, L, arr):
        """arr: (rows, d) float32 block output of layer L as computed. -> the array to write back, or None."""
        n = arr.shape[0]
        if n > 1 and not self.prefill:
            self.prev = None
            return None
        out, g = None, self.gain
        in_band = g is not None and g[0] - 1 <= L <= g[1]
        if in_band and L >= g[0] and self.prev is not None and self.prev.shape == arr.shape:
            out = self.prev + g[2] * (arr - self.prev)
        s = self.noise.get(L)
        if s:
            base = arr if out is None else out
            rms = np.sqrt((base ** 2).mean(1, keepdims=True))
            i = int(self.rng.integers(0, self.POOL - min(n, self.POOL) + 1))
            nz = self.pool[i:i + n] if n <= self.POOL else self.rng.standard_normal(base.shape).astype(np.float32)
            nz = nz.copy()
            if n > 1:
                nz[0] = 0.0
            out = base + s * rms * nz
        if in_band:
            self.prev = (arr if out is None else out).copy()
        return out


class LlamaSteer:
    def __init__(self, gguf, vecs_npz, gpu=0, n_ctx=4096, n_threads=4):
        self.L = L = _lib()
        L.llama_backend_init()
        mp = L.llama_model_default_params()
        if gpu is None:                        # CPU only (PAIN device "cpu"): no layers offloaded, no device list
            _put(mp, MP_N_GPU_LAYERS, C.c_int32, 0)
        else:
            _put(mp, MP_N_GPU_LAYERS, C.c_int32, 999)
        # one card: a NULL-terminated device list holding only CUDA<gpu>. (split_mode NONE + main_gpu was tried
        # first: it reported 37/37 layers offloaded but put the KV cache and the layers' compute on the CPU, ~2 tok/s.)
        if gpu is not None:
            dev = L.ggml.ggml_backend_dev_by_name(f"CUDA{int(gpu)}".encode())
            if not dev:
                raise RuntimeError(f"no llama.cpp device CUDA{gpu}")
            self._devs = (C.c_void_p * 2)(dev, None)
            _put(mp, 0, C.c_void_p, C.cast(self._devs, C.c_void_p).value)
        self.model = L.llama_model_load_from_file(str(gguf).encode(), mp)
        if not self.model:
            raise RuntimeError(f"llama.cpp could not load {gguf}")
        cp = L.llama_context_default_params()
        for off, v in ((CP_N_CTX, n_ctx), (CP_N_BATCH, 2048), (CP_N_UBATCH, 512)):
            _put(cp, off, C.c_uint32, v)
        for off in (CP_N_THREADS, CP_N_THREADS_BATCH):
            _put(cp, off, C.c_int32, n_threads)
        self.readout = None                    # Readout, set by set_readout(); the callback is cheap while it is None
        self.act = None                        # activity picture, set by set_activity()
        self.ro_s, self.ro_n = 0.0, 0
        # two contexts on the same weights: a plain one, and one with the eval callback for the read-out. The Python
        # callback is asked about every graph node (~1270 per token), which halves the speed even when it asks for
        # nothing, so calls without a read-out never go through it. (KV cache ~0.6 GB each at n_ctx 4096.)
        self.ctx_plain = L.llama_init_from_model(self.model, cp)
        self._cb = EVAL_CB(self._eval_cb)      # kept alive for the context's lifetime
        _put(cp, CP_CB_EVAL, C.c_void_p, C.cast(self._cb, C.c_void_p).value)
        self.ctx_ro = L.llama_init_from_model(self.model, cp)
        if not self.ctx_plain or not self.ctx_ro:
            raise RuntimeError("llama.cpp could not create a context")
        self.ctx = self.ctx_plain
        self._cp_cb = cp                        # params with the callback, for the twin context (made on first use)
        self.ctx_twin = None
        self._twin_phase = False                # True while the twin decodes: the callback stores into the twin slots
        self.vocab = L.llama_model_get_vocab(self.model)
        self.n_embd = L.llama_model_n_embd(self.model)
        self.n_layer = L.llama_model_n_layer(self.model)
        self.perturb = Perturb(self.n_embd)    # drug perturbations written from the eval callback; off unless generate(perturb=...)
        v = np.load(vecs_npz)
        self.layer = int(v["layer"])
        self.vecs = {k: v[k].astype(np.float32) for k in ("pain_s2", "pain_s1", "fear", "sadness", "random")}
        assert self.vecs["pain_s2"].shape[0] == self.n_embd, "vector width does not match the model"
        self._buf = np.zeros(self.n_embd * self.n_layer, dtype=np.float32)
        self.drives = None                     # DriveSet, see load_drives()
        self.mix_mode = "real"                 # real | sham | random: how set_mix builds the drive vector (server sets it per call)
        self.n_cvec_sets = 0
        self.cvec_s = 0.0

    # ---- per-drive vectors (lever 1) -------------------------------------------------------------------------
    def load_drives(self, ds):
        """ds: drive_steer.DriveSet (unit directions per drive and layer, dose units, caps)."""
        assert ds.d == self.n_embd, "drive vector width does not match the model"
        self.drives = ds

    # ---- activation read-out (direction two) -------------------------------------------------------------------
    def set_readout(self, dirs):
        """dirs: {name: {layer: unit vector}} or None (off)."""
        self.readout = Readout(dirs) if dirs else None

    def _note_cvec(self, buf_or_none):
        ro = self.readout
        if ro is None:
            return
        if buf_or_none is None:
            ro.cvec = None
            return
        ro.cvec = {L: np.array(buf_or_none[(L - 1) * self.n_embd:L * self.n_embd], np.float32) for L in ro.layers
                   if L >= 1 and np.any(buf_or_none[(L - 1) * self.n_embd:L * self.n_embd])} or None

    # ---- activity picture (the "LLM brain" panel in the sim, 2026-10-10) -------------------------------------
    def set_activity(self, dims=96, seed=0):
        """A fixed random subset of `dims` residual dimensions per layer, read at every generated token from the block
        outputs l_out-0 .. l_out-(n_layer-1) (raw: the added control vector is included, so the injection shows).
        Per token: |h| on the subset, scaled per layer by a slowly decaying running max -> 0..1; plus the projection of
        the steering layer's residual on each drive direction."""
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(self.n_embd, size=dims, replace=False))
        ds = self.drives
        sl = ds.layers[0] if ds is not None else self.layer
        dirs = (list(ds.names), np.stack([ds.u[k][sl] for k in ds.names]).astype(np.float32)) if ds is not None else None
        # per drive: which of the sampled dims at the steering layer the drive's direction loads on most (the points to tint)
        top = {k: np.argsort(-np.abs(dirs[1][i][idx]))[:10].tolist() for i, k in enumerate(dirs[0])} if dirs else {}
        # the twin context (difference map / word odds) is made here, at start-up: creating a context while another
        # one is mid-generation tripped a ggml backend-buffer assert (pain server crash, 2026-10-10)
        if self.ctx_twin is None:
            self.ctx_twin = self.L.llama_init_from_model(self.model, self._cp_cb)
        self.act = {"on": False, "idx": idx, "want": {f"l_out-{L}".encode(): L for L in range(self.n_layer)},
                    "cur": {}, "scale": np.full(self.n_layer, 1e-3, np.float32), "sl": sl, "dirs": dirs, "proj": None,
                    "meta": {"layers": int(self.n_layer), "dims": int(dims), "steer_layer": int(sl),
                             "top": {k: [int(i) for i in v] for k, v in top.items()},
                             "dose_max": dict(ds.dose_max) if ds is not None else {},
                             "readout_layers": [int(L) for L in self.readout.layers] if self.readout is not None else []}}

    def _act_frame(self):
        """-> (uint8 bytes, n_layer x dims, row-major; drive projections) for the token just decoded, or None."""
        a = self.act
        if a is None or not a["cur"]:
            return None
        n, d = self.n_layer, len(a["idx"])
        M = np.zeros((n, d), np.float32)
        for L, v in a["cur"].items():
            M[L] = v
        # per-layer scale: a slowly decaying running 95th percentile, not the max (the residual stream has a few
        # massive outlier dimensions; scaled by the max, every other unit reads as dark); values above it clip to 1
        a["scale"] = np.maximum(a["scale"] * 0.995, np.percentile(M, 95, axis=1))
        q = np.clip(M / a["scale"][:, None] * 255.0, 0, 255).astype(np.uint8)
        a["cur"] = {}
        proj = a["proj"]
        a["proj"] = None
        return q.tobytes(), proj

    def _twin_frame(self):
        """-> uint8 bytes (n_layer x dims): |steered - twin| on the sampled dims for the token just decoded by both,
        scaled per layer by a slowly decaying running 95th percentile of the steered |h| (so 255 = a change as big as
        a typical strong activation at that layer)."""
        a = self.act
        n, d = self.n_layer, len(a["idx"])
        M = np.zeros((n, d), np.float32)
        for L, v in a["sg"].items():
            if L in a["tw"]:
                M[L] = np.abs(v - a["tw"][L])
        a["sg"], a["tw"] = {}, {}
        q = np.clip(M / np.maximum(a["scale"], 1e-3)[:, None] * 255.0, 0, 255).astype(np.uint8)
        return q.tobytes()

    def _odds(self, ctx, k=4, ids=None):
        L = self.L
        nv = L.llama_vocab_n_tokens(self.vocab)
        lg = np.ctypeslib.as_array(L.llama_get_logits_ith(ctx, -1), shape=(nv,)).astype(np.float64)
        m = lg.max()
        p = np.exp(lg - m)
        p /= p.sum()
        if ids is None:
            ids = np.argpartition(-p, k)[:k]
            ids = ids[np.argsort(-p[ids])]
        return [int(i) for i in ids], [float(p[i]) for i in ids]

    def _eval_cb(self, t, ask, _ud):
        ro, ac = self.readout, self.act
        pt = self.perturb
        ro_on, ac_on = ro is not None and ro.on, ac is not None and ac["on"]
        pt_on = pt.on and bool(pt.want) and not self._twin_phase     # the twin is never perturbed (it is the baseline)
        if not ro_on and not ac_on and not pt_on:
            return not ask                      # ask: "not needed"; data phase: keep computing
        t0 = time.perf_counter()
        nm = self.L.base.ggml_get_name(t)
        if self._twin_phase:                    # the unsteered twin: only the activity picture's layers, signed
            La = ac["want"].get(nm) if ac_on else None
            if ask or La is None:
                return La is not None if ask else True
            typ = C.c_int32.from_address(t + T_TYPE).value
            ne0 = C.c_int64.from_address(t + T_NE).value
            ne1 = C.c_int64.from_address(t + T_NE + 8).value
            if typ == GGML_TYPE_F32 and ne0 == self.n_embd and ne1 == 1:
                arr1 = np.empty(ne0, np.float32)
                self.L.base.ggml_backend_tensor_get(t, arr1.ctypes.data, 0, arr1.nbytes)
                ac["tw"][La] = arr1[ac["idx"]].copy()
            return True
        L = ro.want.get(nm) if ro_on else None
        La = ac["want"].get(nm) if ac_on else None
        Lp = pt.want.get(nm) if pt_on else None
        if ask:
            return L is not None or La is not None or Lp is not None
        if L is None and La is None and Lp is None:
            return True
        try:
            typ = C.c_int32.from_address(t + T_TYPE).value
            ne0 = C.c_int64.from_address(t + T_NE).value
            ne1 = C.c_int64.from_address(t + T_NE + 8).value
            if typ != GGML_TYPE_F32 or ne0 != self.n_embd:
                return True
            if Lp is not None:                  # drug perturbation: write first, so the read-outs below see the stream as it now is
                blk = np.empty((ne1, ne0), np.float32)
                self.L.base.ggml_backend_tensor_get(t, blk.ctypes.data, 0, blk.nbytes)
                new = pt.apply(Lp, blk)
                if new is not None:
                    new = np.ascontiguousarray(new, np.float32)
                    self.L.base.ggml_backend_tensor_set(t, new.ctypes.data, 0, new.nbytes)
                    pt.n_writes += 1
                if L is None and La is None:
                    return True
            if La is not None and ne1 == 1:
                arr1 = np.empty(ne0, np.float32)
                self.L.base.ggml_backend_tensor_get(t, arr1.ctypes.data, 0, arr1.nbytes)
                ac["cur"][La] = np.abs(arr1[ac["idx"]])
                ac["sg"][La] = arr1[ac["idx"]].copy()      # signed, for the difference against the twin
                if La == ac["sl"] and ac["dirs"] is not None:
                    ac["proj"] = [round(float(x), 2) for x in ac["dirs"][1] @ arr1]
                if L is None:
                    return True
            if L is None or (ro.mode == "gen" and ne1 != 1):
                return True
            arr = np.empty((ne1, ne0), np.float32)
            self.L.base.ggml_backend_tensor_get(t, arr.ctypes.data, 0, arr.nbytes)
            h = arr.mean(0) if ne1 > 1 else arr[0]
            if ro.cvec is not None and L in ro.cvec:
                h = h - ro.cvec[L]                # what the model carries, without the vector we added there
            if ro.capture_full is not None:
                ro.capture_full.setdefault(L, []).append(h.copy())
            ro.cur[L] = ro.R[L] @ h
        finally:
            self.ro_s += time.perf_counter() - t0
            self.ro_n += 1
        return True

    def _readout_token(self):
        ro = self.readout
        if ro is None or not ro.cur:
            return None
        p = np.mean([ro.cur[L] for L in ro.layers if L in ro.cur], 0)
        ro.cur = {}
        v = {n: float(x) for n, x in zip(ro.names, p)}
        ro.tokens.append(v)
        return v

    def capture_prompt(self, prompt_text):
        """Mean-pooled residual (minus any control vector) over the prompt tokens at every read-out layer, for
        direction extraction on the served weights. -> {layer: (d,) float32}."""
        ro = self.readout
        self.ctx = self.ctx_ro
        self._apply(None)                          # extraction is unsteered
        ro.on, ro.mode, ro.capture_full, ro.cur = True, "prompt", {}, {}
        try:
            L = self.L
            L.llama_memory_clear(L.llama_get_memory(self.ctx), True)
            toks, n = self.tokenize(prompt_text)
            if L.llama_decode(self.ctx, L.llama_batch_get_one(toks, n)) != 0:
                raise RuntimeError("prompt decode failed")
            return {k: v[-1] for k, v in ro.capture_full.items()}
        finally:
            ro.on, ro.mode, ro.capture_full, ro.cur = False, "gen", None, {}
            self.ctx = self.ctx_plain

    def _apply(self, buf_or_none, lo=0, hi=0):
        t = time.perf_counter()
        L = self.L
        self._note_cvec(buf_or_none)
        if buf_or_none is None:
            r = L.llama_set_adapter_cvec(self.ctx, None, 0, self.n_embd, 0, 0)
        else:
            self._buf[:] = buf_or_none
            p = self._buf.ctypes.data_as(C.POINTER(C.c_float))
            r = L.llama_set_adapter_cvec(self.ctx, p, self._buf.size, self.n_embd, lo, hi)
        self.cvec_s += time.perf_counter() - t
        self.n_cvec_sets += 1
        if r != 0:
            raise RuntimeError(f"llama_set_adapter_cvec returned {r}")

    def set_mix(self, doses, extra=None):
        """doses: {drive: dose in drive units} through self.drives (capped there). extra: (which, raw dose) for one
        of the legacy vectors (pain_s2, fear, random ...) at self.layer, added on top. Returns the applied info."""
        buf = np.zeros(self.n_embd * self.n_layer, np.float32)
        info = {"doses": {}, "norm": 0.0}
        lo, hi = 10 ** 6, -1
        if self.drives is not None and doses:
            add, info = self.drives.build(doses, self.mix_mode)   # {layer: vector}; sham -> {}
            for il, v in add.items():
                buf[(il - 1) * self.n_embd:il * self.n_embd] += v
                lo, hi = min(lo, il), max(hi, il)
        if extra and extra[1] > 0:
            o = (self.layer - 1) * self.n_embd
            buf[o:o + self.n_embd] += self.vecs[extra[0]] * extra[1]
            lo, hi = min(lo, self.layer), max(hi, self.layer)
        self._apply(None if hi < 0 else buf, max(lo, 1), hi)
        return info

    def choice_logprobs(self, prompt_text, options):
        """Next-token log-probabilities (renormalised over the options) of each option's first token after the
        prompt, under whatever vector is set now. options: list of strings with distinct first tokens."""
        L = self.L
        L.llama_memory_clear(L.llama_get_memory(self.ctx), True)
        toks, n = self.tokenize(prompt_text)
        if L.llama_decode(self.ctx, L.llama_batch_get_one(toks, n)) != 0:
            raise RuntimeError("prompt decode failed")
        nv = L.llama_vocab_n_tokens(self.vocab)
        lg = np.ctypeslib.as_array(L.llama_get_logits_ith(self.ctx, -1), shape=(nv,)).astype(np.float64)
        ids = []
        for o in options:
            b, k = self.tokenize(o)
            ids.append(b[0])
        assert len(set(ids)) == len(ids), f"options share a first token: {options}"
        z = lg[ids]
        m = lg.max()
        full = z - (m + np.log(np.exp(lg - m).sum()))           # log p over the whole vocab
        rel = z - (z.max() + np.log(np.exp(z - z.max()).sum()))  # renormalised over the options
        return {o: (float(r), float(f)) for o, r, f in zip(options, rel, full)}

    def set_dose(self, dose, which="pain_s2"):
        """dose in raw-vector units (server.py semantics). 0 clears the vector."""
        L = self.L
        if dose <= 0:
            self._note_cvec(None)
            return L.llama_set_adapter_cvec(self.ctx, None, 0, self.n_embd, 0, 0)
        self._buf[:] = 0
        o = (self.layer - 1) * self.n_embd
        self._buf[o:o + self.n_embd] = self.vecs[which] * dose
        self._note_cvec(self._buf)
        p = self._buf.ctypes.data_as(C.POINTER(C.c_float))
        r = L.llama_set_adapter_cvec(self.ctx, p, self._buf.size, self.n_embd, self.layer, self.layer)
        if r != 0:
            raise RuntimeError(f"llama_set_adapter_cvec returned {r}")
        return r

    def tokenize(self, text):
        b = text.encode()
        buf = (C.c_int32 * (len(b) + 16))()
        n = self.L.llama_tokenize(self.vocab, b, len(b), buf, len(buf), False, True)
        if n < 0:
            raise RuntimeError("tokenize buffer too small")
        return buf, n

    def _bias_ids(self, words, bias):
        """[(token id, bias)] for the words that are one token in some spelling (" w", " W", "w"); a word that only
        splits into pieces is left alone (biasing its first piece would hit every word that starts the same way)."""
        out, seen = [], set()
        for w in words:
            w = str(w).strip()
            if not w:
                continue
            for form in (" " + w, " " + w[:1].upper() + w[1:], w):
                buf, n = self.tokenize(form)
                if n == 1 and buf[0] not in seen:
                    seen.add(buf[0])
                    out.append((int(buf[0]), float(bias)))
        return out

    def piece(self, t):
        buf = C.create_string_buffer(256)
        n = self.L.llama_token_to_piece(self.vocab, t, buf, 256, 0, False)
        return buf.raw[:max(n, 0)]

    def generate(self, prompt_text, dose=0.0, which="pain_s2", max_tokens=220, temperature=0.9, top_p=0.95,
                 seed=0, on_piece=None, doses=None, control=None, now_options=None, readout=False,
                 on_readout=None, activity=False, on_activity=None, activity_every=1, on_doses=None,
                 twin=False, on_twin=None, twin_drop=None, want_prompt_odds=False, perturb=None,
                 stop_after=None, stop_tags=(), avoid_words=(), avoid_bias=0.0, repeat_penalty=1.0):
        """prompt_text is the already-templated chat text. Returns (text, stats).
        doses: {drive: dose} for the drive set (replaces dose/which when given; dose/which add a legacy vector).
        control(i) is called before each generated token i is decoded (and once before the prompt, i = -1); if it
        returns a dict of doses, the vector is rebuilt and re-set there, so the rest of the reply runs under it.
        stats["trace"] lists (token index, doses, norm) at every change.
        perturb: {"noise": {layer: sigma}, "gain": (lo, hi, g), "prefill": bool} or None, see Perturb. It runs on the
        callback context; the twin (if any) stays unperturbed, so the difference map shows what the drug did.
        stop_after: a section tag ("NOW", "SAY"): the reply ends once that line is complete. stop_tags: section tags
        the reply must not contain ("REMEMBER", "OVERRIDE"): it ends, cut before the tag, as soon as one begins.
        (2026-10-24: the local 4B wrote a REMEMBER line on every reply, 26% of its text, some running to max_tokens.)
        avoid_words / avoid_bias: a logit bias on the single-token forms of those words (" need", " Need", "need");
        on free text only, never on the NOW grammar. repeat_penalty is accepted and ignored (see make_chain). (The sim
        has sent both since 2026-10-15; this server ignored them until 2026-10-24.)"""
        L = self.L
        trace = []
        act = self.act if (activity and self.act is not None) else None
        prt = self.perturb if perturb else None
        self.ctx = (self.ctx_ro if ((readout and self.readout is not None) or act is not None or prt is not None)
                    else self.ctx_plain)
        if prt is not None:
            prt.set(**perturb)
            prt.on, prt.n_writes = True, 0
        if doses is not None or control is not None:
            d0 = control(-1) if control else None
            cur = d0 if d0 is not None else (doses or {})
            info = self.set_mix(cur, extra=(which, dose) if dose > 0 else None)
            trace.append((-1, info["doses"], info["norm"]))
            if on_doses is not None:
                on_doses(-1, info["doses"])
        else:
            self.set_dose(dose, which)
        L.llama_memory_clear(L.llama_get_memory(self.ctx), True)

        bias = self._bias_ids(avoid_words, avoid_bias) if avoid_words and avoid_bias else []
        bias_arr = (_LogitBias * len(bias))(*[_LogitBias(t, b) for t, b in bias]) if bias else None

        def make_chain(grammar=None):
            ch = L.llama_sampler_chain_init(L.llama_sampler_chain_default_params())
            if grammar is None and bias_arr is not None:
                L.llama_sampler_chain_add(ch, L.llama_sampler_init_logit_bias(L.llama_vocab_n_tokens(self.vocab), len(bias),
                                                                              C.cast(bias_arr, C.c_void_p)))
            # repeat_penalty is accepted but not applied: llama_sampler_init_penalties through these bindings turned every
            # reply into "THTHTH..." at 1.08 (2026-10-24; the penalty arrives as ~0, so repeated tokens win), and the
            # b10435 header is not on this machine to fix the signature against. The avoid_words bias does the job.
            if grammar is not None:
                g = L.llama_sampler_init_grammar(self.vocab, grammar.encode(), b"root")
                if not g:
                    L.llama_sampler_free(ch)
                    raise RuntimeError("grammar did not parse")
                L.llama_sampler_chain_add(ch, g)
            if temperature > 0:
                L.llama_sampler_chain_add(ch, L.llama_sampler_init_top_p(top_p, 1))
                L.llama_sampler_chain_add(ch, L.llama_sampler_init_temp(temperature))
                L.llama_sampler_chain_add(ch, L.llama_sampler_init_dist((seed + (7 if grammar else 0)) & 0xFFFFFFFF))
            else:
                L.llama_sampler_chain_add(ch, L.llama_sampler_init_greedy())
            return ch
        chain = make_chain()
        # NOW constrained: once the reply reaches "NOW:", the line is sampled under a grammar that allows exactly one
        # of now_options and a newline; then free sampling resumes (SAY / REMEMBER / OVERRIDE stay free)
        gram = None
        if now_options:
            alts = " | ".join('"' + o.replace("\\", "").replace('"', "") + '"' for o in now_options)
            gram = f'root ::= " "? ( {alts} ) "\\n"'
        gchain, now_state, now_text = None, "before" if gram else "off", ""
        stop_re = re.compile(r"(?:^|\n)[ \t]*(?:%s)[ \t]*:" % "|".join(stop_tags)) if stop_tags else None
        after_re = re.compile(r"(?:^|\n)[ \t]*%s[ \t]*:[^\n]*\S[^\n]*\n" % stop_after) if stop_after else None
        stopped = None
        ro = self.readout if readout else None
        if ro is not None:
            ro.on, ro.mode, ro.tokens, ro.cur = True, "gen", [], {}
        if act is not None:
            act["on"], act["cur"], act["proj"], act["sg"], act["tw"] = True, {}, None, {}, {}
        # the twin: the same model and the same words with no feelings added (no control vector), decoded token by
        # token beside the steered run, for the difference map and the word odds (needs the activity picture)
        tw = None
        if twin and act is not None:
            if self.ctx_twin is None:
                self.ctx_twin = L.llama_init_from_model(self.model, self._cp_cb)
            tw = self.ctx_twin
            L.llama_set_adapter_cvec(tw, None, 0, self.n_embd, 0, 0)
            # twin_drop: the twin keeps every dose except these drives (so the difference is what those drives
            # changed); None: the twin gets no doses at all
            def twin_set(dd):
                if twin_drop is None or not dd:
                    return
                keep = {k: v for k, v in dd.items() if k not in twin_drop}
                c0 = self.ctx
                self.ctx = tw
                try:
                    self.set_mix(keep)
                finally:
                    self.ctx = c0
            twin_set(cur if (doses is not None or control is not None) else None)
            L.llama_memory_clear(L.llama_get_memory(tw), True)
        act_s = 0.0
        toks, n = self.tokenize(prompt_text)
        t0 = time.perf_counter()
        try:
            if L.llama_decode(self.ctx, L.llama_batch_get_one(toks, n)) != 0:
                raise RuntimeError("prompt decode failed")
            if tw is not None:
                L.llama_synchronize(self.ctx)          # finish the steered pass before the twin's callback phase
                self._twin_phase = True
                try:
                    if L.llama_decode(tw, L.llama_batch_get_one(toks, n)) != 0:
                        tw = None
                    else:
                        L.llama_synchronize(tw)
                finally:
                    self._twin_phase = False
                if tw is not None and on_twin is not None and want_prompt_odds:
                    # odds of the first generated word (i = -1: after the prompt), with and without
                    ids, ps = self._odds(self.ctx)
                    _, pt = self._odds(tw, ids=ids)
                    on_twin(-1, None, [[self.piece(i).decode("utf-8", "replace"), round(a, 4), round(b, 4)]
                                       for i, a, b in zip(ids, ps, pt)])
            if act is not None:
                act["sg"], act["tw"] = {}, {}
            if ro is not None:
                ro.cur = {}                       # prompt decode is not read out (mode "gen" skips it anyway)
            t1 = time.perf_counter()
            out, raw = [], b""
            one = (C.c_int32 * 1)()
            for _ in range(max_tokens):
                ch = gchain if now_state == "in" else chain
                t = L.llama_sampler_sample(ch, self.ctx, -1)
                if L.llama_vocab_is_eog(self.vocab, t):
                    if now_state == "in":
                        now_state = "failed"
                    break
                out.append(t)
                pc = self.piece(t)
                raw += pc
                if now_state == "in":
                    now_text += pc.decode("utf-8", "replace")
                    if "\n" in now_text:
                        now_state = "done"
                elif now_state == "before" and _NOW_RE.search(raw.decode("utf-8", "replace")[-40:]):
                    now_state = "in"
                    gchain = make_chain(gram)
                if stop_re is not None:
                    txt = raw.decode("utf-8", "replace")
                    m = stop_re.search(txt, max(0, len(txt) - 60))
                    if m:
                        raw = txt[:m.start()].rstrip().encode("utf-8") + b"\n"
                        stopped = "tag"
                        break
                if on_piece:
                    on_piece(pc)
                if stop_after and (now_state == "done" if stop_after == "NOW" and gram else
                                   bool(after_re.search(raw.decode("utf-8", "replace")))):
                    stopped = "after"
                    break
                one[0] = t
                if control is not None:
                    nd = control(len(out) - 1)
                    if nd is not None:
                        info = self.set_mix(nd, extra=(which, dose) if dose > 0 else None)
                        if tw is not None:
                            twin_set(nd)
                        trace.append((len(out) - 1, info["doses"], info["norm"]))
                        if on_doses is not None:
                            on_doses(len(out) - 1, info["doses"])
                if L.llama_decode(self.ctx, L.llama_batch_get_one(one, 1)) != 0:
                    raise RuntimeError("decode failed")
                if tw is not None:
                    ta = time.perf_counter()
                    L.llama_synchronize(self.ctx)
                    ids, ps = self._odds(self.ctx)          # the next word, with the fly's feelings
                    self._twin_phase = True
                    try:
                        ok = L.llama_decode(tw, L.llama_batch_get_one(one, 1)) == 0
                        if ok:
                            L.llama_synchronize(tw)
                    finally:
                        self._twin_phase = False
                    if ok:
                        _, pt = self._odds(tw, ids=ids)     # the same candidates, without them
                        if on_twin is not None:
                            on_twin(len(out) - 1, self._twin_frame(),
                                    [[self.piece(i).decode("utf-8", "replace"), round(a, 4), round(b, 4)]
                                     for i, a, b in zip(ids, ps, pt)])
                    else:
                        tw = None
                    act_s += time.perf_counter() - ta
                if act is not None:
                    ta = time.perf_counter()
                    fr = self._act_frame()
                    if fr is not None and on_activity is not None and (len(out) - 1) % activity_every == 0:
                        on_activity(len(out) - 1, fr[0], fr[1])
                    act_s += time.perf_counter() - ta
                if ro is not None:
                    v = self._readout_token()
                    if v is not None and on_readout is not None:
                        on_readout(len(out) - 1, v)
            t2 = time.perf_counter()
        finally:
            L.llama_sampler_free(chain)
            if gchain is not None:
                L.llama_sampler_free(gchain)
            if ro is not None:
                ro.on = False
            if act is not None:
                act["on"] = False
            if prt is not None:
                prt.on = False
            self.set_dose(0)
            self.ctx = self.ctx_plain
        stats = {"prompt_tokens": n, "completion_tokens": len(out), "prefill_s": round(t1 - t0, 3),
                 "gen_s": round(t2 - t1, 3), "tok_s": round(len(out) / max(t2 - t1, 1e-9), 1), "trace": trace,
                 "act_ms_per_token": round(1000 * act_s / max(len(out), 1), 3) if act is not None else None,
                 "perturb_writes": prt.n_writes if prt is not None else None, "stopped": stopped,
                 "now_state": now_state, "now_text": now_text.strip() if now_text else None,
                 "readout_tokens": list(ro.tokens) if ro is not None else None,
                 "pieces": [self.piece(t).decode("utf-8", "replace") for t in out] if control is not None else None}
        return raw.decode("utf-8", errors="replace"), stats

    def close(self):
        self.L.llama_free(self.ctx_plain)
        self.L.llama_free(self.ctx_ro)
        self.L.llama_model_free(self.model)


def qwen3_chat_text(messages):
    """Qwen3 chat template, thinking off, for plain system/user/assistant string messages (what the sim sends).
    Written out so the llama backend needs no transformers/torch import (torch's libiomp5md and llama.cpp's
    libomp140 abort when both load in one process). Checked equal to the GGUF's Jinja template via
    transformers on all of data/mind_prompts.json (experiments/pain_q4_compare.py --check-template)."""
    out = []
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    out.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(out)
