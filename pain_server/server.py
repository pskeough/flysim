"""OpenAI-compatible chat server that adds the pain axis to the residual stream, scaled per request
(research/PAIN_DESIGN.md 2.1, Option A).

    .venv_mind/Scripts/python.exe pain_server/server.py [--device cuda:0|cpu] [--port 8095] [--backend llama|hf]

Backends: llama (default) = see below; hf = transformers, the GGUF de-quantised to bf16 (~8 GB), a forward hook; llama = the GGUF run
as-is by llama.dll (pain_server/llama_steer.py), the same vector set at the same layer through llama.cpp's control
vector, re-set on every call (~3 GB). The llama backend logs no pain projection (pain_projection null).

Then in the sim: Mind settings -> URL http://127.0.0.1:8095/v1, model qwen3-4b; Pain tab -> steer on.

Request body: the usual chat fields, plus
  drives        lever 1: the sim's drive levels (embodiment.steer_drives); with drive_steer (default true) they
                replace the pain dose. steer_mode real | sham (doses computed and logged, nothing added) | random
                (fixed random unit directions, same doses and cap); drive_mask [drives allowed to steer] (default all).
                Both also apply to the live re-sets from /v1/drives (experiments/behav controls).
  pain          0..1, the sim's pain intensity I (nociception.py). Absent or 0 = unsteered.
  steer_vector  pain_s2 (default) | pain_s1 | fear | sadness | random   (norm-matched controls)
Dose = pain x dose_max, in units of the raw pain vector at the steering layer (dose 1 adds the mean difference
between pain and control sentences). dose_max comes from calibrate.py and is capped at DOSE_CAP whatever is asked:
there is no maximum-dose mode. The prompt is re-encoded on every call (the sim keeps no chat history), so one
dose applies to the whole call and no unsteered prefix is ever reused under a different dose.
Every call is logged to pain_server/logs/calls.jsonl with the dose and the pre-steer projection of the last
prompt token onto the pain direction (probe 5: does the sensor-line text alone engage the axis?).
"""
from __future__ import annotations

import argparse
import os
import asyncio
import base64
import json
import sys
import threading
import time
from pathlib import Path

import codecs
import queue

import numpy as np
from aiohttp import web

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pain_server.model import GGUF, MODEL_ID, NAME, chat_ids, layers, load  # noqa: E402

# torch is imported only by the hf backend (Steer, main): torch's libiomp5md and llama.cpp's libomp140 abort
# when both load in one process.

LIVE_THRESH = 0.05             # a live drive level must move this much (0..1) before the vector is re-set
LIVE_STALE_S = 1.0             # live posts older than this are ignored
LIVE_MIN_TOKENS = 4            # at most one re-set per 4 tokens: each re-set costs ~45 ms of graph rebuild in
                               # llama.cpp (experiments/cvec_overhead.py: 87 tok/s fixed, 17 re-set every token)
DOSE_CAP = 8.0                # absolute ceiling; the working cap is dose_max = 0.75 x the measured break point


class Steer:
    def __init__(self, model, device, name=NAME):
        path = HERE / "vectors" / f"{name}.npz"
        if not path.exists():
            have = sorted(p.stem for p in (HERE / "vectors").glob("*.npz"))
            raise SystemExit(f"no steering vectors for {GGUF.name} ({path.name}); have {have}. Run extract.py and "
                             f"calibrate.py with PAIN_GGUF set to these weights, or set PAIN_VECTORS to one of those.")
        v = np.load(path)
        self.name = name
        self.layer = int(v["layer"])
        import torch
        self.vecs = {k: torch.tensor(v[k], dtype=torch.bfloat16, device=device)
                     for k in ("pain_s2", "pain_s1", "fear", "sadness", "random")}
        self.unit = (self.vecs["pain_s2"].float() / self.vecs["pain_s2"].float().norm())
        cal = HERE / "vectors" / f"{name}_calibration.json"
        self.dose_max = min(DOSE_CAP, json.loads(cal.read_text())["dose_max"]) if cal.exists() else 1.0
        self.calibrated = cal.exists()
        self.add = None                       # tensor added at every position, or None
        self.proj = None                      # pre-steer projection of the last prompt token (first pass only)
        self._first = False
        layers(model)[self.layer].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if self._first:
            self._first = False
            self.proj = float(h[0, -1].float() @ self.unit)
        if self.add is None:
            return out
        h = h + self.add
        return (h, *out[1:]) if isinstance(out, tuple) else h

    def set(self, pain, which):
        dose = float(np.clip(pain, 0.0, 1.0)) * self.dose_max
        self.add = None if dose <= 0 else self.vecs.get(which, self.vecs["pain_s2"]) * dose
        self._first = True
        return round(dose, 4)


class _Cancelled(Exception):
    """The client hung up: generation stops at the next token."""


class QueueStreamer:
    """Iterator of text pieces fed from the generating thread (stands in for TextIteratorStreamer)."""
    def __init__(self):
        self.q, self.dec = queue.Queue(), codecs.getincrementaldecoder("utf-8")("replace")
        self.readout = None          # the latest per-token read-out (set from the generating thread)
        self.act = None              # the latest activity frame (body['activity']), sent once then cleared
        self.doses = None            # the doses in force now {"i": token, "d": {drive: dose}}, sent once per change
        self.twin = None             # {"i", "b": |steered - unsteered| frame (base64), "odds": [[word, p, p_without]]}

    def put_bytes(self, b):
        t = self.dec.decode(b)
        if t:
            self.q.put(t)

    def end(self):
        self.q.put(None)

    def __iter__(self):
        return iter(self.q.get, None)


class LlamaSteerAdapter:
    """Same surface as Steer (set / layer / dose_max / calibrated / proj / add) over llama_steer.LlamaSteer."""
    def __init__(self, device, name=NAME):
        from pain_server.llama_steer import LlamaSteer
        path = HERE / "vectors" / f"{name}.npz"
        if not path.exists():
            raise SystemExit(f"no steering vectors {path.name}; run extract.py / calibrate.py for {GGUF.name}")
        gpu = None if str(device) == "cpu" else int(str(device).split(":")[-1]) if ":" in str(device) else 0
        self.m = LlamaSteer(GGUF, path, gpu=gpu, n_threads=int(os.environ.get("PAIN_THREADS", "4")))
        self.name, self.layer, self.proj, self.add = name, self.m.layer, None, None
        cal = HERE / "vectors" / f"{name}_calibration.json"
        self.dose_max = min(DOSE_CAP, json.loads(cal.read_text())["dose_max"]) if cal.exists() else 1.0
        self.calibrated = cal.exists()
        self.dose, self.which = 0.0, "pain_s2"
        self.ds = None
        try:
            from pain_server.drive_steer import DriveSet
            self.ds = DriveSet(name)
            if not self.ds.dose_max:
                print(f"drive vectors for {name} are not calibrated (run drive_calibrate.py); drives ignored", flush=True)
                self.ds = None
            else:
                self.m.load_drives(self.ds)
        except FileNotFoundError as e:
            print(f"no drive vectors ({e}); body['drives'] ignored, pain only", flush=True)
        self.doses = None                      # {drive: dose} for this call, or None = legacy pain path
        self.mask = None                       # drives allowed to steer this call (None = all)
        self.live = {"t": 0.0, "levels": {}, "n": 0}
        # direction two (readout.py): residual read-out during generation, when the request asks for it
        from pain_server.readout import load_calib, load_dirs
        dirs, self.ro_cal = load_dirs(), load_calib()
        if dirs and self.ro_cal:
            self.m.set_readout(dirs)
            print(f"read-out channels {sorted(dirs)} on layers {self.ro_cal['layers']}", flush=True)
        else:
            self.ro_cal = None
            print("no read-out directions / calibration (readout.py extract, calibrate): body['readout'] ignored",
                  flush=True)
        # the activity picture for the sim's LLM-brain panel (body['activity']): 96 fixed dims per layer, every 3rd token
        self.m.set_activity(dims=96, seed=0)

    def set(self, pain, which):
        self.dose = float(np.clip(pain, 0.0, 1.0)) * self.dose_max
        self.which = which if which in self.m.vecs else "pain_s2"
        return round(self.dose, 4)

    def set_drives(self, drives, mask=None, mode="real"):
        """body["drives"] -> per-drive doses (replaces the legacy pain dose for this call). Returns the doses.
        mask: drives allowed to steer (None = all); mode: real | sham | random (drive_steer.DriveSet.build)."""
        from pain_server.drive_steer import from_sim
        self.dose = 0.0
        self.mask, self.m.mix_mode = mask, mode
        self.doses = self.ds.doses_for(from_sim(drives), mask)
        return self.doses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--backend", choices=("hf", "llama"), default="llama")   # hf needs torch + transformers (not installed by install.bat)
    ap.add_argument("--log", default=None, help="call log (default pain_server/logs/calls.jsonl)")
    a = ap.parse_args()
    if a.backend == "llama":
        tok = None                                      # no transformers/torch in this process (OpenMP clash)
        st = LlamaSteerAdapter(a.device)
    else:
        import torch
        from transformers import TextIteratorStreamer
        tok, model = load(a.device)
        st = Steer(model, a.device)
    lock = threading.Lock()
    logf = Path(a.log) if a.log else HERE / "logs" / "calls.jsonl"
    logf.parent.mkdir(parents=True, exist_ok=True)
    print(f"{MODEL_ID} ({GGUF.name}, vectors {NAME}, backend {a.backend}) on {a.device}, steering layer {st.layer}, dose_max {st.dose_max} "
          f"({'calibrated' if st.calibrated else 'NOT calibrated: run calibrate.py'})", flush=True)

    def live_control(body):
        """Per-token control for a live call: re-read the drives the sim last POSTed to /v1/drives and re-set the
        vector when any level moved by more than LIVE_THRESH since the last set (checked every token; stale
        posts, older than LIVE_STALE_S, are ignored and the call keeps its current doses)."""
        from pain_server.drive_steer import from_sim
        cur = {"levels": from_sim(body.get("drives") or {}), "i": -LIVE_MIN_TOKENS}

        def control(i):
            lv = st.live
            if time.time() - lv["t"] > LIVE_STALE_S or i - cur["i"] < LIVE_MIN_TOKENS:
                return None
            new = lv["levels"]
            keys = set(new) | set(cur["levels"])
            if max((abs(new.get(k, 0.0) - cur["levels"].get(k, 0.0)) for k in keys), default=0.0) <= LIVE_THRESH:
                return None
            cur["levels"], cur["i"] = dict(new), i
            return st.ds.doses_for(new, getattr(st, "mask", None))
        return control

    def generate_llama(body, streamer):
        from pain_server.llama_steer import qwen3_chat_text
        text = qwen3_chat_text(body.get("messages") or [])
        temp = float(body.get("temperature", 0.7))
        ctl = live_control(body) if (st.doses is not None and body.get("live")) else None
        cal = getattr(st, "ro_cal", None)
        want_ro = bool(body.get("readout")) and cal is not None
        ro = {"sum": {}, "n": 0, "run": [], "last": None}

        def on_readout(i, v):
            # per token: z against unsteered replies; running reply mean in units of the reply-level sd
            ro["n"] += 1
            for c, x in v.items():
                ro["sum"][c] = ro["sum"].get(c, 0.0) + (x - cal["mu"][c]) / cal["sd"][c]
            r = {c: round(ro["sum"][c] / ro["n"] / max(cal["reply_z_sd"][c], 1e-3), 3) for c in ro["sum"]}
            ro["run"].append(r)
            ro["last"] = {"i": i, "r": r}
            streamer.readout = ro["last"]
        opts = body.get("now_options")
        opts = [str(o) for o in opts][:40] if isinstance(opts, list) and opts else None
        want_act = bool(body.get("activity")) and getattr(st.m, "act", None) is not None
        sent_meta = {"v": False}

        def on_activity(i, q, proj):
            fr = {"i": i, "b": base64.b64encode(q).decode(), "p": proj}
            if not sent_meta["v"]:
                fr["meta"] = st.m.act["meta"] | {"drives": st.m.act["dirs"][0] if st.m.act["dirs"] else []}
                sent_meta["v"] = True
            streamer.act = fr
        cancel = body.get("_cancel")

        def on_piece(pc):
            if cancel is not None and cancel.is_set():
                raise _Cancelled()          # the client hung up (an interrupted thought): stop at this token
            streamer.put_bytes(pc)
        try:
            _, s = st.m.generate(text, dose=st.dose, which=st.which, max_tokens=int(body.get("max_tokens", 400)),
                                 temperature=temp, top_p=float(body.get("top_p", 0.95)),
                                 seed=int(body.get("seed", time.time_ns() & 0x7FFFFFFF)), on_piece=on_piece,
                                 doses=st.doses, control=ctl, now_options=opts, readout=want_ro,
                                 on_readout=on_readout if want_ro else None, activity=want_act,
                                 on_activity=on_activity if want_act else None,
                                 on_doses=lambda i, d: setattr(streamer, "doses", {"i": i, "d": d}),
                                 twin=bool(body.get("twin")) and want_act,
                                 on_twin=lambda i, q, odds: setattr(streamer, "twin", {
                                     "i": i, "b": base64.b64encode(q).decode(), "odds": odds}))
        finally:
            streamer.end()
        s["readout"] = ({"mean_r": ro["run"][-1] if ro["run"] else None, "n": ro["n"], "run": ro["run"],
                         "channels": sorted(ro["sum"])} if want_ro else None)
        s.pop("readout_tokens", None)
        st.last_stats = s
        return s["prompt_tokens"], s["completion_tokens"]

    def generate(body, streamer):
        if a.backend == "llama":
            return generate_llama(body, streamer)
        msgs = body.get("messages") or []
        ids = chat_ids(tok, msgs, a.device)
        temp = float(body.get("temperature", 0.7))
        kw = dict(max_new_tokens=int(body.get("max_tokens", 400)), do_sample=temp > 0, streamer=streamer)
        if temp > 0:
            kw.update(temperature=temp, top_p=float(body.get("top_p", 0.95)))
        with torch.no_grad():
            out = model.generate(ids, **kw)
        return ids.shape[1], out.shape[1] - ids.shape[1]

    async def chat(req):
        body = await req.json()
        pain = float(body.get("pain") or 0.0)
        which = str(body.get("steer_vector") or "pain_s2")
        stream = bool(body.get("stream"))
        loop = asyncio.get_running_loop()
        t0 = time.time()
        await loop.run_in_executor(None, lock.acquire)
        try:
            dose = st.set(pain, which)
            st.last_stats = None
            if a.backend == "llama":
                st.doses = None
                st.mask, st.m.mix_mode = None, "real"
                if st.ds is not None and body.get("drives") is not None and body.get("drive_steer", True):
                    from pain_server.drive_steer import STEER_MODES
                    mode = str(body.get("steer_mode") or "real")
                    mask = body.get("drive_mask")
                    st.set_drives(body["drives"], mask=set(mask) if isinstance(mask, list) else None,
                                  mode=mode if mode in STEER_MODES else "real")
                    st.live = {"t": 0.0, "levels": {}, "n": 0}     # only posts made during this call count
                    dose = 0.0
            streamer = (QueueStreamer() if a.backend == "llama" else
                        TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True))
            res = {}
            body["_cancel"] = cancel = threading.Event()

            def run():
                try:
                    res.update(n=generate(body, streamer))
                except _Cancelled:
                    res.update(n=(0, 0), cancelled=True)
            th = threading.Thread(target=run, daemon=True)
            th.start()
            parts = []
            if stream:
                resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
                await resp.prepare(req)
            it = iter(streamer)
            while True:
                piece = await loop.run_in_executor(None, lambda: next(it, None))
                if piece is None:
                    break
                parts.append(piece)
                if stream and piece:
                    ch = {"choices": [{"index": 0, "delta": {"content": piece}}]}
                    rv = getattr(streamer, "readout", None)
                    if rv is not None:
                        ch["readout"] = rv
                    av = getattr(streamer, "act", None)
                    if av is not None:
                        ch["act"], streamer.act = av, None
                    tv = getattr(streamer, "twin", None)
                    if tv is not None:              # the twin's difference frame and word odds for this token
                        ch["twin"], streamer.twin = tv, None
                    dv = getattr(streamer, "doses", None)
                    if dv is not None:              # the live doses, the moment they are set (the sim's brain panel)
                        ch["doses"], streamer.doses = dv, None
                    await resp.write(b"data: " + json.dumps(ch).encode() + b"\n\n")
            await loop.run_in_executor(None, th.join)
            n_in, n_out = res.get("n", (0, 0))
            meta = {"model": MODEL_ID, "vectors": NAME, "backend": a.backend, "dose": dose, "pain": pain, "steer_vector": which if dose > 0 else None,
                    "layer": st.layer, "pain_projection": st.proj, "calibrated": st.calibrated}
            ls0 = getattr(st, "last_stats", None) or {}
            if ls0.get("readout") is not None:
                meta["readout"] = ls0["readout"]
            if ls0.get("now_state") not in (None, "off"):
                meta["now_state"], meta["now_text"] = ls0["now_state"], ls0.get("now_text")
            if ls0.get("act_ms_per_token") is not None:
                meta["act_ms_per_token"] = ls0["act_ms_per_token"]
            if getattr(st, "doses", None) is not None:
                ls = st.last_stats or {}
                tr = ls.get("trace") or []
                meta.update({"drive_doses": st.doses, "drive_layers": st.ds.layers, "drive_variant": st.ds.variant,
                             "steer_mode": st.m.mix_mode, "drive_mask": sorted(st.mask) if st.mask is not None else None,
                             "seed": body.get("seed"), "tok_s": ls.get("tok_s"),
                             "live": bool(body.get("live")), "trace": tr, "n_updates": max(0, len(tr) - 1),
                             "cvec_ms_mean": round(1000 * st.m.cvec_s / max(st.m.n_cvec_sets, 1), 3)})
                if tr:
                    # per-token summed-vector norm (dose units): the curve to draw under the text
                    per, j, n_out = [], 0, ls.get("completion_tokens", 0)
                    for i in range(n_out):
                        while j + 1 < len(tr) and tr[j + 1][0] < i:
                            j += 1
                        per.append(tr[j][2])
                    meta["norm_per_token"] = per
                    meta["pieces"] = ls.get("pieces")
            usage = {"prompt_tokens": n_in, "completion_tokens": n_out, "cost": 0.0}
            with open(logf, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"t": time.time(), **meta, "latency_s": round(time.time() - t0, 2), "usage": usage,
                                    "prompt": (body.get("messages") or [{}])[-1].get("content", "")[-1500:],
                                    "reply": "".join(parts)}) + "\n")
        finally:
            # a client that hangs up mid-stream (an interrupted thought) lands here with the generator still running:
            # stop it and wait, so the next call never runs beside it on the same context
            c_ = body.get("_cancel")
            th_ = locals().get("th")
            if c_ is not None and th_ is not None and th_.is_alive():
                c_.set()
                th_.join(timeout=float(os.environ.get("PAIN_CANCEL_JOIN_S", "30")))   # CPU: a long prefill outlasts 30 s; never release the lock under a live generator
            st.add = None
            lock.release()
        if stream:
            await resp.write(b"data: " + json.dumps({"choices": [], "usage": usage, "steer": meta}).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            return resp
        return web.json_response({"choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                                               "finish_reason": "stop"}], "usage": usage, "steer": meta})

    async def drives_post(req):
        """Live channel (lever 1): the sim POSTs {"drives": {...}} at ~15 Hz while a thought streams."""
        j = await req.json()
        if getattr(st, "ds", None) is not None:
            from pain_server.drive_steer import from_sim
            st.live = {"t": time.time(), "levels": from_sim(j.get("drives") or {}), "n": st.live["n"] + 1}
        return web.json_response({"ok": True})

    async def models(_):
        return web.json_response({"data": [{"id": MODEL_ID, "object": "model"}]})

    app = web.Application(client_max_size=4 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/v1/caps", lambda req: web.json_response({"abort": True}))
    app.router.add_post("/v1/drives", drives_post)
    web.run_app(app, host="127.0.0.1", port=a.port, print=lambda *_: print(f"listening on :{a.port}", flush=True))


if __name__ == "__main__":
    main()
