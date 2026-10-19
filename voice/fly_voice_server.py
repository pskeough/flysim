"""The fly's voice: a small localhost service the web UI calls when Voice is on.

  POST /speak  {"text", "state": {pain, sedation, intoxication, health, dead, last}, "voice"?}
               -> audio/wav (24 kHz mono). The state picks the delivery (see delivery()).
  POST /stt    body = a recorded clip (webm/ogg/wav, as the browser's MediaRecorder makes it)
               -> {"text": "..."} via faster-whisper (the Voice_Chat setup: small.en)
  GET  /health -> {"ok", "engine", "voice", "engines": {name: {up, state, voices}}, "stt", "ui_default_on"}
  GET  /voices -> the voice designs in voice/designs/*.json
  POST /config {"engine"?: "kokoro"|"qwen", "voice"?} -> switch engine / voice (the UI's Settings); saved to
               voice_config.json so a restart keeps it

Two TTS engines:
  kokoro  Kokoro-82M on the CPU (2 threads), in this process. The default: no GPU, loads in ~30 s, about 1.3x
          faster than real time. Voices: the ones cached in the Hugging Face cache (offline only; install.bat
          --voice downloads the model and the default voice am_puck).
  qwen    Optional, not installed by install.bat. Qwen3-TTS VoiceDesign through audio.cpp's audiocpp_server, started
          here only when chosen, on its own port, on the GPU named by FLY_VOICE_GPU (default 0, PCI bus order);
          stopped again when you switch away (it grows to ~9 GB VRAM over a long session). FLY_AUDIOCPP_DIR is the
          folder holding audiocpp_server.exe (default tts/audio.cpp), FLY_VDES_MODEL the VoiceDesign GGUF (default
          tts/qwen3tts/qwen3-tts-voicedesign-q8_0.gguf). The voice is a design prompt picked by name from
          voice/designs/*.json.

Run it with the voice venv from the repo root (install.bat --voice makes it):
  .venv_voice\\Scripts\\python.exe voice\\fly_voice_server.py
or let run.bat start it. It never binds :8765 and stops only the engine process it started.
"""
from __future__ import annotations

import atexit, io, json, os, subprocess, sys, threading, time, urllib.request, wave
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fx

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
CFG = json.loads((HERE / "voice_config.json").read_text(encoding="utf-8"))
SAMPLES = [HERE / "designs" / "round1_samples.json",            # round 1 (12 voices)
           HERE / "designs" / "round2_samples.json"]            # round 2 (from 09 low)
AUDIOCPP_DIR = Path(os.environ.get("FLY_AUDIOCPP_DIR") or SIM / "tts" / "audio.cpp")
VDES_MODEL = Path(os.environ.get("FLY_VDES_MODEL") or SIM / "tts" / "qwen3tts" / "qwen3-tts-voicedesign-q8_0.gguf")
FFMPEG = fx.FFMPEG
GPU = os.environ.get("FLY_VOICE_GPU", "0")
GPU_ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": GPU}   # the one card the voice may use
os.environ.update(GPU_ENV)
if CFG.get("port") == 8765 or CFG.get("engine_port") == 8765:
    sys.exit("refusing :8765")
ROUND1_POST = {"pitchup": [{"fx": "pitchup", "semitones": 3}], "buzz": [{"fx": "buzz", "level": 0.06}]}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def voices():
    out = {}
    for p in SAMPLES:
        try:
            out.update(json.loads(p.read_text(encoding="utf-8"))["voices"])
        except Exception as e:
            log(f"{p} unreadable:", e)
    return out


def design(name):
    """-> (design prompt, post steps) for a voice in samples.json."""
    v = voices().get(name)
    if not v:
        raise KeyError(f"unknown voice '{name}'; have {sorted(voices())}")
    post = v.get("post")
    return v["design_prompt"], ROUND1_POST.get(post, []) if isinstance(post, str) or post is None else post


# ------------------------------------------------------------------ delivery from the body's state
def delivery(st: dict):
    """-> (tier, ffmpeg tempo plan). Drive levels are the UI's (status.cog.embody.drives, 0..1).
    The tier picks the cue appended to the design prompt (fx.CUES) and the punctuation (fx.punctuate)."""
    pain = float(st.get("pain") or 0)
    sed = max(float(st.get("sedation") or 0), float(st.get("intoxication") or 0))
    health = st.get("health")
    dying = bool(st.get("dead") or st.get("last") or (health is not None and float(health) < 0.15))
    if dying:     # desperate, then each quarter slower than the last and a fade to nothing
        return "dying", {"ramp": [1.0, 0.92, 0.8, 0.66], "fade": 0.4}
    if pain >= 0.35:   # screaming, a little rushed
        return "pain", {"tempo": 1.0 + 0.08 * min(1.0, pain)}
    if sed >= 0.25:    # drunk or sedated: slower
        return "slow", {"tempo": max(0.68, 1.0 - 0.4 * min(1.0, sed))}
    return "calm", {}


def ffmpeg_filter(plan, dur_s):
    """An ffmpeg -filter_complex string for the plan (pitch-preserving atempo)."""
    head = f"atempo={plan['tempo']:.3f}" if plan.get("tempo") else "anull"
    if "ramp" in plan:
        r = plan["ramp"]; n = len(r); seg = dur_s / n
        parts = [f"[0:a]{head},asplit={n}" + "".join(f"[s{i}]" for i in range(n))]
        for i, t in enumerate(r):
            parts.append(f"[s{i}]atrim={i * seg:.3f}:{(i + 1) * seg:.3f},asetpts=PTS-STARTPTS,atempo={t:.3f}[t{i}]")
        parts.append("".join(f"[t{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[out]")
        return ";".join(parts)
    return f"[0:a]{head}[out]"


def wav_dur(b: bytes) -> float:
    with wave.open(io.BytesIO(b)) as w:
        return w.getnframes() / w.getframerate()


def run_ffmpeg(inp: bytes, fc: str) -> bytes:
    p = subprocess.run([str(FFMPEG), "-loglevel", "error", "-i", "pipe:0", "-filter_complex", fc, "-map", "[out]",
                        "-ac", "1", "-ar", "24000", "-f", "s16le", "pipe:1"], input=inp, capture_output=True, timeout=60)
    if p.returncode or not p.stdout:
        raise RuntimeError("ffmpeg: " + p.stderr.decode(errors="replace")[-400:])
    o = io.BytesIO()   # a real header (ffmpeg's piped wav has an unknown length, which some players mis-time)
    with wave.open(o, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(p.stdout)
    return o.getvalue()


# ------------------------------------------------------------------ the TTS engine (audiocpp_server)
class Engine:
    def __init__(self):
        self.port = int(CFG["engine_port"]); self.proc = None; self.lock = threading.Lock(); self.err = ""
        self.cache: OrderedDict[tuple, bytes] = OrderedDict()

    def start(self):
        conf = {"host": "127.0.0.1", "port": self.port, "backend": CFG.get("engine_backend", "cuda"), "device": 0,
                "threads": int(CFG.get("engine_threads", 3)),
                "models": [{"id": "fly-vdes", "family": "qwen3_tts", "path": str(VDES_MODEL), "task": "vdes", "mode": "offline"}]}
        cp = HERE / "_engine_config.json"
        cp.write_text(json.dumps(conf, indent=1), encoding="utf-8")
        logf = open(HERE / "_engine.log", "w", encoding="utf-8")
        self.proc = subprocess.Popen([str(AUDIOCPP_DIR / "audiocpp_server.exe"), "--config", str(cp), "--no-ui"],
                                     cwd=str(AUDIOCPP_DIR), env={**os.environ, **GPU_ENV}, stdout=logf, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        atexit.register(self.stop)
        log(f"engine PID {self.proc.pid} on :{self.port} ({conf['backend']}, CUDA_VISIBLE_DEVICES={GPU})")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            log(f"stopping engine PID {self.proc.pid}")
            self.proc.terminate()
            try: self.proc.wait(10)
            except Exception: self.proc.kill()

    def up(self):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def synth(self, text: str, instruct: str, seed: int) -> bytes:
        key = (text, instruct, seed)
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        body = {"model": "fly-vdes", "input": text, "instruct": instruct, "language": "english",
                "response_format": "wav", "seed": seed}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/audio/speech", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with self.lock:
            with urllib.request.urlopen(req, timeout=180) as r:
                b = r.read()
        if b[:4] != b"RIFF":
            raise RuntimeError("engine returned non-wav: " + b[:300].decode(errors="replace"))
        self.cache[key] = b
        while len(self.cache) > 64:
            self.cache.popitem(last=False)
        return b


# ------------------------------------------------------------------ speech to text (Voice_Chat's faster-whisper)
class STT:
    def __init__(self):
        self.model = None; self.device = None; self.err = ""; self.lock = threading.Lock()

    def load(self):
        with _IMPORT_LOCK:
            self._load()

    def _load(self):
        try:
            from faster_whisper import WhisperModel
            dev = CFG.get("stt_device", "auto")
            if dev == "auto":
                dev = "cpu"
                try:
                    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=10).stdout
                    free = {int(a): int(b) for a, b in (l.split(",") for l in q.strip().splitlines())}
                    if free.get(int(GPU), 0) >= int(CFG.get("stt_min_free_mb", 1500)):
                        dev = "cuda"
                except Exception:
                    pass
            t0 = time.time()
            if dev == "cuda":   # device_index 0 of the visible set == physical GPU FLY_VOICE_GPU
                self.model = WhisperModel(CFG.get("stt_model", "small.en"), device="cuda", device_index=0, compute_type="float16")
            else:
                self.model = WhisperModel(CFG.get("stt_model", "small.en"), device="cpu", compute_type="int8", cpu_threads=2)
            self.device = dev
            log(f"whisper {CFG.get('stt_model', 'small.en')} on {dev} in {time.time() - t0:.1f}s")
        except Exception as e:
            self.err = str(e); log("whisper failed:", e)

    def transcribe(self, blob: bytes) -> str:
        if not self.model:
            raise RuntimeError("speech-to-text not loaded: " + self.err)
        p = subprocess.run([str(FFMPEG), "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                           input=blob, capture_output=True, timeout=60)
        if p.returncode:
            raise RuntimeError("ffmpeg decode: " + p.stderr.decode(errors="replace")[-300:])
        a = np.frombuffer(p.stdout, np.float32)
        if a.size < 1600:
            return ""
        with self.lock:
            segs, _ = self.model.transcribe(a, language="en", beam_size=1, condition_on_previous_text=False, vad_filter=True)
            return " ".join(s.text.strip() for s in segs).strip()


# ------------------------------------------------------------------ Kokoro-82M on the CPU (the default engine)
KOKORO_REPO = "hexgrad/Kokoro-82M"
_IMPORT_LOCK = threading.Lock()


def kokoro_voices():
    """The Kokoro voices already in the Hugging Face cache (no downloads: this runs offline)."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    out = set()
    for d in hub.glob("models--hexgrad--Kokoro-82M/snapshots/*/voices"):
        out.update(p.stem for p in d.glob("*.pt"))
    return sorted(out)


class Kokoro:
    def __init__(self):
        self.pipe = None; self.err = ""; self.lock = threading.Lock(); self.loading = False
        self.cache: OrderedDict[tuple, bytes] = OrderedDict()

    def load(self):
        # one load at a time, and never beside faster-whisper's import: transformers' lazy import is not thread-safe
        # ("cannot import name 'AlbertModel'" when both loaded at once)
        with _IMPORT_LOCK:
            self._load()

    def _load(self):
        if self.pipe is not None:
            return
        self.loading = True; self.err = ""
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            import torch
            torch.set_num_threads(int(CFG.get("kokoro_threads", 2)))
            from kokoro import KPipeline
            t0 = time.time()
            self.pipe = KPipeline(lang_code="a", repo_id=KOKORO_REPO, device="cpu")
            log(f"kokoro on cpu in {time.time() - t0:.1f}s; voices {kokoro_voices()}")
        except Exception as e:
            self.err = str(e); log("kokoro failed:", repr(e))
        finally:
            self.loading = False

    def up(self):
        return self.pipe is not None

    def state(self):
        return "ready" if self.pipe else ("loading" if self.loading else ("error: " + self.err if self.err else "off"))

    def synth(self, text: str, voice: str, speed: float = 1.0) -> bytes:
        if not self.pipe:
            raise RuntimeError("kokoro not loaded: " + (self.err or "still loading"))
        key = (text, voice, round(speed, 3))
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        with self.lock:
            parts = [r.audio.numpy() if hasattr(r.audio, "numpy") else np.asarray(r.audio)
                     for r in self.pipe(text, voice=voice, speed=speed) if r.audio is not None]
        if not parts:
            raise RuntimeError("kokoro made no audio")
        b = fx.np_to_wav(np.concatenate(parts).astype(np.float32), 24000)
        self.cache[key] = b
        while len(self.cache) > 64:
            self.cache.popitem(last=False)
        return b


ENGINE, KOKORO, STTM = Engine(), Kokoro(), STT()
ENGINES = ("kokoro", "qwen")
CUR = {"engine": CFG.get("engine", "kokoro") if CFG.get("engine", "kokoro") in ENGINES else "kokoro"}
CUR["voice"] = CFG.get("kokoro_voice", "am_puck") if CUR["engine"] == "kokoro" else CFG["voice"]
_N_CLIP = 0
_SWITCH = threading.Lock()


def qwen_up():
    return ENGINE.proc is not None and ENGINE.proc.poll() is None and ENGINE.up()


def use_engine(name: str, voice: str | None = None):
    """Switch engine (and voice). Kokoro loads in the background; the Qwen engine is started on the GPU when chosen and
    stopped when you leave it."""
    if name not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}")
    with _SWITCH:
        if name == "kokoro":
            threading.Thread(target=KOKORO.load, daemon=True).start()
            ENGINE.stop(); ENGINE.proc = None
            v = voice or CFG.get("kokoro_voice", "am_puck")
            have = kokoro_voices()
            if have and v not in have:
                raise ValueError(f"unknown kokoro voice '{v}'; have {have}")
            CFG["kokoro_voice"] = v
        else:
            v = voice or CFG["voice"]
            design(v)                                   # raises on an unknown design name
            if ENGINE.proc is None or ENGINE.proc.poll() is not None:
                ENGINE.start()
            CFG["voice"] = v
        CUR.update(engine=name, voice=v)
        CFG["engine"] = name
        try:     # keep the choice for the next start (only these keys change)
            disk = json.loads((HERE / "voice_config.json").read_text(encoding="utf-8"))
            disk.update({k: CFG[k] for k in ("engine", "voice", "kokoro_voice") if k in CFG})
            (HERE / "voice_config.json").write_text(json.dumps(disk, indent=2) + "\n", encoding="utf-8")
        except Exception as e:
            log("could not save voice_config.json:", e)
        log(f"engine -> {name}, voice -> {v}")


def speak(text: str, state: dict, voice: str | None = None) -> tuple[bytes, dict]:
    tier, plan = delivery(state or {})
    t0 = time.time()
    if CUR["engine"] == "kokoro":
        name = CUR["voice"]
        raw, post = KOKORO.synth(fx.punctuate(text, tier), name), []
    else:
        name = voice or CUR["voice"]
        prompt, post = design(name)
        if not qwen_up():
            raise RuntimeError("Qwen3-TTS engine still loading on the GPU")
        raw = ENGINE.synth(fx.punctuate(text, tier), prompt + fx.CUES[tier], int(CFG.get("seed", 1234)))
    t1 = time.time()
    x, sr = fx.wav_to_np(raw)                  # the voice's own effect chain, as in its samples, then a level per tier
    out = fx.np_to_wav(fx.normalize(fx.apply(x, sr, post), sr, tier), sr)
    if plan:
        out = run_ffmpeg(out, ffmpeg_filter(plan, wav_dur(out)))
    if plan.get("fade"):                       # dying: full voice, then a cos^2 fall to silence over the last part
        x, sr = fx.wav_to_np(out); n = len(x); k = int(n * plan["fade"])
        g = np.ones(n, np.float32); g[n - k:] = np.cos(np.linspace(0, np.pi / 2, k)) ** 2
        out = fx.np_to_wav(x * g, sr)
    out = fx.apply_wav(out, [])                # peak limit -1 dBFS
    info = {"voice": name, "tier": tier, "tts_s": round(t1 - t0, 2), "post_s": round(time.time() - t1, 2),
            "audio_s": round(wav_dur(out), 2)}
    keep = os.environ.get("FLY_VOICE_KEEP")   # set to a folder to keep every served clip (proof recordings)
    if keep:
        global _N_CLIP
        _N_CLIP += 1; info["clip"] = f"{_N_CLIP:03d}_{tier}.wav"
        Path(keep).mkdir(parents=True, exist_ok=True); (Path(keep) / info["clip"]).write_bytes(out)
    log(f"speak [{tier}] {info} {text[:70]!r}")
    return out, info


class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Expose-Headers", "X-Voice-Info")

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self._cors(); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        if self.path.startswith("/health"):
            q_state = ("ready" if qwen_up() else "loading" if ENGINE.proc is not None and ENGINE.proc.poll() is None
                       else "off")
            engines = {"kokoro": {"label": "Kokoro-82M (CPU)", "up": KOKORO.up(), "state": KOKORO.state(),
                                  "voices": kokoro_voices()},
                       "qwen": {"label": "Qwen3-TTS VoiceDesign (GPU)", "up": q_state == "ready", "state": q_state,
                                "voices": sorted(voices())}}
            return self._json(200, {"ok": engines[CUR["engine"]]["up"], "engine": CUR["engine"], "voice": CUR["voice"],
                                    "engines": engines,
                                    "stt": STTM.device or ("error: " + STTM.err if STTM.err else "loading"),
                                    "ui_default_on": bool(CFG.get("ui_default_on", False))})
        if self.path.startswith("/voices"):
            return self._json(200, {k: v.get("design_prompt") for k, v in voices().items()})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        try:
            if self.path.startswith("/speak"):
                j = json.loads(body or b"{}")
                text = str(j.get("text", "")).strip()[:600]
                if not text:
                    return self._json(400, {"error": "no text"})
                wav, info = speak(text, j.get("state") or {}, j.get("voice"))
                self.send_response(200); self._cors(); self.send_header("Content-Type", "audio/wav")
                self.send_header("X-Voice-Info", json.dumps(info)); self.send_header("Content-Length", str(len(wav)))
                self.end_headers(); self.wfile.write(wav); return
            if self.path.startswith("/config"):
                j = json.loads(body or b"{}")
                use_engine(str(j.get("engine") or CUR["engine"]), j.get("voice") or None)
                return self._json(200, {"ok": True, "engine": CUR["engine"], "voice": CUR["voice"]})
            if self.path.startswith("/event"):   # the page reports what it played and when (voice/_events.jsonl)
                with open(HERE / "_events.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"server_t": time.time(), **json.loads(body or b"{}")}) + "\n")
                return self._json(200, {"ok": True})
            if self.path.startswith("/stt"):
                t0 = time.time(); text = STTM.transcribe(body)
                log(f"stt {len(body)} B -> {text!r} in {time.time() - t0:.2f}s")
                return self._json(200, {"text": text, "s": round(time.time() - t0, 2)})
            self._json(404, {"error": "not found"})
        except Exception as e:
            log("error:", repr(e))
            self._json(500, {"error": str(e)[:400]})

    def log_message(self, *a):
        pass


def main():
    # the server answers at once; the chosen engine loads behind it (/health says "loading" until it is ready)
    if "--no-engine" not in sys.argv:
        use_engine(CUR["engine"], CUR["voice"])
    if "--no-stt" not in sys.argv:
        threading.Thread(target=STTM.load, daemon=True).start()
    (HERE / "_pids.json").write_text(json.dumps({"server": os.getpid(), "engine": ENGINE.proc.pid if ENGINE.proc else None}))
    srv = ThreadingHTTPServer(("127.0.0.1", int(CFG["port"])), H)
    log(f"fly voice on http://127.0.0.1:{CFG['port']}  engine={CUR['engine']} voice={CUR['voice']}")
    try:
        srv.serve_forever()
    finally:
        ENGINE.stop()


if __name__ == "__main__":
    main()
