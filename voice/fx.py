"""Post-processing for the fly's voice, shared by the sample scripts and fly_voice_server.py so the app sounds
like the samples. A voice's "post" in samples.json is a list of steps applied in order, e.g.
  [{"fx": "formant", "shift": 1.18, "pitch": 1.06}, {"fx": "buzz", "level": 0.10, "flutter": 19}]
Every output is peak-limited to -1 dBFS (0.891).
  formant  rubberband: formants up by `shift` (a smaller body, less human), pitch times `pitch`, same tempo
  pitchup  plain resample pitch-up by `semitones` (formants move with it), same tempo
  buzz     a 190 Hz sawtooth wing-buzz under the voice, gated by the voice envelope; `flutter` Hz amplitude
           modulation makes it flutter like wingbeats instead of a steady drone
  ring     ring modulation at `f` Hz mixed in at `mix`: a dry, metallic, chitinous edge
"""
from __future__ import annotations

import io, subprocess, wave
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
FFMPEG = SIM / ".venv" / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries" / "ffmpeg-win-x86_64-v7.1.exe"
PEAK = 0.891   # -1 dBFS


# The state cues (appended to the voice design prompt), shared by round-2 samples and the app.
# Measured on 09 low: these plus punctuation() lift pain loudness ~35% and pitch ~2x over the round-1 cues.
CUES = {
    "calm": "",
    "slow": " Speaking slowly and drowsily, words slightly slurred and heavy.",
    "pain": " Now SCREAMING at the top of his lungs in agony, shrieking, panicked, loud and frantic, gasping between the words.",
    "dying": " Desperate, terrified and sobbing, crying out and begging, voice cracking, then breaking down into a weak, hoarse, dying whisper.",
}


def punctuate(text: str, tier: str) -> str:
    """Same words, delivery punctuation: pain gets exclamations, dying breaks into trailing fragments."""
    import re
    t = text.strip()
    if tier == "pain":
        t = re.sub(r"[.,;]+(\s|$)", r"!\1", t).rstrip()
        return t if t.endswith(("!", "?")) else t + "!"
    if tier == "dying":
        t = re.sub(r"[.,;]+\s+", "... ", t)
        return re.sub(r"[.!]?$", "...", t, count=1)
    return t


def wav_to_np(b: bytes):
    with wave.open(io.BytesIO(b)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, np.int16 if sw == 2 else np.int32).astype(np.float32) / (32768.0 if sw == 2 else 2147483648.0)
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x, sr


def np_to_wav(x, sr) -> bytes:
    o = io.BytesIO()
    with wave.open(o, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return o.getvalue()


def ffmpeg_np(x, sr, af: str):
    p = subprocess.run([str(FFMPEG), "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                        "-af", af, "-ar", str(sr), "-ac", "1", "-f", "f32le", "pipe:1"],
                       input=x.astype(np.float32).tobytes(), capture_output=True, timeout=60)
    if p.returncode:
        raise RuntimeError("ffmpeg: " + p.stderr.decode(errors="replace")[-300:])
    return np.frombuffer(p.stdout, np.float32).copy()


def envelope(x, sr, ms=30):
    fl = max(1, int(sr * ms / 1000))
    e = np.convolve(np.abs(x), np.ones(fl) / fl, mode="same")
    return e / (e.max() + 1e-9)


def buzz(x, sr, level=0.06, flutter=0.0, f0=190.0, seed=0):
    t = np.arange(len(x)) / sr
    rng = np.random.default_rng(seed)
    jitter = np.convolve(rng.standard_normal(len(x)), np.ones(int(sr * 0.05)) / int(sr * 0.05), mode="same")
    f = f0 + 6 * np.sin(2 * np.pi * 5.5 * t) + 25 * jitter
    ph = 2 * np.pi * np.cumsum(f) / sr
    saw = sum(np.sin(k * ph) / k for k in range(1, 9))
    if flutter:   # wingbeat flutter: 60% amplitude modulation, rate wandering +-15%
        fr = flutter * (1 + 0.15 * np.sin(2 * np.pi * 0.7 * t))
        saw = saw * (0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(fr) / sr)))
    return x + level * saw * envelope(x, sr) * np.abs(x).max()


def ring(x, sr, f=210.0, mix=0.12):
    t = np.arange(len(x)) / sr
    return (1 - mix) * x + mix * 1.6 * x * np.sin(2 * np.pi * f * t)


def apply(x, sr, steps):
    for s in steps or []:
        fx = s["fx"]
        if fx == "formant":
            k, p = float(s.get("shift", 1.18)), float(s.get("pitch", 1.0))
            # resample up by k (formants and pitch up, faster), then rubberband the pitch back to p with the
            # formants held, then restore the tempo
            x = ffmpeg_np(x, sr, f"asetrate={sr}*{k:.4f},aresample={sr},rubberband=pitch={p / k:.4f}:formant=preserved,atempo={1 / k:.4f}")
        elif fx == "pitchup":
            r = 2 ** (float(s.get("semitones", 3)) / 12)
            x = ffmpeg_np(x, sr, f"asetrate={sr}*{r:.4f},aresample={sr},atempo={1 / r:.4f}")
        elif fx == "buzz":
            x = buzz(x, sr, float(s.get("level", 0.06)), float(s.get("flutter", 0)))
        elif fx == "ring":
            x = ring(x, sr, float(s.get("f", 210)), float(s.get("mix", 0.12)))
        else:
            raise ValueError("unknown fx " + fx)
    return limit(x)


# loudness per tier (RMS over the clip): the model renders some lines very quietly, so each clip is brought to a
# level; pain sits louder than calm. Gain is capped at 10x; peaks go through a lookahead limiter at -1 dBFS.
TARGET_RMS = {"calm": 0.07, "slow": 0.06, "pain": 0.11, "dying": 0.08}


def normalize(x, sr, tier="calm"):
    rms = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
    if rms < 1e-5:
        return x
    y = x * min(10.0, TARGET_RMS.get(tier, 0.07) / rms)
    if np.abs(y).max() > PEAK:
        y = ffmpeg_np(y, sr, f"alimiter=limit={PEAK}:level=0:attack=3:release=60:latency=1")
    return limit(y)


def limit(x):
    pk = float(np.abs(x).max()) if len(x) else 0.0
    return x * (PEAK / pk) if pk > PEAK else x


def apply_wav(b: bytes, steps) -> bytes:
    x, sr = wav_to_np(b)
    return np_to_wav(apply(x, sr, steps), sr)
