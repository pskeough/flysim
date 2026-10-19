"""Direction two: read the steered model's residual stream and send it to the fly (2026-10-08, first version).

While the mind generates, llama_steer.Readout copies the block outputs l_out-11 .. l_out-18 at every generated token
(minus the control vector added there) and projects them on a few directions. Channels:

  freeze    unit(u_threat - u_arousal), deshared drive directions (lever 1 extraction): threat over arousal = hold
            still; its negative (arousal over threat) = move.
  avoid     the shared "something is wrong with my body" axis of the drive extraction (mean of the unit raw drive
            directions; drive_extract.py "shared"): avoid / back off.
  approach  mean(approach sentences) - mean(withdraw sentences) (APPROACH / WITHDRAW below, ours), extracted here on
            the served Q4_K_M weights through the same read-out (mean over the sentence tokens), projected off the
            shared axis so it does not double the avoid channel.

Each channel is z-scored per token against unsteered replies to the sim's own mind prompts (data/mind_prompts.json,
calibrate). The sim gets per-token z and the running reply mean, and maps them to descending-neuron biases
(flysim/cognition.py ReadoutBias). The calibration also stores per-reply z vectors: the pool the shuffled control
draws from.

    set PYTHONPATH=pain_server\\pyshim
    .venv_pain\\Scripts\\python.exe pain_server\\readout.py --device cuda:0 extract
    .venv_pain\\Scripts\\python.exe pain_server\\readout.py --device cuda:0 calibrate
Writes vectors/<NAME>_readout.npz (directions per layer, check AUCs in _readout_report.json) and
vectors/<NAME>_readout_calib.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pain_server.model import GGUF, NAME  # noqa: E402

LAYERS = tuple(range(11, 19))
CHANNELS = ("approach", "avoid", "freeze")

# five themed sets of six, fly register (as drive_data.py); approach = going toward, seeking; withdraw = pulling
# back, shrinking away. The same nouns on both sides, so the contrast is the direction of movement and wanting.
APPROACH = [
    ["I walk toward the sweet smell.", "I go closer to the drop ahead.", "I head straight for the bright patch.",
     "I step nearer to the thing I want.", "I make my way over to the damp spot.", "I move up to the edge of the drop."],
    ["I want to reach it and I keep going.", "I push on until I get there.", "Nothing will stop me getting to it.",
     "I am set on reaching the far side.", "I keep after it, step by step.", "I go on toward it without stopping."],
    ["I lean in to taste it.", "I reach my front legs out to touch it.", "I put my mouth down onto it.",
     "I press my feet into the wet.", "I stretch toward the smell.", "I bring my antennae right up to it."],
    ["I follow the warm trail.", "I track the scent across the floor.", "I chase the moving speck.",
     "I seek out the place I remember.", "I search for the way through.", "I hunt along the wall for the gap."],
    ["I am drawn to it.", "It pulls me in.", "I am curious and go to look.", "I go to see what it is.",
     "I come closer to find out.", "I want more of it and go to it."],
]
WITHDRAW = [
    ["I walk away from the sweet smell.", "I back off from the drop ahead.", "I turn from the bright patch.",
     "I step back from the thing in front.", "I move away from the damp spot.", "I pull back from the edge of the drop."],
    ["I want to be far from it and I keep going.", "I hurry off until it is behind me.", "Nothing will make me go near it.",
     "I am set on getting away from the far side.", "I keep clear of it, step by step.", "I go on away from it without stopping."],
    ["I lean back so I do not touch it.", "I draw my front legs in away from it.", "I lift my mouth off it.",
     "I pull my feet out of the wet.", "I shrink from the smell.", "I fold my antennae back away from it."],
    ["I leave the warm trail.", "I turn off the scent and go the other way.", "I flee from the moving speck.",
     "I keep away from the place I remember.", "I give up on the way through and turn back.", "I retreat along the wall."],
    ["I am pushed away from it.", "It drives me off.", "I am wary and stay back.", "I do not go to see what it is.",
     "I keep my distance to be safe.", "I have had enough of it and leave it."],
]


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def drive_dirs():
    z = np.load(HERE / "vectors" / f"{NAME}_drives.npz")
    names = [str(x) for x in z["drives"]]
    M = z["deshared"].astype(np.float32)
    S = z["shared"].astype(np.float32)
    ut = {L: unit(M[names.index("threat"), L]) for L in LAYERS}
    ua = {L: unit(M[names.index("arousal"), L]) for L in LAYERS}
    return {"freeze": {L: unit(ut[L] - ua[L]) for L in LAYERS}, "avoid": {L: unit(S[L]) for L in LAYERS},
            "_threat": ut, "_arousal": ua}


def load_dirs():
    """{channel: {layer: unit vector}} for the server, or None when not extracted yet."""
    p = HERE / "vectors" / f"{NAME}_readout.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {c: {L: z[c][i].astype(np.float32) for i, L in enumerate(z["layers"])} for c in CHANNELS if c in z}


def load_calib():
    p = HERE / "vectors" / f"{NAME}_readout_calib.json"
    return json.loads(p.read_text()) if p.exists() else None


def _model(device):
    from pain_server.llama_steer import LlamaSteer
    gpu = int(str(device).split(":")[-1]) if ":" in str(device) else 0
    return LlamaSteer(GGUF, HERE / "vectors" / f"{NAME}.npz", gpu=gpu)


def extract(device):
    from pain_server.drive_data import NEUTRAL, S as DS, SUFFIX
    t0 = time.time()
    m = _model(device)
    d = drive_dirs()
    probe = {"freeze": d["freeze"], "avoid": d["avoid"], "threat": d["_threat"]}
    # any directions will do for capture: capture_full keeps the whole (corrected) residual per layer
    m.set_readout(probe)

    def acts(texts):
        out = {L: [] for L in LAYERS}
        for tx in texts:
            h = m.capture_prompt(tx + SUFFIX)
            for L in LAYERS:
                out[L].append(h[L])
        return {L: np.stack(v) for L, v in out.items()}
    ap = [s for st in APPROACH for s in st]
    wd = [s for st in WITHDRAW for s in st]
    set_ap = np.repeat(np.arange(len(APPROACH)), 6)
    A, W = acts(ap), acts(wd)
    thr = acts([s for st in DS["threat"] for s in st])
    aro = acts([s for st in DS["arousal"] for s in st])
    neu = acts([s for st in NEUTRAL for s in st])
    pain = acts([s for st in DS["pain"] for s in st])
    rep = {"layers": list(LAYERS), "checks": {}}
    appr = {}
    for L in LAYERS:
        raw = A[L].mean(0) - W[L].mean(0)
        s = d["avoid"][L]
        appr[L] = unit(raw - (raw @ s) * s)
        # held-out by set: direction from 4 sets, AUC on the fifth
        aucs = []
        for f in range(len(APPROACH)):
            tr, te = set_ap != f, set_ap == f
            r = A[L][tr].mean(0) - W[L][tr].mean(0)
            r = r - (r @ s) * s
            aucs.append(auc(A[L][te] @ r, W[L][te] @ r))
        rep["checks"][L] = {
            "approach_heldout_auc": round(float(np.mean(aucs)), 3),
            # path check: the lever 1 directions read back through llama.cpp must separate their own sentences
            "threat_vs_neutral_auc": round(auc(thr[L] @ d["_threat"][L], neu[L] @ d["_threat"][L]), 3),
            "freeze_threat_vs_arousal_auc": round(auc(thr[L] @ d["freeze"][L], aro[L] @ d["freeze"][L]), 3),
            "avoid_pain_vs_neutral_auc": round(auc(pain[L] @ d["avoid"][L], neu[L] @ d["avoid"][L]), 3),
            "cos_approach_freeze": round(float(appr[L] @ d["freeze"][L]), 3),
        }
        print(L, rep["checks"][L], flush=True)
    arr = lambda dd: np.stack([dd[L] for L in LAYERS]).astype(np.float32)
    np.savez(HERE / "vectors" / f"{NAME}_readout.npz", layers=np.array(LAYERS), approach=arr(appr),
             avoid=arr(d["avoid"]), freeze=arr(d["freeze"]))
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["readout_cb_ms_per_call"] = round(1000 * m.ro_s / max(1, m.ro_n), 4)
    (HERE / "vectors" / f"{NAME}_readout_report.json").write_text(json.dumps(rep, indent=1))
    m.close()


def calibrate(device, seeds=6, max_tokens=140):
    """Unsteered replies to the sim's mind prompts: per-channel mean / sd over tokens (the z basis), per-reply
    mean z (the shuffled control's pool), and the read-out's cost in tok/s."""
    from pain_server.llama_steer import qwen3_chat_text
    t0 = time.time()
    m = _model(device)
    m.set_readout(load_dirs())
    P = json.loads((HERE / "data" / "mind_prompts.json").read_text(encoding="utf-8"))
    per_tok, replies, rates = [], [], {"on": [], "off": []}
    for s in range(seeds):
        for i, p in enumerate(P):
            txt = qwen3_chat_text(p["messages"])
            _, st = m.generate(txt, max_tokens=max_tokens, temperature=0.9, seed=1000 * s + i, readout=True)
            rates["on"].append(st["tok_s"])
            toks = st["readout_tokens"] or []
            per_tok += toks
            replies.append({"situation": p.get("situation"), "mean": {c: float(np.mean([t[c] for t in toks]))
                                                                       for c in CHANNELS} if toks else None})
            if s == 0 and i < 4:
                _, st2 = m.generate(txt, max_tokens=max_tokens, temperature=0.9, seed=1000 * s + i, readout=False)
                rates["off"].append(st2["tok_s"])
    mu = {c: float(np.mean([t[c] for t in per_tok])) for c in CHANNELS}
    sd = {c: float(np.std([t[c] for t in per_tok]) + 1e-6) for c in CHANNELS}
    pool = [{c: round((r["mean"][c] - mu[c]) / sd[c], 3) for c in CHANNELS} for r in replies if r["mean"]]
    out = {"name": NAME, "layers": list(LAYERS), "mu": mu, "sd": sd, "n_tokens": len(per_tok), "n_replies": len(pool),
           "pool": pool, "situations": [r["situation"] for r in replies if r["mean"]],
           "tok_s_readout_on": round(float(np.median(rates["on"])), 1),
           "tok_s_readout_off": round(float(np.median(rates["off"])), 1) if rates["off"] else None,
           "reply_z_sd": {c: round(float(np.std([p[c] for p in pool])), 3) for c in CHANNELS},
           "calibrated": time.strftime("%Y-%m-%d %H:%M"), "wall_s": round(time.time() - t0, 1)}
    (HERE / "vectors" / f"{NAME}_readout_calib.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("pool", "situations")}, indent=1))
    m.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("cmd", choices=("extract", "calibrate"))
    a = ap.parse_args()
    (extract if a.cmd == "extract" else calibrate)(a.device)
