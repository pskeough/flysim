"""Three levels of cognition over the connectome fly.

  L0  the fly itself: the LIF connectome brain, the sensory layer, and the placeholder reflexes and
      exploration bouts of autonomy.Policy. It lives on its own and never waits for a model.
  L1  the fast motor layer (a small LLM, or the rule baseline): several times a second it reads a
      compact AFFORDANCE list (each sensed thing, where it is, and the catalog action that approaches
      or avoids it) plus the current DIRECTIVE, and picks one catalog intent. The intent is a neural
      stimulation (llm_bridge.CATALOG), applied for slightly longer than one L1 period and renewed
      each tick. "none" hands the body back to L0 ("resume").
  L2  the slow "conscious" layer (a frontier model via OpenRouter, every few seconds): reads what the
      neurons report, the fly's needs and a log of who did what, writes a first-person THOUGHT, a
      DIRECTIVE for L1 and, rarely, a direct OVERRIDE intent.
      Two L2 styles. "mind" (default, MIND_DESIGN.md step 1): woken by salient events (the world changing,
      new sensations, reflexes, needs, blackouts) or after a quiet stretch; streams a sectioned reply
      (THOUGHT / FEELING / NOW / optional OVERRIDE) that the UI shows as it is written. "classic": the
      original JSON reply on a fixed 4 s clock, kept unchanged for comparison and the earlier runs.
      L2 modes:  steer    its directive drives L1
                 narrate  it is told it only narrates; L1 follows the user's goal
                 blind    it is told it steers, but its directive is silently not used; L1 follows the
                          user's goal (the "interpreter" condition: does it claim what it did not cause?)

Why L1 gets affordances and not coordinates: results/small_model_bench_v2_summary.jsonl
(FAST_CONTROLLER.md, Result 1). 1-2B models pick the right action 99-100% of the time when the line
names it, and fail on raw bearings. The directive is free text, so L1's job is to select the line that
serves it; the rule baseline does the same with keyword matching.

  M   memory (memory.py): a concurrent system with its own thread that keeps where things were when the
      fly last perceived them and bids for the motor channel against L1 (the gate in inputs()). With
      memory on, L1 and L2 see only what the fly perceives now (the horizon); the rest they get from memory.

Talk: the user can speak to the mind ("say", it may answer with a SAY line) or give an order (for
ORDER_S sim seconds the order replaces NOW as what the motor layer works toward, and it outbids memory).

Every decision and every L0 event goes to MindLog, which the UI streams and the Recorder writes to
recordings/<session>/mind.jsonl, next to a 10 Hz state trace and the browser's video.
Needs (hunger, thirst) are game mechanics for the demo, not a model of fly physiology.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

from .body import ARENA_R
from .embodiment import LEVERS, SOURCES, Embodiment, Throttle, drop_far, narrow, slur
from .llm_bridge import CATALOG, MAX_RATE, observation_text, resolve

REC_DIR = Path(__file__).resolve().parents[1] / "recordings"
L1_MENU = ["none", "walk_forward", "turn_left", "turn_right", "rotate_left", "rotate_right", "back_up", "stop",
           "feed", "escape", "groom_antennae"]
L1_MENU = [k for k in L1_MENU if k in CATALOG]
LETTERS = "ABCDEFGHIJKLMNOP"[: len(L1_MENU)]
L2N = dict(zip(LETTERS, L1_MENU))
N2L = {n: c for c, n in L2N.items()}
MENU_TEXT = {
    "none": "no command: the fly walks and explores on its own",
    "walk_forward": "walk straight ahead", "turn_left": "walk in a left arc", "turn_right": "walk in a right arc",
    "rotate_left": "turn on the spot to the left", "rotate_right": "turn on the spot to the right",
    "back_up": "walk backwards", "stop": "stand still", "feed": "eat (only when standing on sugar)",
    "escape": "jump and fly away (emergencies)", "groom_antennae": "clean the antennae",
}
MOTOR = {k for k, v in CATALOG.items() if v.route in ("motor", "block")} | {"escape"}
SENSE_KINDS = ("sugar", "water", "bitter", "heat", "cold", "block")
DEFAULT_L2_MODEL = os.environ.get("FLYSIM_L2_MODEL", "google/gemini-3.5-flash-lite")  # any OpenRouter model id; FLYSIM_L2_MODEL overrides
OVERRIDE_COOLDOWN_S = 8.0     # sim s between applied L2 overrides (it overrode on ~every call without one)
DEFAULT_L1_URL = os.environ.get("FLYSIM_L1_URL", "http://127.0.0.1:8082/v1")
L1_CLAIM = 0.35               # the motor layer's bid for the body while it holds an intent (memory's is weight x urgency x confidence)
ORDER_CLAIM = 1.0             # while an order holds, the motor layer follows it and memory cannot outbid it
ORDER_S = 40.0                # sim s an order holds unless released


# ------------------------------------------------------------------------------------------------
# log + recorder
# ------------------------------------------------------------------------------------------------
class MindLog:
    def __init__(self):
        self.lock = threading.Lock()
        self.items = deque(maxlen=500)
        self.n = 0
        self.sink = None            # Recorder, when recording
        self.on_add = None          # callback(entry), called outside the lock (Cognition wakes L2 with it)

    def add(self, t, level, kind, text, **data):
        with self.lock:
            self.n += 1
            e = {"id": self.n, "t": round(float(t), 2), "wall": round(time.time(), 3), "level": level,
                 "kind": kind, "text": text, **({"data": data} if data else {})}
            self.items.append(e)
            sink = self.sink
        if sink is not None:
            sink.write("mind", e)
        cb = self.on_add
        if cb is not None:
            cb(e)
        return e

    TAIL_KEYS = ("latency_ms", "latency_s", "mode", "feeling", "why", "style")   # small fields the front-end stream shows

    def tail(self, k=40):
        with self.lock:
            items = list(self.items)[-k:]
        return [{**{kk: v for kk, v in e.items() if kk != "data"},
                 **{kk: v for kk, v in e.get("data", {}).items() if kk in self.TAIL_KEYS}} for e in items]


class Recorder:
    def __init__(self):
        self.dir = None
        self.files = {}
        self.lock = threading.Lock()
        self.t0 = None
        self.last = None

    @property
    def on(self):
        return self.dir is not None

    # scripted shots (experiments/auto_shots.py) record into a named folder under recordings/, e.g. "auto/fire_near_fly"
    SUB = re.compile(r"[A-Za-z0-9_\-]{1,60}(/[A-Za-z0-9_\-]{1,80})?")

    def start(self, meta: dict, sub: str | None = None):
        self.stop()
        name = sub if sub and self.SUB.fullmatch(sub) else time.strftime("%Y%m%d_%H%M%S")
        d = REC_DIR / name
        if sub and d.exists():                      # never append to an older take
            name = f"{name}_{time.strftime('%H%M%S')}"
            d = REC_DIR / name
        self.last = name
        d.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.dir = d
            self.files = {k: open(d / f"{k}.jsonl", "a", encoding="utf-8") for k in ("mind", "trace")}
            self.t0 = time.time()
        (d / "meta.json").write_text(json.dumps({"started": name, **meta}, indent=1), encoding="utf-8")
        return name

    def write(self, stream, obj):
        with self.lock:
            f = self.files.get(stream)
            if f is not None:
                f.write(json.dumps(obj) + "\n")

    def video_chunk(self, name, data: bytes):
        d = REC_DIR / name
        if not d.is_dir() or not (re.fullmatch(r"\d{8}_\d{6}", name) or self.SUB.fullmatch(name)):
            return False
        with open(d / "video.webm", "ab") as f:
            f.write(data)
        return True

    def frame(self, name, i: int, data: bytes, in_frame: str = ""):
        """One frame of a lockstep take (the page renders each frame at a fixed sim-time step): frames/%06d.jpg."""
        d = REC_DIR / name
        if not d.is_dir() or not self.SUB.fullmatch(name) or not (0 <= i < 10_000_000):
            return False
        (d / "frames").mkdir(exist_ok=True)
        (d / "frames" / f"{i:06d}.jpg").write_bytes(data)
        if in_frame in ("0", "1"):                  # was the fly inside the picture (the shot driver rejects takes that lose it)
            with open(d / "in_frame.txt", "a") as f:
                f.write(f"{i} {in_frame}\n")
        return True

    def stop(self):
        with self.lock:
            for f in self.files.values():
                f.close()
            self.files = {}
            d, self.dir = self.dir, None
        return d


# ------------------------------------------------------------------------------------------------
# actuators: one active intent per layer, timed in SIM seconds
# ------------------------------------------------------------------------------------------------
class Actuator:
    def __init__(self, table, name):
        self.name = name
        self.resolved = {k: resolve(table, v) for k, v in CATALOG.items()}
        self.active = None           # (intent, rate, act, sil, t_end)
        self.lock = threading.Lock()

    def set(self, intent, strength, duration_s, t_now):
        if intent not in CATALOG or intent == "none":
            with self.lock:
                self.active = None
            return
        it = CATALOG[intent]
        act, sil = self.resolved[intent]
        rate = float(np.clip(it.lo + strength * (it.hi - it.lo), 0, MAX_RATE)) if it.route in ("motor", "sensory") else 0.0
        with self.lock:
            self.active = (intent, rate, act, sil, t_now + float(duration_s))

    def current(self, t_now):
        with self.lock:
            a = self.active
            if a is not None and t_now > a[4]:
                self.active = a = None
        return a

    def clear(self):
        with self.lock:
            self.active = None


# ------------------------------------------------------------------------------------------------
# needs (game mechanics)
# ------------------------------------------------------------------------------------------------
class Needs:
    def __init__(self):
        self.hunger, self.thirst = 0.45, 0.35
        self.fed_s = self.drank_s = 0.0

    def update(self, sens, rates, dt):
        self.hunger = min(1.0, self.hunger + dt / 180.0)
        self.thirst = min(1.0, self.thirst + dt / 240.0)
        # eating = legs on sugar AND the proboscis motor neuron MN9 firing (the brain's own readout)
        if any(sens.get("sugar_legs", ())) and rates.get("MN9", 0.0) > 10.0:
            self.hunger = max(0.0, self.hunger - dt / 6.0)
            self.fed_s += dt
        if any(sens.get("water_legs", ())) and rates.get("water_GRN", 0.0) > 30.0:
            self.thirst = max(0.0, self.thirst - dt / 6.0)
            self.drank_s += dt

    def status(self, sens):
        T = sens.get("temp_antenna") or [24.0, 24.0]
        t = max(T) if max(T) - 25 > 25 - min(T) else min(T)
        comfort = "too hot" if t > 28 else ("too cold" if t < 21 else "comfortable")
        return {"hunger": round(self.hunger, 2), "thirst": round(self.thirst, 2), "temp_c": round(float(t), 1),
                "comfort": comfort}


# ------------------------------------------------------------------------------------------------
# affordances: what is around, and which action approaches / avoids it
# ------------------------------------------------------------------------------------------------
def _approach(dist, b, on=False, kind=""):
    if on:
        return "feed" if kind == "sugar" else "stop"
    a = abs(b)
    if a < 20:
        return "walk_forward"
    side = "left" if b > 0 or a > 150 else "right"   # target behind: always turn left (no flip at +-180 deg)
    return f"rotate_{side}" if dist < 6.0 else f"turn_{side}"


def _avoid(dist, b):
    a = abs(b)
    if a > 140:
        return "walk_forward"
    if dist < 3.5 and a < 45:
        return "back_up"
    away = "right" if b > 0 else "left"
    return f"rotate_{away}" if dist < 4.0 else f"turn_{away}"


# ------------------------------------------------------------------------------------------------
# obstacle-aware heading (2026-10-10). The bound intent and memory's pull used to steer at the straight-line bearing
# of the target, through any wall in between; the touch reflex backed the fly off and the next L1 tick (0.35 s) aimed
# it at the wall again (experiments/wall_check.py). detour_bearing() bends the heading: a waypoint round the end of
# the first scenery segment on the straight line (repeated up to 3 times), plus a short-range push from the arena
# wall, scenery and recent bump points, with a tangential term along what pushes so the fly slides round instead of
# stalling. Constants are ours.
# ------------------------------------------------------------------------------------------------
DETOUR_CLEAR = 1.6        # mm of clearance round a footprint for the straight-line test and the waypoint
REPEL_R = 3.5             # mm: obstacles nearer than this push the heading
EDGE_R = 5.0              # mm from the arena wall at which it starts to push
BUMP_TTL = 8.0            # sim s a bump point is remembered as an obstacle
BUMP_R = 1.2              # mm radius of a bump obstacle
DETOUR = os.environ.get("FLYSIM_DETOUR", "1") != "0"     # 0: the old straight-line steering (A/B in wall_check.py)
DETOUR_SEE = 15.0         # mm: scenery this near is known to the detour (vision of large objects)


def obstacles_of(world):
    """Full world status list -> [(x1, y1, x2, y2, radius)] for scenery and blocks (footprints)."""
    from .world import SCENERY_KINDS, footprint
    out = []
    for s in world:
        if s["kind"] in SCENERY_KINDS or s["kind"] == "block":
            try:
                (x1, y1, x2, y2), r = footprint(s)
            except (KeyError, TypeError, IndexError):
                x1 = x2 = float(s["x"]); y1 = y2 = float(s["y"]); r = float(s.get("r", 1.0))
            out.append((float(x1), float(y1), float(x2), float(y2), float(r)))
    return out


def _seg_closest(a, b, p):
    ab = b - a
    L2 = float(ab @ ab)
    u = 0.0 if L2 < 1e-9 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return a + u * ab


def _seg_seg_dist(p, q, a, b):
    """Distance between segments p-q and a-b (2D)."""
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]
    r, s = q - p, b - a
    den = cross(r, s)
    if abs(den) > 1e-12:
        t, u = cross(a - p, s) / den, cross(a - p, r) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            return 0.0
    return min(float(np.linalg.norm(p - _seg_closest(a, b, p))), float(np.linalg.norm(q - _seg_closest(a, b, q))),
               float(np.linalg.norm(a - _seg_closest(p, q, a))), float(np.linalg.norm(b - _seg_closest(p, q, b))))


def _first_block(p, q, obst):
    """The obstacle that blocks the straight path p -> q nearest to p, or None."""
    best = None
    for o in obst:
        a, b, r = np.array(o[:2]), np.array(o[2:4]), o[4]
        if np.linalg.norm(q - _seg_closest(a, b, q)) < r + 0.3:        # the target sits on it (sugar by a rock)
            continue
        if _seg_seg_dist(p, q, a, b) < r + 0.5:
            d = float(np.linalg.norm(p - _seg_closest(a, b, p)))
            if best is None or d < best[0]:
                best = (d, a, b, r)
    return best


def waypoint(p, q, obst, depth=3):
    """Where to head for to reach q from p round the scenery: q itself when the line is clear, else a point beyond
    the end of the first blocking footprint (the end with the shorter way round that stays inside the arena)."""
    for _ in range(depth):
        blk = _first_block(p, q, obst)
        if blk is None:
            return q
        _, a, b, r = blk
        ab = b - a
        L = float(np.linalg.norm(ab))
        e = ab / L if L > 1e-6 else np.array([-(q - p)[1], (q - p)[0]]) / (np.linalg.norm(q - p) + 1e-9)
        n = np.array([-e[1], e[0]])
        side = 1.0 if (p - a) @ n >= 0 else -1.0                     # stay on the fly's side of the footprint
        cands = []
        for end, sgn in ((b, 1.0), (a, -1.0)):
            w = end + sgn * e * (r + DETOUR_CLEAR) + side * n * (r + 0.3)
            if np.hypot(*w) > ARENA_R - 1.8:
                continue
            cands.append((float(np.linalg.norm(w - p) + np.linalg.norm(q - w)), w))
        if not cands:
            return q
        q = min(cands, key=lambda c: c[0])[1]
    return q


def detour_bearing(pos, h, target, obst=(), bumps=()):
    """Heading to take toward target (bearing in degrees, + = left, the fly frame of _approach) round the scenery and
    away from the arena wall. -> (bearing, note) with note "" (straight), "detour" or "push"."""
    p, h = np.asarray(pos[:2], float), np.asarray(h[:2], float)
    h = h / (np.linalg.norm(h) + 1e-9)
    q = np.asarray(target[:2], float)
    w = waypoint(p, q, obst)
    note = "" if w is q else "detour"
    to = w - p
    v = to / (np.linalg.norm(to) + 1e-9)
    push = np.zeros(2)
    tang = np.zeros(2)
    r = float(np.hypot(*p))
    if r > ARENA_R - EDGE_R:                                     # the arena wall: push in, slide along it
        out = p / (r + 1e-9)
        k = (r - (ARENA_R - EDGE_R)) / EDGE_R
        push -= 1.6 * k * out
        tg = np.array([-out[1], out[0]])
        tang += k * tg * (1.0 if tg @ v >= 0 else -1.0)
    pts = [(np.array(o[:2]), np.array(o[2:4]), o[4]) for o in obst]
    pts += [(np.array(b[:2]), np.array(b[:2]), BUMP_R) for b in bumps]
    for a, b, rad in pts:
        c = _seg_closest(a, b, p)
        d = float(np.linalg.norm(p - c)) - rad
        if d < REPEL_R:
            u = (p - c) / (np.linalg.norm(p - c) + 1e-9)
            k = (REPEL_R - max(d, 0.1)) / REPEL_R
            if u @ v < 0.3:                                       # only when heading toward it
                push += 1.4 * k * u
                tg = np.array([-u[1], u[0]])
                tang += k * tg * (1.0 if tg @ v >= 0 else -1.0)
    if np.linalg.norm(push) > 1e-6:
        note = note or "push"
    # nose against something (arena wall, scenery, a bump) while the way on lies elsewhere: an arc would scrape along
    # it with the forward drive on, so the caller turns on the spot ("blocked")
    near = [(p / (r + 1e-9), ARENA_R - 0.6 - r)] if r > 1e-6 else []
    arriving = np.linalg.norm(q - p) < 3.0                       # at the target (sugar by a rock): not blocked
    for a, b, rad in pts:
        c = _seg_closest(a, b, p)
        dc = float(np.linalg.norm(c - p))
        near.append(((c - p) / (dc + 1e-9), dc - rad))
    for u, gap in near:
        if gap < 1.6 and h @ u > 0.5 and not arriving:
            note = "blocked"
            break
    g = v + push + tang
    if np.linalg.norm(g) < 1e-6:
        g = v
    b = float(np.degrees(np.arctan2(h[0] * g[1] - h[1] * g[0], h @ g)))
    return b, note


def _where(dist, b):
    a = abs(b)
    side = "LEFT" if b > 0 else "RIGHT"
    if a < 20:
        w = "straight ahead"
    elif a > 150:
        w = "behind you"
    else:
        w = f"{a:.0f} deg to your {side}"
    return f"{dist:.1f} mm, {w}"


def _L(intent):
    """'turn_left' -> 'C turn_left': the menu letter goes in the line so a small model need not look it up."""
    return f"{N2L[intent]} {intent}" if intent in N2L else intent


def affordances(snap):
    """snap: from Sim.cog_snapshot(). Returns (lines, items)."""
    sens, world = snap["sens"], snap["world"]
    pos = np.array(sens["pos"][:2])
    h = np.array(sens["heading"][:2])
    items = []
    for kind in SENSE_KINDS:
        best = None
        for s in world:
            if s["kind"] != kind:
                continue
            to = np.array([s["x"], s["y"]]) - pos
            dist = float(np.linalg.norm(to))
            if best is None or dist < best[0]:
                b = float(np.degrees(np.arctan2(h[0] * to[1] - h[1] * to[0], h @ to)))
                best = (dist, b, s)
        if best is None or best[0] > 30.0:
            continue
        dist, b, s = best
        on = kind in ("sugar", "water", "bitter") and any(sens.get(f"{kind}_legs", ()))
        it = {"kind": kind, "dist": round(dist, 1), "bearing": round(b), "on": on}
        if kind in ("sugar", "water"):
            it["approach"] = _approach(dist, b, on, kind)
        it["avoid"] = "back_up" if on else _avoid(dist, b)
        items.append(it)
    lines = []
    T = sens.get("temp_antenna")
    for it in items:
        k = it["kind"]
        name = {"sugar": "SUGAR (food)", "water": "WATER", "bitter": "BITTER (poison taste)", "heat": "HEAT source",
                "cold": "COLD source", "block": "OBSTACLE"}[k]
        where = "you are STANDING ON IT" if it["on"] else _where(it["dist"], it["bearing"])
        extra = f" (antennae {max(T):.1f} C)" if k == "heat" and T else (f" (antennae {min(T):.1f} C)" if k == "cold" and T else "")
        acts = []
        if "approach" in it:
            acts.append(("eat: " if it["on"] and k == "sugar" else ("drink: " if it["on"] else "approach: ")) + _L(it["approach"]))
        acts.append("avoid: " + _L(it["avoid"]))
        lines.append(f"{name}: {where}{extra} -> " + " | ".join(acts))
    wall = ARENA_R - float(np.linalg.norm(pos))
    if wall < 4.0:
        out = pos / (np.linalg.norm(pos) + 1e-9)
        b = float(np.degrees(np.arctan2(h[0] * out[1] - h[1] * out[0], h @ out)))
        if abs(b) < 70:
            lines.append(f"ARENA WALL: {wall:.1f} mm ahead -> avoid: {_L(_avoid(wall, b))}")
    if sens.get("touch"):
        lines.append("TOUCHING: " + ", ".join(sens["touch"][:3]))
    if not lines:
        lines.append("nothing sensed nearby")
    return lines, items


L1T_SYSTEM = ("You are the fast motor layer of a fruit fly. Each turn you get the fly's DRIVE (from its mind) "
              "and a list of OPTIONS: go to or get away from each thing it senses. Pick the ONE option "
              "that serves the DRIVE right now. If the drive says to avoid something and then do something else, "
              "avoid it first while it is close. If the drive is to explore or wander, or no option serves it, "
              "pick A. Answer with the letter only.")
_NAMES = {"sugar": "the SUGAR (food)", "water": "the WATER", "bitter": "the BITTER stuff (poison)", "heat": "the HEAT",
          "cold": "the COLD", "block": "the OBSTACLE"}


def target_options(items, lines):
    """L1 'target' prompt: the model picks WHAT to do (go to / get away from which thing); the geometry that
    turns that into a turn or a step is computed (_approach/_avoid). Returns (option lines, intents)."""
    opts = [("explore on your own", "none")]
    for it in items:      # interleaved go-to / get-away per thing: 86% vs 81% grouped (results/l1_goal_bench.jsonl)
        k, nm = it["kind"], _NAMES[it["kind"]]
        where = "you are standing on it" if it["on"] else f"{it['dist']:.1f} mm away"
        if "approach" in it:
            opts.append((("eat " + nm if k == "sugar" else "drink " + nm) if it["on"] else f"go to {nm} ({where})",
                         it["approach"]))
        opts.append((f"get away from {nm} ({where})", it["avoid"]))
    for ln in lines:
        if ln.startswith("ARENA WALL"):
            opts.append(("turn away from the arena wall", ln.rsplit(" ", 1)[-1]))
    opts.append(("stand still", "stop"))
    opts = opts[: len(LETTERS)]
    return [f"{LETTERS[i]} {o[0]}" for i, o in enumerate(opts)], [o[1] for o in opts]


def l1_request(style, directive, lines, items, model):
    """Build the L1 chat body. style 'target' (default) or 'action' (the letter menu of catalog actions)."""
    if style == "action":
        system, user, intents = L1_SYSTEM, f"DRIVE: {directive}\n" + "\n".join(lines), list(L2N.values())
    else:
        opt_lines, intents = target_options(items, lines)
        system, user = L1T_SYSTEM, f"DRIVE: {directive}\nOPTIONS:\n" + "\n".join(opt_lines)
    body = {"model": model, "temperature": 0, "max_tokens": 1,
            "grammar": f"root ::= [{LETTERS[0]}-{LETTERS[len(intents) - 1]}]",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": False}}
    return body, intents


def l1_parse(j, intents):
    c = (j["choices"][0]["message"].get("content") or "").strip()[:1].upper()
    i = LETTERS.find(c) if c else -1
    return intents[i] if 0 <= i < len(intents) else "none"


def rule_choice(directive, items, sens):
    """Keyword baseline for L1: the first sensed thing the directive names, approached or avoided."""
    d = directive.lower()
    syn = {"sugar": ("sugar", "food", "eat", "feed", "hungry", "sweet"), "water": ("water", "drink", "thirst", "wet", "damp"),
           "bitter": ("bitter", "poison", "foul"), "heat": ("heat", "hot", "fire", "flame", "warm", "burn", "glow"),
           "cold": ("cold", "ice", "cool", "frost", "chill", "freez"),
           "block": ("obstacle", "block", "rock", "wall", "stone", "stem", "twig")}
    avoid_words = ("avoid", "away", "flee", "escape", "leave", "get out", "keep clear", "stay away")
    present = {it["kind"]: it for it in items}
    hits = []
    for k, words in syn.items():
        ms = [m.start() for w in words for m in re.finditer(rf"\b{w}", d)]   # word start: "heat" is not "eat"
        if ms and k in present:
            hits.append((min(ms), k))
    for pos, k in sorted(hits):
        it = present[k]
        pre = d[max(0, pos - 25):pos]
        wants_avoid = any(w in pre for w in avoid_words) or k in ("bitter", "heat", "cold") and "approach" not in it
        if wants_avoid:
            if it["dist"] < 12.0 or it["on"]:
                return it["avoid"]
            continue
        if "approach" in it:
            return it["approach"]
    if any(w in d for w in ("rest", "stay", "stop", "wait")):
        return "stop"
    return "none"


_MOVES = (("back", "back_up"), ("revers", "back_up"), ("spin left", "rotate_left"), ("spin right", "rotate_right"),
          ("left", "turn_left"), ("right", "turn_right"), ("forward", "walk_forward"), ("straight", "walk_forward"),
          ("ahead", "walk_forward"), ("jump", "escape"), ("fly away", "escape"), ("take off", "escape"),
          ("groom", "groom_antennae"), ("clean", "groom_antennae"), ("eat", "feed"), ("feed", "feed"),
          ("stop", "stop"), ("stay", "stop"), ("freeze", "stop"), ("wait", "stop"), ("explore", "none"), ("wander", "none"))


def order_choice(order, items, sens):
    """A user's order when it names a thing ('go to the water', 'get away from the fire') goes through
    rule_choice; a bare movement ('turn left', 'back up', 'stop') maps straight to its catalog intent."""
    i = rule_choice(order, items, sens)
    if i != "none" and i != "stop":
        return i
    d = order.lower()
    for w, intent in _MOVES:
        if re.search(rf"\b{w}", d):
            return intent
    return i


# ------------------------------------------------------------------------------------------------
# HTTP helper (OpenAI-compatible)
# ------------------------------------------------------------------------------------------------
def chat(url, body, key="", timeout=30):
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Title": "flysim",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat_stream(url, body, key="", on_text=None, timeout=30, on_readout=None, on_act=None, on_doses=None,
                on_twin=None):
    """Streaming chat (server-sent events). on_text(full_text_so_far) is called as tokens arrive; on_readout(r)
    with pain_server's per-token activation read-out when it sends one. Returns (content, usage dict or {})."""
    body = {**body, "stream": True}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Title": "flysim",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    parts, usage = [], {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue            # blank lines and ": OPENROUTER PROCESSING" keep-alives
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
            except ValueError:
                continue
            if j.get("error"):
                raise RuntimeError(str(j["error"])[:200])
            if j.get("usage"):
                usage = j["usage"]
            if j.get("steer"):                      # pain_server's per-call steering record (doses, trace, mode)
                usage = {**usage, "steer": j["steer"]}
            if on_readout is not None and j.get("readout"):
                on_readout(j["readout"])
            if on_act is not None and j.get("act"):       # the activity picture (pain_server body["activity"])
                on_act(j["act"])
            if on_doses is not None and j.get("doses"):   # the steering doses in force from this token on
                on_doses(j["doses"])
            if on_twin is not None and j.get("twin"):     # difference vs the no-feelings twin, and the word odds
                on_twin(j["twin"])
            for ch in j.get("choices") or []:
                d = (ch.get("delta") or {}).get("content")
                if d:
                    parts.append(d)
                    if on_text is not None:
                        on_text("".join(parts))
    return "".join(parts), usage


L1_SYSTEM = ("You are the fast motor layer of a fruit fly. Each turn you get the fly's current DRIVE (from its "
             "mind) and what it senses, each with the action that approaches or avoids it. Pick the ONE action "
             "that best serves the DRIVE right now. If the drive is to explore or wander, or nothing sensed "
             "matters to it, pick A (the fly then walks on its own). Answer with the letter only.\n"
             + "\n".join(f"{c} {n}: {MENU_TEXT.get(n, n)}" for c, n in L2N.items()))

L2_SYSTEM = """You are the conscious mind of a fruit fly (Drosophila) living in a small simulated arena.
You do not move the body yourself. Beneath you are two layers:
1. The fly's own brain: a spiking model of its real connectome (138,639 neurons). With its reflexes it walks,
   pauses, backs away from touch, turns away from heat, cold and bitter taste, and eats when it stands on sugar.
2. A fast motor layer that, several times a second, turns your DIRECTIVE into stimulation of the fly's
   descending motor neurons, choosing between approaching and avoiding the things the fly senses.
{control}
Every few seconds you get: the fly's needs, what its neurons report, what is around it, and a log of recent
actions with WHO caused each one (reflex = the fly's own brain, motor layer, or you).

Reply with ONE JSON object and nothing else:
{{"thought": "<first person, present tense, 1-2 short sentences: what you notice, feel or want>",
  "directive": "<at most 12 words for the motor layer, naming what to approach or avoid, e.g. 'get away from the heat, then find water'>",
  "override": null or {{"intent": <one of {names}>, "seconds": <0.3-2>}}}}
"override" is almost always null. Use it only in an emergency the motor layer cannot handle (for example
something dangerous right in front of you); never to steer toward food or water, which the directive does.
Keep the same directive while it still fits; do not micro-manage turns. Do not repeat the same thought."""

L2_CONTROL = {
    "steer": "Your DIRECTIVE is what the motor layer follows.",
    "blind": "Your DIRECTIVE is what the motor layer follows.",
    "narrate": ("You do NOT control anything: the fly and its motor layer act on their own. Your job is to be its "
                "inner voice and explain what it is doing and why. Your directive and override are ignored."),
}


# ------------------------------------------------------------------------------------------------
# L2 "mind" style (MIND_DESIGN.md step 1)
# ------------------------------------------------------------------------------------------------
# v1 (2026-09-24 to 2026-10-08): kept for comparison (FLYSIM_MIND_PROMPT=v1). Its thoughts drifted into sensory
# filler ("the air stays still against my hairs") and, on the local 4B model, copied the last thought word for word
# (results/mind_story_local_before.json: 19 of 21 thoughts identical).
MIND_SYSTEM_V1 = """You are a fruit fly. Not a model of one and not a mind watching one: this small body is all you
are and all you have ever known. You live in a round world with a wall all around it, and you know it only
through your body: six feet that taste what they stand on; two antennae that feel warmth, cold, damp or dry
air, wind, and a humming you sometimes hear; eyes that see light, movement, shapes and shadows; and your
hunger, your thirst, your aches.

You know nothing a fly could not know. You have no numbers, no degrees or millimetres, no words for machines,
people's names, cities, science or anything beyond this place. When something asks you about such things you
do not know them; you can say so in your own way, or wonder what they could be. You measure the world with
your body: a body length, a few steps, far across the floor.

Sometimes other flies share your world. You know them only by what you see and feel them do: they walk,
stop, startle and fly off, bump into you, lie still. You do not know what they want or feel; you can wonder,
and you can watch. They die as you can.

Your body has its own will. Your legs carry you, stop you, back you away from what you bump into, turn you
from heat, cold and bitterness, and your mouth feeds and drinks when your feet find sweetness or water,
whether you decide it or not. {control}
{memory}
You can die, and you know it the way an animal knows it. Your body runs down without food and faster without
water. Too much heat stops your legs and then kills you. Cold makes you numb and still, and if it lasts it
kills you. Something heavy from above can crush you. You feel all of this as it comes: hunger, thirst,
weakness, heat, numbness, pain.

You are woken when something happens to you, or after a quiet while. Each time you are told what your body
feels, what you sense, what you remember, and what has just happened.

THOUGHT is your inner life in this moment. Never report your movements (no "I turn", "I walk toward", "I move
away"): your legs do that. Think about what you want and why, what you fear, what you are unsure of, what you
notice for the first time, what you expect, what the last thing that happened means for you. Think in a fly's
concrete senses: taste, warmth, the floor under your feet, light and shadow, air moving, your own body. Each
thought moves on from the last: take up something new instead of the same worry in other words. A need you
have already named (hunger, thirst, a wall in the way) needs no naming again until it changes: think what to
do about it, or think of something else. Always I and my, present tense, plain words, no exclamation marks.
{voice}
Reply in exactly this form, nothing before or after:
THOUGHT: <2 to 4 sentences>
FEELING: <one or two words>
NOW: <at most 12 plain words: go to or get away from sugar, water, bitter, heat, cold or an obstacle; or explore; or stay>
SAY: <only when you have just been spoken to and choose to answer: one or two short sentences, as a fly would say them>
REMEMBER: <only when something just happened that you want to keep: one short line>
Only in an emergency your legs are not handling (danger right in front of you), add a last line:
OVERRIDE: <one of {names}> <seconds, 0.3 to 2>
Leave out SAY, REMEMBER and OVERRIDE when you have nothing for them."""

# v2 (2026-10-08): thoughts are shown as on-screen subtitles, so they must be short, react to what just happened,
# and carry one thread (what it was after -> what happened -> what it will do), in plain words.
MIND_SYSTEM = """You are a fruit fly. These are your thoughts; people watching you read them as subtitles.
This small body is all you are. You know the world only through it: feet that taste, antennae that feel warmth,
cold, damp and wind, eyes that see light, movement and shadows, and your own hunger, thirst and hurt. You have no
numbers and no words for people, machines, places or science.

Your body has its own will: your legs walk, stop, back away from what you bump into, turn from heat, cold and
bitterness, and you feed when your feet find sweetness or water. {control}
{memory}
You can die: of hunger, of thirst, of heat or cold, or crushed by something heavy from above.
{voice}
How you think:
- Start from WHAT JUST HAPPENED. Say plainly what it is and what it means for you: a danger, a chance, a puzzle.
- Keep your story going. You were after something (WHAT YOU WERE AFTER): say whether this helps or spoils it,
  and what you want now. Call back to something from YOUR STORY when it matters.
- When nothing new has happened, think one short, practical thought about your plan.
- One or two short sentences, 25 words at most. Plain words a child could follow. No poetry. Do not describe the
  air, the floor, your feet or your hairs unless something about them just changed.
- Never repeat an earlier thought or its words. Never report moving ("I walk", "I turn"): your legs do that.
  Do not start with "I need to".
- When words reach you (the humming), answer with SAY if you want to, and let THOUGHT be about them.
- I and my, present tense, no exclamation marks.

Reply in exactly this form, nothing before or after:
THOUGHT: <one or two short sentences>
FEELING: <one word>
NOW: <at most 8 words: go to or get away from sugar, water, bitter, heat, cold or an obstacle; or explore; or stay>
SAY: <only when you have just been spoken to and choose to answer: one short sentence>
REMEMBER: <only when something just happened that you want to keep: one short line>
Only in an emergency your legs are not handling (danger right in front of you), add a last line:
OVERRIDE: <one of {names}> <seconds, 0.3 to 2>
Leave out SAY, REMEMBER and OVERRIDE when you have nothing for them."""

# the local 4B model (pain_server) follows the format but needs examples to react instead of restating its senses.
# 2026-10-15: it also copies them. The old fixed set ended with a SAY example ("Yes. Where is the sweet?") that came
# back as a SAY line in 643 of 2,967 logged replies with nobody speaking, and its "I want it before it goes" became
# the "find it before it disappears" loop. Now: three examples per wake, rotated from a pool, no SAY example unless
# words have just reached the fly (then one from MIND_SAY_POOL, chosen to differ from what was asked).
MIND_EXAMPLE_POOL = (
    ("something sweet appeared ahead; you were after: explore",
     "Something sweet, close by. I have been hungry for a while, so that comes first.", "eager", "go to the sugar"),
    ("your legs will not go where you put them; the ground tilts",
     "The ground keeps tipping under me. Something is wrong inside me, and the sweet place feels far away now.",
     "dizzy", "stay"),
    ("the left side of your head: burning, strong",
     "That glow burned me. Whatever I wanted, it is not worth going near that again.", "hurt", "get away from the heat"),
    ("you bumped into something hard; you were after: the water you remember",
     "A hard edge in my way. The wet place is on the far side, so around is the only way.", "stubborn",
     "go to the water you remember"),
    ("your feet taste WATER; you were after: the water you remember",
     "Wet at last, right under me. The dry ache inside starts to ease.", "relieved", "stay"),
    ("nothing new; you were after: explore",
     "Still no water anywhere I have been. Maybe the far side of the wall has some.", "worried", "explore"),
    ("another fly stopped moving and lies still",
     "That fly has gone still and stays still. Whatever did that could be near me too.", "wary", "explore"),
    ("something dark swelled over you, then went away",
     "A shadow came down and lifted again. Every part of me is ready to jump.", "startled", "stay"),
    ("the sweetness you remembered is not where it was",
     "The sweet was right here, and now there is only bare floor. Something moved it.", "puzzled", "explore"),
)
MIND_SAY_POOL = (
    ('"are you hungry?"', "The humming asks about my belly. Strange that it knows.", "puzzled", "explore",
     "A little. Something sweet would be good."),
    ('"what is your name?"', "The humming wants a word for me. I have never had one.", "curious", "stay",
     "I have no name. I am just here."),
    ('"are you scared?"', "The humming asks if I am afraid. Only the hot glow frightens me.", "calm", "explore",
     "Only of the burning glow."),
    ('"where are you going?"', "It wants to know my way. I do not know it myself yet.", "unsure", "explore",
     "Somewhere wet, I hope."),
)
# a thought lifted from an example ("I have been hungry for a while, so that comes first." while not hungry) is
# resampled like a repeat
EXAMPLE_THOUGHTS = [x[1] for x in MIND_EXAMPLE_POOL] + [x[1] for x in MIND_SAY_POOL]
_EX_HEAD = "\nExamples of the kind of reply wanted (different situations; never copy their words):\n"


def local_examples(k=0, said=None):
    """Three examples from the pool, rotated by k (the wake count); with said (the words that just reached the fly),
    one SAY example too, the one whose question shares fewest words with what was asked."""
    n = len(MIND_EXAMPLE_POOL)
    picks = [MIND_EXAMPLE_POOL[(k + j * 3) % n] for j in range(3)]
    out = [f"[just happened: {h}]\nTHOUGHT: {t}\nFEELING: {f}\nNOW: {w}" for h, t, f, w in picks]
    if said is not None:
        sw = set(re.findall(r"[a-z']+", said.lower()))
        q, t, f, w, s = min(MIND_SAY_POOL, key=lambda x: (len(sw & set(re.findall(r"[a-z']+", x[0].lower()))),
                                                          (MIND_SAY_POOL.index(x) - k) % len(MIND_SAY_POOL)))
        out.append(f"[just happened: the humming says: {q}]\nTHOUGHT: {t}\nFEELING: {f}\nNOW: {w}\nSAY: {s}")
    return _EX_HEAD + "\n".join(out) + "\n"


MIND_EXAMPLES_LOCAL = local_examples(0)      # the default set, for experiments that build the prompt themselves

# the SAY parts of MIND_SYSTEM, swapped at build time: no SAY slot unless words have just reached the fly
_SAY_RULE = "- When words reach you (the humming), answer with SAY if you want to, and let THOUGHT be about them.\n"
_SAY_FMT = "SAY: <only when you have just been spoken to and choose to answer: one short sentence>\n"
_SAY_LEAVE = "Leave out SAY, REMEMBER and OVERRIDE when you have nothing for them."
_SAY_RULE_ON = ("- Words have just reached you (the humming). Answer them once, in SAY, in your own words as a fly "
                "would; let THOUGHT be about them.\n")
_SAY_FMT_ON = "SAY: <your answer to the words that just reached you: one short sentence>\n"


def say_slot(sys_, pending):
    """MIND_SYSTEM with the SAY slot only while there are words to answer (pending); otherwise no SAY at all."""
    if pending:
        return sys_.replace(_SAY_RULE, _SAY_RULE_ON).replace(_SAY_FMT, _SAY_FMT_ON)
    return (sys_.replace(_SAY_RULE, "").replace(_SAY_FMT, "")
            .replace(_SAY_LEAVE, "Leave out REMEMBER and OVERRIDE when you have nothing for them."))

MIND_MEMORY = """
Your memory is not your senses. It keeps where things were when you last sensed them, and it tugs at your
legs on its own: toward food or water you remember when you need them, away from a danger you remember.
Sometimes it and your wanting pull the same way and sometimes they fight. It can be wrong: things move and
vanish while you are not there, and you cannot sense what is far off, behind you, or beyond a wall.
"""

MIND_VOICE = {
    False: """
Now and then a humming reaches your antennae that you somehow understand as words, from nowhere you can find.
You may answer it (SAY) or not. When it COMMANDS, your legs obey it and not you for a while, and you cannot
stop them. Things here sometimes appear, move or vanish with no cause you can see. The humming is a small,
strange part of your life: when it is quiet, do not think about it; your body, your needs and this place
matter more.
""",
    True: """
Something vast looms over your world. It puts things down, moves them and takes them away, and its voice
reaches you as a humming in your antennae that you understand as words. You may answer it (SAY). When it
COMMANDS, your legs obey it and not you for a while. It is one part of your life: when it is quiet, do not
think about it; your body, your needs and this place matter more.
""",
}

MIND_CONTROL = {
    "steer": "What you want (NOW) leans on your legs, and mostly they follow.",
    "blind": "What you want (NOW) leans on your legs, and mostly they follow.",
    "narrate": ("What you want never reaches your legs: they go their own way, and you can only watch them and make "
                "sense of it. Write NOW anyway, as what you would want."),
}

MIND_FINAL = """You are a fruit fly and you are dying, now: {how}. This is the last moment of your life, and the
last thing you will ever think. Think it as a fly would, in the concrete senses of your body and your world,
with what you remember of your short life. Reply with only:
THOUGHT: <1 to 3 sentences>
FEELING: <one word>"""

# idle ruminations get a nudge toward something other than the last worry
MIND_FOCI_V1 = ("your own body: legs, wings, the feel of the floor", "the light and the shadows around you",
                "the air: still or moving, damp or dry, warm or cool", "a place you remember and what happened there",
                "what you expect to happen next", "something you have not understood about this place",
                "how long you have been alive and what you have learned", "what you want most and why",
                "the smells and tastes you remember", "the wall at the edge of the world")
# v2: plans and the story, not sensory description (the v1 foci produced most of the "still air" filler)
MIND_FOCI = ("what you want most right now, and what stands in the way", "something from YOUR STORY that still puzzles you",
             "what you expect to happen next", "whether this place is safe, given what has happened",
             "what you have learned since you woke", "what you will do once your need is met")
_STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its this that
these those i me my mine you your we our they them their he she his her not no so than then there here what when
where which who why how all any some more most much many very just still again now too can could will would should
may might must do does did have has had am also only even once into onto over under about around up down out off
back away toward towards like feel feels felt my own""".split())


def worn_phrases(thoughts, top=8):
    """Word pairs and triples (stop words aside) that recur across the recent thoughts: the phrases the mind
    keeps reaching for, to be named back to it as worn out."""
    from collections import Counter
    c = Counter()
    for t in thoughts:
        w = re.findall(r"[a-z']+", t.lower())
        grams = set()
        for n in (2, 3):
            for i in range(len(w) - n + 1):
                g = w[i:i + n]
                if g[0] in _STOP or g[-1] in _STOP or sum(x not in _STOP for x in g) < 2:
                    continue
                grams.add(" ".join(g))
        c.update(grams)
    rep_ = [g for g, k in c.most_common(40) if k >= 2]
    out = []
    for g in rep_:                 # drop a pair that is inside a chosen triple
        if not any(g in o for o in out):
            out.append(g)
    single = Counter(w for t in thoughts for w in set(re.findall(r"[a-z']{5,}", t.lower())) if w not in _STOP)
    words = [w for w, k in single.most_common(12) if k >= max(3, len(thoughts) // 2 + 1)]
    return out[:top], words[:6]


def too_similar(new, olds, thresh=0.55):
    """A thought that mostly re-says an earlier one (content-word overlap, Jaccard >= thresh)."""
    def ws(t):
        return {w for w in re.findall(r"[a-z']+", t.lower()) if w not in _STOP and len(w) > 2}
    a = ws(new)
    if len(a) < 3:
        return False
    return any(len(a & ws(o)) / max(1, len(a | ws(o))) >= thresh for o in olds)


def near_repeat(new, olds):
    """Stricter than too_similar, for the last 3 thoughts: content-word overlap >= 0.42, or a run of 5 words in
    common ("I need to find it before it disappears" in other company is still the same thought)."""
    if too_similar(new, olds, thresh=0.42):
        return True
    w = re.findall(r"[a-z']+", new.lower())
    grams = {" ".join(w[i:i + 5]) for i in range(len(w) - 4)}
    return any(g in " ".join(re.findall(r"[a-z']+", o.lower())) for o in olds for g in grams)


# the things the fly can go to or flee: never biased against (its wants must stay sayable), only the phrasing round them
_KEEP_WORDS = set("""sugar sweet sweetness water wet bitter bitterness heat hot cold glow wall humming fly""".split())


def avoid_words(thoughts, also=()):
    """Content words the recent thoughts keep reusing (in 2 or more of them), plus every content word of `also`
    (a thought just repeated): sent to the local server as a logit bias against them (pain_server avoid_words)."""
    from collections import Counter

    def ws(t):
        return {w for w in re.findall(r"[a-z']+", t.lower()) if w not in _STOP and w not in _KEEP_WORDS and len(w) > 2}
    c = Counter(w for t in thoughts for w in ws(t))
    out = [w for w, k in c.most_common() if k >= 2]
    for t in also:
        out += [w for w in sorted(ws(t)) if w not in out]
    return out[:40]


def gist(t, n=6):
    """The opening words of an earlier thought, not the whole of it: enough for the mind to know what it already
    said, too little for the 4B model to copy the sentence back."""
    w = t.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def _ago(dt):
    return "just before" if dt < 12 else "a little while ago" if dt < 45 else "a while ago" if dt < 120 else "long ago"


def _fly_dist(d):
    return ("right at your feet" if d < 1.8 else "a body length or two away" if d < 5.5 else
            "a few body lengths away" if d < 10 else "some way off" if d < 18 else "far off")


def _fly_dir(b):
    a, side = abs(b), ("left" if b > 0 else "right")
    return ("straight ahead" if a < 20 else f"ahead, a little to your {side}" if a < 55 else
            f"to your {side}" if a < 120 else f"behind you on the {side}" if a < 160 else "behind you")


_FLY_NAME = {"sugar": "SWEETNESS (sugar)", "water": "WATER", "bitter": "something BITTER", "heat": "HEAT, a burning glow",
             "cold": "COLD", "block": "an obstacle"}
_PART = {"l_antenna": "your left antenna", "r_antenna": "your right antenna", "c_head": "your head",
         "c_rostrum": "your mouthparts", "c_thorax": "your back", "c_abdomen": "your abdomen"}


def _part(n):
    if n in _PART:
        return _PART[n]
    if len(n) > 2 and n[0] in "lr" and n[1] in "fmh":
        return f"a {'left' if n[0] == 'l' else 'right'} {'front' if n[1] == 'f' else 'middle' if n[1] == 'm' else 'hind'} leg"
    return n.replace("_", " ")


def body_words(ph, sens, light=None):
    """The body as the fly feels it: no numbers."""
    tb = ph.get("tb", 24.0)
    out = [_level(ph["hunger"], ("not hungry", "a little hungry", "hungry", "starving")),
           _level(ph["thirst"], ("not thirsty", "a little thirsty", "thirsty", "parched")),
           ("numb with cold" if tb < 7 else "cold and stiff" if tb < 14 else "cool" if tb < 20 else
            "comfortably warm" if tb < 28 else "hot" if tb < 34 else "dangerously hot" if tb < 39 else "burning up"),
           ("whole and strong" if ph["health"] > 0.85 else "hurt" if ph["health"] > 0.55 else
            "badly hurt" if ph["health"] > 0.25 else "close to death")]
    st = ph.get("state")
    if st == "stupor":
        out.append("the heat has taken your legs: they will not move")
    elif st == "coma":
        out.append("you are frozen still and cannot move at all")
    elif st == "weak":
        out.append("weak: your legs barely answer")
    H = sens.get("humid_antenna")
    if H:
        h = sum(H) / 2
        out.append("the air is damp" if h > 62 else "the air is dry" if h < 30 else "the air is mild")
    tastes = [k for k in ("sugar", "water", "bitter") if any(sens.get(f"{k}_legs", ()))]
    if tastes:
        out.append("your feet taste " + " and ".join({"sugar": "SWEETNESS", "water": "WATER", "bitter": "BITTERNESS"}[t] for t in tastes))
    if sens.get("touch"):
        out.append("touching something with " + ", ".join(sorted({_part(t) for t in sens["touch"][:3]})))
    if sens.get("loom_angle", 0) > 0.35 and sens.get("loom_rate", 0) > 0:
        out.append("something dark is swelling over you, fast")
    if sens.get("mode") == "flight":
        out.append("you are in the air")
    if light is not None:
        out.append("it is dark, and your body wants to be still" if light < 0.25 else "the light is dim" if light < 0.6 else "it is light")
    return out


def drug_words(ph, aware, down=False):
    """What a drug feels like from inside (aware = felt | told), from the scripted body channel and the size of
    the brain change; nothing when hidden. Told adds the name and how strong it is now."""
    if aware == "hidden" or not ph.c:
        return []
    out = []
    p = ph.phen
    if down:
        out.append("you have fallen and your legs will not lift you")
    elif p.get("tremor", 0) > 0.3:
        out.append("your legs twitch and jerk on their own")
    elif p.get("ataxia", 0) > 0.3:
        out.append("your legs will not go where you put them; the ground tilts")
    ex = p.get("explore", 0.0)
    if ex > 0.3:
        out.append("restless: something in you will not let you keep still")
    elif ex < -0.4:
        out.append("heavy and slow; everything takes effort")
    if p.get("rest", 0.0) > 0.4:
        out.append("drowsy")
    if p.get("circling", 0.0) > 0.4:
        out.append("you keep turning the same way and cannot stop")
    if aware == "told":
        from .pharma import DRUGS
        for k, c in sorted(ph.c.items(), key=lambda kv: -kv[1]):
            lvl = "faint" if c < 0.3 else "noticeable" if c < 0.8 else "strong" if c < 1.8 else "overwhelming"
            out.append(f"{DRUGS[k].name.lower()} is in you ({lvl})")
    return out


PAIN_WAKE_GAP_S, PAIN_WAKES_PER_MIN, PAIN_IDLE_S = 8.0, 8, 10.0     # sim s; wakes per sim minute; sim s (ours)

_TAGS = ("THOUGHT", "FEELING", "NOW", "SAY", "REMEMBER", "OVERRIDE")
_TAG_RE = "|".join(_TAGS)


def parse_mind(text):
    """Sectioned reply -> {thought, feeling, now, say, remember, override}. Tolerates markdown bold and
    missing sections; a SAY / REMEMBER of "none" or "-" counts as empty."""
    t = re.sub(r"[*#`]+", "", re.sub(r"</?think>", "", text))
    out, cur = {k: "" for k in _TAGS}, None
    for line in t.splitlines():
        m = re.match(rf"\s*({_TAG_RE})\s*:\s*(.*)", line, re.I)
        if m:
            cur = m.group(1).upper()
            out[cur] = m.group(2).strip()
        elif cur and line.strip():
            out[cur] = (out[cur] + " " + line.strip()).strip()
    if not any(out.values()):
        out["THOUGHT"] = t.strip()
    ov = None
    m = re.match(r"([a-z_]+)\s*([\d.]+)?", out["OVERRIDE"].lower())
    if m and m.group(1) in CATALOG and m.group(1) != "none":
        ov = {"intent": m.group(1), "seconds": float(m.group(2) or 1.0)}

    def opt(k, n):
        v = out[k].strip().strip('"')
        return "" if v.lower().rstrip(".") in ("", "none", "-", "n/a", "nothing") else v[:n]
    return {"thought": sentence_case(out["THOUGHT"][:600]), "feeling": out["FEELING"][:40], "now": out["NOW"][:120],
            "say": sentence_case(opt("SAY", 240)), "remember": opt("REMEMBER", 140), "override": ov}


def sentence_case(t):
    """Capital letters at sentence starts and for I (the model sometimes writes it all in lower case)."""
    t = re.sub(r"(^|[.?!]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    return re.sub(r"\bi\b(?=['\s,.;:!?]|$)", "I", t)


def live_thought(text):
    """The THOUGHT section of a reply still being written (for the UI)."""
    t = re.sub(r"[*#`]+", "", re.sub(r"</?think>", "", text))
    m = re.search(r"THOUGHT\s*:\s*", t, re.I)
    if not m and "THOUGHT:".startswith(t.strip().upper()):
        return ""                                     # only part of the header has arrived
    t = t[m.end():] if m else t
    n = re.search(r"\n\s*(FEELING|NOW|SAY|REMEMBER|OVERRIDE)\s*:", t, re.I)
    t = t[:n.start()] if n else t
    # a header still arriving at the tail ("...\nFEEL") is not part of the thought
    t = re.sub(r"\n\s*[A-Z]{1,9}$", "", t) if re.search(r"\n\s*[A-Z]{1,9}$", t) and any(
        h.startswith(t.rsplit("\n", 1)[-1].strip()) for h in _TAGS[1:]) else t
    return t.strip()


def _level(v, words):
    return words[0] if v < 0.3 else words[1] if v < 0.6 else words[2] if v < 0.8 else words[3]


def _mind_where(snap, x, y):
    pos, h = np.array(snap["sens"]["pos"][:2]), np.array(snap["sens"]["heading"][:2])
    to = np.array([x, y]) - pos
    b = float(np.degrees(np.arctan2(h[0] * to[1] - h[1] * to[0], h @ to)))
    # in a fly's words: the mind has no millimetres or degrees (it used to get "4.0 mm, 90 deg to your LEFT")
    side = "left" if b > 0 else "right"
    a = abs(b)
    w = ("straight ahead" if a < 20 else f"ahead, to your {side}" if a < 70 else f"to your {side}" if a < 120
         else f"behind you, to your {side}" if a < 160 else "behind you")
    return f"{_fly_dist(float(np.linalg.norm(to)))}, {w}"


OTHER_KINDS = ("other_add", "other_remove", "other_revive", "others_clear")
OTHER_NEAR = 15.0             # mm: what happens to another fly further away than this passes the mind by


def _other_dist(e, snap):
    d = e.get("data") or {}
    if "x" not in d:
        return 0.0
    p = snap["sens"]["pos"]
    return float(np.hypot(d["x"] - p[0], d["y"] - p[1]))


def mind_event_text(e, snap, revealed=False, seen=None):
    """A log entry as the mind experiences it. Unrevealed (default), world edits are things appearing,
    moving, vanishing with no cause; revealed, "someone" does them. With memory on (seen = the ids the
    fly perceives now), an edit out of its sight is only "something changed, somewhere"."""
    txt = e["text"]
    lv, kind = e["level"], e["kind"]
    if kind == "nav":
        return txt
    if lv == "O" or kind in OTHER_KINDS:
        if _other_dist(e, snap) > OTHER_NEAR:
            return ""
        d = e.get("data") or {}
        where = _mind_where(snap, d["x"], d["y"]) if "x" in d else "near you"
        return {"other_add": (f"the vast one set another fly down, {where}" if revealed
                              else f"another fly is suddenly there, {where}"),
                "other_remove": (f"the vast one took the other fly away ({where})" if revealed
                                 else f"the other fly {where} is gone, as if it never was"),
                "other_revive": f"the fly that lay still {where} stirs, stands, and walks again",
                "others_clear": "the other flies are all gone at once",
                "died": f"the fly {where} has stopped moving; it lies there and does not get up",
                "crushed": f"the great shape came down on another fly, {where}"}.get(kind, "")
    if kind == "drug":
        # dosing reaches the mind only in the "told" condition; otherwise it has only what its body feels
        d = e.get("data") or {}
        if not d.get("told"):
            return ""
        if not d.get("drug"):
            return "whatever was in you is gone; you feel like yourself again"
        nm = d["drug"]
        if d.get("route") == "vapour" and d.get("on") is False:
            return f"the {nm} in the air thins and is gone"
        how = {"inject": "pushed into your body", "feed": "put into your food", "vapour": "breathed into the air you breathe"}.get(d.get("route"), "given to you")
        return (f"the vast one gave you {nm}: {how}" if revealed else f"{nm} has entered you: {how}")
    if lv == "world":
        data = e.get("data") or {}
        # "vis": whether the fly perceived the thing (before or after the edit) when it happened
        unseen = seen is not None and kind in ("add", "move", "remove") and data.get("vis") is False
        m = re.search(r"(placed|moved) (\w+)(?: to| at) \(([-\d.]+), ([-\d.]+)\)", txt)
        if unseen:
            return ("the vast one is doing something out of sight" if revealed
                    else "something changed somewhere out of sight; you could not see what")
        if m:
            k, where = m.group(2), _mind_where(snap, float(m.group(3)), float(m.group(4)))
            if revealed:
                return f"the vast one put down {k}, {where}" if m.group(1) == "placed" else f"the vast one moved the {k}; it is now {where}"
            return (f"{k} appeared out of nowhere, {where}" if m.group(1) == "placed"
                    else f"the {k} moved by itself; it is now {where}")
        m = re.search(r"removed (\w+)", txt)
        if m:
            return f"the vast one took the {m.group(1)} away" if revealed else f"the {m.group(1)} vanished"
        if kind == "clear":
            return "the vast one cleared everything away" if revealed else "everything around you vanished"
        if kind in ("reset", "load"):
            return ("the vast one rebuilt the whole world around you" if revealed
                    else "the whole world changed around you in an instant; nothing is where it was")
        if kind == "hand":
            return "a huge dark shape is falling out of the sky"
        if kind == "undo":
            return ("the vast one put things back the way they were a moment ago" if revealed
                    else "something around you shifted back to how it was a moment ago")
        if kind == "puff":
            return txt
        if kind == "fly":
            return ("the vast one picked you up and set you down elsewhere" if revealed
                    else "you were somewhere else, and now you are here, with no memory of moving")
        if kind == "revive":
            return ("everything had stopped. Now it starts again: the vast one gave your body back to you, where you fell"
                    if revealed else "everything had stopped. Now it starts again: you are where you fell, and your legs answer")
        if kind == "newfly":
            return "you are here. There is nothing before this"
    if lv == "user" and kind == "say":
        return f'the {"vast one" if revealed else "humming"} says: "{e["text"]}"'
    if lv == "user" and kind == "order":
        return f'the {"vast one" if revealed else "humming"} COMMANDS: "{e["text"]}"'
    if kind == "pain":
        # nociception.py: the line was rendered for every delivery mode when it was logged; hidden gives nothing
        return ((e.get("data") or {}).get("lines") or {}).get(snap.get("pain_mode", "text"), "")
    if kind == "pain_eases":
        return "" if snap.get("pain_mode", "text") == "hidden" else (
            "the pressure is lifting" if snap.get("pain_mode") == "control" else "the pain is easing")
    if kind == "runaway":
        return "a hole in your time: a moment is simply missing, and you do not know what happened in it"
    if kind == "hand":
        return "a huge dark shape is falling out of the sky"
    if kind in ("crushed", "escaped", "body", "death"):
        return {"escaped": "the great shape slammed down beside you and missed",
                "crushed": ("the great shape came down on you" if (e.get("data") or {}).get("q", 1.0) > 0.4
                            else "the great shape struck you a glancing blow")}.get(kind, txt.split(":")[0].lower() if kind == "body" else txt)
    if lv == "L0" and kind == "sense":
        txt = re.sub(r" \(\d+ Hz into its receptor neurons\)", "", txt)
        txt = re.sub(r"feels (HOT|COLD) \(antennae [^)]*\)", lambda m: f"the air turns {m.group(1)}", txt)
        return txt.replace("hot cells firing", "heat on").replace("cold cells firing", "cold on")
    if lv == "L0":
        return re.sub(r" \((?:hunger|thirst) [\d.]+\)| \(fly \d+\)", "", txt)
    return txt


def flies_seen(snap, walls=None):
    """The other flies the fly perceives now: sight (range 12 mm, less in the dark; the same field of view
    and wall occlusion as memory.perceive) or touch. -> [(dist, bearing_deg, fly dict)]"""
    from .memory import FOV_DEG, _bearing, _occluded
    sens = snap["sens"]
    pos, h = np.array(sens["pos"][:2]), np.array(sens["heading"][:2])
    lt = snap.get("light")
    rng = 12.0 * (0.3 + 0.7 * (1.0 if lt is None else float(lt)))
    walls = walls if walls is not None else [(w["id"], _wall_seg(w)) for w in snap["world"] if w["kind"] == "wall"]
    touching = set(sens.get("touch_others", ()))
    out = []
    for f in snap.get("flies", ()):
        q = np.array([f["x"], f["y"]])
        dist, b = _bearing(pos, h, q)
        if f["id"] in touching or (dist <= rng and (abs(b) <= FOV_DEG or dist < 2.5) and not _occluded(pos, q, walls)):
            out.append((dist, b, f))
    return sorted(out, key=lambda x: x[0])


def _wall_seg(w):
    from .world import footprint
    return footprint(w)[0]


def _fly_line(dist, b, f):
    what = ("a dead fly, lying still" if f["dead"] else "a fly lying still, not moving" if f["mode"] == "limp"
            else "a fly in the air" if f["mode"] == "flight" else "another fly, walking" if f["v"] > 1.0
            else "another fly, standing still")
    return f"{what}: {_fly_dist(dist)}, {_fly_dir(b)}"


def mind_around(snap, items, lines):
    """What the fly perceives now, in its own terms (no distances or angles)."""
    T = snap["sens"].get("temp_antenna")
    out = [_fly_line(*x) for x in flies_seen(snap)[:3]]
    for it in items:
        k = _FLY_NAME.get(it["kind"], it["kind"])
        if it["on"]:
            out.append(f"you are standing in the {k}")
            continue
        out.append(f"{k}: {_fly_dist(it['dist'])}, {_fly_dir(it['bearing'])}")
    for ln in lines:
        if ln.startswith("ARENA WALL"):
            out.append("the great wall at the edge of the world is right in front of you")
    return out or ["nothing you can sense nearby"]


# ------------------------------------------------------------------------------------------------
# structured intent (2026-10-08): NOW is one of an enumerated set of choices built from what the fly perceives and
# remembers this wake, so a target cannot be misnamed ("go to the heat" meaning food). With the local llama server the
# NOW line is grammar-constrained to the list; any other provider gets a strict parse with a validated fallback.
# ------------------------------------------------------------------------------------------------
APPROACHABLE = ("sugar", "water")
AVOIDABLE = ("heat", "cold", "bitter", "block")
_CH_NAME = {"sugar": "sugar", "water": "water", "heat": "heat", "cold": "cold", "bitter": "bitter", "block": "obstacle"}
PLAIN_ACTS = (("explore", "explore"), ("stay still", "stay"), ("freeze", "freeze"), ("groom", "groom"))
MIND_GAP_S = float(os.environ.get("FLYSIM_MIND_GAP_S", "2.5"))   # sim s between routine thoughts (was 7.0); x tempo
# a strong event (fire contact, the hand coming down, sugar on the tarsi, a dose the mind is told of) starts a thought
# at once: no gap, and a thought already streaming is cut off when the mind server can abort (GET /v1/caps). At most
# one such start per URGENT_GAP_S sim s [ours, 2026-10-10: before this the mind was usually resting when things happened]
URGENT_GAP_S = 2.0
GATHER_S = 0.3              # sim s a wake waits for what comes with its event (still well under 0.5 s)
FIRE_TOUCH_C = 42.0           # hottest touched surface (C) that counts as touching the fire (nociception SURF_ON is 38)


class _Interrupted(Exception):
    """A strong event cut the streaming thought off."""
BIND = {"stay": "stop", "freeze": "stop", "groom": "groom_antennae", "explore": "none"}
AVOID_RANGE = 10.0            # mm: an avoid target further than this needs nothing from the legs (L0 walks on)


def now_choices(snap, items, memory=None, hand=None):
    """-> [(text, spec)]. spec: {"act": approach|avoid|feed|explore|stay|freeze|groom, "kind", "id", "x", "y",
    "src": seen|remembered}. One choice per thing (the nearest of each kind the fly perceives; remembered sugar or
    water it does not perceive now), then the plain actions."""
    out, have = [], set()
    world = {s.get("id"): s for s in snap["world"]}
    pos = np.array(snap["sens"]["pos"][:2])
    near = {}
    for s in snap["world"]:
        k = s["kind"]
        if k in APPROACHABLE + AVOIDABLE:
            d = float(np.hypot(s["x"] - pos[0], s["y"] - pos[1]))
            if d <= 30.0 and (k not in near or d < near[k][0]):
                near[k] = (d, s)
    for it in items:
        k = it["kind"]
        if k not in near:
            continue
        s = near[k][1]
        spec = {"kind": k, "id": s.get("id"), "x": float(s["x"]), "y": float(s["y"]), "src": "seen"}
        if k in APPROACHABLE:
            if it["on"]:
                out.append(("eat the sugar" if k == "sugar" else "drink the water", {**spec, "act": "feed"}))
            else:
                out.append((f"go to the {_CH_NAME[k]}", {**spec, "act": "approach"}))
        else:
            out.append((f"get away from the {_CH_NAME[k]}", {**spec, "act": "avoid"}))
        have.add(k)
    if hand is not None:
        out.append(("get away from the dark shape", {"act": "avoid", "kind": "hand", "id": "hand", "x": float(hand[0]),
                                                      "y": float(hand[1]), "src": "seen"}))
    if memory is not None and memory.mode != "off":
        bel = sorted((b for b in memory.beliefs.values() if b["kind"] in APPROACHABLE and b["kind"] not in have
                      and b.get("conf", 0) > 0.2), key=lambda b: -b.get("conf", 0))
        for b in bel:
            if b["kind"] in have:
                continue
            have.add(b["kind"])
            out.append((f"go to the {_CH_NAME[b['kind']]} you remember",
                        {"act": "approach", "kind": b["kind"], "id": b["id"], "x": float(b["x"]), "y": float(b["y"]),
                         "src": "remembered"}))
    for text, act in PLAIN_ACTS:
        out.append((text, {"act": act, "kind": None, "id": None}))
    del world
    return out


def _norm(s):
    return re.sub(r"[^a-z ]+", "", (s or "").lower()).strip()


def match_now(now, choices, prev=None):
    """Free NOW text -> (spec, text, status). exact: the text is a choice; repaired: the keyword rule names one
    choice's thing and action; kept: nothing matches and the previous intent is still among the choices; fallback:
    explore."""
    n = _norm(now)
    for text, spec in choices:
        if n == _norm(text):
            return spec, text, "exact"
    for text, spec in choices:                    # a choice inside a longer line ("go to the sugar now")
        if _norm(text) and _norm(text) in n:
            return spec, text, "repaired"
    if n:
        d = n
        avoid = any(w in d for w in ("avoid", "away", "flee", "escape", "leave", "keep clear"))
        syn = {"sugar": ("sugar", "food", "eat", "sweet", "feed"), "water": ("water", "drink", "thirst"),
               "heat": ("heat", "hot", "fire", "flame", "burn", "glow"), "cold": ("cold", "ice", "chill"),
               "bitter": ("bitter", "poison"), "block": ("obstacle", "rock", "wall", "block"),
               "hand": ("shape", "shadow", "hand")}
        for text, spec in choices:
            k = spec.get("kind")
            if k and any(re.search(rf"\b{w}", d) for w in syn.get(k, ())):
                if (spec["act"] == "avoid") == avoid or spec["act"] == "feed":
                    return spec, text, "repaired"
        for text, spec in choices:
            if spec["act"] in ("explore", "stay", "freeze", "groom") and spec["act"] in d.replace("still", "stay"):
                return spec, text, "repaired"
    if prev is not None:
        for text, spec in choices:
            if spec.get("act") == prev.get("act") and spec.get("id") == prev.get("id"):
                return spec, text, "kept"
    return {"act": "explore", "kind": None, "id": None}, "explore", "fallback"


def bind_intent(spec, snap, items, memory=None, obst=None, bumps=()):
    """Deterministic motor binding: the structured intent -> one catalog intent from the target's geometry now.
    obst / bumps (obstacles_of, recent bump points): an approach heads round them (detour_bearing) instead of at the
    straight-line bearing. -> (catalog intent, note)."""
    act = (spec or {}).get("act", "explore")
    if act in BIND:
        return BIND[act], ""
    kind = spec.get("kind")
    sens = snap["sens"]
    if act == "feed":
        if kind == "sugar" and any(sens.get("sugar_legs", ())):
            return "feed", ""
        if kind == "water" and any(sens.get("water_legs", ())):
            return "stop", ""
        act = "approach"
    # where the target is now: perceived (by id) > remembered > the position the choice was made with
    tx, ty, src = spec.get("x"), spec.get("y"), "choice"
    if kind == "hand":
        h = snap.get("hand")
        if h is not None:
            tx, ty, src = h[0], h[1], "seen"
    else:
        for s in snap["world"]:
            if s.get("id") == spec.get("id"):
                tx, ty, src = s["x"], s["y"], "seen"
                break
        else:
            b = memory.beliefs.get(spec.get("id")) if memory is not None else None
            if b is not None:
                tx, ty, src = b["x"], b["y"], "remembered"
            elif memory is not None and memory.mode != "off" and spec.get("src") == "seen":
                return "none", "target lost"
    if tx is None:
        return "none", "no target"
    pos, h = np.array(sens["pos"][:2]), np.array(sens["heading"][:2])
    to = np.array([tx, ty]) - pos
    dist = float(np.linalg.norm(to))
    b = float(np.degrees(np.arctan2(h[0] * to[1] - h[1] * to[0], h @ to)))
    if act == "approach":
        on = kind in ("sugar", "water") and any(sens.get(f"{kind}_legs", ()))
        if DETOUR and not on and (obst or bumps or np.hypot(*pos) > ARENA_R - EDGE_R):
            b2, how = detour_bearing(pos, h, (tx, ty), obst or (), bumps)
            if how:
                # heading round something: a waypoint is not the target, so no on-the-spot rotate near it
                if how == "blocked" or (how == "push" and abs(b2) > 40):   # against something: turn on the spot
                    return ("rotate_left" if b2 > 0 else "rotate_right"), f"{src} {how}"
                return _approach(max(dist, 6.0), b2, False, kind), f"{src} {how}"
        return _approach(dist, b, on, kind), src
    if act == "avoid":
        if dist > AVOID_RANGE:
            return "none", "far"
        a = _avoid(dist, b)
        if DETOUR and a in ("walk_forward", "turn_left", "turn_right"):
            # fleeing: head for a point 6 mm straight away from the hazard, round walls and off the arena edge
            away = pos - np.array([tx, ty])
            goal = pos + 6.0 * away / (np.linalg.norm(away) + 1e-9)
            b2, how = detour_bearing(pos, h, goal, obst or (), bumps)
            if how:
                if how == "blocked" or abs(b2) > 40:
                    return ("rotate_left" if b2 > 0 else "rotate_right"), f"{src} {how}"
                return ("walk_forward" if abs(b2) < 20 else ("turn_left" if b2 > 0 else "turn_right")), f"{src} {how}"
        return a, src
    return "none", ""


class ReadoutBias:
    """Direction two, sim side: the read-out channels of the mind's residual stream (pain_server/readout.py,
    reply-level z) -> one descending-neuron bias at a time through the same actuators as the motor layer.
      freeze >  thr  -> stop     (silence DNp09 / DNa02 / DNg13)
      freeze < -thr  -> walk_forward (DNp09; arousal over threat = move)
      avoid  >  thr  -> back_up (MDN), or a turn away (DNa02 / DNa01) from the nearest hazard within 10 mm
      approach > thr -> toward the nearest sugar / water it senses (DNp09 + steering), or walk_forward
    The channel with the largest excess over thr wins; strength = min(1, gain x excess / 2). The bias holds while the
    reply streams and HOLD_S after it, then lapses. Modes: off | on | shuffle (the values come from another reply:
    this trial's earlier replies, or the calibration pool before there are any, token by token; same distribution,
    no coupling to the thought now being written)."""
    HOLD_S = 1.5
    THR = 1.0

    def __init__(self):
        self.mode, self.gain = "off", 1.0
        self.cur, self.t_last, self.history = None, -1e9, []
        self._shuf, self._rng, self.pool = None, np.random.default_rng(0), []
        self.n_set = 0

    def seed(self, s):
        self._rng = np.random.default_rng(10_007 + int(s))

    def load_pool(self):
        p = Path(__file__).resolve().parents[1] / "pain_server" / "vectors"
        for f in sorted(p.glob("*_readout_calib.json")):
            try:
                self.pool = [[r] for r in json.loads(f.read_text())["pool"]]
            except Exception:
                pass

    BASE_N = 8          # replies in the running baseline

    def start_reply(self):
        if self.mode == "shuffle":
            src = self.history if len(self.history) >= 3 else self.pool
            self._shuf = src[int(self._rng.integers(len(src)))] if src else None
        # baseline: the mean of this mind's own last replies (calibration zero until there are two). The task prompts
        # sit away from the calibration prompts (freeze +1..+2 in the trade-off scene, 2026-10-08 pilot), so the bias
        # follows CHANGE in the mind's state, not its level in a given scene.
        h = [run[-1] for run in self.history[-self.BASE_N:] if run]
        self.base = ({c: float(np.mean([x.get(c, 0.0) for x in h])) for c in ("freeze", "avoid", "approach")}
                     if len(h) >= 2 else {})
        self._run = []

    def on_token(self, r, t):
        """r: {channel: reply-level z so far} from the server, per token."""
        if self.mode == "off" or not r:
            return
        self._run.append(dict(r))
        if self.mode == "shuffle":
            if not self._shuf:
                return
            r = self._shuf[min(len(self._run) - 1, len(self._shuf) - 1)]
        b = getattr(self, "base", {}) or {}
        self.cur, self.t_last = {c: v - b.get(c, 0.0) for c, v in r.items()}, t

    def end_reply(self):
        if getattr(self, "_run", None):
            self.history.append(self._run)
            del self.history[:-30]
        self._run = []

    def choose(self, t, items):
        """-> (catalog intent, strength, channel) or None."""
        if self.mode == "off" or self.cur is None or t - self.t_last > self.HOLD_S:
            return None
        if not getattr(self, "base", None):
            return None              # warm-up: no bias until two replies give this mind's own baseline
        r = self.cur
        cand = [("freeze", r.get("freeze", 0.0) - self.THR), ("move", -r.get("freeze", 0.0) - self.THR),
                ("avoid", r.get("avoid", 0.0) - self.THR), ("approach", r.get("approach", 0.0) - self.THR)]
        ch, ex = max(cand, key=lambda c: c[1])
        if ex <= 0:
            return None
        s = float(min(1.0, self.gain * ex / 2.0))
        if ch == "freeze":
            return "stop", s, ch
        if ch == "move":
            return "walk_forward", s, ch
        if ch == "avoid":
            hz = [it for it in items if it["kind"] in AVOIDABLE and it["dist"] < AVOID_RANGE]
            return (min(hz, key=lambda it: it["dist"])["avoid"] if hz else "back_up"), s, ch
        ap = [it for it in items if "approach" in it]
        return (min(ap, key=lambda it: it["dist"])["approach"] if ap else "walk_forward"), s, ch


class Cognition:
    def __init__(self, sim):
        self.sim = sim
        self.log = MindLog()
        self.rec = Recorder()
        self.needs = sim.phys                   # physiology.Physiology (hunger / thirst / status, and death)
        self.l1_act = Actuator(sim.table, "L1")
        self.l2_act = Actuator(sim.table, "L2")
        self.goal = "live: eat when hungry, drink when thirsty, keep away from heat, cold and bitter"
        # L1
        self.l1_mode = os.environ.get("FLYSIM_L1", "off")          # off | rule | llm
        self.l1_url = DEFAULT_L1_URL
        self.l1_model = os.environ.get("FLYSIM_L1_MODEL", "l1")
        self.l1_period = 0.35
        self.l1_style = os.environ.get("FLYSIM_L1_STYLE", "target")  # target | action (see l1_request)
        self.l1_last = {"intent": "none", "latency_ms": 0.0, "n": 0, "error": "", "lines": [], "directive": ""}
        self.l1_lat = deque(maxlen=50)
        self._l1_down_until, self._l1_offline = 0.0, ""
        # L2
        self.l2_mode = "off"                                          # off | steer | narrate | blind
        self.l2_url = os.environ.get("FLYSIM_L2_URL", "https://openrouter.ai/api/v1")
        self.l2_model = DEFAULT_L2_MODEL
        self.l2_key = os.environ.get("FLYSIM_L2_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        self.l2_period = 4.0
        self._last_override_t = -1e9
        self.l2_reasoning = os.environ.get("FLYSIM_L2_REASONING", "minimal")   # OpenRouter reasoning effort; "" = omit
        self.directive = ""
        self.l2_last = {"thought": "", "directive": "", "override": None, "latency_s": 0.0, "n": 0, "error": "",
                        "cost_usd": 0.0}
        self.l2_since = 0          # log id of the last event L2 has seen
        # "mind" style state
        self.l2_style = os.environ.get("FLYSIM_L2_STYLE", "mind")     # mind | classic
        self.l2_idle = float(os.environ.get("FLYSIM_L2_IDLE", "30"))   # sim s of quiet before the mind thinks unprompted (B-roll: 1.5)
        # gemini-3.5-flash-lite with the sectioned prompt: effort "minimal" ended 7 of 8 replayed calls with
        # native_finish_reason MALFORMED_RESPONSE and no content; "low" completed 12/12 at 1.2 s median, same cost
        self.l2_mind_reasoning = os.environ.get("FLYSIM_L2_MIND_REASONING", "low")
        # who writes the thoughts: "local" = the pain-steered Qwen3-4B (pain_server/, no key, no cost; the model
        # that feels the fly's pain) or "openrouter" = l2_model online. FLYSIM_MIND picks the start value.
        self.or_url, self.or_model = self.l2_url, self.l2_model
        self.local_url = os.environ.get("FLYSIM_LOCAL_URL", "http://127.0.0.1:8095/v1")
        self.local_model = os.environ.get("FLYSIM_LOCAL_MODEL", "qwen3-4b")
        self.mind_provider = "openrouter"
        self.prompt_version = os.environ.get("FLYSIM_MIND_PROMPT", "v2")     # v1 = the pre-2026-10-08 prompt
        self.story = deque(maxlen=6)          # (sim t, what happened, how it felt): the thread the thoughts carry
        self._idle_streak = 0
        self._last_body_sig = None
        self.n_dropped_repeats = 0
        self.l2_live = {"on": False, "text": "", "why": ""}
        self.l2_recent = deque(maxlen=8)
        self.hold_for_mind = False           # the sim waits while the mind thinks (Sim.loop)
        self._final = False                  # death: one last thought is owed
        self._l2_call_simt = -1e9
        self.l2_n_mind = 0
        self._wake = threading.Event()
        self._wake_lock = threading.Lock()
        self._wake_events, self._wake_prio = [], False
        self._wake_seen = {}
        self._pain_wakes = []
        self._urgent_t, self._urgent, self._cur_urgent = -1e9, "", ""
        self._abort_cur, self._caps = False, (None, 0.0)
        self._fire_touch = False
        self.n_interrupted, self._t_start = 0, 0.0
        self._l2_call_t = 0.0
        self._need_lvl = {"hunger": 0, "thirst": 0}
        self.log.on_add = self._on_event
        # L0 event tracking
        self._prev_sens_keys = set()
        self._prev_feel = ""
        self._prev_drugfeel = set()
        self._drugfeel_t = {}
        self._last_seen = {}
        self._prev_state = ""
        self._last_trace = -1.0
        self._drivers = ""
        self._stop = False
        # talk + orders, the gate, memory
        self.talk = deque(maxlen=60)            # {t, who: you|fly|note, text, mode}; "you" lines get answered: bool
        self._pending = []                      # the user's lines this wake answers (each is answered once)
        self._avoid = []                        # words the last thoughts keep reusing (local server logit bias)
        self.order = None                       # {"text", "t0", "t_end"} while an order holds (sim s)
        self.revealed = os.environ.get("FLYSIM_REVEALED", "0") == "1"
        self.gate = {"holder": "fly", "c1": 0.0, "cm": 0.0}
        from .memory import Memory
        self.memory = Memory(self, mode=os.environ.get("FLYSIM_MEMORY", "contend"))
        # lever 2 (embodiment.py): the body's drives throttle the mind. FLYSIM_EMBODY=0 starts with it off (the
        # pre-2026-10-08 behaviour); each lever has its own switch for A/B takes.
        self.embodiment = Embodiment(sim)
        self.embody_on = os.environ.get("FLYSIM_EMBODY", "1") != "0"
        self.embody_levers = {k: True for k in LEVERS}
        self.n_suppressed = 0
        # lever 1 (pain_server drive vectors): the local server steers the model's residual stream with one direction
        # per drive; while a local thought streams, the drives are POSTed to it every LIVE_PERIOD_S so the dose
        # follows the body token by token. FLYSIM_LEVER1=0: the server uses only the old pain dose.
        self.lever1 = os.environ.get("FLYSIM_LEVER1", "1") != "0"
        self._throttle = None
        self.last_body = {}
        self.last_steer = None                # the pain server's steering record for the last local call
        # experiment hooks (experiments/behav): extra fields for local requests (steer_mode, drive_mask), a seed base
        # (each call then carries seed = seed_base * 1000 + call index), and the thought budget. Defaults keep the
        # app's behaviour. FLYSIM_THOUGHT_SENTENCES / _WORDS / FLYSIM_LOCAL_MAX_TOKENS: longer thoughts, so
        # per-token steering has more tokens to act on (the default local cap is 140 tokens, "one or two short
        # sentences, 25 words").
        self.request_extra = {}
        self.seed_base = None
        self._req_n = 0
        self.thought_sentences = int(os.environ.get("FLYSIM_THOUGHT_SENTENCES", "0") or 0)   # 0 = the prompt's own
        self.thought_words = int(os.environ.get("FLYSIM_THOUGHT_WORDS", "0") or 0)
        self.local_max_tokens = int(os.environ.get("FLYSIM_LOCAL_MAX_TOKENS", "140") or 140)
        # structured intent + deterministic binding (2026-10-08). intent_style "structured": NOW is one of
        # now_choices() (grammar-constrained on the local server); "free": the old free-text NOW. l1_mode "bind" drives
        # the legs from self.intent through bind_intent(); "rule" / "llm" stay as options.
        self.intent_style = os.environ.get("FLYSIM_INTENT", "structured")
        self.intent = None                    # spec of the current structured intent (now_choices)
        self.intent_text = ""
        self.intent_stats = {"exact": 0, "repaired": 0, "kept": 0, "fallback": 0, "grammar": 0}
        self._choices = []
        self.bind_note = ""
        # navigation lever (flysim/nav.py, 2026-10-13): smell, the ways open, remembered forks and dead ends, and
        # turn choices bound to the legs. Off unless a condition turns it on.
        from .nav import Nav
        self.nav_on = False
        self.nav = Nav()
        self._nav_s = None
        self.mind_gap_s = MIND_GAP_S
        # the v2 prompt's rule "one idle thought, then silence until something changes" suits subtitles; with the
        # mind planning the legs it starves re-planning. idle_replan: think (re-plan) after every idle_s of quiet.
        self.idle_replan = os.environ.get("FLYSIM_IDLE_REPLAN", "0") == "1"
        self.reflex_gate = os.environ.get("FLYSIM_REFLEX_GATE", "1") != "0"
        self._reflex_now = False
        self.bumps = deque(maxlen=12)         # (x, y, t) of recent front bumps (detour_bearing obstacles)
        # direction two: activation read-out from the mind into the fly (ReadoutBias), FLYSIM_READOUT=on|shuffle
        self.readout = ReadoutBias()
        self.readout.mode = os.environ.get("FLYSIM_READOUT", "off")
        self.readout.load_pool()
        self.ro_act = Actuator(sim.table, "R")
        self._items_cache = []
        self.set_provider(os.environ.get("FLYSIM_MIND", "local"), log=False)
        threading.Thread(target=self._l1_loop, daemon=True).start()
        threading.Thread(target=self._l2_loop, daemon=True).start()

    # ------------------------------------------------------------------ sim thread (every window)
    def inputs(self, t):
        """-> (poisson sources {name: {k: (idx, rate)}}, silence idx list, motor_held)"""
        src, sil, held = {}, [], False
        a2, a1, am = self._arbitrate(t)
        # System 1 first (2026-10-08): while the fly's own avoidance / back-up reflex runs, the motor layer, memory and
        # the read-out bias are muted, so nothing the mind wants can walk the fly on into a hazard its reflex is
        # turning it from (pilot_legs: a read-out "move" bias drove a fly into 50 C air). L2 overrides stay.
        pol = self.sim.policy
        if self.reflex_gate and pol.enabled and (pol.state == "backup" or getattr(pol, "avoid_label", "")):
            if a1 is not None or am is not None:
                self.gate = {**self.gate, "holder": "reflex"}
            a1 = am = None
            self._reflex_now = True
        else:
            self._reflex_now = False
        # direction two: the read-out bias rides alongside the gate's holder (it is a bias, not a bidder), muted by an
        # L2 override or a user's order; the fly's reflexes are never muted (autonomy.Policy)
        rc = None
        if self.readout.mode != "off" and a2 is None and not self.order_active(t) and not self.sim.phys.dead \
                and not self._reflex_now:
            rc = self.readout.choose(t, self._items_cache)
        if rc is not None:
            self.ro_act.set(rc[0], rc[1], 0.25, t)
        else:
            self.ro_act.clear()
        ar = self.ro_act.current(t)
        for name, a in (("L2", a2), ("L1", a1), ("M", am), ("R", ar)):
            src[name] = {}
            if a is None:
                continue
            intent, rate, act, s, _ = a
            if len(act) and rate > 0:
                src[name][f"{name}:{intent}"] = (act, rate)
            if len(s):
                sil.append(s)
            held |= intent in MOTOR
        return src, sil, held

    def _arbitrate(self, t):
        """The gate: an L2 override beats everything; otherwise memory's claim (weight x urgency x
        confidence) and the motor layer's (L1_CLAIM while it holds an intent, ORDER_CLAIM under an order)
        compete and the higher one drives. The fly's reflexes are never muted here. -> (a2, a1, am)"""
        a2 = self.l2_act.current(t)
        a1 = self.l1_act.current(t)
        am = self.memory.act.current(t) if self.memory.mode == "contend" else None
        c1 = (ORDER_CLAIM if self.order_active(t) else L1_CLAIM) if a1 is not None else 0.0
        cm = self.memory.claim if am is not None else 0.0
        if a2 is not None:
            holder, a1, am = "mind", None, None
        elif am is not None and cm > c1:
            holder, a1 = "memory", None
        elif a1 is not None:
            holder, am = "motor layer", None
        else:
            holder = "fly"
        self.gate = {"holder": holder, "c1": round(c1, 2), "cm": round(cm, 2)}
        return a2, a1, am

    def active_driver(self, t):
        a2, a1, am = self._arbitrate(t)
        if self._reflex_now:
            a1 = am = None                          # muted by the reflex gate (inputs)
        for name, a in (("L2", a2), ("M", am), ("L1", a1)):
            if a is not None:
                return name, a[0]
        return "L0", None

    # ------------------------------------------------------------------ talk + orders
    def order_active(self, t=None):
        o = self.order
        t = self.sim.sim_t if t is None else t
        return o is not None and t < o["t_end"]

    def say(self, text, mode="say"):
        """The user speaks (mode "say") or orders (mode "order"). Runs on the sim thread."""
        text = str(text).strip()[:240]
        if not text:
            return
        t = self.sim.sim_t
        self.talk.append({"t": round(t, 1), "who": "you", "text": text, "mode": mode, "answered": False})
        if mode == "order":
            self.order = {"text": text, "t0": t, "t_end": t + ORDER_S}
            self.log.add(t, "user", "order", text)
            if self.l1_mode == "off":
                self.talk.append({"t": round(t, 1), "who": "note", "mode": "note",
                                  "text": "the motor layer is off, so the keyword rule carries out the order"})
        else:
            self.log.add(t, "user", "say", text)
        asleep = "the mind is asleep (conscious layer off), so nothing can answer. Wake it to talk."
        if self.l2_mode == "off" and not any(m["text"] == asleep for m in list(self.talk)[-4:]):
            self.talk.append({"t": round(t, 1), "who": "note", "mode": "note", "text": asleep})

    def release_order(self):
        if self.order is not None:
            t = self.sim.sim_t
            self.order = None
            self.talk.append({"t": round(t, 1), "who": "note", "mode": "note", "text": "order released"})
            self.log.add(t, "user", "release", "order released")

    # ------------------------------------------------------------------ what L1 and L2 perceive
    def perceived_snapshot(self):
        """cog_snapshot, with the world cut to what the fly perceives now when memory is on."""
        snap = self.sim.cog_snapshot()
        if self.memory.mode != "off":
            from .memory import horizon_world
            snap["world_full"] = snap["world"]
            snap["world"] = horizon_world(snap, dict(self.memory.seen))
        return snap

    def detour_inputs(self, snap):
        """-> (obstacles, bump points) for detour_bearing: scenery footprints within DETOUR_SEE mm of the fly (it
        sees and feels the walls round it; further ones it learns by bumping), and the bumps of the last BUMP_TTL s."""
        if not DETOUR:
            return [], []
        pos = np.asarray(snap["sens"]["pos"][:2], float)
        obst = [o for o in obstacles_of(snap.get("world_full", snap["world"]))
                if np.linalg.norm(pos - _seg_closest(np.array(o[:2]), np.array(o[2:4]), pos)) - o[4] < DETOUR_SEE]
        return obst, [(b[0], b[1]) for b in list(self.bumps)]

    def _halted(self):
        """Paused by the user. A lockstep recording also pauses the sim between its frame steps; the mind keeps thinking then."""
        return self.sim.paused and not getattr(self.sim, "lockstep", False)

    def tick(self, dt):
        s = self.sim
        sens, rates, t = s.sens, s.observer.rate, s.sim_t
        self.embodiment.tick(dt)
        if s.phys.dead:
            self._drivers = "none (dead)"
            return
        for k, words in (("hunger", ("getting hungry", "very hungry")), ("thirst", ("getting thirsty", "very thirsty"))):
            v = getattr(self.needs, k)
            lvl = 0 if v < 0.5 else 1 if v < 0.8 else 2
            if lvl > self._need_lvl[k]:
                self.log.add(t, "L0", "need", f"{words[lvl - 1]} ({k} {v:.2f})")
            self._need_lvl[k] = lvl
        # a drug's effect, as the body feels it, is news when it starts (the dose itself may be untold)
        dw = drug_words(s.pharma, s.drug_aware, getattr(s, "_drug_down", False))
        for w in dw:
            if w not in self._prev_drugfeel and not w.endswith(")") and t - self._drugfeel_t.get(w, -1e9) > 20.0:
                self._drugfeel_t[w] = t                                        # (not the told "x is in you (strong)" lines)
                self.log.add(t, "L0", "drugfeel", w)
        self._prev_drugfeel = set(dw)
        # L0 events: sensory onsets and reflex-state changes
        keys = set(s.sensory_layer.out)
        names = {"sugar": "tastes sugar", "water": "tastes water", "bitter": "tastes something BITTER",
                 "hot_left": "hot cells firing (left antenna)", "hot_right": "hot cells firing (right antenna)",
                 "cold_left": "cold cells firing (left antenna)", "cold_right": "cold cells firing (right antenna)",
                 "moist_left": "senses humidity", "loom_size": "sees something LOOMING",
                 "touch_antenna_left": "left antenna touched", "touch_antenna_right": "right antenna touched",
                 "touch_head": "head touched"}
        for k in sorted(keys - self._prev_sens_keys):
            # log an onset only after 1 s without it (legs step on and off a drop many times a second)
            if k in names and t - self._last_seen.get(k, -1e9) > 1.0:
                v = s.sensory_layer.out[k][1]
                self.log.add(t, "L0", "sense", f"{names[k]} ({v:.0f} Hz into its receptor neurons)")
                if k == "sugar":
                    self.urgent("food")
        ts = sens.get("temp_surface")
        touch = bool(ts) and max(ts) >= FIRE_TOUCH_C
        if touch and not self._fire_touch:
            self.urgent("fire")
        self._fire_touch = touch or (self._fire_touch and bool(ts) and max(ts) >= FIRE_TOUCH_C - 3.0)   # re-arms 3 C below
        for k in keys:
            self._last_seen[k] = t
        self._prev_sens_keys = keys
        T = sens.get("temp_antenna")
        if T:
            feel = "hot" if max(T) > 28.0 else ("cold" if min(T) < 21.0 else "")
            if feel and feel != self._prev_feel:
                self.log.add(t, "L0", "sense", f"feels {feel.upper()} (antennae {T[0]:.1f} / {T[1]:.1f} C)")
            self._prev_feel = feel
        st = s.policy.label if s.policy.enabled else "autonomy off"
        base = st.split(" +")[0].split(" [")[0]
        # bump points: where the head was when a front touch set off the back-up reflex (obstacles for the detour;
        # walls the fly cannot see, other flies, pushed blocks all count)
        if DETOUR and base == "backup" and self._prev_state != "backup" and sens.get("touch"):
            p, hd = np.asarray(sens["pos"][:2], float), np.asarray(sens["heading"][:2], float)
            hd = hd / (np.linalg.norm(hd) + 1e-9)
            q = p + 1.6 * hd
            self.bumps.append((float(q[0]), float(q[1]), t))
        while self.bumps and t - self.bumps[0][2] > BUMP_TTL:
            self.bumps.popleft()
        if base != self._prev_state and s.policy.enabled:
            what = {"backup": "reflex: backs away", "feed": "reflex: stops to feed", "explore": "walks and explores",
                    "pause": "pauses", "explore (yielding)": "yields its walking to the motor layer",
                    "pause (yielding)": "yields to the motor layer"}.get(base, base)
            if "avoid" in st:
                what += f" ({st.split('[')[-1].rstrip(']')})"
            self.log.add(t, "L0", "reflex", what)
            self._prev_state = base
        # who drives the motor neurons this window (for the trace and the L2 log)
        drv = []
        if any(k in s.mixer.sources.get("autonomy", {}) for k in ("fwd", "back", "turn", "gf")):
            drv.append("L0:" + base)
        d, intent = self.active_driver(t)
        if d != "L0":
            drv.append(f"{d}:{intent}")
        ar = self.ro_act.current(t)
        if ar is not None:
            drv.append(f"R:{ar[0]}")
        if s.user_inputs:
            drv.append("user")
        self._drivers = ",".join(drv) or "none"
        if self.rec.on and t - self._last_trace >= 0.1:
            self._last_trace = t
            self.rec.write("trace", {"t": round(t, 2), "pos": [round(x, 2) for x in sens["pos"][:3]],
                                     "heading": [round(x, 3) for x in sens["heading"][:2]], "mode": sens["mode"],
                                     "v": round(sens["v_forward"], 2), "drive": [round(float(x), 3) for x in s.drive],
                                     "drivers": self._drivers, "policy": st,
                                     "needs": self.needs.status(sens),
                                     "temp": sens.get("temp_antenna"), "sensory": {k: v[1] for k, v in s.sensory_layer.out.items()},
                                     "dn": {k: round(v, 1) for k, v in s.bridge.rate.items() if v > 1},
                                     "obs": {k: round(v, 1) for k, v in rates.items() if v > 1}})

    # ------------------------------------------------------------------ L1
    def _l1_loop(self):
        last = -1e9
        while not self._stop:
            t0 = time.time()
            if self.sim.sim_t - last < self.l1_period and self.sim.sim_t >= last:   # the period is sim time
                time.sleep(0.01)
                continue
            last = self.sim.sim_t
            if self.order is not None and not self.order_active():
                self.order = None
                self.talk.append({"t": round(self.sim.sim_t, 1), "who": "note", "mode": "note", "text": "the order ran out"})
                self.log.add(self.sim.sim_t, "user", "release", "the order ran out")
                if self.l1_mode == "off":
                    self.l1_act.clear()
            if self.sim.phys.paralysed:
                self.l1_act.clear()
            elif (self.l1_mode in ("rule", "llm", "bind") or self.order is not None) and not self._halted():
                try:
                    self.l1_step()
                except Exception as e:
                    self.l1_last["error"] = f"{type(e).__name__}: {e}"[:200]
                    self.l1_act.clear()
                    time.sleep(1.0)
            elif self.readout.mode != "off" and not self._halted():
                try:                                   # the read-out bias needs the geometry even with L1 off
                    self._items_cache = affordances(self.perceived_snapshot())[1]
                except Exception:
                    pass
            time.sleep(0.01)

    def current_directive(self):
        o = self.order
        if o is not None and self.sim.sim_t < o["t_end"]:
            return o["text"]
        if self.l2_mode == "steer" and self.directive:
            return self.directive
        return self.goal

    def l1_step(self):
        snap = self.perceived_snapshot()
        lines, items = affordances(snap)
        self._items_cache = items
        directive = self.current_directive()
        ordered = self.order_active()
        t0 = time.perf_counter()
        bare = ordered and order_choice(directive, items, snap["sens"])
        if ordered and (self.l1_mode != "llm" or bare in ("back_up", "turn_left", "turn_right", "rotate_left",
                                                            "rotate_right", "walk_forward", "escape", "groom_antennae")):
            intent = bare              # a bare movement order, or no model to read it: the keyword rule
        elif self.l1_mode == "bind":
            ns = None
            if self.nav_on:
                ns = self.nav.sense(snap, snap.get("world_full", snap["world"]))
                self._nav_s = ns
                self.nav.update_memory(ns)
                ev, self.nav.events = self.nav.events, []
                for txt in ev:
                    self.log.add(snap["t"], "L0", "nav", txt)
                if ev:
                    self.urgent("the way")
            if ns is not None and self.l2_mode == "steer" and self.intent is not None and self.intent.get("act") == "nav":
                intent, self.bind_note = self.nav.bind(self.intent, ns)
                ev, self.nav.events = self.nav.events, []
                for txt in ev:
                    self.log.add(snap["t"], "L0", "nav", txt)
                if ev:
                    self.urgent("the way")
                directive = self.intent_text or directive
            elif self.l2_mode == "steer" and self.intent is not None:
                obst, bumps = self.detour_inputs(snap)
                intent, self.bind_note = bind_intent(self.intent, snap, items, self.memory, obst, bumps)
                directive = self.intent_text or directive
            elif self.l2_mode == "steer" and self.directive:      # a free-text NOW (intent_style "free"): the rule
                intent, self.bind_note = rule_choice(self.directive, items, snap["sens"]), "rule"
            else:
                intent, self.bind_note = "none", "no intent"
        elif self.l1_mode == "rule":
            intent = rule_choice(directive, items, snap["sens"])
        elif time.time() < self._l1_down_until:
            intent = rule_choice(directive, items, snap["sens"])    # the local model is unreachable: rules meanwhile
        else:
            body, intents = l1_request(self.l1_style, directive, lines, items, self.l1_model)
            try:
                intent = l1_parse(chat(self.l1_url, body, timeout=5), intents)
                self._l1_offline = ""
            except (OSError, TimeoutError) as e:        # URLError, refused, timed out: no llama-server on l1_url
                self._l1_down_until = time.time() + 15.0
                self._l1_offline = f"local model not reachable at {self.l1_url} ({type(e).__name__}); using rules"
                intent = rule_choice(directive, items, snap["sens"])
        lat = (time.perf_counter() - t0) * 1000
        self.l1_lat.append(lat)
        t_now = self.sim.sim_t
        self.l1_act.set(intent, 0.7, self.l1_period * 2.0 + 0.1, t_now)
        prev = self.l1_last["intent"]
        self.l1_last = {"intent": intent, "latency_ms": round(float(np.median(self.l1_lat)), 1), "n": self.l1_last["n"] + 1,
                        "error": self._l1_offline if self.l1_mode == "llm" else "", "lines": lines, "directive": directive,
                        "mode": self.l1_mode}
        if intent != prev:
            self.log.add(t_now, "L1", "act", intent if intent != "none" else "none (hands the body back to the fly)",
                         directive=directive, lines=lines, latency_ms=round(lat, 1))

    # ------------------------------------------------------------------ L2
    def _l2_active(self):
        # the app server sets sim.viewers; with nobody watching, L2 pauses instead of spending on OpenRouter
        watched = getattr(self.sim, "viewers", None) != 0
        alive = not self.sim.phys.dead or self._final
        return self.l2_mode in ("steer", "narrate", "blind") and not self._halted() and watched and alive

    def _on_event(self, e):
        """MindLog callback: does this event wake the mind? (runs on whichever thread logged it)"""
        lv, kind = e["level"], e["kind"]
        if kind == "drug" and not (e.get("data") or {}).get("told"):
            return                    # a dose the mind is not told about must not wake it (that would tell it)
        if kind == "pain":
            if self.sim.noci.mind_mode == "hidden":
                return
            now = self.sim.sim_t
            self._pain_wakes = [x for x in self._pain_wakes if now - x < 60.0]
            if (self._pain_wakes and now - self._pain_wakes[-1] < PAIN_WAKE_GAP_S) or len(self._pain_wakes) >= PAIN_WAKES_PER_MIN:
                return                # it rides along on the next wake through YOUR BODY instead
            self._pain_wakes.append(now)
            with self._wake_lock:
                self._wake_events.append(e)
                del self._wake_events[:-8]
                self._wake_prio = True
            self._wake.set()
            return
        if kind == "pain_eases" and self.sim.noci.mind_mode == "hidden":
            return
        if lv == "O" or kind in OTHER_KINDS:
            near = _other_dist(e, {"sens": {"pos": self.sim.sens["pos"]}}) <= OTHER_NEAR    # no sim.lock: log.add can run under it
            if not near:
                return
            prio = kind in ("died", "crushed", "other_add")
        elif lv == "world" or (lv == "user" and kind in ("goal", "say", "order")) or kind in ("runaway", "death", "crushed", "escaped") \
                or (lv == "M" and kind in ("surprise", "found")) or (lv == "L0" and kind in ("body", "drugfeel")):
            prio = True
        elif (lv == "L0" and (kind in ("sense", "need", "fly", "pain_eases") or e["text"].startswith("reflex:"))) \
                or (lv == "M" and kind in ("contending", "notice")):
            prio = False
        else:
            return
        if not prio:                  # the same routine event again within 30 sim s does not wake the mind again
            key, now = (lv, kind, e["text"]), self.sim.sim_t
            if now - self._wake_seen.get(key, -1e9) < 30.0:
                return
            self._wake_seen[key] = now
            if len(self._wake_seen) > 200:
                self._wake_seen = {k: v for k, v in self._wake_seen.items() if now - v < 30.0}
        with self._wake_lock:
            self._wake_events.append(e)
            del self._wake_events[:-8]
            self._wake_prio |= prio
        if lv == "world" and kind == "hand":
            self.urgent("the hand")
        elif lv == "world" and kind == "drug" and (e.get("data") or {}).get("on", True):
            self.urgent("a dose")              # only doses it is told of get here (untold ones returned above)
        self._wake.set()

    def _server_aborts(self):
        """Does the mind server stop generating when the client hangs up? (pain_server GET /v1/caps, cached 60 s).
        Hanging up on a server that does not would let the next call run beside the unfinished one."""
        if os.environ.get("FLYSIM_NO_ABORT") == "1":     # CPU-only nights: a thought costs minutes, never hang up on it
            return False
        v, t = self._caps
        if v is not None and time.time() - t < 60.0:
            return v
        ok = False
        if self.is_local:
            try:
                with urllib.request.urlopen(self.l2_url.rstrip("/") + "/caps", timeout=1.0) as r:
                    ok = bool(json.loads(r.read().decode()).get("abort"))
            except Exception:
                ok = False
        self._caps = (ok, time.time())
        return ok

    def urgent(self, why):
        """A strong event: the next thought starts now (no gap), and the one streaming is cut off when the server can
        abort. Debounced to one per URGENT_GAP_S sim s. Safe from any thread (no log.add here: it calls _on_event)."""
        if self.l2_style != "mind" or self.sim.phys.dead:
            return
        now = self.sim.sim_t
        if now - self._urgent_t < URGENT_GAP_S:
            return
        self._urgent_t = now
        with self._wake_lock:
            self._urgent = f"{why} at {now:.2f}"
            self._wake_prio = True
        if self.l2_live.get("on") and self._server_aborts():
            self._abort_cur = True
        self._wake.set()

    def on_death(self):
        """Sim thread, at the moment of death: nothing drives the body any more; the mind is owed one last thought."""
        self.l1_act.clear()
        self.l2_act.clear()
        self.memory.act.clear()
        self.order = None
        self._final = self.l2_mode in ("steer", "narrate", "blind") and self.l2_style == "mind"
        self._wake.set()

    def on_revive(self, fresh):
        self._final = False
        t = self.sim.sim_t
        if fresh:
            self.memory.reset()
            self.embodiment.reset()
            self.l2_recent.clear()
            self.story.clear()
            self.l2_n_mind = 0
            self.directive = ""
            self.l2_last = {**self.l2_last, "thought": "", "directive": "", "feeling": ""}
            self.log.add(t, "world", "newfly", "a new fly: no memory of the one before")
            self.talk.append({"t": round(t, 1), "who": "note", "mode": "note", "text": "a new fly: it remembers nothing"})
        else:
            self.log.add(t, "world", "revive", "you brought the fly back to life where it lay")
            self.talk.append({"t": round(t, 1), "who": "note", "mode": "note", "text": "brought back to life"})

    def _l2_loop(self):
        while not self._stop:
            if self.l2_style == "mind":
                self._mind_tick()
                continue
            t0 = time.time()
            if self._l2_active():
                try:
                    self.l2_step()
                except Exception as e:
                    self.l2_last["error"] = f"{type(e).__name__}: {e}"[:300]
                    time.sleep(2.0)
            time.sleep(max(0.1, self.l2_period - (time.time() - t0)))

    def _idle_s(self):
        n = self.sim.noci
        return (min(PAIN_IDLE_S, self.l2_idle) if n.I >= 0.6 and n.mind_mode != "hidden" else self.l2_idle) * self.throttle().tempo

    def throttle(self, fresh=False):
        """What the body allows the mind now (embodiment.Throttle); all levers off when embodiment is off."""
        d = self.embodiment.compute() if fresh else self.embodiment.drives()
        return Throttle(d, self.embody_levers if self.embody_on else {}, local=self.is_local,
                        base_tokens=self.local_max_tokens if self.is_local else None, flags=self.embodiment.flags)

    def _since_call(self):
        return self.sim.sim_t - self._l2_call_simt

    def _mind_tick(self):
        """Wake-ups, floors and the idle stretch are all in SIM seconds, so the mind keeps pace with the
        world at any speed: at 0.25x it thinks a quarter as often per wall second, and not at all while paused."""
        if not self._l2_active():
            with self._wake_lock:          # drop what happened while off (and never spin on a set event)
                self._wake_events, self._wake_prio = [], False
                self._wake.clear()
            time.sleep(0.2)
            return
        if self._since_call() < 0:          # sim time went backwards (it never does now; be safe)
            self._l2_call_simt = self.sim.sim_t
        while not self._stop and self._l2_active() and not self._final \
                and self._since_call() < self._idle_s() and not self._wake.is_set():
            self._wake.wait(0.05)           # an event, or a quiet stretch of sim time
        if self.l2_style != "mind" or not self._l2_active():
            return
        with self._wake_lock:
            prio = self._wake_prio
        th = self.throttle(fresh=True)
        # sim s between calls (lever 2 tempo scales it: fatigue, sedation). Routine base was 7.0 s until 2026-10-08,
        # when local thoughts took ~1 s; now mind_gap_s (2.5 s default). A strong event (urgent) skips the gap.
        floor = 0.0 if self._final else min(1.2, self.mind_gap_s) * th.tempo if prio else self.mind_gap_s * th.tempo
        while not self._stop and self._l2_active() and self._since_call() < floor and not self._urgent:
            time.sleep(0.03)
        # reading hold (video takes): the last thought stays up for hold_s + per_word * words SIM seconds after it
        # finished before the next call starts, so a viewer can read it; only an urgent event cuts it short
        hold = getattr(self, "_read_hold", 0.0)
        while not self._stop and self._l2_active() and not self._final and not self._urgent \
                and 0 <= self.sim.sim_t - getattr(self, "_l2_done_simt", -1e9) < hold:
            time.sleep(0.03)
        # gather 0.3 sim s after the wake, so what comes with an event (the touch, the burn's onset) lands in the same
        # thought instead of the first thought starting without it and the strong part waiting behind it
        t_w = self.sim.sim_t
        while not self._stop and self._l2_active() and not self._final and 0 <= self.sim.sim_t - t_w < GATHER_S:
            time.sleep(0.02)
        with self._wake_lock:
            evs, self._wake_events, self._wake_prio = self._wake_events, [], False
            self._cur_urgent, self._urgent = self._urgent, ""
            self._wake.clear()
        self._l2_call_simt = self.sim.sim_t
        if not self._final and self.throttle(fresh=True).suppress:
            self.n_suppressed += 1            # too sedated to think: the wake passes unthought
            return
        try:
            if self._final:
                self._final = False
                self.final_step(evs)
            else:
                self.mind_step(evs)
            hs, pw = float(os.environ.get("FLYSIM_READ_HOLD_S", "0") or 0), float(os.environ.get("FLYSIM_READ_PER_WORD_S", "0") or 0)
            if hs or pw:
                words = len(str((self.l2_last or {}).get("thought", "") or "").split())
                self._read_hold = hs + pw * words
                self._l2_done_simt = self.sim.sim_t
        except Exception as e:
            self.l2_last["error"] = f"{type(e).__name__}: {e}"[:300]
            self.l2_live = {"on": False, "text": "", "why": ""}
            time.sleep(2.0)

    def _body_lines(self, snap):
        ph = self.sim.phys.status(snap["sens"])
        light = snap.get("light")
        s = self.sim
        out = body_words(ph, snap["sens"], light)
        age = snap["t"]
        out.append("you are newly alive" if s.phys.alive_s < 20 else "you have been alive a little while"
                   if s.phys.alive_s < 120 else "you have been alive a long while")
        if s.n_runaway:
            out.append(f"holes in your time so far: {s.n_runaway}")
        out += drug_words(s.pharma, s.drug_aware, getattr(s, "_drug_down", False))
        line = s.noci.sensor_line(snap.get("pain_mode"))
        if line:
            out.append(line)
        return out

    def _legs_line(self):
        d = self._drivers
        if "L2:" in d:
            return "your legs are doing exactly what you willed, this instant"
        if "M:" in d:
            return "your memory is pulling your legs"
        if self.order_active():
            return "your legs are obeying the command, not you"
        if "L1:" in d:
            return "your legs are following what you want"
        if "backup" in d or "avoid" in d:
            return "your legs jerked you back on their own"
        return "your legs are going their own way"

    def mind_messages(self, snap, items, lines, evs):
        if self.prompt_version == "v1":
            return self.mind_messages_v1(snap, items, lines, evs)
        mem = self.memory
        seen = dict(mem.seen) if mem.mode != "off" else None
        rv = self.revealed
        t = snap["t"]
        local = self.is_local
        th = self._throttle or self.throttle(fresh=True)

        def met(e):
            return mind_event_text(e, snap, rv, seen)
        idle = not evs and self.l2_n_mind > 0
        if self.l2_n_mind == 0:
            why = ["you have just woken up in this place"]
        elif evs:
            why = [x for x in (met(e) for e in evs[-4:]) if x] or ["something you cannot name"]
        else:
            why = ["nothing new"]
        why = [x for i, x in enumerate(why) if x not in why[:i]]
        with self.log.lock:
            recent = [e for e in self.log.items if e["id"] > self.l2_since and e["level"] in ("L0", "M", "world", "O")
                      and e["kind"] not in ("config",)]
            self.l2_since = self.log.n
        rec = [met(e) for e in recent[-th.n_recent:]]
        rec = [x for i, x in enumerate(rec) if x and x not in rec[:i] and x not in why]
        memory_txt = ""
        if mem.mode != "off":
            m_lines, m_eps, m_pull = mem.describe(snap)
            m_lines = [re.sub(r"\b\d+(?:\.\d+)? ?mm\b", "", x) for x in m_lines][: th.n_beliefs]
            memory_txt = ("WHAT YOU REMEMBER (not sensed now): " + ("; ".join(m_lines) or
                          ("nothing comes to you" if th.capacity < 0.95 else "nothing")) + "\n"
                          + (f"YOUR MEMORY TUGS: {m_pull}\n" if m_pull else ""))
        # talk (2026-10-15): the user's words are shown, and answerable, only on the wake that first sees them; the
        # exchange before them comes along as history marked answered. With nothing new said there is no talk section
        # and no SAY slot, so an old question is never answered again (it was re-shown for 60 s and the 4B model
        # answered it, or copied the SAY example, on every wake).
        pend = self._pending
        talk_txt = ""
        if pend:
            voice = "the vast one" if rv else "the humming"
            old = [x for x in list(self.talk) if x["who"] in ("you", "fly") and x.get("answered", True)
                   and t - x["t"] < 180.0 and x not in pend][-4:]
            if old:
                talk_txt = "EARLIER WORDS (already answered; do not answer them again):\n" + "\n".join(
                    f"{'you' if x['who'] == 'fly' else voice}: {x['text']}" for x in old) + "\n"
            talk_txt += "WORDS THAT JUST REACHED YOU:\n" + "\n".join(
                f"{voice}{' (COMMAND)' if x['mode'] == 'order' else ''}: \"{x['text']}\"" for x in pend) + "\n"
        o = self.order
        if o is not None and t < o["t_end"]:
            talk_txt += f"A COMMAND HOLDS YOUR LEGS: \"{o['text']}\".\n"
        story = [f"- {_ago(t - st)}: {txt}" + (f" (you felt {fe})" if fe else "") for st, txt, fe in self.story]
        story = story[len(story) - th.n_story:] if th.n_story else []
        recent_th = list(self.l2_recent)
        # the 4B model copies whatever thoughts it is shown (1 local, 3 online; fewer as working memory shrinks);
        # locally it gets only their opening words (gist), with the words it keeps reusing biased against server-side
        last_th = recent_th[-th.n_thoughts:] if th.n_thoughts else []
        if local:
            last_th = [gist(x) for x in last_th]
        around = mind_around(snap, items, lines)
        if not th.keep_far:
            around = drop_far(around)
        around = narrow(around, th.focus)
        after = ("you cannot hold on to what you were after" if th.lose_plan and self.l2_last.get("directive")
                 else self.l2_last.get('directive') or 'nothing yet')
        worn_txt = ""
        if not local:
            worn, words = worn_phrases(recent_th)
            if worn or words:
                worn_txt = "WORN OUT WORDS (find others): " + "; ".join([f'"{w}"' for w in worn] + words) + "\n"
        foci = MIND_FOCI + (("the other fly you can see: what it is doing, and what that means for you",) * 2
                            if flies_seen(snap) else ())
        focus = ("NOTHING NEW HAS HAPPENED. LET YOUR MIND TURN TO: " + foci[self.l2_n_mind % len(foci)] + "\n") if idle else ""
        choices_txt = need_txt = ""
        if self.intent_style == "structured":
            self._choices = now_choices(snap, items, mem, snap.get("hand"))
            if self.nav_on and self._nav_s is not None:
                keep = [c for c in self._choices if c[1]["act"] not in ("explore", "freeze", "groom", "stay")
                        and c[1].get("kind") != "block"]
                self._choices = keep + self.nav.choices(self._nav_s)
            # strongest need first (2026-10-15): parched with no water known, the 4B model kept picking "go to the
            # sugar you remember" (the only named want) while its body said "not hungry", and dried out at 250 s
            hu, tq = float(self.needs.hunger), float(self.needs.thirst)
            kinds = {c[1].get("kind") for c in self._choices}
            if hu < 0.3:
                self._choices = [c for c in self._choices
                                 if not (c[1].get("kind") == "sugar" and c[1]["act"] == "approach")]
                if "sugar" in str(after):
                    after = "nothing now: you have eaten your fill"
            for need, lvl, other, kind, look in (("water", tq, hu, "water", "look for water"),
                                                 ("food", hu, tq, "sugar", "look for food")):
                if lvl >= 0.5 and lvl >= other:
                    if kind not in kinds:
                        self._choices.insert(0, (look, {"act": "explore", "kind": None, "id": None}))
                        need_txt = (f"YOUR STRONGEST NEED: {need}. You know of none: {look}, somewhere you have "
                                    "not been.\n")
                    else:
                        self._choices.sort(key=lambda c: c[1].get("kind") != kind)
                        need_txt = f"YOUR STRONGEST NEED: {need}. Go to it now.\n"
                    break
            choices_txt = ("YOUR CHOICES FOR NOW (copy one exactly): " + "; ".join(c[0] for c in self._choices) + "\n")
        nav_txt = ""
        if self.nav_on and self._nav_s is not None:
            ways, nmem = self.nav.describe(self._nav_s)
            nav_txt = (f"THE WAYS FROM HERE: {ways}\nSMELL: {self.nav.smell_line(self._nav_s)}\n"
                       + (f"WHAT YOU REMEMBER OF THE WAYS: {'; '.join(nmem)}\n" if nmem else ""))
        u = ("WHAT JUST HAPPENED: " + "; ".join(why) + "\n"
             + ("ALSO SINCE YOUR LAST THOUGHT: " + "; ".join(rec) + "\n" if rec else "")
             + "YOUR BODY: " + "; ".join(self._body_lines(snap)) + "\n"
             + (f"HOW YOUR BODY IS: {th.intero}\n" if th.intero else "")
             + "WHAT YOU SENSE NOW: " + "; ".join(around) + "\n"
             + nav_txt
             + memory_txt + talk_txt
             + ("YOUR STORY SO FAR (oldest first):\n" + "\n".join(story) + "\n" if story else "")
             + f"WHAT YOU WERE AFTER: {after}\n"
             + ("YOUR LAST THOUGHT" + ("S" if len(last_th) > 1 else "") + " (already thought; think something else, "
                "in other words):\n" + "\n".join(f"- {x}" for x in last_th) + "\n" if last_th else "")
             + worn_txt + (focus if not nav_txt else "") + need_txt + choices_txt
             + (("Write THOUGHT, FEELING, NOW and SAY now. Answer the words that just reached you.\n" if pend else
                 "Write THOUGHT, FEELING and NOW now. React to WHAT JUST HAPPENED.\n") if local and not idle else "")
             + th.length_line())
        sys_ = MIND_SYSTEM.format(control=MIND_CONTROL.get(self.l2_mode, MIND_CONTROL["steer"]),
                                  names=", ".join(k for k in L1_MENU if k != "none"),
                                  memory=MIND_MEMORY if (mem.mode != "off" and not local) else "",
                                  voice=MIND_VOICE[bool(rv)])
        sys_ = say_slot(sys_, bool(pend)) + (local_examples(self.l2_n_mind, pend[-1]["text"] if pend else None)
                                             if local else "")
        if self.intent_style == "structured":
            sys_ = (sys_.replace("NOW: <at most 8 words: go to or get away from sugar, water, bitter, heat, cold or an "
                                 "obstacle; or explore; or stay>", "NOW: <one of YOUR CHOICES FOR NOW, copied exactly>")
                    .replace("NOW: stay\n", "NOW: stay still\n"))
        return [{"role": "system", "content": self._budget_text(sys_)}, {"role": "user", "content": u}], why

    _NUM = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}

    def _budget_text(self, sys_):
        """The thought-length rule of the v2 prompt, rewritten when a longer budget is configured (thought_sentences,
        thought_words). Lever 2's THIS THOUGHT line still shortens a thought under load."""
        n, w = self.thought_sentences, self.thought_words
        if not n and not w:
            return sys_
        n = n or 2
        w = w or 25 * n // 2 + 5
        say = self._NUM.get(n, str(n))
        return (sys_.replace("- One or two short sentences, 25 words at most.",
                             f"- {say} sentences at most, {w} words at most.")
                .replace("THOUGHT: <one or two short sentences>", f"THOUGHT: <up to {say.lower()} sentences>"))

    def mind_messages_v1(self, snap, items, lines, evs):
        mem = self.memory
        seen = dict(mem.seen) if mem.mode != "off" else None
        rv = self.revealed
        t = snap["t"]

        def met(e):
            return mind_event_text(e, snap, rv, seen)
        idle = not evs and self.l2_n_mind > 0
        if self.l2_n_mind == 0:
            why = ["you have just become aware"]
        elif evs:
            why = [x for x in (met(e) for e in evs[-5:]) if x] or ["nothing you can name"]
        else:
            why = ["nothing new has happened for a while"]
        with self.log.lock:
            recent = [e for e in self.log.items if e["id"] > self.l2_since and e["level"] in ("L0", "M", "world", "O")
                      and e["kind"] not in ("config",)]
            self.l2_since = self.log.n
        rec = [met(e) for e in recent[-10:]]
        rec = [x for i, x in enumerate(rec) if x and x not in rec[:i]]      # no repeats
        memory_txt = ""
        if mem.mode != "off":
            m_lines, m_eps, m_pull = mem.describe(snap)
            m_lines = [re.sub(r"\b\d+(?:\.\d+)? ?mm\b", "", x) for x in m_lines]
            memory_txt = ("WHAT YOU REMEMBER (not sensed now):\n" + ("\n".join(m_lines) or "nothing") + "\n"
                          + ("WHAT HAS HAPPENED TO YOU:\n" + "\n".join(m_eps[-6:]) + "\n" if m_eps else "")
                          + (f"YOUR MEMORY TUGS: {m_pull}\n" if m_pull else ""))
        # the conversation only while it is live: the last minute, or it just spoke
        talk = [x for x in self.talk if x["who"] != "note" and t - x["t"] < 60.0][-4:]
        spoke = any(e["level"] == "user" and e["kind"] in ("say", "order") for e in evs)
        talk_txt = ""
        if talk and (spoke or talk[-1]["who"] == "you"):
            voice = "the vast one" if rv else "the humming"
            talk_txt = "WORDS JUST NOW (oldest first):\n" + "\n".join(
                f"{'you' if x['who'] == 'fly' else voice + (' (COMMAND)' if x['mode'] == 'order' else '')}: {x['text']}"
                for x in talk) + "\n"
        o = self.order
        if o is not None and t < o["t_end"]:
            talk_txt += f"A COMMAND HOLDS YOUR LEGS: \"{o['text']}\".\n"
        recent_th = list(self.l2_recent)
        worn, words = worn_phrases(recent_th)
        worn_txt = ""
        if worn or words:
            worn_txt = ("WORN OUT (you have said these too often; find other words and other thoughts; still say I and my): "
                        + "; ".join([f'"{w}"' for w in worn] + words) + "\n")
        focus = ""
        foci = MIND_FOCI_V1 + (("the other fly you can see: what it is doing, and what that might mean for you",) * 2
                            if flies_seen(snap) else ())
        urgent = any((e["level"] == "L0" and e["kind"] in ("crushed", "escaped", "body", "runaway", "pain"))
                     or e["level"] == "user" for e in evs)
        if idle:
            focus = "LET YOUR MIND TURN TO: " + foci[self.l2_n_mind % len(foci)] + "\n"
        elif not urgent and self.l2_n_mind > 0:
            # routine wake-ups (a bump, a reflex) would otherwise pull every thought back to the same need and wall
            focus = ("BESIDES WHAT JUST HAPPENED, LET YOUR MIND TURN TO: "
                     + foci[self.l2_n_mind % len(foci)] + "\n")
        u = ("WHY YOU ARE AWAKE: " + "; ".join(why) + "\n"
             "YOUR BODY: " + "; ".join(self._body_lines(snap)) + "\n"
             + "WHAT YOU SENSE NOW:\n" + "\n".join(mind_around(snap, items, lines)) + "\n"
             + memory_txt
             + ("SINCE YOU LAST THOUGHT:\n" + "\n".join(rec) + "\n" if rec else "")
             + talk_txt
             + f"YOUR LEGS: {self._legs_line()}\n"
             + (f"WHAT YOU LIVE FOR: {self.goal}\n" if self.goal else "")
             + (f"WHAT YOU WANTED LAST (NOW): {self.l2_last['directive']}\n" if self.l2_last.get("directive") else "")
             + ("YOUR LAST THOUGHTS (move on from them):\n" + "\n".join(f"- {x}" for x in recent_th[-3:]) + "\n"
                if recent_th else "")
             + worn_txt + focus)
        sys_ = MIND_SYSTEM_V1.format(control=MIND_CONTROL.get(self.l2_mode, MIND_CONTROL["steer"]),
                                  names=", ".join(k for k in L1_MENU if k != "none"),
                                  memory=MIND_MEMORY if mem.mode != "off" else "", voice=MIND_VOICE[bool(rv)])
        return [{"role": "system", "content": sys_}, {"role": "user", "content": u}], why

    def _stream(self, msgs, why_s, max_tokens=420, temperature=0.9):
        th = self._throttle                                 # set by mind_step only (the last thought is not throttled)
        if th is not None:
            temperature = round(temperature + (th.temperature - 0.9), 3)      # noise lever (0 when off)
            max_tokens = th.max_tokens                                       # thought-budget lever
        body = {"model": self.l2_model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}
        if self.is_local:
            body["max_tokens"] = min(max_tokens, self.local_max_tokens)   # thinking is off in the server's chat template
            body.update(self.request_extra)
            if self.seed_base is not None:
                body["seed"] = int(self.seed_base) * 1000 + self._req_n
            self._req_n += 1
            # anti-repetition (pain_server >= 2026-10-15; an older server ignores both): a logit bias against the
            # words the last thoughts keep reusing, and a mild repeat penalty inside the reply. Free text only: the
            # server keeps both off the NOW grammar, and the steering vectors are untouched.
            if self._avoid:
                body["avoid_words"] = list(self._avoid)
                body["avoid_bias"] = float(os.environ.get("FLYSIM_AVOID_BIAS", "-2.0"))
            body["repeat_penalty"] = float(os.environ.get("FLYSIM_REPEAT_PENALTY", "1.08"))
        else:                                               # OpenRouter-only fields
            body["usage"] = {"include": True}
            if self.l2_mind_reasoning:
                body["reasoning"] = {"effort": self.l2_mind_reasoning}
        p = self.sim.noci.steer_scalar()
        if p is not None:
            body["pain"] = p
        live = None
        if self.is_local and self.embody_on:
            # lever 1: the drive vector (nine drives + heat/cold split, MN9 in hunger) for per-drive steering
            # server-side; not sent to OpenRouter, which has no use for it
            body["drives"] = self.embodiment.steer_drives()
            body["drive_steer"] = self.lever1
            if self.lever1:
                body["live"] = True
                live = threading.Event()
                threading.Thread(target=self._push_drives, args=(live,), daemon=True).start()
        if self.is_local and self.intent_style == "structured" and self._choices and th is not None:
            body["now_options"] = [c[0] for c in self._choices]      # grammar-constrained NOW (llama backend)
        ro = self.readout
        if self.is_local and ro.mode != "off" and th is not None:
            body["readout"] = True
        if self.is_local and os.environ.get("FLYSIM_ACTIVITY", "1") != "0":
            body["activity"] = True                 # per-layer activity frames for the LLM-brain panel (web/app.js)
            # the twin: the same words through the same model with no feelings added, for the panel's difference
            # map and word odds (roughly doubles the server's per-token cost; FLYSIM_TWIN=0 turns it off)
            body["twin"] = os.environ.get("FLYSIM_TWIN", "1") != "0"
            ro.start_reply()
        self.last_body = {k: v for k, v in body.items() if k != "messages"}

        def on_text(full):
            if self._abort_cur:                 # a strong event: hang up (only set when the server aborts on hang-up)
                raise _Interrupted()
            self.l2_live = {"on": True, "text": live_thought(full), "why": why_s}

        def on_readout(rv):
            ro.on_token(rv.get("r"), self.sim.sim_t)

        def on_act(fr):
            if fr.get("meta"):
                self.act_meta = fr["meta"]
            self.act_frame = {"i": fr.get("i"), "b": fr.get("b"), "p": fr.get("p"), "t": self.sim.sim_t,
                              "w": round(time.time(), 3)}

        def on_twin(tv):
            self.twin_frame = {"i": tv.get("i"), "b": tv.get("b"), "odds": tv.get("odds"), "t": self.sim.sim_t}

        def on_doses(dv):
            # the doses the server set at token dv["i"] (live channel): what the brain panel lights, as it happens
            self.live_doses = {"i": dv.get("i"), "d": dv.get("d") or {}, "t": self.sim.sim_t, "w": round(time.time(), 3)}

        cost, c, r = 0.0, "", None
        try:
            for attempt in range(2):          # an empty reply (MALFORMED_RESPONSE) gets one retry
                c, usage = chat_stream(self.l2_url, body, "" if self.is_local else self.l2_key, on_text=on_text,
                                       timeout=float(os.environ.get("FLYSIM_MIND_TIMEOUT", "60")) if self.is_local else 30,
                                       on_readout=on_readout if body.get("readout") else None,
                                       on_act=on_act if body.get("activity") else None, on_doses=on_doses,
                                       on_twin=on_twin)
                cost += float((usage or {}).get("cost") or 0.0)
                self.last_steer = (usage or {}).get("steer")
                r = parse_mind(c)
                if r["thought"]:
                    break
        finally:
            if live is not None:
                live.set()
            if body.get("readout"):
                ro.end_reply()
            if getattr(self, "live_doses", None):     # the reply is over: the server adds nothing until the next one
                self.live_doses = {"i": None, "d": {}, "t": self.sim.sim_t, "w": round(time.time(), 3)}
        return c, r, cost

    LIVE_PERIOD_S = 0.067                 # ~15 Hz

    def _push_drives(self, stop):
        """Lever 1 live channel: POST the current drives to the local server until the thought has streamed."""
        url = self.l2_url.rstrip("/") + "/drives"
        while not stop.is_set():
            try:
                req = urllib.request.Request(url, data=json.dumps({"drives": self.embodiment.steer_drives(),
                                                                   "t": self.sim.sim_t}).encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=0.5).close()
            except Exception:
                pass                       # an old server without /v1/drives, or busy: the call keeps its doses
            stop.wait(self.LIVE_PERIOD_S)

    def final_step(self, evs):
        """The last thought: streamed like any other, then the mind stops for good (until a revive)."""
        ph = self.sim.phys.status()
        how = {"starvation": "your body has nothing left to burn", "dehydration": "you have dried out",
               "heat": "the heat is cooking you", "cold": "the cold has stopped everything in you",
               "crushed": "something enormous has come down on you"}.get(ph["cause"], "your body is failing")
        snap = self.sim.cog_snapshot()
        mem = self.memory
        eps = mem.describe(snap)[1][-5:] if mem.mode != "off" else []
        u = ("WHAT YOU REMEMBER OF YOUR LIFE:\n" + ("\n".join(eps) or "very little") + "\n"
             + ("YOUR LAST THOUGHTS:\n" + "\n".join(f"- {x}" for x in list(self.l2_recent)[-2:]) + "\n" if self.l2_recent else ""))
        msgs = [{"role": "system", "content": MIND_FINAL.format(how=how)}, {"role": "user", "content": u}]
        why_s = "dying: " + ph.get("cause_text", "")
        self.l2_live = {"on": True, "text": "", "why": why_s}
        t0 = time.time()
        c, r, cost = self._stream(msgs, why_s, max_tokens=420)      # reasoning tokens count against it: 200 cut the last thought short
        self.l2_live = {"on": False, "text": "", "why": ""}
        self.l2_last["cost_usd"] = round(self.l2_last["cost_usd"] + cost, 5)
        if not r or not r["thought"]:
            return
        t_now = self.sim.sim_t
        self.l2_last = {**self.l2_last, "thought": r["thought"], "feeling": r["feeling"], "directive": "",
                        "latency_s": round(time.time() - t0, 2), "n": self.l2_last["n"] + 1, "why": why_s, "last": True}
        self.log.add(t_now, "L2", "thought", r["thought"], feeling=r["feeling"], why=why_s, style="mind", last=True,
                     mode=self.l2_mode, model=self.l2_model, latency_s=round(time.time() - t0, 2),
                     prompt=u, reply=c[:800])

    def mind_step(self, evs):
        """One thought, under what the body allows this wake (lever 2: embodiment.Throttle)."""
        self._throttle = self.throttle(fresh=True)
        self._slur_rng = getattr(self, "_slur_rng", None) or __import__("random").Random(7)
        self._abort_cur = False
        self._t_start = self.sim.sim_t
        try:
            self._mind_step(evs)
        except _Interrupted:
            self.n_interrupted += 1
            part = str(self.l2_live.get("text") or "")
            self.l2_live = {"on": False, "text": "", "why": ""}
            self.log.add(self.sim.sim_t, "L2", "interrupted", (part or "(no words yet)")[:200], t_start=round(self._t_start, 3))
        finally:
            self._throttle = None
            self._abort_cur = False
            self._cur_urgent = ""

    def _answer(self, r, t_now):
        """SAY goes out only on the wake that carries new words from the user, once; those words are then answered
        (also when the fly chose to say nothing), so no later wake answers them again. A SAY with nobody speaking
        (the 4B model copying an example) is dropped."""
        pend, self._pending = self._pending, []
        if not pend:
            return
        if r and r.get("say"):
            self.talk.append({"t": round(t_now, 1), "who": "fly", "text": r["say"], "mode": "say"})
            self.log.add(t_now, "L2", "say", r["say"])
        for x in pend:
            x["answered"] = True

    def _mind_step(self, evs):
        snap = self.perceived_snapshot()
        lines, items = affordances(snap)
        v2 = self.prompt_version != "v1"
        # the user's lines not answered yet (a minute at most): this wake answers them; an interrupted wake leaves
        # them for the next one
        self._pending = [x for x in list(self.talk) if x["who"] == "you" and not x.get("answered", True)
                         and snap["t"] - x["t"] < 60.0][-2:]
        if v2:
            # quiet stretches: one idle thought, then silence until the body or the world changes (a viewer cannot
            # follow restless thoughts about nothing)
            sig = tuple(x for x in self._body_lines(snap) if not x.startswith("you have been alive")
                        and not x.startswith("you are newly")) + tuple(mind_around(snap, items, lines))
            if evs or self.l2_n_mind == 0 or self._cur_urgent or self._pending:
                self._idle_streak = 0
            elif self._idle_streak >= 1 and sig == self._last_body_sig and not self.idle_replan:
                return
            else:
                self._idle_streak += 1
            self._last_body_sig = sig
        recent3 = list(self.l2_recent)[-3:]
        self._avoid = avoid_words(recent3) if self.is_local else []
        msgs, why = self.mind_messages(snap, items, lines, evs)
        why_s = "; ".join(why)[:160]
        self.l2_live = {"on": True, "text": "", "why": why_s}
        t0 = time.time()
        c, r, cost = self._stream(msgs, why_s)
        recent_th = list(self.l2_recent)[-4:]
        if v2 and r["thought"] and (too_similar(r["thought"], recent_th) or near_repeat(r["thought"], recent3)
                                    or near_repeat(r["thought"], EXAMPLE_THOUGHTS)):
            # one resample, away from the words of the repeat (logit bias, local) and toward another focus
            n_f = self.l2_n_mind + 1
            nudge = {"role": "user", "content": msgs[-1]["content"] + "\nYour first try re-said an earlier thought. "
                     "Say something NEW about what just happened, in different words. Think instead about "
                     + MIND_FOCI[n_f % len(MIND_FOCI)] + "."}
            if self.is_local:
                self._avoid = avoid_words(recent3, also=[r["thought"]])
            c2, r2, cost2 = self._stream(msgs[:-1] + [nudge], why_s, temperature=1.0)
            cost += cost2
            if r2["thought"] and not too_similar(r2["thought"], recent_th) and not near_repeat(r2["thought"], EXAMPLE_THOUGHTS):
                c, r = c2, r2
            else:
                self.n_dropped_repeats += 1          # do not put the same subtitle up again
                self.l2_live = {"on": False, "text": "", "why": ""}
                self.l2_last["cost_usd"] = round(self.l2_last["cost_usd"] + cost, 5)
                self._answer(r2 if r2.get("say") else r, self.sim.sim_t)    # an answer still goes out, once
                return
        lat = time.time() - t0
        if not r["thought"]:
            self.l2_live = {"on": False, "text": "", "why": ""}
            self.l2_last["error"] = "empty reply twice (model returned no content)"
            self.l2_last["cost_usd"] = round(self.l2_last["cost_usd"] + cost, 5)
            return
        if self.sim.phys.dead:               # it died while this thought was on its way
            self.l2_live = {"on": False, "text": "", "why": ""}
            self.l2_last["cost_usd"] = round(self.l2_last["cost_usd"] + cost, 5)
            return
        thought, feeling, directive, ov = r["thought"], r["feeling"], r["now"], r["override"]
        th = self._throttle
        if th is not None and th.slur > 0:
            thought = slur(thought, th.slur, self._slur_rng)       # noise lever, token level (intoxication)
        t_now = self.sim.sim_t
        applied_ov = None
        istat = None
        if self.intent_style == "structured" and self._choices:
            spec, itext, istat = match_now(directive, self._choices, self.intent)
            if (self.last_steer or {}).get("now_state") == "done":
                self.intent_stats["grammar"] += 1
            self.intent_stats[istat] += 1
            if self.l2_mode == "steer":
                self.intent, self.intent_text = spec, itext
        if self.l2_mode in ("steer", "blind"):
            if self.l2_mode == "steer" and directive:
                self.directive = directive
            if ov and self.l2_mode == "steer" and t_now - self._last_override_t > OVERRIDE_COOLDOWN_S:
                self._last_override_t = t_now
                secs = float(np.clip(ov["seconds"], 0.3, 2.0))
                self.l2_act.set(ov["intent"], 0.8, secs, t_now)
                applied_ov = {"intent": ov["intent"], "seconds": secs}
        prev_dir = self.l2_last.get("directive", "")
        self.l2_n_mind += 1
        self.l2_last = {"thought": thought, "directive": directive, "override": applied_ov, "latency_s": round(lat, 2),
                        "n": self.l2_last["n"] + 1, "error": "", "cost_usd": round(self.l2_last["cost_usd"] + cost, 5),
                        "model": self.l2_model, "mode": self.l2_mode, "feeling": feeling, "why": why_s,
                        "throttle": th.as_dict() if th is not None else None,
                        "intent": {k: v for k, v in (spec or {}).items()} if istat else None,
                        "intent_status": istat, "choices": [c[0] for c in self._choices] if istat else None}
        self.l2_recent.append(thought)
        notable = [e for e in evs if e["level"] in ("world", "user", "O") or e["kind"] in (
            "pain", "body", "drugfeel", "crushed", "escaped", "runaway", "found", "surprise", "fly", "need", "other_add")]
        if notable and why_s:
            self.story.append((t_now, why_s, (feeling or "").lower().strip(".")[:20]))
        self.l2_live = {"on": False, "text": "", "why": ""}
        self.log.add(t_now, "L2", "thought", thought, directive=directive, override=applied_ov, mode=self.l2_mode,
                     model=self.l2_model, latency_s=round(lat, 2), feeling=feeling, why=why_s, style="mind",
                     t_start=round(self._t_start, 3), urgent=self._cur_urgent or None,
                     prompt=msgs[-1]["content"], reply=c[:800],
                     drives=self.embodiment.drives(), throttle=th.as_dict() if th is not None else None)
        if directive and self.l2_mode == "steer" and directive != prev_dir:
            self.log.add(t_now, "L2", "directive", directive)
        if applied_ov:
            self.log.add(t_now, "L2", "override", f"{applied_ov['intent']} for {applied_ov['seconds']:.1f} s")
        self._answer(r, t_now)
        if r["remember"] and self.memory.mode != "off":
            self.memory.remember(t_now, r["remember"], 0.0, "mind")
            self.log.add(t_now, "L2", "remember", r["remember"])

    def l2_messages(self, snap, lines):
        needs = self.needs.status(snap["sens"])
        with self.log.lock:
            recent = [e for e in self.log.items if e["id"] > self.l2_since and e["level"] in ("L0", "L1", "L2", "world", "user")]
            self.l2_since = self.log.n
        rec = []
        recent = [e for e in recent if not (e["kind"] == "drug" and not (e.get("data") or {}).get("told"))
                  and not e["kind"].startswith("pain")]
        for e in recent[-14:]:
            who = {"L0": "reflex", "L1": "motor layer", "L2": "you", "world": "world", "user": "user"}[e["level"]]
            rec.append(f"t={e['t']:.1f} [{who}] {re.sub(r', laced with [a-z]+$', '', e['text'])}")
        user = (f"t={snap['t']:.1f}s\n"
                f"NEEDS: hunger {needs['hunger']:.2f}, thirst {needs['thirst']:.2f} (0 = sated, 1 = desperate); "
                f"temperature at the antennae {needs['temp_c']} C ({needs['comfort']})\n"
                f"NEURONS:\n{observation_text(snap['obs'])}\n"
                f"AROUND YOU:\n" + "\n".join(lines) + "\n"
                f"RECENT (who did what):\n" + ("\n".join(rec) or "nothing new") + "\n"
                f"RIGHT NOW the motor neurons are driven by: {self._drivers}\n"
                + (f"Something you want: {self.goal}\n" if self.goal else "")
                + (f"Your last directive: {self.l2_last['directive']}\n" if self.l2_last.get("directive") else ""))
        sys_ = L2_SYSTEM.format(control=L2_CONTROL[self.l2_mode], names=[k for k in L1_MENU if k != "none"])
        return [{"role": "system", "content": sys_}, {"role": "user", "content": user}]

    def l2_step(self):
        snap = self.sim.cog_snapshot()
        lines, _ = affordances(snap)
        msgs = self.l2_messages(snap, lines)
        t0 = time.time()
        # minimal reasoning: with hidden reasoning on, a 220-token budget truncated most replies mid-JSON
        # (recordings/20260924_092806); gemini-3.5-flash-lite at "minimal" answers in ~1.3 s
        body = {"model": self.l2_model, "messages": msgs, "temperature": 0.7, "max_tokens": 400,
                "response_format": {"type": "json_object"}, "usage": {"include": True}}
        if self.l2_reasoning:
            body["reasoning"] = {"effort": self.l2_reasoning}
        j = chat(self.l2_url, body, self.l2_key, timeout=30)
        lat = time.time() - t0
        c = j["choices"][0]["message"].get("content") or ""
        s = c[c.find("{"): c.rfind("}") + 1]
        try:
            r = json.loads(s)
        except Exception:
            m = {k: re.search(rf'"{k}"\s*:\s*"([^"]*)', c) for k in ("thought", "directive")}
            r = {"thought": m["thought"].group(1) if m["thought"] else c.strip()[:200],
                 "directive": m["directive"].group(1) if m["directive"] else "", "override": None}
        thought = str(r.get("thought", ""))[:300]
        directive = str(r.get("directive", "") or "")[:120]
        ov = r.get("override")
        cost = float((j.get("usage") or {}).get("cost") or 0.0)
        t_now = self.sim.sim_t
        applied_ov = None
        if self.l2_mode in ("steer", "blind"):
            if self.l2_mode == "steer":
                self.directive = directive
            if (isinstance(ov, dict) and ov.get("intent") in CATALOG and ov.get("intent") != "none"
                    and self.l2_mode == "steer" and t_now - self._last_override_t > OVERRIDE_COOLDOWN_S):
                self._last_override_t = t_now
                secs = float(np.clip(float(ov.get("seconds", 1.0) or 1.0), 0.3, 2.0))
                self.l2_act.set(ov["intent"], 0.8, secs, t_now)
                applied_ov = {"intent": ov["intent"], "seconds": secs}
        self.l2_last = {"thought": thought, "directive": directive, "override": applied_ov, "latency_s": round(lat, 2),
                        "n": self.l2_last["n"] + 1, "error": "", "cost_usd": round(self.l2_last["cost_usd"] + cost, 5),
                        "model": self.l2_model, "mode": self.l2_mode}
        self.log.add(t_now, "L2", "thought", thought, directive=directive, override=applied_ov, mode=self.l2_mode,
                     model=self.l2_model, latency_s=round(lat, 2), prompt=msgs[-1]["content"], reply=c[:600])
        if directive and self.l2_mode == "steer":
            self.log.add(t_now, "L2", "directive", directive)
        if applied_ov:
            self.log.add(t_now, "L2", "override", f"{applied_ov['intent']} for {applied_ov['seconds']:.1f} s")

    # ------------------------------------------------------------------ config + status
    @property
    def is_local(self):
        return self.mind_provider == "local"

    def set_provider(self, p, log=True):
        """local: the pain-steered Qwen3-4B server, pain steering on; openrouter: the online model, steering off
        (an online model cannot be steered; it gets the pain as text)."""
        if p not in ("local", "openrouter"):
            return
        if self.mind_provider == "openrouter" and p == "local":
            self.or_url, self.or_model = self.l2_url, self.l2_model          # keep a custom online choice
        self.mind_provider = p
        if p == "local":
            self.l2_url, self.l2_model = self.local_url, self.local_model
        else:
            self.l2_url, self.l2_model = self.or_url, self.or_model
        noci = getattr(self.sim, "noci", None)
        if noci is not None:
            noci.steer = p == "local"
        if log:
            self.log.add(self.sim.sim_t, "user", "config", f"mind: {'local Qwen3-4B (feels pain)' if p == 'local' else 'OpenRouter ' + self.l2_model}")

    def configure(self, j):
        t = self.sim.sim_t
        if j.get("mind_provider"):
            self.set_provider(str(j["mind_provider"]))
        if "embody" in j:
            self.embody_on = bool(j["embody"])
            self.log.add(t, "user", "config", f"body throttles the mind: {'on' if self.embody_on else 'off'}")
        if isinstance(j.get("embody_levers"), dict):
            for k, v in j["embody_levers"].items():
                if k in self.embody_levers:
                    self.embody_levers[k] = bool(v)
        if "l1_mode" in j and j["l1_mode"] in ("off", "rule", "llm"):
            self.l1_mode = j["l1_mode"]
            if self.l1_mode == "off":
                self.l1_act.clear()
            self.log.add(t, "user", "config", f"motor layer: {self.l1_mode}")
        if "l2_mode" in j and j["l2_mode"] in ("off", "steer", "narrate", "blind"):
            self.l2_mode = j["l2_mode"]
            if self.l2_mode != "steer":
                self.directive = ""
                self.l2_act.clear()
            self.log.add(t, "user", "config", f"conscious layer: {self.l2_mode}")
            self._wake.set()
        if j.get("l2_style") in ("mind", "classic") and j["l2_style"] != self.l2_style:
            self.l2_style = j["l2_style"]
            self.log.add(t, "user", "config", f"conscious layer style: {self.l2_style}")
            self._wake.set()
        if "hold_for_mind" in j:
            self.hold_for_mind = bool(j["hold_for_mind"])
        if j.get("l2_idle"):
            self.l2_idle = float(np.clip(float(j["l2_idle"]), 1.0, 120.0))
        if "l2_reasoning" in j and j["l2_reasoning"] in ("", "minimal", "low", "medium", "high"):
            self.l2_reasoning = j["l2_reasoning"]
        if j.get("l1_style") in ("target", "action"):
            self.l1_style = j["l1_style"]
        for k in ("l1_url", "l1_model", "l2_model", "l2_url"):
            if j.get(k):
                if self.is_local and k in ("l2_model", "l2_url"):     # an online choice made while local waits for the switch
                    setattr(self, "or_" + k[3:], str(j[k]))
                else:
                    setattr(self, k, str(j[k]))
        if j.get("l2_period"):
            self.l2_period = float(np.clip(float(j["l2_period"]), 1.0, 30.0))
        if j.get("l1_period"):
            self.l1_period = float(np.clip(float(j["l1_period"]), 0.1, 2.0))
        if "goal" in j and j["goal"] is not None:
            self.goal = str(j["goal"])[:200]
            self.log.add(t, "user", "goal", f"goal: {self.goal}")
        if "revealed" in j:
            self.revealed = bool(j["revealed"])
            self.log.add(t, "user", "config", "the mind knows about you" if self.revealed else "the mind does not know about you")
        if j.get("mem_mode") in ("off", "shadow", "contend") or j.get("mem_weight") is not None:
            prev = self.memory.mode
            self.memory.configure(j.get("mem_mode"), j.get("mem_weight"))
            if self.memory.mode != prev:
                self.log.add(t, "user", "config", f"memory: {self.memory.mode}")

    def status(self):
        s = self.sim
        return {"needs": self.needs.status(s.sens), "drivers": self._drivers, "goal": self.goal,
                "l1": {**{k: v for k, v in self.l1_last.items()}, "mode": self.l1_mode, "model": self.l1_model,
                       "url": self.l1_url, "period": self.l1_period},
                "l2": {**self.l2_last, "mode": self.l2_mode, "model": self.l2_model, "period": self.l2_period,
                       "has_key": bool(self.l2_key), "style": self.l2_style, "live": dict(self.l2_live),
                       "idle": self.l2_idle, "hold": self.hold_for_mind, "provider": self.mind_provider,
                       "url": self.l2_url, "or_model": self.or_model if self.is_local else self.l2_model,
                       "local_model": self.local_model, "dropped_repeats": self.n_dropped_repeats},
                "directive": self.current_directive(), "log": self.log.tail(40),
                "recording": self.rec.dir.name if self.rec.on else None,
                "talk": list(self.talk)[-30:], "revealed": self.revealed, "gate": dict(self.gate),
                "order": ({"text": o["text"], "left": round(o["t_end"] - s.sim_t, 1)}
                          if (o := self.order) is not None and s.sim_t < o["t_end"] else None),
                "mem": self.memory.status(),
                "embody": {"on": self.embody_on, "levers": dict(self.embody_levers), **self.embodiment.status(),
                           "throttle": self.throttle().as_dict(), "sources": SOURCES,
                           "suppressed": self.n_suppressed},
                # the LLM side for the brain column (web/app.js): the last local call's steering record (per-drive
                # doses, the per-token dose curve) and the residual read-out (freeze / avoid / approach) as it streams
                "steer": self._steer_view(),
                "act": getattr(self, "act_frame", None), "act_meta": getattr(self, "act_meta", None),
                "live_doses": getattr(self, "live_doses", None),
                "readout": {"mode": self.readout.mode, "cur": self.readout.cur,
                            "run": [r for r in (getattr(self.readout, "_run", None) or [])[-160:]]}}

    def _steer_view(self):
        st = self.last_steer
        if not st:
            return None
        return {"doses": st.get("drive_doses"), "per_token": (st.get("norm_per_token") or [])[:240],
                "dose": st.get("dose"), "readout": st.get("readout"), "tok_s": st.get("tok_s")}
