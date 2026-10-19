"""LLM <-> fly intermediate layer.

    stimulus -> LLM   NeuralObserver reads the brain's spike counts every window and decodes the
                      fly's situation from neuron populations (looming detectors, taste neurons,
                      antennal and bristle mechanoreceptors, descending motor neurons, MN9), plus a
                      few body facts that have no brain pathway in the model yet (marked [body]).
                      observation_text() renders it as the prompt the LLM sees.
    LLM -> fly        The LLM answers with one intent from CATALOG. Each intent is a neural
                      intervention (Poisson drive to named cell types, or silencing), capped in size
                      and rate so it stays below the ignition threshold found in
                      experiments/attractor.py. It enters the brain through the same input mixer
                      as the sensory layer and the user; nothing writes the body directly.

The agent runs in its own thread against any OpenAI-compatible chat endpoint (local llama-server
by default), so the 10 ms sim loop never waits on the model. Evidence for every catalog entry is
in GROUNDING.md (item numbers in brackets below).
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------------------------------------
# LLM -> fly: intervention catalog
# ------------------------------------------------------------------------------------------------
# route: 'motor' drives descending neurons (a command); 'sensory' makes the fly sense something
# (a hallucinated stimulus the connectome then processes); 'block' silences a population.
# rate(strength) = lo + strength * (hi - lo), strength in [0, 1].


@dataclass
class Intent:
    name: str
    route: str
    targets: list                 # list of (cell_type, side|None)
    lo: float = 0.0
    hi: float = 0.0
    what: str = ""
    evidence: str = ""
    silence: list = field(default_factory=list)


CATALOG = {i.name: i for i in [
    Intent("none", "none", [], what="no intervention; the fly's own sensory-driven brain acts alone"),
    Intent("walk_forward", "motor", [("DNp09", "left"), ("DNp09", "right")], 30, 70,
           "forward walking command (P9 / DNp09, both sides)",
           "Bidaye 2020 Neuron: P9 drives forward walking; bridge maps DNp09 -> forward [1]"),
    Intent("turn_left", "motor", [("DNp09", "left"), ("DNa02", "left"), ("DNg13", "left")], 40, 90,
           "walk in a left arc: left P9 plus the left steering DNs (DNa02, DNg13)",
           "Bidaye 2020 Neuron: unilateral P9 -> forward + ipsilateral turn; Yang 2024 Cell: DNa02 and "
           "DNg13 drive ipsiversive rotation [1]"),
    Intent("turn_right", "motor", [("DNp09", "right"), ("DNa02", "right"), ("DNg13", "right")], 40, 90,
           "walk in a right arc: right P9 plus the right steering DNs (DNa02, DNg13)",
           "Bidaye 2020 Neuron; Yang 2024 Cell [1]"),
    Intent("back_up", "motor", [("MDN", "left"), ("MDN", "right")], 60, 120,
           "backward walking command (moonwalker DNs)",
           "Bidaye 2014 Science: MDN sufficient for backward walking [1]"),
    Intent("stop", "block", [], what="silence the forward and steering command DNs (a halt)",
           evidence="Sapkal 2024 Nature: halting by inhibiting descending walking commands (walk-OFF) [1]",
           silence=[("DNp09", None), ("DNa02", None), ("DNg13", None)]),
    Intent("escape", "sensory", [("LPLC2", "left"), ("LPLC2", "right")], 60, 150,
           "show a looming object to both eyes (LPLC2 looming detectors) -> giant fibre -> take-off",
           "Ache 2019 Curr Biol: LPLC2 synapses onto the GF; probe_loom.json dose ladder [4]"),
    Intent("feed", "sensory", [("sugar_GRN", None)], 100, 200,
           "taste sugar (20 sugar GRNs) -> MN9 proboscis motor neuron",
           "Shiu 2024 Nature; validate_mn9.json dose-response [3]"),
    Intent("groom_antennae", "sensory", [("JO_CE", None)], 150, 220,
           "deflect the antennae (JO-C/E mechanoreceptors) -> aBN1 -> aDN1/aDN2 grooming DNs",
           "Hampel 2015 eLife; Shiu 2024 Nature; probe_touch.json [2]"),
    Intent("calm", "block", [], what="blind the looming pathway (silence LPLC2 + LC4): no escape",
           evidence="Ache 2019 Curr Biol: LPLC2 + LC4 are the GF's looming inputs [4]",
           silence=[("LPLC2", None), ("LC4", None)]),
]}

# rotate on the spot: needs bridge v3 (under v2 a steering DN without forward drive does nothing)
if os.environ.get("FLYSIM_BRIDGE", "v3").lower() == "v3":
    for _side in ("left", "right"):
        CATALOG[f"rotate_{_side}"] = Intent(
            f"rotate_{_side}", "motor", [("DNa01", _side), ("DNa02", _side), ("DNg13", _side)], 60, 150,
            f"turn on the spot to the {_side}, without walking forward: the {_side} steering DNs (DNa01, DNa02, DNg13), no P9",
            "Braun 2024 Nature: DNa01 activation triggers in-place turning; Yang 2024 Cell: DNa02 shortens inside "
            "steps, DNg13 lengthens outside steps [1]")

MAX_NEURONS = 400     # per intervention; ignition needs ~1% of all sensory neurons (~170) at 5 Hz for
MAX_RATE = 220.0      # 300 ms *broadly*; targeted sets of this size stayed local in every probe run


def resolve(table, intent: Intent):
    """-> (poisson {key: (idx, rate_fn)}, silence idx)"""
    def idx_of(t, s):
        return table.find(t if s is None else f"{t}_{s}")
    act = [idx_of(t, s) for t, s in intent.targets]
    act = np.unique(np.concatenate(act)) if act else np.zeros(0, np.int64)
    sil = [idx_of(t, s) for t, s in intent.silence]
    sil = np.unique(np.concatenate(sil)) if sil else np.zeros(0, np.int64)
    return act[:MAX_NEURONS], sil


# ------------------------------------------------------------------------------------------------
# stimulus -> LLM: neural observation
# ------------------------------------------------------------------------------------------------
READ_POPS = {
    # name: (cell_type or group, side)
    "LPLC2_left": ("LPLC2", "left"), "LPLC2_right": ("LPLC2", "right"),
    "LC4_left": ("LC4", "left"), "LC4_right": ("LC4", "right"),
    "sugar_GRN": ("sugar_GRN", None),
    "JO_CE": ("JO_CE", None),
    "eye_bristle_left": ("BM_InOm_left", None), "eye_bristle_right": ("BM_InOm_right", None),
    "DNp09_left": ("DNp09", "left"), "DNp09_right": ("DNp09", "right"),
    "MDN_left": ("MDN", "left"), "MDN_right": ("MDN", "right"),
    "DNa02_left": ("DNa02", "left"), "DNa02_right": ("DNa02", "right"),
    "DNg13_left": ("DNg13", "left"), "DNg13_right": ("DNg13", "right"),
    "GF_left": ("DNp01", "left"), "GF_right": ("DNp01", "right"),
    "MN9": ("MN9", None),
    "aDN1": ("aDN1", None), "aDN2": ("aDN2", None), "aBN1": ("aBN1", None),
    "DNg02": ("DNg02", None),
    # world senses (world.py)
    "water_GRN": ("water_GRN", None), "bitter_GRN": ("bitter_GRN", None),
    "TRN_hot_left": ("TRN_hot_left", None), "TRN_hot_right": ("TRN_hot_right", None),
    "TRN_cold_left": ("TRN_cold_left", None), "TRN_cold_right": ("TRN_cold_right", None),
    "HRN_moist_left": ("HRN_moist_left", None), "HRN_moist_right": ("HRN_moist_right", None),
    # nociception (nociception.py): the stand-in entry set and the PPL1 read-out
    "AN_noci": ("AN_noci", None), "PPL1": ("PPL1", None),
}


class NeuralObserver:
    """Exponentially filtered population rates (Hz) from each 10 ms window of spike counts."""

    def __init__(self, table, tau_ms=150.0):
        self.tau = tau_ms
        self.idx = {}
        for k, (t, s) in READ_POPS.items():
            i = table.find(t if s is None else f"{t}_{s}")
            if len(i):
                self.idx[k] = i
        self.rate = {k: 0.0 for k in self.idx}
        self.active_frac = 0.0
        # all populations in one gather + reduceat (one call per window, not one per population)
        self._keys = list(self.idx)
        self._cat = np.concatenate([self.idx[k] for k in self._keys]) if self._keys else np.zeros(0, int)
        lens = np.array([len(self.idx[k]) for k in self._keys])
        self._starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.intp)
        self._lens = lens.astype(float)

    def update(self, counts, window_ms):
        a = 1.0 - np.exp(-window_ms / self.tau)
        s = window_ms * 1e-3
        if self._keys:
            means = np.add.reduceat(counts[self._cat].astype(np.float64), self._starts) / self._lens
            for k, m in zip(self._keys, means.tolist()):
                self.rate[k] += a * (m / s - self.rate[k])
        self.active_frac += a * (np.count_nonzero(counts) / counts.size - self.active_frac)

    def observation(self, sens: dict, extra: dict) -> dict:
        r = {k: round(v, 1) for k, v in self.rate.items()}
        return {"neural": r, "active_frac": round(self.active_frac, 5), "body": _body_facts(sens), **extra}


def _body_facts(sens):
    """Facts with no brain pathway in the model yet; the LLM is told they are [body]-derived."""
    pos = np.array(sens["pos"][:2])
    h = np.array(sens["heading"][:2])
    from .body import SUGAR, ARENA_R
    to = np.array(sens.get("food_pos", SUGAR["pos"])) - pos
    dist = float(np.linalg.norm(to))
    bearing = float(np.degrees(np.arctan2(h[0] * to[1] - h[1] * to[0], h @ to)))  # + = to the left
    wall = float(ARENA_R - np.linalg.norm(pos))
    return {"pos": [round(float(x), 2) for x in pos], "heading_deg": round(float(np.degrees(np.arctan2(h[1], h[0])))),
            "mode": sens["mode"], "speed_mm_s": round(sens["v_forward"], 1),
            "food_bearing_deg": round(bearing), "food_dist_mm": round(dist, 1),
            "wall_dist_mm": round(wall, 1), "upright": sens["upright"] > 0.5,
            "touch": [p for p in sens["touch"] if not p.endswith(("tarsus5", "tarsus4"))][:6],
            "on_sugar_legs": int(sum(sens["sugar_legs"])),
            "on_water_legs": int(sum(sens.get("water_legs", ()))), "on_bitter_legs": int(sum(sens.get("bitter_legs", ()))),
            "temp_c": [round(x, 1) for x in sens.get("temp_antenna", [])]}


# Food-line format (see experiments/bias_eval.py and GROUNDING.md):
#   A: "food drop: 6.7 mm away, RIGHT (ahead-ish) (bearing -45 deg, + = left)"   (original)
#   B: egocentric "food: 6.7 mm away, 45 deg to your RIGHT (0 = straight ahead, 180 = directly behind)"
#   C: B + the action that faces it ("to face it: turn_right")
#   D: C, but when legs are on the sugar it says so and points to feeding instead of turning away
#   E: D, but within 6 mm the action that faces the food is rotate_x (needs bridge v3)
FOOD_FORMAT = os.environ.get("FLYSIM_LLM_FOOD", "E").upper()  # C: bias_eval_context.json; D (+ on-food cue): on_food_eval.json; E (+ rotate hint within 6 mm, bridge v3): authorship_*_v3e (2026-09-24)


def _food_text(b):
    bearing, dist = b["food_bearing_deg"], b["food_dist_mm"]
    if FOOD_FORMAT == "A":
        return (f"food drop: {dist} mm away, {_direction(bearing)} (bearing {bearing:+d} deg, + = left)")
    if FOOD_FORMAT in ("D", "E") and b.get("on_sugar_legs", 0) > 0:
        return (f"food: you are STANDING ON IT ({b['on_sugar_legs']} legs on the sugar, centre {dist} mm away); "
                f"to eat it: feed (walking or turning takes you off it)")
    side = "LEFT" if bearing > 0 else "RIGHT"
    a = abs(bearing)
    if a < 20:
        where, act = "straight AHEAD", "walk_forward"
    elif a > 150:
        where, act = f"BEHIND you ({a:.0f} deg, slightly to your {side})", f"turn_{side.lower()}"
    else:
        where, act = f"{a:.0f} deg to your {side}", f"turn_{side.lower()}"
    if FOOD_FORMAT == "E" and a >= 20 and dist < 6.0 and "rotate_left" in CATALOG:
        act = f"rotate_{side.lower()} (a walking turn would overshoot this close)"
    s = f"food: {dist} mm away, {where} (0 = straight ahead, 180 = directly behind)"
    return s + (f"; to face it: {act}" if FOOD_FORMAT in ("C", "D", "E") else "")


def _direction(bearing):
    a = abs(bearing)
    if a < 20:
        return "AHEAD"
    if a > 150:
        return "BEHIND"
    return ("LEFT" if bearing > 0 else "RIGHT") + (" (ahead-ish)" if a < 60 else "")


def _lvl(hz, lo=5, hi=60):
    return "none" if hz < lo else ("weak" if hz < hi else "STRONG")


def observation_text(o: dict) -> str:
    n, b = o["neural"], o["body"]
    own = set(o.get("self_driven", []))
    g = lambda k: n.get(k, 0.0)
    # sensory neurons the LLM is itself driving report its own input, not the world: hide the number
    mine = "(your own input, not the world)"

    def loom(s):
        if f"LPLC2_{s}" in own:
            return mine
        return f"{_lvl(g(f'LPLC2_{s}') + g(f'LC4_{s}'))} (LPLC2 {g(f'LPLC2_{s}'):.0f} Hz, LC4 {g(f'LC4_{s}'):.0f} Hz)"

    hz = lambda k: mine if k in own else f"{g(k):.0f} Hz"
    lines = [
        f"t={o.get('t', 0):.1f}s mode={b['mode']} speed={b['speed_mm_s']} mm/s upright={b['upright']}",
        f"VISION (neurons): looming LEFT {loom('left')}; looming RIGHT {loom('right')}",
        f"TASTE (neurons): sugar receptors {hz('sugar_GRN')}" + ("" if "sugar_GRN" in own else f" ({_lvl(g('sugar_GRN'), 5, 100)})"),
        f"TOUCH (neurons): antennal deflection JO-C/E {hz('JO_CE')}; eye bristles L {hz('eye_bristle_left')} / R {hz('eye_bristle_right')}",
        f"MOTOR (descending neurons): forward DNp09 {g('DNp09_left') + g('DNp09_right'):.0f} | backward MDN "
        f"{g('MDN_left') + g('MDN_right'):.0f} | turn-left DNa02+DNg13 {g('DNa02_left') + g('DNg13_left'):.0f} | "
        f"turn-right {g('DNa02_right') + g('DNg13_right'):.0f} | giant fibre {max(g('GF_left'), g('GF_right')):.0f} | "
        f"proboscis MN9 {g('MN9'):.0f} | antennal grooming aDN1/aDN2 {g('aDN1'):.0f}/{g('aDN2'):.0f} Hz",
        f"BRAIN: {100 * o['active_frac']:.2f}% of neurons active per 10 ms" + (" RUNAWAY STATE (was reset)" if o.get("runaway") else ""),
        f"[body] {_food_text(b)}; "
        f"arena wall {b['wall_dist_mm']} mm; touching: {', '.join(b['touch']) or 'nothing'}; legs on sugar: {b['on_sugar_legs']}",
    ]
    # world senses (world.py): shown only when active, so the default prompt is unchanged
    if g("water_GRN") + g("bitter_GRN") > 0.5:
        lines.insert(3, f"TASTE (neurons): water receptors {hz('water_GRN')}; bitter receptors {hz('bitter_GRN')}")
    th = ["TRN_cold_left", "TRN_cold_right", "HRN_moist_left", "HRN_moist_right"]
    T = b.get("temp_c") or []
    if sum(g(k) for k in th) > 0.5 or (T and max(abs(x - 24.0) for x in T) > 1.0):
        lines.insert(3, f"TEMPERATURE: [body] antennae L {T[0]:.1f} / R {T[1]:.1f} C (hot cells not modelled); "
                        f"(neurons) cold cells L {g('TRN_cold_left'):.0f} / R {g('TRN_cold_right'):.0f} Hz; "
                        f"moist cells {g('HRN_moist_left') + g('HRN_moist_right'):.0f} Hz")
    if o.get("active_intervention"):
        lines.append(f"YOUR ACTIVE INTERVENTION: {o['active_intervention']}. These readings are your own input, "
                     f"not the world: {', '.join(sorted(own)) or 'none'}")
    if o.get("effect"):
        lines.append("EFFECT of your last choice: " + o["effect"])
    return "\n".join(lines)


SYSTEM = """You are the executive layer above a simulated fruit fly (Drosophila). Its brain is a spiking model of the \
full FlyWire connectome (138,639 neurons) driving a physics body. You cannot move the body directly. You can only \
intervene on neurons: each option below is a real neural manipulation that the brain then processes.

Every ~{period:.0f} s you get an observation decoded from the fly's neurons (lines marked [body] come from the physics, \
not the brain). Reply with ONE JSON object and nothing else:
{{"intent": <one of {names}>, "strength": <0..1>, "duration_s": <0.3..1>, "why": <max 15 words>}}

Options:
{options}

Goal set by the user: {goal}
Keep interventions short and reactive; "none" is a good answer when the fly is already doing the right thing.
Read the "EFFECT of your last choice" line: if an intervention did not move the fly the way you wanted, change it.
The fly does not move by itself: with "none" or "stop" it stands still. Each intervention is a short bout
(at most 1 s) and the fly stands still while you decide the next one. Walking at strength 1 covers about
10-15 mm per second, so use low strength when the target is close."""


class LLMAgent:
    """Background thread: observation -> chat completion -> validated intent -> neural inputs."""

    def __init__(self, table, url=None, model=None, period_s=1.5, goal="explore the arena and find the food"):
        self.table = table
        self.url = (url or os.environ.get("FLYSIM_LLM_URL", "http://127.0.0.1:8081/v1")).rstrip("/")
        self.model = model or os.environ.get("FLYSIM_LLM_MODEL", "qwen3-8b")
        self.key = os.environ.get("FLYSIM_LLM_KEY", "")
        self.think = os.environ.get("FLYSIM_LLM_THINK", "0") == "1"
        self.last_thought = ""
        self.period = period_s
        self.goal = goal
        self.enabled = False
        self.resolved = {k: resolve(table, v) for k, v in CATALOG.items()}
        self.obs_fn = None          # set by the sim: () -> observation dict
        self.lock = threading.Lock()
        self.active = None          # (intent, rate, sil_idx, act_idx, t_end_wall)
        self.last = {"obs": "", "reply": "", "intent": "none", "why": "", "latency_s": 0.0, "error": "", "n": 0}
        # past turns in the prompt; 0 by default: small models copy their own history, and the
        # EFFECT line already closes the loop (llm_loop_food.json runs, GROUNDING.md)
        self.history = deque(maxlen=int(os.environ.get("FLYSIM_LLM_HISTORY", "0")) or None)             if os.environ.get("FLYSIM_LLM_HISTORY", "0") != "0" else deque(maxlen=0)
        self._prev = None           # (body facts, intent) at the previous decision
        self.log = deque(maxlen=200)
        self.forced = False
        self._th = None

    # ---- sim side (called every window, cheap) -------------------------------------------------
    def inputs(self):
        """-> (poisson {name: (idx, rate)}, silence idx or None)"""
        with self.lock:
            a = self.active
            if a is None or time.time() > a[4] or not (self.enabled or self.forced):
                self.active = None
                self.forced = False
                return {}, None
            name, rate, sil, act, _ = a
        return ({f"llm:{name}": (act, rate)} if len(act) and rate > 0 else {}), (sil if len(sil) else None)

    def driven(self):
        """(intent name, driven idx) of the active intervention, for the observer."""
        with self.lock:
            a = self.active
        if a is None or time.time() > a[4]:
            return None, np.zeros(0, np.int64)
        return a[0], a[3]

    def force(self, intent, strength=1.0, duration_s=1.0):
        """Apply one catalog intervention directly (UI buttons, tests): same path as an LLM choice."""
        if intent not in CATALOG:
            raise KeyError(intent)
        it = CATALOG[intent]
        act, sil = self.resolved[intent]
        rate = float(np.clip(it.lo + strength * (it.hi - it.lo), 0, MAX_RATE)) if it.route in ("motor", "sensory") else 0.0
        with self.lock:
            self.active = None if intent == "none" else (intent, rate, sil, act, time.time() + float(duration_s))
        self.forced = True

    # ---- thread ------------------------------------------------------------------------------
    def start(self):
        if self._th is None:
            self._th = threading.Thread(target=self._run, daemon=True)
            self._th.start()

    def set(self, on=None, goal=None, url=None, model=None, period=None):
        if goal:
            self.goal = str(goal)[:300]
        if url:
            self.url = str(url).rstrip("/")
        if model:
            self.model = str(model)
        if period:
            self.period = float(np.clip(period, 0.5, 10))
        if on is not None:
            self.enabled = bool(on)
            if not self.enabled:
                with self.lock:
                    self.active = None
                self._prev = None
                self.history.clear()
        self.start()

    def _run(self):
        while True:
            t0 = time.time()
            if self.enabled and self.obs_fn is not None:
                try:
                    self.step()
                except Exception as e:  # keep the loop alive; show the error in the UI
                    self.last["error"] = f"{type(e).__name__}: {e}"[:300]
            time.sleep(max(0.05, self.period - (time.time() - t0)))

    def prompt(self, obs_text):
        opts = "\n".join(f"- {i.name}: {i.what}" for i in CATALOG.values())
        sys_ = SYSTEM.format(period=self.period, names=list(CATALOG), options=opts, goal=self.goal)
        msgs = [{"role": "system", "content": sys_}]
        for o, rep in self.history:
            msgs += [{"role": "user", "content": o}, {"role": "assistant", "content": rep}]
        msgs.append({"role": "user", "content": obs_text})
        return msgs

    def step(self):
        obs = self.obs_fn()
        if self._prev is not None:
            pb, pi = self._prev
            b = obs["body"]
            moved = float(np.linalg.norm(np.subtract(b["pos"], pb["pos"])))
            turned = int((b["heading_deg"] - pb["heading_deg"] + 180) % 360 - 180)
            obs["effect"] = (f"after '{pi}' the fly moved {moved:.1f} mm and turned {turned:+d} deg (+ = left); "
                             f"food distance {pb['food_dist_mm']} -> {b['food_dist_mm']} mm")
        text = observation_text(obs)
        t0 = time.time()
        reply = self.chat(self.prompt(text))
        dt = time.time() - t0
        intent, strength, dur, why = self.parse(reply)
        it = CATALOG[intent]
        act, sil = self.resolved[intent]
        rate = float(np.clip(it.lo + strength * (it.hi - it.lo), 0, MAX_RATE)) if it.route in ("motor", "sensory") else 0.0
        with self.lock:
            self.active = None if intent == "none" else (intent, rate, sil, act, time.time() + dur)
        self._prev = (obs["body"], intent)
        self.history.append((text, json.dumps({"intent": intent, "strength": round(strength, 2),
                                                "duration_s": round(dur, 1), "why": why})))
        self.last = {"obs": text, "reply": reply[:400], "intent": intent, "strength": round(strength, 2),
                     "rate_hz": round(rate), "n_neurons": int(len(act) or len(sil)), "duration_s": round(dur, 1),
                     "why": why, "latency_s": round(dt, 2), "error": "", "n": self.last["n"] + 1,
                     "route": it.route, "evidence": it.evidence, "thought": self.last_thought}
        self.log.append({"t": obs.get("t"), "wall": time.time(), "intent": intent, "strength": strength,
                         "dur": dur, "why": why, "latency": dt, "obs": obs})

    def chat(self, messages):
        if self.think:   # reasoning models: let them think, then parse the JSON from the answer
            body = {"model": self.model, "messages": messages, "temperature": 0.6, "max_tokens": 2048,
                    "chat_template_kwargs": {"enable_thinking": True}}
        else:
            body = {"model": self.model, "messages": messages, "temperature": 0.4, "max_tokens": 120,
                    "response_format": {"type": "json_object"},
                    "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        with urllib.request.urlopen(req, timeout=60 if self.think else 20) as r:
            j = json.load(r)
        msg = j["choices"][0]["message"]
        self.last_thought = (msg.get("reasoning_content") or "")[-600:]
        c = msg.get("content") or ""
        return c.split("</think>")[-1]

    @staticmethod
    def parse(reply):
        """Tolerant parse; anything invalid becomes 'none' (never an unvetted intervention)."""
        s = reply[reply.find("{"): reply.rfind("}") + 1] if "{" in reply else ""
        try:
            j = json.loads(s)
        except Exception:
            return "none", 0.0, 1.0, "unparseable reply"
        intent = str(j.get("intent", "none")).strip()
        if intent not in CATALOG:
            return "none", 0.0, 1.0, f"unknown intent {intent[:30]}"
        try:
            strength = float(np.clip(float(j.get("strength", 0.5)), 0, 1))
            dur = float(np.clip(float(j.get("duration_s", 1.0)), 0.3, 1.0))
        except (TypeError, ValueError):
            strength, dur = 0.5, 1.0
        return intent, strength, dur, str(j.get("why", ""))[:120]
