"""Lever 2: the body throttles cognition (2026-10-08).

Each wake the mind (cognition.Cognition.mind_step) reads a small DRIVE VECTOR computed here from the sim, and the
drives limit what the mind can hold, notice, say, and how often it thinks. The fly's state (System 1: reflexes,
needs, the connectome's own activity) sets the room the language model (System 2) has to plan in; the plan it
writes (NOW) still leans on the legs through the motor layer, so System 2 guides the next System 1 actions only as
far as the body lets it think.

Drives, each 0..1. [neural] = read from connectome population rates (llm_bridge.READ_POPS via sim.observer.rate,
filtered with tau 150 ms); [body] = from physiology / pharmacology / nociception state, because the connectome
model has no meaningful read-out for it. All mappings and constants are ours (demo choices, not measurements).

  hunger        [body]   1 - energy reserve (physiology.Physiology). The model has no hunger state in its neurons.
  thirst        [body]   1 - water reserve.
  threat        [neural] looming detectors and the escape command: LPLC2 (angular size) and LC4 (angular velocity)
                         population rates, max over sides, and the giant fibre (DNp01). Ache et al. 2019 (Curr Biol,
                         doi:10.1016/j.cub.2019.01.079): LPLC2 and LC4 are the GF's looming inputs.
                [body]   max'd with a geometric term (the hand coming down: horizontal distance and height; the
                         body's loom angle), so a hand overhead that is not yet looming still counts a little.
  arousal       [neural] overall firing (observer.active_frac, the share of the 138,639 neurons spiking per window,
                         relative to ACT_HI) and the walking / backing / escape commands (DNp09, MDN, GF).
  fatigue       [body]   low energy, low health, and an exertion store that fills with walking and flight and empties
                         at rest (tau FATIGUE_TAU_S); time awake adds slowly (AWAKE_FULL_S).
  pain          [body]   nociception.Nociception.I (itself read out from the PPL1 stand-in, so partly [neural]).
  intoxication  [body]   pharma phenotypes: ataxia, tremor, circling, the explore lift of a rising ethanol level, and
                         a seizure (drugged brain in its runaway state) pins it to 1.
  sedation      [body]   pharma phenotypes: rest gain, negative explore (sluggish), knock-down; chill coma counts.
  temp_stress   [body]   body temperature tb against the comfort band 20..30 C, scaled to CT_MAX 41 C / CT_MIN 7 C
                         (physiology.py; those two limits are measured, the band is ours).

Why hunger, thirst, fatigue and sedation are [body]: the brain model (Shiu et al. 2024, Nature,
doi:10.1038/s41586-024-07763-9) has no neuropeptides or neuromodulation and a 0 Hz basal rate, so need LEVELS,
sleep pressure and octopaminergic arousal have no read-out in it; they are hormonal / modulatory states and are
taken from physiology. Connectome read-outs are used only where the model carries the pathway: looming
(LPLC2 / LC4 -> giant fibre) and the locomotor command DNs (DNp09, MDN) as a motor-arousal read-out. "arousal"
is therefore motor/firing arousal, not octopamine arousal.

TODO (interface only, not built): interoceptive_inputs() could inject physiology states as Poisson drive into
identified interoceptive neurons (ISNs for hunger / thirst, R5 ring neurons for sleep pressure), the way
nociception.py drives its stand-in entry set, so these drives pass through the connectome before a read-out.

The throttle (Throttle) is a pure function of the drives and the lever toggles, so it can be tested without a
model, and the same drive vector goes to the local server in the request body (body["drives"]) for lever 1.
"""
from __future__ import annotations

import os
import random
import re

import numpy as np

# ---------------------------------------------------------------- drive constants [ours]
LOOM_HZ = 40.0          # LPLC2 / LC4 population mean rate (Hz) read as full threat (recruited cells fire 60-150 Hz,
                        # the population mean is recruited fraction x rate; autonomy.py)
GF_HZ = 60.0            # giant fibre filtered rate read as full threat / arousal
WALK_HZ = 150.0         # DNp09 / MDN filtered rate read as full motor arousal (60 saturated in ordinary walking:
                        # results/embodiment_check.json, first run)
ACT_LO, ACT_HI = 0.002, 0.015   # active_frac read as 0 and 1 arousal (the runaway watchdog trips at 0.02)
HAND_NEAR_MM = 12.0     # horizontal distance at which an overhead hand starts to count
FATIGUE_TAU_S = 90.0    # exertion store time constant (sim s)
AWAKE_FULL_S = 600.0    # sim s alive that adds the full time-awake share
COMFORT_LO, COMFORT_HI = 20.0, 30.0
MN9_HZ = 60.0           # MN9 rate that counts as full feeding drive (feeding gate is 10 Hz; ours)
CT_MAX, CT_MIN = 41.0, 7.0

FINAL_HOLD_S = 2.5          # sim s after death that the last thought keeps the dying feelings [ours, cinema]
DRIVE_KEYS = ("hunger", "thirst", "threat", "arousal", "fatigue", "pain", "intoxication", "sedation", "temp_stress")
SOURCES = {
    "hunger": "body: 1 - energy",
    "thirst": "body: 1 - water",
    "threat": "neural: LPLC2/LC4/GF rates; body: hand proximity, loom angle",
    "arousal": "neural: active fraction, DNp09/MDN/GF rates",
    "fatigue": "body: energy, health, exertion, time awake",
    "pain": "body: nociception I (PPL1 read-out)",
    "intoxication": "body: pharma ataxia/tremor/circling/ethanol, seizure",
    "sedation": "body: pharma rest/sluggish/knock-down, chill coma",
    "temp_stress": "body: tb outside 20-30 C",
}
LEVERS = ("wm", "attn", "budget", "noise", "tempo", "intero")
LEVER_NAMES = {"wm": "working memory", "attn": "attention", "budget": "thought budget", "noise": "noise",
               "tempo": "tempo", "intero": "body line"}


def _c(x):
    return float(np.clip(x, 0.0, 1.0))


class Embodiment:
    """Computes the drive vector from a Sim. tick() runs on the sim thread (cheap, every 10th window) to keep the
    exertion store; drives() returns the latest vector."""

    def __init__(self, sim):
        self.sim = sim
        self.exertion = 0.0
        self._acc = 0.0
        self.d = {k: 0.0 for k in DRIVE_KEYS}
        self.flags = {"seizure": False, "dominant": ""}

    def reset(self):
        self.exertion = 0.0
        self.d = {k: 0.0 for k in DRIVE_KEYS}

    def tick(self, dt):
        self._acc += dt
        if self._acc < 0.1:
            return
        dt, self._acc = self._acc, 0.0
        s = self.sim
        v = abs(float(s.sens.get("v_forward", 0.0)))
        work = min(1.0, v / 12.0) + (1.5 if s.sens.get("mode") == "flight" else 0.0)
        self.exertion += (dt / FATIGUE_TAU_S) * (work - self.exertion * 0.6)     # fills with work, empties at rest
        self.exertion = float(np.clip(self.exertion, 0.0, 1.0))
        self.compute()

    def compute(self):
        s = self.sim
        r = s.observer.rate
        ph, sens = s.phys, s.sens
        g = lambda k: float(r.get(k, 0.0))
        # threat
        loom_n = max(g("LPLC2_left"), g("LPLC2_right"), g("LC4_left"), g("LC4_right")) / LOOM_HZ
        gf = max(g("GF_left"), g("GF_right"))
        geo = 0.0
        h = getattr(s, "hand", None)
        if h is not None and not ph.dead:
            p = sens["pos"]
            dist = float(np.hypot(p[0] - h["x"], p[1] - h["y"]))
            geo = _c(1.0 - max(0.0, dist - h["r"]) / HAND_NEAR_MM) * (0.5 + 0.5 * _c(1.0 - h["z"] / 26.0))
        la = float(sens.get("loom_angle", 0.0) or 0.0)
        geo = max(geo, _c((la - 0.2) / 0.8) if (sens.get("loom_rate", 0) or 0) > 0 else 0.0)
        threat = max(_c(loom_n), _c(gf / GF_HZ), 0.8 * geo)
        # arousal
        act = _c((s.observer.active_frac - ACT_LO) / (ACT_HI - ACT_LO))
        motor = max(g("DNp09_left"), g("DNp09_right"), g("MDN_left"), g("MDN_right")) / WALK_HZ
        arousal = _c(0.55 * act + 0.45 * _c(motor) + 0.5 * _c(gf / GF_HZ))
        # pharma
        pm = s.pharma
        phen = dict(pm.phen)
        c = dict(pm.c)
        seizure = bool(getattr(pm, "seizing", False))
        intox = max(phen.get("ataxia", 0.0), phen.get("tremor", 0.0), 0.6 * phen.get("circling", 0.0),
                    0.5 * c.get("ethanol", 0.0) / 1.5)
        intox = 1.0 if seizure else _c(intox)
        sed = max(phen.get("rest", 0.0) / 1.2, -phen.get("explore", 0.0) / 1.5, 1.0 if pm.down else 0.0,
                  1.0 if ph.state == "coma" else 0.0, 0.6 if ph.state == "stupor" else 0.0)
        # fatigue
        fat = _c(0.45 * _c((0.5 - ph.energy) / 0.5) + 0.25 * _c((0.6 - ph.health) / 0.6)
                 + 0.35 * self.exertion + 0.15 * _c(ph.alive_s / AWAKE_FULL_S))
        tb = ph.tb if ph.tb is not None else 24.0
        temp = max(_c((tb - COMFORT_HI) / (CT_MAX - COMFORT_HI)), _c((COMFORT_LO - tb) / (COMFORT_LO - CT_MIN)))
        d = {"hunger": _c(ph.hunger), "thirst": _c(ph.thirst), "threat": threat, "arousal": arousal, "fatigue": fat,
             "pain": _c(s.noci.I), "intoxication": intox, "sedation": _c(sed), "temp_stress": temp}
        if ph.dead:
            d = {k: 0.0 for k in DRIVE_KEYS}
        self.d = {k: round(v, 3) for k, v in d.items()}
        self.flags = {"seizure": seizure, "dominant": dominant(self.d)}
        return self.d

    def drives(self):
        return dict(self.d)

    def steer_drives(self):
        """The drive levels lever 1 steers with (pain_server/drive_steer.py STEER_DRIVES), next to the nine drives:
        threat, arousal, pain as above (neural read-outs); hunger also rises with the feeding motor neuron MN9
        (sugar GRN -> MN9, the connectome's feeding read-out) so tasting sugar while fed still pushes the feeding
        direction; heat and cold are temp_stress split by sign (tb above 30 C / below 20 C)."""
        d = dict(self.d)
        s = self.sim
        if s.phys.dead:
            # the last thought is steered by the feelings it died with (held for FINAL_HOLD_S sim s after death),
            # then nothing: a dead fly sends no drives
            if getattr(self, "_dead_t", None) is None:
                self._dead_t = s.sim_t
            last = getattr(self, "_last_steer", None)
            if last and s.sim_t - self._dead_t < FINAL_HOLD_S:
                return dict(last)
            return {k: 0.0 for k in (*DRIVE_KEYS, "heat", "cold")}
        self._dead_t = None
        d = self._steer_now()
        self._last_steer = dict(d)
        return d

    def _steer_now(self):
        """steer_drives' values for a live fly, with no side effects (the stage's feelings panel reads them)."""
        d = dict(self.d)
        s = self.sim
        mn9 = float(s.observer.rate.get("MN9", 0.0))
        tb = s.phys.tb if s.phys.tb is not None else 24.0
        d["hunger"] = round(max(d["hunger"], _c(mn9 / MN9_HZ)), 3)
        d["heat"] = round(_c((tb - COMFORT_HI) / (CT_MAX - COMFORT_HI)), 3)
        d["cold"] = round(_c((COMFORT_LO - tb) / (COMFORT_LO - CT_MIN)), 3)
        return d

    def interoceptive_inputs(self):
        """TODO hook (see module docstring): {name: (neuron idx, Hz)} for the mixer, driving ISNs (hunger, thirst)
        and R5 (sleep pressure) from physiology. Returns nothing until those cell sets are identified in the table."""
        return {}

    def status(self):
        try:                                   # what goes to the AI as drives now (frozen at death: the last ones)
            if self.sim.phys.dead:
                steer = dict(self._last_steer)
            else:
                steer = self._last_steer = self._steer_now()     # also keeps the dying feelings current
        except Exception:
            steer = dict(getattr(self, "_last_steer", None) or {})
        return {"drives": dict(self.d), "flags": dict(self.flags), "exertion": round(self.exertion, 3),
                "steer": {k: round(float(v), 3) for k, v in steer.items()}}


def dominant(d, thresh=0.55):
    """The drive that narrows attention: the strongest of hunger, thirst, threat, pain (urgent ones first on ties)."""
    cand = {"threat": d["threat"], "pain": d["pain"], "thirst": d["thirst"] * 0.95, "hunger": d["hunger"] * 0.9}
    k = max(cand, key=cand.get)
    return k if cand[k] >= thresh else ""


# ---------------------------------------------------------------- the throttle [ours]
# kinds the dominant drive keeps in WHAT YOU SENSE NOW (cognition._FLY_NAME keys; "fly" = other flies, "wall")
ATTEND = {"hunger": {"sugar"}, "thirst": {"water"}, "threat": {"fly", "heat", "block", "wall"},
          "pain": {"heat", "cold", "bitter"}}


class Throttle:
    """What the body allows the mind this wake. Every field is derived from the drives; a lever that is off leaves
    its fields at the unthrottled values (cap 1, the base budget, the base temperature, tempo 1)."""

    def __init__(self, d, levers, local=True, base_tokens=None, base_temp=0.9, flags=None):
        self.d = d
        on = lambda k: bool(levers.get(k, False))
        self.levers = {k: on(k) for k in LEVERS}
        need = _c((max(d["hunger"], d["thirst"]) - 0.5) / 0.5)
        self.stress = max(d["threat"], d["pain"], d["temp_stress"], 0.7 * need)
        # working memory capacity
        cap = 1.0 - 0.7 * self.stress - 0.6 * d["intoxication"] - 0.55 * d["sedation"] - 0.3 * d["fatigue"]
        self.capacity = float(np.clip(cap, 0.1, 1.0)) if on("wm") else 1.0
        cp = self.capacity
        self.n_beliefs = (3 if local else 8) if cp >= 0.95 else int(round((3 if local else 8) * max(0.0, cp - 0.2) / 0.75))
        self.n_thoughts = (1 if local else 3) if cp >= 0.95 else (0 if cp < 0.45 else 1)
        self.n_story = 6 if cp >= 0.95 else (0 if cp < 0.5 else int(round(6 * (cp - 0.4) / 0.55)))
        self.keep_far = cp >= 0.6          # below: things "some way off" / "far off" drop out of perception
        self.lose_plan = cp < 0.25
        self.n_recent = 6 if cp >= 0.95 else max(1, int(round(6 * cp)))
        # attention
        dom = dominant(d)
        self.focus = dom if on("attn") else ""
        # thought budget
        b = max(0.6 * d["arousal"], d["threat"], d["pain"], 0.5 * d["temp_stress"])
        self.load = b if on("budget") else 0.0
        bt = base_tokens if base_tokens is not None else (140 if local else 420)
        if not on("budget") or b < 0.4:
            self.length = "normal"
        elif b < 0.75:
            self.length = "short"
        else:
            self.length = "fragment"
        if on("budget"):
            # local: the output is the whole budget; OpenRouter: reasoning tokens share it, so keep a floor of 240
            floor = 48 if local else 240
            self.max_tokens = int(max(floor, round(bt * (1.0 - 0.6 * b))))
        else:
            self.max_tokens = bt
        # noise
        seiz = bool((flags or {}).get("seizure"))
        self.temperature = base_temp
        self.slur = 0.0
        if on("noise"):
            self.temperature = round(float(min(1.6, base_temp + 0.25 * d["intoxication"] + (0.4 if seiz else 0.0))), 3)
            # (0.5 x intoxication until 2026-10-24; halved when ethanol also became noise inside the model, Perturb)
            self.slur = round(0.3 * d["intoxication"], 3)
        # tempo: the floor and idle stretch between thoughts; sedation can stop wakes
        if on("tempo"):
            self.tempo = round(1.0 + 2.0 * d["fatigue"] + 4.0 * d["sedation"], 2)
            self.suppress = d["sedation"] >= 0.85
        else:
            self.tempo, self.suppress = 1.0, False
        self.intero = intero_line(d) if on("intero") else ""

    def length_line(self):
        return {"normal": "",
                "short": "THIS THOUGHT: one short sentence, 12 words at most. There is no room in you for more.\n",
                "fragment": "THIS THOUGHT: only a few broken words, a fragment, 6 words at most.\n"}[self.length]

    def as_dict(self):
        return {"capacity": round(self.capacity, 3), "stress": round(self.stress, 3), "n_beliefs": self.n_beliefs,
                "n_thoughts": self.n_thoughts, "n_story": self.n_story, "n_recent": self.n_recent,
                "keep_far": self.keep_far, "lose_plan": self.lose_plan, "focus": self.focus,
                "length": self.length, "max_tokens": self.max_tokens, "temperature": self.temperature,
                "slur": self.slur, "tempo": self.tempo, "suppress": self.suppress, "intero": self.intero,
                "levers": dict(self.levers)}


def intero_line(d):
    """How the body is, in plain body words (not what to feel): the strongest three drives above 0.35."""
    words = {
        "hunger": ("your belly is empty", "your belly is empty and gnawing"),
        "thirst": ("your mouth is dry", "you are dry all through"),
        "threat": ("every part of you is braced to flee", "your whole body is ready to spring away"),
        "arousal": ("your body is buzzing and quick", "your body is racing, every nerve lit"),
        "fatigue": ("your legs are heavy", "your legs are heavy and slow, everything is an effort"),
        "pain": ("it hurts", "the hurt fills everything"),
        "intoxication": ("the world tilts and swims", "nothing stays still, your body will not hold a line"),
        "sedation": ("you are slow and dim", "you are sinking, everything far away and soft"),
        "temp_stress": ("your body is not at the right warmth", "your body is far too hot or too cold to work"),
    }
    top = sorted(((v, k) for k, v in d.items() if v >= 0.35 and k in words), reverse=True)[:3]
    return "; ".join(words[k][1 if v >= 0.7 else 0] for v, k in top)


def item_kind(line):
    """Which ATTEND kind a WHAT-YOU-SENSE line is about (cognition.mind_around wording)."""
    s = line.lower()
    if re.search(r"\bfly\b", s.split(":")[0]):
        return "fly"
    for k, w in (("sugar", "sweetness"), ("water", "water"), ("bitter", "bitter"), ("heat", "heat"),
                 ("cold", "cold"), ("block", "obstacle"), ("wall", "wall")):
        if w in s:
            return k
    return ""


def narrow(lines, focus):
    """Attentional narrowing: keep the lines about the dominant drive's objects; the rest fade."""
    if not focus:
        return lines
    keep = ATTEND.get(focus, set())
    out = [x for x in lines if item_kind(x) in keep or x.startswith("you are standing")]
    n = len(lines) - len(out)
    if n > 0:
        out.append("other things are there at the edge of notice, but you cannot attend to them")
    return out or ["nothing you can attend to"]


FAR = ("some way off", "far off")


def drop_far(lines):
    out = [x for x in lines if not any(f in x for f in FAR)]
    return out or ["nothing close enough to notice"]


def slur(text, p, rng=None):
    """Token-level degradation for intoxication: stretched vowels, dropped and run-together words. p = per-word
    chance. Applied to the finished thought only (not the format lines)."""
    if p <= 0 or not text:
        return text
    rng = rng or random.Random()
    ws = text.split(" ")
    out, i = [], 0
    while i < len(ws):
        w = ws[i]
        x = rng.random()
        if x < p * 0.4 and len(w) > 2:
            w = re.sub(r"([aeiou])", lambda m: m.group(1) * 3, w, count=1)
        elif x < p * 0.7 and i + 1 < len(ws) and os.environ.get("FLYSIM_SLUR_NO_JOIN") != "1":   # video takes: off (reads as a typo)
            w = w.rstrip(",.") + ws[i + 1]
            i += 1
        elif x < p * 0.85 and len(ws) > 4:
            i += 1
            continue
        out.append(w)
        i += 1
    return " ".join(out)
