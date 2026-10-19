"""Nociception: injury -> a nociceptor-like drive -> the connectome -> one pain signal read from the brain.

Design: research/PAIN_DESIGN.md (converged method). Every constant marked [ours] is a design choice, not a measurement.

1. Periphery (the fly has no VNC in this model, so this module stands in for the abdominal md neurons). Per side,
   n in [0, 1]:
     heat   clip((tb - 38) / 6): from the sim's harm onset (physiology.HEAT_HARM) to 1.0 at 44 C [ours; Jones et al.
            2025 put the adult md threshold near 40 C]. Both sides. tb lags the air (physiology.TAU_BODY 1.5 s), so a
            fly that touches a flame and darts off never warmed past 38 C and felt nothing (fire_check.json before
            2026-10-08: contact with antennae at 69 C, I = 0).
     contact heat (added 2026-10-08) [ours]: per side, clip((Ts - 38) / 18) with Ts the hottest surface that side
            touches (its antenna or any of its three tarsi, world.temp_surface), no lag: the cuticle in contact is at
            the air's temperature at once. 0.1 about 4 mm from a flame, 0.8 at 3 mm, 1.0 in it. Same entry route.
            Since 2026-10-10 full at 50 C (was 56), and a burn leaves an after-discharge like the hand's: 0.8 of the
            contact peak, decaying with tau 1.5 s [ours]. Before, a fly that touched and darted off drove the entry
            set for a few windows only and the pain group barely showed in the brain view. Caps unchanged (n <= 1,
            150 Hz).
     cold   0.5 * clip((7 - tb) / 10), a weak term and a guess (adult cold nociception is not in our sources).
     mech   the hand's graded hit (Sim._hand_tick): n = q at impact on the side it landed, decaying with tau 1.5 s
            (an injury discharge after the 0.35 s press [ours]; a press-only pulse was too brief to register
            through the 1 s smoothing below: pain_check.json, 2026-10-03 quick run, I peak 0.20).
     ache   0.3 * clip((1 - health) / 0.5): damaged tissue keeps a low tonic drive while it heals [ours], so the
            lingering ache also passes through the circuit and silencing the circuit removes it.
     opto   a user-set drive with no injury at all (an optogenetic-style control).
2. Entry: n -> Poisson drive on that side's half of the 20-neuron STAND-IN ascending set (data/noci_stand_in.json:
   the ascending neurons with the highest path weight onto PPL1, not the real md targets), 150 Hz at n = 1,
   quantised to 10 Hz. results/pain_probe.json: the full set at 200 Hz for 1 s never ignited (0/5 seeds).
3. Read-out: the PPL1 aversive dopamine neurons (16), filtered by the neural observer (tau 150 ms), compressed
   P = log(1 + r / R0) / log(1 + R_MAX / R0). PPL1 is part of the runaway attractor core: above 40 Hz the
   brain is igniting, not hurting, and P is gated to 0 (the watchdog resets it).
4. State: A = P low-passed (tau 1 s); L follows A up (tau 3 s) and decays slowly (tau 30 sim s); I = max(A, 0.6 L).
   The brain itself returns to baseline within about 100 ms (probe), so all lingering lives in L and in the ache.

Circuit checks: silencing the entry set (mixer silencing: their synapses are cut) or the read-out (PPL1 reads 0)
must remove the signal with the injury unchanged; opto must produce it with no injury. experiments/pain_check.py.
"""
from __future__ import annotations

from collections import deque

import numpy as np

NOCI_MAX_HZ = 150.0            # Hz at n = 1 [ours; 200 Hz tested safe for 1 s, sustained drive not soak-tested]
HEAT_ON, HEAT_FULL = 38.0, 44.0
SURF_ON, SURF_FULL = 38.0, 50.0    # contact heat [ours, 2026-10-08; full at 50 C since 2026-10-10, was 56]
BURN_TAU, BURN_KEEP = 1.5, 0.8     # burn after-discharge [ours, 2026-10-10]: s; share of the contact peak kept
COLD_ON, COLD_SPAN, COLD_GAIN = 7.0, 10.0, 0.5
ACHE_GAIN, ACHE_SPAN = 0.3, 0.5
MECH_TAU, MECH_MIN = 1.5, 0.03       # s; below this the discharge is over
R0 = 0.5                       # Hz, the knee of the compression
R_MAX = 14.0                   # Hz of PPL1 that reads as P = 1 (pain_check.json: both sides at n = 1 plateau at 14.2 Hz)
IGNITION_HZ = 40.0
TAU_A, TAU_L_UP, TAU_L_DOWN = 0.4, 3.0, 30.0   # TAU_A was 1.0 [ours]; 0.4 since 2026-10-08 so a burn reads within ~0.5 s
ONSET, SHARPEN, REARM_A, REARM_L = 0.25, 0.6, 0.1, 0.2
LEVELS = ((0.08, ""), (0.25, "faint"), (0.45, "mild"), (0.65, "moderate"), (0.85, "strong"), (9.0, "severe"))
PRESSURE = {"faint": "light", "mild": "light", "moderate": "firm", "strong": "heavy", "severe": "very heavy"}
QUALITY = {"heat": "burning", "cold": "aching cold", "mech": "sharp, crushing", "ache": "dull, aching",
           "opto": "sharp", "": "dull"}
MIND_MODES = ("text", "control", "numeric", "hidden")


def level_word(i):
    for hi, w in LEVELS:
        if i < hi:
            return w
    return "severe"


def hand_hit(fly_xy, heading_xy, hand_xy, r):
    """The hand's fingertip against the fly: (q in [0, 1], side, part). q = 1 - d / (r + 0.4) with d the distance
    from the thorax to the fingertip's centre [ours]; a fingertip centred within r - 1.2 mm covers the whole fly."""
    rel = np.asarray(hand_xy, float) - np.asarray(fly_xy, float)
    d = float(np.hypot(*rel))
    q = float(np.clip(1.0 - d / (r + 0.4), 0.0, 1.0))
    if d < r - 1.2:
        return q, "both", "whole body"
    h = np.asarray(heading_xy, float)
    h = h / (np.linalg.norm(h) + 1e-9)
    fwd = float(rel @ h)
    lat = float(h[0] * rel[1] - h[1] * rel[0])          # + = to the fly's left
    side = "both" if abs(lat) < 0.25 * d else ("left" if lat > 0 else "right")
    part = "abdomen" if fwd < -0.35 * d else "head" if fwd > 0.6 * d else "thorax" if abs(lat) < 0.5 * d else "legs"
    return q, side, part


def hand_damage(q):
    """Health lost to a hit of strength q: 0.1 for a glancing blow, lethal near the centre [ours]."""
    return 0.1 + 1.1 * q * q


class Nociception:
    def __init__(self, table):
        self.idx = {s: table.find(f"AN_noci_{s}") for s in ("left", "right")}
        self.all = table.find("AN_noci")
        self.ppl1 = table.find("PPL1")
        self.silence = ""              # "" | "entry" (the stand-in set) | "readout" (PPL1)
        self.mind_mode = "text"        # how the online mind gets it: text | control | numeric | hidden
        self.steer = False             # also send the pain scalar to a steering server (pain_server/)
        self.hist = deque(maxlen=240)  # (t, I, ppl1 Hz) at 2 Hz for the panel
        self.reset()

    def reset(self):
        self.mech = []                 # [n, side, part, t_end]
        self.burn = {}                 # side -> [n, part, t]: the burn after-discharge
        self.opto = None               # (n, side, t_end)
        self.n = {"left": 0.0, "right": 0.0}
        self.drive = {"left": 0.0, "right": 0.0}
        self.src, self.side, self.part = "", "", ""
        self.last_part = "body"
        self.ppl1_hz = 0.0
        self.P = self.A = self.L = self.I = 0.0
        self.gated = False
        self._armed = [True, True]
        self._onset = False
        self._hist_t = -1.0
        self.hist.clear()

    # ------------------------------------------------------------------ inputs
    def hit(self, q, side, part, t):
        self.mech.append([float(q), side, part, t])
        self.last_part = part

    def set_opto(self, n, side, seconds, t):
        self.opto = (float(np.clip(n, 0, 1)), side, t + float(seconds)) if n > 0 and seconds > 0 else None

    def periphery(self, phys, t, sens=None):
        """Per-side nociceptor drive n and which source dominates it."""
        tb = phys.tb if phys.tb is not None else 24.0
        terms = []                                           # (n, side, src, part)
        heat = float(np.clip((tb - HEAT_ON) / (HEAT_FULL - HEAT_ON), 0, 1))
        if heat > 0:
            terms.append((heat, "both", "heat", "whole body"))
        ts = (sens or {}).get("temp_surface")
        if ts:
            ta = (sens or {}).get("temp_antenna") or ts
            for i, s in enumerate(("left", "right")):
                v = float(np.clip((ts[i] - SURF_ON) / (SURF_FULL - SURF_ON), 0, 1))
                if v > 0:
                    part = "head" if ta[i] >= ts[i] - 0.5 else "legs"
                    terms.append((v, s, "heat", part))
                    b = self.burn.get(s)
                    if b is None or BURN_KEEP * v >= b[0] * float(np.exp(-(t - b[2]) / BURN_TAU)):
                        self.burn[s] = [BURN_KEEP * v, part, t]
        for s, b in list(self.burn.items()):
            v = b[0] * float(np.exp(-(t - b[2]) / BURN_TAU))
            if v < MECH_MIN:
                del self.burn[s]
            else:
                terms.append((v, s, "heat", b[1]))
        cold = COLD_GAIN * float(np.clip((COLD_ON - tb) / COLD_SPAN, 0, 1))
        if cold > 0:
            terms.append((cold, "both", "cold", "whole body"))
        live = []
        for m in self.mech:
            v = m[0] * float(np.exp(-(t - m[3]) / MECH_TAU))
            if v >= MECH_MIN:
                live.append(m)
                terms.append((v, m[1], "mech", m[2]))
        self.mech = live
        ache = ACHE_GAIN * float(np.clip((1.0 - phys.health) / ACHE_SPAN, 0, 1))
        if ache > 0:
            terms.append((ache, "both", "ache", self.last_part))
        if self.opto and self.opto[2] > t:
            terms.append((self.opto[0], self.opto[1], "opto", "abdomen"))
        elif self.opto:
            self.opto = None
        n = {"left": 0.0, "right": 0.0}
        for v, side, _, _ in terms:
            for s in ("left", "right"):
                if side in (s, "both"):
                    n[s] = max(n[s], v)
        # the place and quality the mind is told: an acute source while it lasts, else the ache of the healing tissue
        acute = [x for x in terms if x[2] != "ache"]
        top = max(acute or terms, key=lambda x: x[0]) if terms else (0.0, "", "", "")
        self.src, self.side, self.part = (top[2], top[1], top[3]) if top[0] > 0 else ("", "", "")
        return n

    def inputs(self, phys, t, sens=None):
        """Mixer source {name: (idx, Hz)} for this window."""
        self.n = self.periphery(phys, t, sens) if not phys.dead else {"left": 0.0, "right": 0.0}
        out = {}
        for s in ("left", "right"):
            hz = float(round(NOCI_MAX_HZ * self.n[s] / 10.0) * 10.0)
            self.drive[s] = hz
            if hz > 0 and len(self.idx[s]):
                out[f"noci_{s}"] = (self.idx[s], hz)
        return out

    def silenced_idx(self):
        return self.all if self.silence == "entry" else self.ppl1 if self.silence == "readout" else None

    # ------------------------------------------------------------------ read-out and state
    def update(self, rates, t, dt, dead=False):
        """After the brain window: read PPL1, update A / L / I. Returns events [(kind, text, prio)] to log."""
        r = 0.0 if dead else float(rates.get("PPL1", 0.0))
        self.ppl1_hz = r
        self.gated = r > IGNITION_HZ
        self.P = 0.0 if self.gated else float(min(1.0, np.log1p(r / R0) / np.log1p(R_MAX / R0)))
        if dead:
            self.A = self.L = 0.0
        self.A += (1.0 - np.exp(-dt / TAU_A)) * (self.P - self.A)
        tau = TAU_L_UP if self.A > self.L else TAU_L_DOWN
        self.L += (1.0 - np.exp(-dt / tau)) * (self.A - self.L)
        self.I = float(max(self.A, 0.6 * self.L))
        if t - self._hist_t >= 0.5:
            self._hist_t = t
            self.hist.append((round(t, 1), round(self.I, 3), round(r, 2)))
        ev = []
        if dead:
            return ev
        if self._armed[0] and self.A >= ONSET:
            self._armed[0], self._onset = False, True
            ev.append(("pain", self.sensor_line("text") or "pain", True))
        if self._armed[1] and self.A >= SHARPEN:
            self._armed[1] = False
            ev.append(("pain", self.sensor_line("text") or "pain", True))
        if self._onset and self.A < REARM_A:
            self._onset = False
            ev.append(("pain_eases", "the pain is easing", False))
        if not self._armed[0] and self.A < REARM_A and self.L < REARM_L:
            self._armed = [True, True]
        return ev

    # ------------------------------------------------------------------ what the mind gets
    def where(self):
        side, part = self.side, self.part
        if not part:                      # lingering after the source stopped, or PPL1 moved by something else
            part, side = self.last_part, "both"
        if part in ("whole body", "body"):
            return "all over your body"
        if part == "legs":
            return {"left": "your left legs", "right": "your right legs"}.get(side, "your legs")
        return {"left": f"the left side of your {part}", "right": f"the right side of your {part}"}.get(side, f"your {part}")

    def sensor_line(self, mode=None):
        """One line in the register of temperature and hunger: place, quality, level, present tense, no emotion
        words, no imperatives. mode: text (nociceptive) | control (matched non-nociceptive) | numeric (Keeling
        et al. 2024 arm) | hidden."""
        mode = mode or self.mind_mode
        lvl = level_word(self.I)
        if not lvl or mode == "hidden":
            return ""
        where = self.where()
        where = where[0].upper() + where[1:]
        if mode == "control":
            return f"{where}: pressure, {PRESSURE[lvl]}"
        if mode == "numeric":
            return f"{where}: pain {int(round(10 * self.I))} out of 10"
        q = QUALITY.get(self.src if self.src else "ache" if self.L > 0.1 else "", "dull")
        easing = ", easing" if self.A < 0.5 * self.L else ""
        return f"{where}: {q}, {lvl}{easing}"

    def steer_scalar(self):
        return round(self.I, 3) if self.steer else None

    def status(self):
        return {"n": [round(self.n["left"], 3), round(self.n["right"], 3)],
                "drive_hz": [self.drive["left"], self.drive["right"]],
                "ppl1_hz": round(self.ppl1_hz, 2), "P": round(self.P, 3), "A": round(self.A, 3),
                "L": round(self.L, 3), "I": round(self.I, 3), "gated": self.gated,
                "src": self.src, "where": self.where() if self.I >= 0.08 else "", "level": level_word(self.I),
                "line": self.sensor_line(), "silence": self.silence, "mind_mode": self.mind_mode, "steer": self.steer,
                "opto": list(self.opto[:2]) if self.opto else None, "hist": list(self.hist)}
