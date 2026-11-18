"""Exogenous inputs that make the fly behave on its own. Everything here enters the BRAIN as
Poisson drive to named neurons; nothing writes the body's drive directly.

Two layers, kept separate so the research lane can replace either:

SensoryLayer (body state -> sensory neurons)
  - looming (GROUNDING.md item 4): the nearest moving object, seen from the head, drives the
    two looming-selective visual projection types on that side, split the way Ache et al.
    2019 (Curr Biol) split the GF's looming input: LPLC2 carries angular SIZE, LC4 carries
    angular VELOCITY. Angular size sets the fraction of that side's LPLC2 and LC4 cells
    recruited (the share of receptive fields the object covers: 20 deg -> 0, 90 deg -> all;
    fixed cell order, so recruitment is nested); recruited LPLC2 fire at 100 Hz and recruited
    LC4 at 60 Hz per rad/s of expansion (cap 150 Hz). loom_ladder2.json: GF silent below
    ~30 deg, crosses the take-off threshold near 35 deg, strong from 45 deg; the contralateral
    DNa02 (turn away) grows with size. The ranges are our calibration, not measured values.
  - sugar: any tarsus inside the sugar drop drives the 20 sugar GRNs from Shiu et al.
    2024 at 200 Hz (their 'activation' protocol).
  - touch (item 2): antenna contact drives that side's Johnston's-organ wind/gravity neurons
    (JO-C/E) and antennal bristles (BM_Ant) at 150 Hz; head contact drives the eye bristle
    mechanoreceptors (BM_InOm) of both sides at 80 Hz. probe_touch.json: these reach the
    antennal/eye grooming pathway (aBN1, aDN1, aDN2) without ignition. Leg contact has no
    brain pathway (the VNC is not modelled) and is left to the placeholder policy below.
  - world senses (world.py): water / bitter tarsal contact -> the Shiu et al. water / bitter GRN
    sets at 200 Hz; per-antenna temperature -> hot cells (TRN_VP2) or cold cells (TRN_VP3a/b) on
    that side, tonic above/below a threshold plus a phasic term on warming/cooling (hot and cold
    cells respond to temperature change, Gallio et al. 2011 Cell); humidity -> moist cells
    (HRN_VP5). Thresholds and gains are ours.
    Caps from results/world_sense_ignition.json: in this model the antennal-lobe attractor (item 6)
    ignites from the hot cells at ANY rate (2 Hz on the 3 left TRN_VP2 for 2 s), so heat is NOT
    sent to neurons: it stays a [body] fact. Cold cells and moist cells stayed quiet up to 10 Hz
    and ignited at 20 Hz; cold on BOTH sides at 10 Hz still reset once in 3 s. Water and bitter GRNs
    were safe to 200 Hz. Held for 4 s and combined (results/world_sense_combo.json, 3 trials each),
    moist cells at 10 Hz on both sides ignited 3/3 on their own, cold 4 + moist 4 + water 200 Hz 1/3,
    and cold 2 + moist 4 + water 200 Hz 0/3. So cold is capped at 4 Hz, moist at 4 Hz, and their sum
    per side at 6 Hz (moist gives way first). The runaway watchdog stays as the backstop.

Policy (PLACEHOLDER, not a model of fly decision making)
  - explore bouts: DNp09 L+R at 40-60 Hz for 1.5-4 s, with occasional DNa02 turn pulses
  - pauses of 0.5-2 s
  - touch reflex: MDN L+R 90 Hz for 0.5 s (back away), then a turn pulse
  - on sugar: stop walking for 3 s (the GRNs keep firing)
  - rare spontaneous take-off: DNp01 (giant fibre) 200 Hz for 30 ms, about once a minute
  - self-righting: if the fly is on its back for 1.5 s it is set upright (body.right()).
  - thermal / bitter avoidance (added with world.py): when an antenna reads above 29 C or below
    20 C, turn away from the warmer (colder) side (random if they are within 0.05 C), backing up
    first when it is extreme; bitter contact triggers the back-away reflex. A placeholder for innate
    avoidance, read from the [body] temperature because heat cannot enter the brain model
    (see the SensoryLayer caps above).
  - yielding: while a higher layer (cognition.py) holds a motor intent, the explore/pause/turn
    bouts are muted and the fly "resumes" when it lets go. Reflexes (touch, avoidance, GF,
    self-righting) are never muted.
"""
from __future__ import annotations

import os

import numpy as np

# arena-edge avoidance and a turn-away after a bump (2026-10-10, experiments/wall_check.py). FLYSIM_DETOUR=0: off.
EDGE_AVOID = os.environ.get("FLYSIM_DETOUR", "1") != "0"
ARENA_R_MM = float(__import__('os').environ.get('FLYSIM_ARENA_R', '30'))          # body.ARENA_R (not imported: autonomy stays free of the MuJoCo body module)
EDGE_MM = 4.5              # start turning in this far from the wall's inner face

LOOM_GAIN, LOOM_MAX = 60.0, 150.0               # LC4: Hz per rad/s of expansion
SIZE_LO, SIZE_HI, LPLC2_RATE = np.radians(20), np.radians(90), 100.0
TOUCH_ANT_RATE, TOUCH_HEAD_RATE = 150.0, 80.0
TASTE_RATE = 200.0
HOT_ON, COLD_ON = 26.0, 22.0             # deg C at the antenna
THERMO_TONIC, THERMO_PHASIC = 2.0, 5.0   # Hz per deg C; Hz per deg C/s
HOT_MAX, COLD_MAX, MOIST_MAX = 0.0, 4.0, 4.0     # Hz; hot = 0: any hot-cell drive ignites the AL attractor
AL_SIDE_MAX = 6.0                                  # Hz, cold + moist on one antenna (see docstring)
MOIST_ON, MOIST_GAIN = 50.0, 0.25        # % RH; Hz per % RH


def thermo_rates(T, dT):
    """(hot Hz, cold Hz) for one antenna at temperature T (deg C) changing at dT (deg C/s)."""
    hot = THERMO_TONIC * max(0.0, T - HOT_ON) + THERMO_PHASIC * max(0.0, dT) * (T > HOT_ON - 2)
    cold = THERMO_TONIC * max(0.0, COLD_ON - T) + THERMO_PHASIC * max(0.0, -dT) * (T < COLD_ON + 2)
    q = lambda r, cap: float(round(min(r, cap) / 2) * 2)
    return q(hot, HOT_MAX), q(cold, COLD_MAX)


class SensoryLayer:
    def __init__(self, table, seed=0):
        rng = np.random.default_rng(seed)
        self.lplc2 = {s: rng.permutation(table.type_side("LPLC2", s)) for s in ("left", "right")}
        self.lc4 = {s: rng.permutation(table.type_side("LC4", s)) for s in ("left", "right")}
        self.ant = {s: np.concatenate([table.find(f"JO_wind_{s}"), table.find(f"BM_Ant_{s}")]) for s in ("left", "right")}
        self.eye = np.concatenate([table.find("BM_InOm_left"), table.find("BM_InOm_right")])
        self.sugar = table.find("sugar_GRN")
        self.water = table.find("water_GRN")
        self.bitter = table.find("bitter_GRN")
        self.hot = {s: table.find(f"TRN_hot_{s}") for s in ("left", "right")}
        self.cold = {s: table.find(f"TRN_cold_{s}") for s in ("left", "right")}
        self.moist = {s: table.find(f"HRN_moist_{s}") for s in ("left", "right")}
        self.enabled = True
        self.out = {}

    def update(self, sens, body) -> dict:
        """Returns {name: (idx, rate)} for the input mixer."""
        out = {}
        if not self.enabled:
            self.out = out
            return out
        rate = float(np.clip(sens["loom_rate"] * LOOM_GAIN, 0, LOOM_MAX))
        rate = round(rate / 10) * 10
        frac = float(np.clip((sens["loom_angle"] - SIZE_LO) / (SIZE_HI - SIZE_LO), 0, 1))
        frac = round(frac * 20) / 20
        if frac > 0 and sens["loom_rate"] > 0:            # a receding object is not a loom
            side = sens.get("loom_side") or self._loom_side(body)
            k, c = self.lplc2[side], self.lc4[side]
            out["loom_size"] = (k[: max(1, int(frac * len(k)))], LPLC2_RATE)
            if rate > 0:
                out["loom_speed"] = (c[: max(1, int(frac * len(c)))], rate)
        for s in ("left", "right"):
            if f"{s[0]}_antenna" in sens["touch"]:
                out[f"touch_antenna_{s}"] = (self.ant[s], TOUCH_ANT_RATE)
        if any(p in sens["touch"] for p in ("c_head", "c_rostrum")):
            out["touch_head"] = (self.eye, TOUCH_HEAD_RATE)
        if any(sens["sugar_legs"]):
            out["sugar"] = (self.sugar, 200.0)
        if any(sens.get("water_legs", ())):
            out["water"] = (self.water, TASTE_RATE)
        if any(sens.get("bitter_legs", ())):
            out["bitter"] = (self.bitter, TASTE_RATE)
        if "temp_antenna" in sens:
            for i, s in enumerate(("left", "right")):
                hot, cold = thermo_rates(sens["temp_antenna"][i], sens["dtemp_antenna"][i])
                if hot > 0:
                    out[f"hot_{s}"] = (self.hot[s], hot)
                if cold > 0:
                    out[f"cold_{s}"] = (self.cold[s], cold)
                moist = float(round(min(MOIST_GAIN * max(0.0, sens["humid_antenna"][i] - MOIST_ON), MOIST_MAX) / 2) * 2)
                moist = min(moist, AL_SIDE_MAX - cold)
                if moist > 0:
                    out[f"moist_{s}"] = (self.moist[s], moist)
        self.out = out
        return out

    @staticmethod
    def _loom_side(body):
        # side of the fly the nearest live object is on (+y in the fly frame = left)
        d = body.d
        hp = d.xpos[body.head]
        R = d.xmat[body.thorax].reshape(3, 3)
        if body.ext_loom:                              # the hand overrides pool objects while it comes down
            rel = np.asarray(body.ext_loom[0][0]) - hp
            return "left" if (R.T @ rel)[1] >= 0 else "right"
        live = np.flatnonzero(body._obj_live)
        if not len(live):
            return "left"
        k = live[np.argmin(np.linalg.norm(d.xpos[body.obj_bodies[live]] - hp, axis=1))]
        rel = d.xpos[body.obj_bodies[k]] - hp
        return "left" if (R.T @ rel)[1] >= 0 else "right"


class Policy:
    def __init__(self, table, seed=0):
        self.rng = np.random.default_rng(seed)
        g = lambda t, s: table.type_side(t, s)
        self.n = {"DNp09": np.concatenate([g("DNp09", "left"), g("DNp09", "right")]),
                  "MDN": np.concatenate([g("MDN", "left"), g("MDN", "right")]),
                  "DNa02_left": g("DNa02", "left"), "DNa02_right": g("DNa02", "right"),
                  "GF": np.concatenate([g("DNp01", "left"), g("DNp01", "right")])}
        self.enabled = True
        self.neural = {}          # set by the sim: NeuralObserver rates (Hz), for the avoidance reflex
        self.yield_motor = False  # set by the sim while a cognition layer holds a motor intent
        self.activity = 1.0       # set by the sim from the time of day (World.activity): < 1 = longer rests
        # drugs, the scripted body channel (pharma.py): walking bouts, pauses, spontaneous take-off
        self.drug_explore = self.drug_rest = self.drug_takeoff = 1.0
        self.reset()

    def reset(self):
        self.state, self.t_left = "pause", 0.5
        self.fwd_rate = 50.0
        self.turn = None          # (side, rate, t_left)
        self.gf_left = 0.0
        self.backup_cool = 0.0
        self.feed_cool = 0.0
        self.upside_t = 0.0
        self.avoid_label = ""
        self.post_turn = None     # the side to turn after an avoidance back-up (None = random)
        self.label = "pause"

    def update(self, sens, dt, body) -> dict:
        if not self.enabled:
            self.label = "off"
            return {}
        self.t_left -= dt
        self.backup_cool -= dt
        self.feed_cool -= dt
        # self-righting
        self.upside_t = self.upside_t + dt if sens["upright"] < 0.2 else 0.0
        if self.upside_t > 1.5 and sens["mode"] == "walk":
            body.right()
            self.upside_t = 0.0
        # touch reflex
        front = [p for p in sens["touch"] if p.startswith(("c_head", "lf_", "rf_", "c_rostrum", "l_antenna", "r_antenna"))]
        resting = self.state == "pause" and self.activity < 0.35   # a fly at rest leans on what it touches
        if front and not resting and self.state != "backup" and self.backup_cool <= 0 and sens["mode"] == "walk":
            self.state, self.t_left = "backup", 0.5
            self.backup_cool = 1.5
            if EDGE_AVOID:
                # then turn away from the touched side (random when both or the head), as an avoidance turn: it is
                # not yielded to a higher layer, and the reflex gate holds the mind's intent off while it runs
                lt = any(p.startswith(("l_antenna", "lf_")) for p in front)
                rt = any(p.startswith(("r_antenna", "rf_")) for p in front)
                self.post_turn = "right" if lt and not rt else "left" if rt and not lt else self.rng.choice(["left", "right"])
                self._touch_turn = True
        if any(sens.get("bitter_legs", ())) and self.state != "backup" and self.backup_cool <= 0 and sens["mode"] == "walk":
            self.state, self.t_left = "backup", 0.5
            self.backup_cool = 1.5
        T = sens.get("temp_antenna")
        if T:
            for kind, excess in (("heat", max(T) - 29.0), ("cold", 20.0 - min(T))):
                if excess > 0 and self.turn is None and self.state != "backup" and self.backup_cool <= 0                         and sens["mode"] == "walk":
                    diff = (T[0] - T[1]) * (1 if kind == "heat" else -1)    # > 0: the left side is worse
                    away = ("right" if diff > 0 else "left") if abs(diff) > 0.05 else self.rng.choice(["left", "right"])
                    self.turn = (away, 120.0, 0.8)
                    # back up only when the worse air is AHEAD (antennae hotter / colder than the thorax); with the
                    # source behind or under the fly, backing up walked it into the flame and the turn-away walked
                    # it out again, over and over (fire_check.json, 2026-10-08) -- then walk on, away
                    tb_air = sens.get("temp_body")
                    ahead = tb_air is None or (max(T) >= tb_air if kind == "heat" else min(T) <= tb_air)
                    if excess > 5.0 and ahead:
                        self.state, self.t_left = "backup", 0.4
                        self.post_turn = away
                    elif excess > 5.0:
                        self.state, self.t_left = "explore", 1.5
                        self.fwd_rate = 60.0
                    self.backup_cool = 1.2
                    self.avoid_label = f"avoid {kind}"
        # arena edge: walking toward the wall near it, turn back in (the side that faces the centre sooner)
        self._pivot = False
        if EDGE_AVOID and sens["mode"] == "walk" and self.state == "explore" and "pos" in sens \
                and (self.turn is None or self.avoid_label in ("", "avoid touch")):
            p, hd = np.asarray(sens["pos"][:2], float), np.asarray(sens["heading"][:2], float)
            r = float(np.hypot(*p))
            if r > ARENA_R_MM - 0.6 - EDGE_MM and r > 1e-6:
                out = p / r
                hn = hd / (np.linalg.norm(hd) + 1e-9)
                if hn @ out > 0.2:                                   # heading outward
                    cr = hn[0] * out[1] - hn[1] * out[0]             # > 0: the wall is to the left
                    self.turn = ("right" if cr > 0 else "left", 120.0, 0.7)
                    self.avoid_label = "avoid edge"
            if self.avoid_label in ("avoid edge", "avoid touch") and r > ARENA_R_MM - 0.6 - 2.2 and hn @ out > 0.5:
                self._pivot = True              # right at the wall and facing it: turn on the spot, no forward drive
        if EDGE_AVOID and self.avoid_label == "avoid touch" and front:
            self._pivot = True                  # still touching what it bumped: turn on the spot until free
        if any(sens["sugar_legs"]) and self.state in ("explore", "pause") and self.backup_cool <= 0 and self.feed_cool <= 0:
            self.state, self.t_left = "feed", 3.0
            if EDGE_AVOID:     # its own cool-down: the old shared 8 s also turned the touch reflex off, so a fly
                self.feed_cool = 8.0   # walking off the sugar pushed into whatever it met (wall_check.py, 2026-10-11)
            else:
                self.backup_cool = 8.0  # walk off afterwards
        dx, dr = self.drug_explore, self.drug_rest
        if self.t_left <= 0:
            if self.state == "backup":
                self.state, self.t_left = "explore", self.rng.uniform(1.5, 4.0) * (0.5 + 0.5 * self.activity) * dx ** 0.5
                side = self.post_turn or self.rng.choice(["left", "right"])
                self.turn = (side, 120.0 if self.post_turn else 100.0, 0.9 if self.post_turn else 0.6)
                self.post_turn = None
                if getattr(self, "_touch_turn", False):
                    self.avoid_label = "avoid touch"
                    self._touch_turn = False
            elif self.state == "explore":
                # rests lengthen as activity falls (night, midday); bouts of walking shorten
                self.state, self.t_left = "pause", min(30.0, self.rng.uniform(0.5, 2.0) / max(self.activity, 0.1) ** 1.5
                                                       * dr / dx ** 0.5)
            else:
                self.state, self.t_left = "explore", self.rng.uniform(1.5, 4.0) * (0.5 + 0.5 * self.activity) * dx ** 0.5
                self.fwd_rate = self.rng.uniform(40, 60)      # speed under a drug: the sim scales the drive (vigour)
        # occasional turn pulses while exploring
        if self.state == "explore" and self.turn is None and self.rng.random() < dt / 1.5:
            self.turn = (self.rng.choice(["left", "right"]), self.rng.uniform(40, 100), self.rng.uniform(0.2, 0.8))
        # rare spontaneous take-off
        if sens["mode"] == "walk" and self.rng.random() < dt * self.activity * self.drug_takeoff / 60.0:
            self.gf_left = 0.03
        out = {}
        if self.state == "explore" and not self.yield_motor and not self._pivot:
            out["fwd"] = (self.n["DNp09"], round(self.fwd_rate / 5) * 5)
        elif self.state == "backup":
            out["back"] = (self.n["MDN"], 90.0)
        if self.turn is not None and self.yield_motor and self.state != "backup" and not getattr(self, "avoid_label", ""):
            self.turn = None   # exploratory turn pulses yield too; avoidance turns do not
        if self.turn is not None:
            side, rate, tl = self.turn
            out["turn"] = (self.n[f"DNa02_{side}"], round(rate / 5) * 5)
            tl -= dt
            self.turn = None if tl <= 0 else (side, rate, tl)
            if self.turn is None:
                self.avoid_label = ""
        if self.gf_left > 0:
            out["gf"] = (self.n["GF"], 200.0)
            self.gf_left -= dt
        state = "rest" if self.state == "pause" and self.activity < 0.35 else self.state
        self.label = (state + (" (yielding)" if self.yield_motor and self.state in ("explore", "pause") else "")
                      + (f" + turn {self.turn[0]}" if self.turn else "") + (f" [{self.avoid_label}]" if getattr(self, "avoid_label", "") else "")
                      + (" + GF" if "gf" in out else ""))
        return out


class InputMixer:
    """Combines Poisson drive from several sources (max rate per neuron) plus user
    silencing / current injection, and pushes to the brain only when something changed."""

    def __init__(self, n):
        self.n = n
        self.sources: dict[str, dict] = {}
        self.silenced = np.zeros(n, bool)
        self.i_ext = np.zeros(n, np.float32)
        self.drug_i = np.zeros(n, np.float32)       # resting-potential shift from drugs (pharma.py), mV
        self._last_key = None

    def set_source(self, name, entries: dict):
        self.sources[name] = entries

    def apply(self, brain):
        key = (tuple((s, k, float(r), len(i)) for s, e in sorted(self.sources.items())
                     for k, (i, r) in sorted(e.items())),
               int(np.count_nonzero(self.silenced)), int(np.count_nonzero(self.i_ext != 0)), float(self.i_ext.sum()),
               float(self.drug_i.sum()), float(np.abs(self.drug_i).max(initial=0.0)))
        # (self.i_ext != 0): the same count, ~9x faster than count_nonzero on a float array (138k every window)
        if key == self._last_key:
            return False
        rate = np.zeros(self.n)
        for e in self.sources.values():
            for idx, r in e.values():
                if len(idx):
                    rate[idx] = np.maximum(rate[idx], r)
        brain.set_inputs(rate, self.silenced, self.i_ext + self.drug_i)
        self._last_key = key
        return True

    def force(self):
        self._last_key = None
