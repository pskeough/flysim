"""Other flies in the arena: each has its OWN whole-brain LIF state on the GPU (a Brain.clone of the
main fly's: same FlyWire connectome, shared read-only on the device, separate membranes, inputs and
noise), and the same sensory layer, placeholder policy, DN bridge, runaway watchdog and physiology as
the main fly. Hunger, thirst, heat, cold and the hand kill them the same way. They have no LLM: `mind`
is an empty slot for a later per-fly mind (local model or another provider).

What they do NOT have is the main fly's MuJoCo body, which is what limits the speed of the whole sim
(about 1 s of wall time per simulated second on this CPU). A LiteBody walks kinematically:
  - its forward speed and yaw rate for a descending drive (left, right) come from a lookup table
    measured on the full FlyBody (experiments/lite_calib.py -> data/lite_body.npz, 7 x 7 drives,
    1 s each: 15.6 mm/s forward at (1, 1), 9.0 mm/s back at (-1, -1), 3.7 rad/s at (1, -1)), with a
    first-order lag of 80 ms (ours);
  - legs, head and wings are drawn from a walking gait and standing / limp / dead poses recorded
    from the same full body, so they look like it;
  - it is a mocap capsule in the MuJoCo world (body.py, oth*), so the main fly bumps into it and feels
    it through ordinary contacts, and it feels walls, scenery, blocks, the main fly and the other flies
    by short rays from its thorax (head and antenna touch, and it cannot walk through them);
  - take-off (giant fibre) is a jump and a short flight to a landing spot away from the threat
    (placeholder path and speed, ours), during which it looms at the other flies' eyes.
"""
from __future__ import annotations

from pathlib import Path

import os
import mujoco as mj
import numpy as np

from .autonomy import InputMixer, Policy, SensoryLayer
from .body import ARENA_R, MAX_OTHERS
from .bridge import Bridge
from .llm_bridge import NeuralObserver
from .physiology import Physiology

CAL_PATH = Path(__file__).resolve().parents[1] / "data" / "lite_body.npz"
TAU_V = 0.08                  # s, lag of speed and turning behind the drive (ours)
RAY_Z = (0.45, 1.05)          # ray heights (mm): low things (a stem, a twig) and body height
HEAD_X = 1.0                  # thorax to antenna tip, mm (stand pose: antennae at x = 0.96)
TAIL_X = 1.7                  # thorax to abdomen tip, mm
TOUCH_GAP, BLOCK_GAP = 0.35, 0.08
FLY_R = 1.2                   # a flying fly as a looming object: radius, mm (wings spread)


class Calib:
    def __init__(self, path=CAL_PATH):
        z = np.load(path)
        self.grid, self.v, self.w = z["grid"], z["v"], z["yaw"]
        self.names = [str(x) for x in z["names"]]
        self.walk, self.stand, self.limp, self.dead = z["walk"], z["stand"], z["limp"], z["dead"]
        # the main body's death animation as frames (experiments/lite_death_frames.py): curl, tuck, roll onto the back
        self.dying = z["dying"] if "dying" in z.files else None
        self.dying_dt = float(z["dying_dt"]) if "dying_dt" in z.files else 0.02
        self.walk_hz = float(z["walk_hz"])
        n = self.names
        self.tarsi = [n.index(f"{l}_tarsus5") for l in ("lf", "lm", "lh", "rf", "rm", "rh")]
        self.ant = [n.index("l_antenna"), n.index("r_antenna")]

    def lookup(self, dl, dr):
        """Bilinear interpolation of (v_forward mm/s, yaw rate rad/s) measured on the full body."""
        g = self.grid
        x, y = float(np.clip(dl, g[0], g[-1])), float(np.clip(dr, g[0], g[-1]))
        i = min(int(np.searchsorted(g, x, side="right")) - 1, len(g) - 2)
        j = min(int(np.searchsorted(g, y, side="right")) - 1, len(g) - 2)
        tx, ty = (x - g[i]) / (g[i + 1] - g[i]), (y - g[j]) / (g[j + 1] - g[j])

        def bl(a):
            return ((1 - tx) * (1 - ty) * a[i, j] + tx * (1 - ty) * a[i + 1, j]
                    + (1 - tx) * ty * a[i, j + 1] + tx * ty * a[i + 1, j + 1])
        return float(bl(self.v)), float(bl(self.w))

    def frames(self):
        """For the front end: body names and the pose tables (ground frame, pos xyz + quat wxyz)."""
        r = lambda a: np.round(a, 4).tolist()
        return {"names": self.names, "walk": r(self.walk), "stand": r(self.stand), "limp": r(self.limp),
                "dead": r(self.dead), "walk_hz": self.walk_hz,
                "dying": r(self.dying) if self.dying is not None else None, "dying_dt": self.dying_dt}


class LiteBody:
    def __init__(self, main_body, k, cal: Calib, x, y, heading):
        self.mb, self.k, self.cal = main_body, k, cal
        self.bid = int(main_body.oth_bodies[k])
        self.x, self.y, self.yaw, self.z = float(x), float(y), float(heading), 0.0
        self.mode = "walk"
        self.v = self.w = 0.0
        self.phase = 0.0                  # gait frame (float)
        self.t = 0.0
        self.flight = None                # {p0, p1, t, dur}
        self.touch, self.touch_others = [], []
        self.last_drive = np.zeros(2)
        self._gid = np.zeros(1, np.int32)
        self.place(x, y, heading)

    # ------------------------------------------------------------------ helpers
    def hvec(self):
        return np.array([np.cos(self.yaw), np.sin(self.yaw)])

    def place(self, x, y, heading=None):
        self.x, self.y = float(x), float(y)
        if heading is not None:
            self.yaw = float(heading)
        self._sync()

    def _sync(self):
        dz = (0.0 if self.cal.dying is not None else -0.25) if self.mode == "dead" else -0.15 if self.mode == "limp" else 0.0
        self.mb.other_set(self.k, self.x, self.y, self.z + dz, self.yaw)

    def _ray(self, ang, z, back=False):
        """Distance (mm) from the thorax along heading+ang at height z to the nearest geom, and what it is."""
        a = self.yaw + ang + (np.pi if back else 0.0)
        p = np.array([self.x, self.y, self.z + z])
        v = np.array([np.cos(a), np.sin(a), 0.0])
        d = mj.mj_ray(self.mb.m, self.mb.d, p, v, None, 1, self.bid, self._gid)
        g = int(self._gid[0])
        return (d if d >= 0 else 1e9), g

    def _what(self, g):
        m = self.mb
        if g in m._oth_of_geom:
            return "fly", m._oth_of_geom[g]
        if g >= 0 and int(m.m.geom_bodyid[g]) in m._fly_body_set:
            return "fly", -1                                    # the main fly
        return "thing", None

    # ------------------------------------------------------------------ stepping
    def takeoff(self, away=None, rng=None):
        if self.mode != "walk":
            return
        rng = rng or np.random.default_rng()
        a = (np.arctan2(away[1], away[0]) + rng.normal(0, 0.5)) if away is not None else rng.uniform(-np.pi, np.pi)
        p0 = np.array([self.x, self.y])
        for _ in range(8):
            d = rng.uniform(8.0, 20.0)
            p1 = p0 + d * np.array([np.cos(a), np.sin(a)])
            if np.linalg.norm(p1) < ARENA_R - 5.0:
                break
            a += rng.uniform(0.6, 1.4)
        else:
            p1 = p0 * 0.3
        dist = float(np.linalg.norm(p1 - p0))
        self.flight = {"p0": p0, "p1": p1, "t": 0.0, "dur": 0.15 + dist / 60.0}   # 60 mm/s: placeholder, ours
        self.yaw = float(np.arctan2(*(p1 - p0)[::-1]))
        self.mode = "flight"

    def right(self):
        """Self-righting (Policy calls it): a light body never lies on its back while alive."""

    def set_state(self, state):
        if state == self.mode or (state == "walk" and self.mode == "flight"):
            return
        if self.mode == "flight":
            self.flight, self.z = None, 0.0
        self.mode = state
        self.v = self.w = 0.0
        self.dead_t = 0.0                               # sim s since death: the front end plays the curl from it
        self._sync()

    def step(self, drive, dt):
        self.t += dt
        self.last_drive = np.asarray(drive, float)
        if self.mode == "dead":
            self.dead_t = getattr(self, "dead_t", 0.0) + dt
        if self.mode in ("limp", "dead", "held"):      # held (cinema pin): standing still, stand pose
            self.v = self.w = 0.0
            self.touch, self.touch_others = [], []
            return
        if self.mode == "flight":
            f = self.flight
            f["t"] += dt
            s = min(1.0, f["t"] / f["dur"])
            e = 0.5 - 0.5 * np.cos(np.pi * s)
            p = f["p0"] + e * (f["p1"] - f["p0"])
            self.x, self.y = float(p[0]), float(p[1])
            self.z = 5.0 * np.sin(np.pi * s)
            self.v = float(np.linalg.norm(f["p1"] - f["p0"]) / f["dur"])
            if s >= 1.0:
                self.mode, self.flight, self.z, self.v = "walk", None, 0.0, 0.0
            self._sync()
            return
        v, w = self.cal.lookup(*drive)
        k = 1.0 - np.exp(-dt / TAU_V)
        self.v += k * (v - self.v)
        self.w += k * (w - self.w)
        # touch and blocking: rays ahead (centre at two heights, and 30 deg to each side), and behind
        dc, gc = min((self._ray(0.0, z) for z in RAY_Z), key=lambda r: r[0])
        dl, gl = self._ray(0.5, RAY_Z[1])
        dr, gr = self._ray(-0.5, RAY_Z[1])
        touch, oth = [], set()
        for d, g, part in ((dc, gc, "c_head"), (dl, gl, "l_antenna"), (dr, gr, "r_antenna")):
            if d < HEAD_X + TOUCH_GAP:
                touch.append(part)
                what, o = self._what(g)
                if what == "fly":
                    oth.add(o)
        self.touch, self.touch_others = touch, sorted(oth)
        if self.v > 0 and min(dc, dl * np.cos(0.5), dr * np.cos(0.5)) < HEAD_X + BLOCK_GAP:
            self.v = 0.0
        if self.v < 0 and self._ray(0.0, RAY_Z[1], back=True)[0] < TAIL_X + BLOCK_GAP:
            self.v = 0.0
        self.yaw += self.w * dt
        self.x += np.cos(self.yaw) * self.v * dt
        self.y += np.sin(self.yaw) * self.v * dt
        r = np.hypot(self.x, self.y)
        if r > ARENA_R - 1.4:                               # never through the wall (the rays should stop it first)
            self.x, self.y = self.x * (ARENA_R - 1.4) / r, self.y * (ARENA_R - 1.4) / r
        # gait: the stepping rate follows the drive's magnitude (the CPG's), backwards when backing up
        act = float(np.clip(np.abs(self.last_drive).mean(), 0.0, 1.5))
        self.phase = (self.phase + (1 if self.v >= -0.2 else -1) * act * self.cal.walk_hz * dt) % len(self.cal.walk)
        self._sync()

    # ------------------------------------------------------------------ senses
    def _points(self, idx):
        f = self.cal.walk[int(self.phase)] if self.mode == "walk" and abs(self.v) + abs(self.w) > 0.05 else self.cal.stand
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        p = f[idx, :2]
        return np.stack([self.x + c * p[:, 0] - s * p[:, 1], self.y + s * p[:, 0] + c * p[:, 1]], axis=1)

    def tarsi_xy(self):
        return self._points(self.cal.tarsi)

    def antennae_xy(self):
        return self._points(self.cal.ant)

    def head_pos(self):
        h = self.hvec()
        return np.array([self.x + 0.6 * h[0], self.y + 0.6 * h[1], self.z + 1.2])

    def sensory(self, looms):
        """The same keys as Body.sensory(); looms: [(pos, radius, vel)] of things that could loom."""
        hp = self.head_pos()
        loom, rate, at = 0.0, 0.0, None
        for p, r, vel in looms:
            rel = hp - np.asarray(p)
            dist = max(np.linalg.norm(rel) - r, 0.05)
            ang = 2 * np.arctan(r / dist)
            closing = float(np.asarray(vel) @ rel / (np.linalg.norm(rel) + 1e-9))
            if ang > loom:
                loom, rate, at = ang, 2 * r * closing / (dist ** 2 + r ** 2), np.asarray(p, float)
        side = None
        if at is not None:
            h = self.hvec()
            rel = at[:2] - hp[:2]
            side = "left" if h[0] * rel[1] - h[1] * rel[0] >= 0 else "right"
        h = self.hvec()
        return {"t": self.t, "mode": self.mode, "pos": [self.x, self.y, self.z + 1.27], "heading": [h[0], h[1], 0.0],
                "v_forward": self.v, "v_lateral": 0.0, "yaw_rate": self.w, "leg_stance": [True] * 6,
                "touch": list(self.touch), "touch_others": list(self.touch_others),
                "sugar_legs": [False] * 6, "loom_angle": float(loom), "loom_rate": float(rate), "loom_side": side,
                "upright": -1.0 if self.mode == "dead" else 1.0, "drive": self.last_drive.tolist()}

    def looming_self(self):
        """This fly as a looming object while it flies (for everyone else's eyes)."""
        if self.mode != "flight":
            return None
        f = self.flight
        vel = (f["p1"] - f["p0"]) / f["dur"]
        return (np.array([self.x, self.y, self.z + 1.2]), FLY_R, np.array([vel[0], vel[1], 0.0]))


class Other:
    """One more fly: its own brain state, senses, reflexes, physiology; a LiteBody; no mind (yet)."""

    def __init__(self, sim, k, x, y, heading, brain, seed):
        self.sim, self.k = sim, k
        self.name = f"fly {k + 2}"
        self.brain = brain
        self.sensory_layer = SensoryLayer(sim.table)
        self.policy = Policy(sim.table, seed=seed)
        self.mixer = InputMixer(brain.n)
        self.bridge = Bridge(sim.table)
        self.observer = NeuralObserver(sim.table)
        self.phys = Physiology()
        self.phys.rate = sim.phys.rate
        from .pharma import Pharmacology
        self.pharma = Pharmacology(sim.drug_targets, brain)     # its own drugs, on its own synapse gains
        self._fed_prev = 0.0
        self.body = LiteBody(sim.body, k, sim.others.cal, x, y, heading)
        self.world_st = {"prev_t": None, "prev_T": None, "dTdt": np.zeros(2)}
        self.rng = np.random.default_rng(seed)
        self.runaway_ms = 0.0
        self.n_runaway = self.n_takeoff = 0
        self.mind = None                  # a later per-fly mind: {"provider": ..., "model": ...}
        self.sens = None
        self.drive = np.zeros(2)
        self.running = False
        self.born_t = sim.sim_t
        brain.reset_state()
        self.sense(sim, [])

    def sense(self, sim, looms):
        s = self.body.sensory(looms)
        s = sim.world.senses_at(s, self.body.tarsi_xy(), self.body.antennae_xy(), self.world_st)
        self.sens = s

    def launch(self, sim, dt_ms):
        """First half of a window: inputs to its brain, and start the brain on the GPU."""
        self.running = False
        if self.phys.dead:
            return
        sens = self.sens
        self.policy.neural = self.observer.rate
        self.policy.activity = sim.world.activity()
        self.mixer.set_source("sensory", self.sensory_layer.update(sens, None))
        ph = self.pharma
        eating = sens.get("drug_legs", ()) if self.phys.fed_s > self._fed_prev else ()
        self._fed_prev = self.phys.fed_s
        if ph.active(eating):
            ph.tick(sim.sim_t, dt_ms * 1e-3, eating)
            np.copyto(self.mixer.drug_i, ph.rest)
            ph.apply_policy(self.policy)
            down = ph.knocked_down()
            if down != ph.down:
                ph.down = down
                if down and not self.phys.paralysed:
                    self.body.set_state("limp")
                elif not down and not self.phys.paralysed and self.body.mode == "limp":
                    self.body.set_state("walk")
        self.mixer.set_source("autonomy", self.policy.update(sens, dt_ms * 1e-3, self.body))
        if self.runaway_ms >= (sim.SEIZURE_MS if self.pharma.any() else sim.RUNAWAY_MS):
            self.brain.reset_state()
            self.runaway_ms = 0.0
            self.n_runaway += 1
            self.observer.active_frac = 0.0
        self.mixer.apply(self.brain)
        self.brain.launch(dt_ms)
        self.running = True

    def finish(self, sim, dt_ms, looms):
        """Second half: read its brain's DNs, move, sense, live."""
        dt = dt_ms * 1e-3
        if self.running:
            counts = self.brain.collect()
            self.bridge.update(counts, dt_ms)
            self.observer.update(counts, dt_ms)
            self.runaway_ms = self.runaway_ms + dt_ms if self.observer.active_frac > sim.RUNAWAY_FRAC else 0.0
            drive = self.bridge.drive() * self.phys.motor_gain()
            drive, _ = self.pharma.shape_drive(drive, self.rng, self.observer.active_frac > sim.RUNAWAY_FRAC, sim.sim_t, dt_ms)
            if self.bridge.takeoff() and self.body.mode == "walk" and self.phys.motor_gain() > 0 and not self.pharma.seizing:
                away = None
                if self.sens["loom_angle"] > 0 and self.sens.get("loom_side"):
                    h = self.body.hvec()
                    left = np.array([-h[1], h[0]])
                    away = -left if self.sens["loom_side"] == "left" else left
                self.body.takeoff(away=away, rng=self.rng)
                self.n_takeoff += 1
        else:
            drive = np.zeros(2)
            self.bridge.update(np.zeros(self.brain.n, np.uint32), dt_ms)
        self.drive = drive
        self.body.step(drive, dt)
        self.sense(sim, looms)
        p = self.sens["pos"]
        ch = self.phys.update(self.sens, self.observer.rate, dt, sim.world.temperature(p), sim.world.humidity(p),
                              abs(self.sens["v_forward"]), self.body.mode == "flight")
        if ch in ("stupor", "coma"):
            self.body.set_state("limp")
        elif ch == "dead":
            self.body.set_state("dead")
            sim.cog.log.add(sim.sim_t, "O", "died", f"{self.name} died: {self.phys.status()['cause_text']}, "
                            f"after {self.phys.alive_s:.0f} s", who=self.k, x=self.body.x, y=self.body.y)
        elif ch is not None and self.body.mode == "limp" and not self.pharma.down:
            self.body.set_state("walk")

    def revive(self):
        self.phys.reset()
        self.pharma.clear()
        self.pharma.down = False
        self.body.set_state("walk")
        self.brain.reset_state()
        self.runaway_ms = 0.0
        self.policy.reset()
        self.bridge.reset()

    def status(self):
        b, ph = self.body, self.phys.status(self.sens)
        return {"id": self.k, "name": self.name, "x": round(b.x, 3), "y": round(b.y, 3), "z": round(b.z, 3),
                "yaw": round(b.yaw, 4), "mode": b.mode, "phase": round(b.phase, 2), "v": round(b.v, 2),
                "policy": self.policy.label, "state": ph["state"], "energy": ph["energy"], "water": ph["water"],
                "health": ph["health"], "tb": ph["tb"], "alive_s": ph["alive_s"], "cause_text": ph["cause_text"],
                "active": round(float(self.observer.active_frac), 6), "drive": [round(float(x), 2) for x in self.drive],
                "touch": b.touch, "mind": "none", "runaways": self.n_runaway, "takeoffs": self.n_takeoff,
                "drugs": {k: round(v, 3) for k, v in dict(self.pharma.c).items()}, "down": self.pharma.down}


class Others:
    """The other flies of one sim, their brains (pooled: a removed fly's GPU state is reused) and slots."""

    def __init__(self, sim):
        self.sim = sim
        self.cal = Calib() if CAL_PATH.exists() else None
        self.flies: dict[int, Other] = {}
        self._pool = []                   # spare brains
        self._seed = 1000
        sim.body._fly_body_set = set(int(i) for i in sim.body.fly_body_ids)

    def add(self, x, y, heading=None):
        if self.cal is None:
            raise RuntimeError("data/lite_body.npz is missing: run experiments/lite_calib.py")
        free = [k for k in range(MAX_OTHERS) if k not in self.flies]
        if not free:
            return None
        k = free[0]
        self._seed += 1
        if self._pool:
            brain = self._pool.pop()
        else:
            brain = self.sim.brain.clone(seed=self._seed)
            brain.run(float(os.environ.get("FLYSIM_TWIN_WARM", "10")))   # captures its CUDA graph now, not inside a live window
        if heading is None:
            heading = float(np.random.default_rng(self._seed).uniform(-np.pi, np.pi))
        o = Other(self.sim, k, x, y, heading, brain, self._seed)
        self.flies[k] = o
        return o

    def remove(self, k):
        o = self.flies.pop(k, None)
        if o is None:
            return False
        self._pool.append(o.brain)
        self.sim.body.other_park(k)
        return True

    def clear(self):
        for k in list(self.flies):
            self.remove(k)

    def looms_for(self, exclude=None):
        """Things that could loom at a fly's eyes: the hand, falling objects, the flies in the air."""
        out = list(self.sim.body.ext_loom) + self.sim.body.movers()
        if self.sim.body.mode == "flight":
            d = self.sim.body.d
            out.append((d.xpos[self.sim.body.thorax].copy(), FLY_R, d.qvel[self.sim.body.fly_vadr:self.sim.body.fly_vadr + 3].copy()))
        for k, o in self.flies.items():
            if k != exclude:
                ls = o.body.looming_self()
                if ls is not None:
                    out.append(ls)
        return out

    def flying(self):
        return [ls for o in self.flies.values() if (ls := o.body.looming_self()) is not None]

    def status(self):
        return [o.status() for o in self.flies.values()]

    MODE_CODE = {"walk": 0, "flight": 1, "limp": 2, "dead": 3, "held": 0}

    def poses(self):
        """For the 30 Hz pose stream: per fly (id, x, y, z, yaw, gait phase, mode code, speed) as float32
        (a dead fly sends sim s since death in the speed slot: the front end plays the death curl from it)."""
        rows = [(o.k, o.body.x, o.body.y, o.body.z, o.body.yaw, o.body.phase, self.MODE_CODE.get(o.body.mode, 0),
                 getattr(o.body, "dead_t", 0.0) if o.body.mode == "dead" else abs(o.body.v) + abs(o.body.w)) for o in list(self.flies.values())]
        return np.array(rows, np.float32).reshape(-1, 8)
