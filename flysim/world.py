"""Editable world: stimulus sources and scenery the user places, moves, turns and removes (the "sim game"
layer), and prebuilt worlds (worlds.py) loaded through the same calls.

Each source is a physical-world thing with a sensory consequence. Everything it does to the fly goes
through the body's sense organs into named receptor neurons (autonomy.SensoryLayer); nothing here
writes the body's drive.

  kind     what the fly senses                                   receptor neurons (FlyWire v783)
  sugar    taste on tarsal contact                               20 sugar GRNs (Shiu et al. 2024)
  water    taste on contact + humidity field around it           18 water GRNs (Shiu 2024); moist HRN_VP5
  bitter   taste on contact                                      20 bitter GRNs (Shiu 2024)
  heat     temperature field (a warm spot / flame)               hot cells TRN_VP2, per antenna
  cold     temperature field (an ice crystal)                    cold cells TRN_VP3a/b, per antenna
  block    a loose obstacle it can push (a MuJoCo free body)     antennal / head mechanoreceptors on touch
  wall, rock, pillar, log
           fixed scenery (MuJoCo mocap bodies, body.scenery_*)   the same mechanoreceptors on touch

Receptor identities: Marin et al. 2020 (Curr Biol, doi:10.1016/j.cub.2020.06.028) for the VP
thermo/hygrosensory types; Gallio et al. 2011 (Cell, doi:10.1016/j.cell.2011.01.028) for hot and
cold cells in the arista; Knecht et al. 2017 (eLife, doi:10.7554/eLife.26654) for moist and dry
cells. The FIELDS (Gaussian temperature and humidity bumps), the ambient values of each world and
every gain below are ours, chosen for a readable demo, not measured values. Heat and cold have a
steep core (the flame, the ice itself, about a body length across) on top of the wide field, so the
field is survivable and the thing itself is not: a fire's core passes the heat-knockdown limit and an
ice crystal's the chill-coma limit (physiology.py). Temperatures are sampled
at each antenna, so the two sides differ by the local gradient.
"""
from __future__ import annotations

import copy
import itertools

import numpy as np

from .body import ARENA_R, SUGAR

T_AMBIENT, T_PREF = 24.0, 25.0          # deg C; flies prefer ~25 deg C
H_AMBIENT = 40.0                        # % relative humidity
KINDS = {
    #          contact radius mm, field amplitude, field sigma mm
    "sugar":  {"r": 1.6},
    "water":  {"r": 2.2, "humid": 45.0, "sigma": 4.0},
    "bitter": {"r": 1.6},
    "heat":   {"r": 1.2, "temp": 16.0, "sigma": 5.0, "core": 34.0, "core_sigma": 1.1},   # glow + the flame itself
    "cold":   {"r": 1.2, "temp": -14.0, "sigma": 5.0, "core": -14.0, "core_sigma": 1.1},  # chilled air + the ice
    "block":  {"r": 1.0},
    # scenery: geom type, default size at strength 1 (see body.scenery_set)
    "wall":   {"r": 1.0, "geom": "box", "size": (2.0, 0.35, 1.6)},
    "rock":   {"r": 1.8, "geom": "ellipsoid", "size": (2.0, 1.5, 1.2)},
    "pillar": {"r": 0.8, "geom": "cylinder", "size": (0.6, 2.6)},
    "log":    {"r": 1.0, "geom": "cylinder", "size": (0.45, 3.5), "lie": True},
}
TASTES = ("sugar", "water", "bitter")
SCENERY_KINDS = ("wall", "rock", "pillar", "log")
UNDO_MAX = 40


def footprint(s):
    """A thing's footprint on the floor as a segment (x1, y1, x2, y2) plus a radius: walls and logs are
    long, everything else is a point with its contact radius."""
    x, y = s["x"], s["y"]
    if s["kind"] in ("wall", "log"):
        h = s["size"][0] if s["kind"] == "wall" else s["size"][1]
        c, sn = np.cos(s.get("yaw", 0.0)), np.sin(s.get("yaw", 0.0))
        w = s["size"][1] if s["kind"] == "wall" else s["size"][0]
        return (x - c * h, y - sn * h, x + c * h, y + sn * h), w
    if s["kind"] == "rock":
        return (x, y, x, y), float(max(s["size"][0], s["size"][1]))
    if s["kind"] == "pillar":
        return (x, y, x, y), float(s["size"][0])
    return (x, y, x, y), float(s.get("r", 1.0))


def nearest_point(s, p):
    """Nearest point of a thing's footprint to p (for distances to walls and logs)."""
    (x1, y1, x2, y2), _ = footprint(s)
    a, b = np.array([x1, y1]), np.array([x2, y2])
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 < 1e-9 else float(np.clip((np.asarray(p[:2]) - a) @ ab / L2, 0.0, 1.0))
    q = a + t * ab
    return float(q[0]), float(q[1])


class World:
    def __init__(self, body):
        self.body = body
        self._ids = itertools.count(1)
        self.sources: dict[int, dict] = {}
        self._prev_t = None
        self.T0, self.H0 = T_AMBIENT, H_AMBIENT
        self.name, self.title = "classic", "Classic"
        self._prev_T = np.array([self.T0, self.T0])
        self.dTdt = np.zeros(2)
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self.version = 0                     # bumped on every change (the front end redraws the fields)
        # day and night: off by default (always day). day_len = sim seconds for a whole 24 h day (a demo
        # compression); hour = the time of day now. Kept across world loads.
        self.day_len, self.hour, self._day_t = 0.0, 9.0, None
        from .worlds import PRESETS
        self.load(PRESETS["classic"])

    # ------------------------------------------------------------------ day and night
    def tick_day(self, t):
        """Advance the time of day to sim time t (called every window)."""
        if self._day_t is not None and self.day_len > 0 and t > self._day_t:
            self.hour = (self.hour + 24.0 * (t - self._day_t) / self.day_len) % 24.0
        self._day_t = t

    def set_day(self, day_len=None, hour=None):
        if day_len is not None:
            self.day_len = float(np.clip(day_len, 0.0, 3600.0))
        if hour is not None:
            self.hour = float(hour) % 24.0

    def sun(self):
        """+1 at noon, -1 at midnight; 1 when the cycle is off."""
        return float(np.sin(2 * np.pi * (self.hour - 6.0) / 24.0)) if self.day_len > 0 else 1.0

    def light(self):
        """0.04 (moonlight) .. 1 (day), with about an hour of twilight either side of 6:00 and 18:00."""
        return float(np.clip(0.5 + 1.8 * self.sun(), 0.04, 1.0)) if self.day_len > 0 else 1.0

    def activity(self):
        """How active the fly's own walking is by time of day (Policy.activity): morning and evening peaks
        around lights-on and lights-off, a midday lull, and rest at night (the bimodal pattern of
        Grima et al. 2004, Nature, doi:10.1038/nature02935; Stoleru et al. 2004, Nature,
        doi:10.1038/nature02926). The shape and numbers are a demo choice."""
        if self.day_len <= 0:
            return 1.0
        h = self.hour
        peak = max(np.exp(-((h - 7.0) / 1.6) ** 2), np.exp(-((h - 19.0) / 1.6) ** 2))
        a = 0.12 + 0.88 * peak
        if self.light() > 0.5:
            a = max(a, 0.45)
        return float(a)

    def _T_amb(self):
        return self.T0 + (1.5 * self.sun() if self.day_len > 0 else 0.0)       # warmer by day, cooler by night (small: the reflex thresholds are absolute)

    def _H_amb(self):
        return self.H0 - (8.0 * self.sun() if self.day_len > 0 else 0.0)       # damper at night

    def reset_default(self):
        """The original arena: the four props and one sugar drop at body.SUGAR (so old experiments see
        the same world; the props were static geoms before they became scenery)."""
        from .worlds import PRESETS
        self.load(PRESETS["classic"])

    # ------------------------------------------------------------------ editing (sim thread)
    @staticmethod
    def _inside(x, y, margin=1.5):
        """Clamp a point to inside the arena wall (a click past the wall lands just inside it)."""
        d = float(np.hypot(x, y)) + 1e-9
        k = min(1.0, (ARENA_R - margin) / d)
        return float(x) * k, float(y) * k

    def add(self, kind, x, y, r=None, strength=1.0, yaw=0.0, size=None, x2=None, y2=None, sid=None, drug=None):
        """Place a thing. Walls can be given as a segment (x, y) -> (x2, y2). sid: keep an id (undo / redo
        restore things under their old ids, so memory still knows them)."""
        if kind not in KINDS:
            raise KeyError(kind)
        if kind == "wall" and x2 is not None and y2 is not None:
            L = float(np.hypot(x2 - x, y2 - y))
            if L < 0.8:
                L = 4.0
            yaw = float(np.arctan2(y2 - y, x2 - x)) if np.hypot(x2 - x, y2 - y) >= 0.8 else yaw
            x, y = (x + x2) / 2, (y + y2) / 2
            size = [L / 2, KINDS["wall"]["size"][1], KINDS["wall"]["size"][2]]
        x, y = self._inside(x, y)
        strength = float(np.clip(strength, 0.1, 2.0))
        sid = int(sid) if sid is not None and int(sid) not in self.sources else next(self._ids)
        s = {"id": sid, "kind": kind, "x": float(x), "y": float(y),
             "r": float(r if r is not None else KINDS[kind]["r"]), "strength": strength}
        from .pharma import DRUGS
        if kind == "sugar" and drug in DRUGS:
            s["drug"] = str(drug)          # laced food: eating it doses the fly (pharma.py)
        if kind == "block":
            k = self.body.free_object(prefer="box")
            if k is None:
                return None
            self.body.place_object(k, x, y)
            s["obj"] = int(k)
        elif kind in SCENERY_KINDS:
            k = self.body.scenery_free(KINDS[kind]["geom"])
            if k is None:
                return None
            if size is None:
                sc = strength if kind != "wall" else 1.0
                size = [v * sc for v in KINDS[kind]["size"]]
            s.update(obj=int(k), yaw=float(yaw), size=[round(float(v), 3) for v in size])
            s["r"] = footprint(s)[1]
            self._place_scenery(s)
        self.sources[s["id"]] = s
        self.version += 1
        return s["id"]

    def _place_scenery(self, s):
        self.body.scenery_set(s["obj"], s["x"], s["y"], s["size"], s.get("yaw", 0.0), KINDS[s["kind"]].get("lie", False))

    def move(self, sid, x, y):
        s = self.sources.get(int(sid))
        if s is None:
            return False
        x, y = self._inside(x, y)
        s["x"], s["y"] = x, y
        if s["kind"] == "block":
            self.body.place_object(s["obj"], x, y)
        elif s["kind"] in SCENERY_KINDS:
            self._place_scenery(s)
        self.version += 1
        return True

    def set(self, sid, strength=None, r=None, yaw=None, scale=None):
        """Change strength / contact radius (stimuli), or turn (yaw, rad) / scale (scenery)."""
        s = self.sources.get(int(sid))
        if s is None:
            return False
        if strength is not None:
            s["strength"] = float(np.clip(strength, 0.1, 2.0))
        if r is not None and s["kind"] in TASTES:
            s["r"] = float(np.clip(r, 0.5, 6.0))
        if s["kind"] in SCENERY_KINDS:
            if yaw is not None:
                s["yaw"] = float(yaw)
            if scale is not None:
                f = float(np.clip(scale, 0.5, 2.0))
                if s["kind"] == "wall":      # a wall grows in length only
                    s["size"][0] = float(np.clip(s["size"][0] * f, 0.4, 30.0))
                else:
                    s["size"] = [float(np.clip(v * f, 0.2, 8.0)) for v in s["size"]]
                s["r"] = footprint(s)[1]
            self._place_scenery(s)
        self.version += 1
        return True

    def remove(self, sid):
        s = self.sources.pop(int(sid), None)
        if s and s["kind"] == "block":
            self.body.remove_object(s["obj"])
        elif s and s["kind"] in SCENERY_KINDS:
            self.body.scenery_park(s["obj"])
        self.version += 1
        return s is not None

    def clear(self):
        for sid in list(self.sources):
            self.remove(sid)

    def reapply(self):
        """After body.reset(): put blocks back (the reset parks the pool); scenery re-places itself."""
        for s in self.sources.values():
            if s["kind"] == "block":
                self.body.place_object(s["obj"], s["x"], s["y"])

    # ------------------------------------------------------------------ whole worlds, undo / redo
    def snapshot(self, ids=True):
        items = []
        for s in self.status():
            it = {k: s[k] for k in ("id", "kind", "x", "y", "strength") if k in s and (ids or k != "id")}
            if s["kind"] in TASTES:
                it["r"] = s["r"]
            if s.get("drug"):
                it["drug"] = s["drug"]
            if s["kind"] in SCENERY_KINDS:
                it.update(yaw=s.get("yaw", 0.0), size=list(s["size"]))
            items.append(it)
        return {"name": self.name, "title": self.title, "ambient": {"T": self.T0, "H": self.H0}, "items": items}

    def load(self, spec):
        """Replace everything with a world spec: {name, title, ambient: {T, H}, items: [...]} (worlds.py).
        The fly's start pose (spec["fly"]) is applied by the caller, which owns the body reset."""
        self.clear()
        amb = spec.get("ambient") or {}
        self.T0, self.H0 = float(amb.get("T", T_AMBIENT)), float(amb.get("H", H_AMBIENT))
        self.name, self.title = spec.get("name", "custom"), spec.get("title", spec.get("name", "Custom"))
        for it in spec.get("items", []):
            self.add(it["kind"], it["x"], it["y"], r=it.get("r"), strength=it.get("strength", 1.0),
                     yaw=it.get("yaw", 0.0), size=it.get("size"), x2=it.get("x2"), y2=it.get("y2"), sid=it.get("id"),
                     drug=it.get("drug"))
        self._prev_T = np.array([self.T0, self.T0])
        self.version += 1

    def checkpoint(self):
        """Call before an edit: the world as it was goes on the undo stack."""
        self.undo_stack.append(self.snapshot())
        del self.undo_stack[:-UNDO_MAX]
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.snapshot())
        self.load(self.undo_stack.pop())
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.snapshot())
        self.load(self.redo_stack.pop())
        return True

    # ------------------------------------------------------------------ fields
    def temperature(self, p):
        T = self._T_amb()
        for s in self.sources.values():
            k = KINDS[s["kind"]]
            if "temp" in k:
                d2 = (p[0] - s["x"]) ** 2 + (p[1] - s["y"]) ** 2
                T += s["strength"] * (k["temp"] * np.exp(-d2 / (2 * k["sigma"] ** 2))
                                      + k["core"] * np.exp(-d2 / (2 * k["core_sigma"] ** 2)))
        return float(T)

    def humidity(self, p):
        H = self._H_amb()
        for s in self.sources.values():
            k = KINDS[s["kind"]]
            if "humid" in k:
                d2 = (p[0] - s["x"]) ** 2 + (p[1] - s["y"]) ** 2
                H += s["strength"] * k["humid"] * np.exp(-d2 / (2 * k["sigma"] ** 2))
        return float(min(H, 100.0))

    def augment(self, sens, body):
        """Add world senses to the body's sensory dict (called after every body step)."""
        d = body.d
        st = {"prev_t": self._prev_t, "prev_T": self._prev_T, "dTdt": self.dTdt}
        out = self.senses_at(sens, d.xpos[body.tarsus5, :2], d.xpos[body.antennae, :2], st)
        self._prev_t, self._prev_T, self.dTdt = st["prev_t"], st["prev_T"], st["dTdt"]
        return out

    def senses_at(self, sens, txy, ant, st):
        """World senses for a fly whose six tarsi are at txy and antennae at ant (xy). st holds that fly's
        own temperature history ({"prev_t", "prev_T", "dTdt"}), so several flies can share one world."""
        taste = {k: [False] * 6 for k in TASTES}
        drugs = set()
        for s in list(self.sources.values()):
            if s["kind"] in TASTES:
                on = np.linalg.norm(txy - np.array([s["x"], s["y"]]), axis=1) < s["r"]
                taste[s["kind"]] = [a or bool(b) for a, b in zip(taste[s["kind"]], on)]
                if s.get("drug") and on.any():
                    drugs.add(s["drug"])
        T = np.array([self.temperature(a) for a in ant])
        H = np.array([self.humidity(a) for a in ant])
        t = sens["t"]
        if st["prev_t"] is not None and t > st["prev_t"]:
            a = 1.0 - np.exp(-(t - st["prev_t"]) / 0.1)            # 100 ms smoothing of dT/dt
            st["dTdt"] = st["dTdt"] + a * ((T - st["prev_T"]) / (t - st["prev_t"]) - st["dTdt"])
        st["prev_t"], st["prev_T"] = t, T
        sens["sugar_legs"] = taste["sugar"]        # replaces body.SUGAR contact: any sugar source counts
        sens["water_legs"] = taste["water"]
        sens["bitter_legs"] = taste["bitter"]
        sens["drug_legs"] = sorted(drugs)          # laced food under the feet (eaten only while it feeds)
        sens["temp_antenna"] = [round(float(x), 2) for x in T]
        sens["dtemp_antenna"] = [round(float(x), 2) for x in st["dTdt"]]
        sens["humid_antenna"] = [round(float(x), 1) for x in H]
        # the hottest / coldest surface each side touches (antenna + that side's three tarsi), for contact
        # nociception (nociception.periphery) and the escape direction (autonomy); the body's own temperature
        # lags the air by physiology.TAU_BODY, the cuticle in contact does not
        Tt = np.array([self.temperature(p) for p in txy])
        sens["temp_surface"] = [round(float(max(T[0], Tt[:3].max())), 2), round(float(max(T[1], Tt[3:].max())), 2)]
        sens["temp_surface_min"] = [round(float(min(T[0], Tt[:3].min())), 2), round(float(min(T[1], Tt[3:].min())), 2)]
        sens["temp_body"] = round(self.temperature(sens["pos"][:2]), 2)
        food = self.nearest(sens["pos"][:2], "sugar")
        sens["food_pos"] = [food["x"], food["y"]] if food else list(SUGAR["pos"])
        return sens

    def nearest(self, p, kind):
        best, bd = None, 1e9
        for s in list(self.sources.values()):
            if s["kind"] == kind:
                dd = (p[0] - s["x"]) ** 2 + (p[1] - s["y"]) ** 2
                if dd < bd:
                    best, bd = s, dd
        return best

    def status(self):
        out = []
        for s in list(self.sources.values()):
            if s["kind"] == "block":
                if not self.body._obj_placed[s["obj"]]:     # its pool object was reused by a drop
                    continue
                s["x"], s["y"] = self.body.object_xy(s["obj"])   # it can be pushed around
            out.append({k: (round(v, 3) if isinstance(v, float) else copy.copy(v)) for k, v in s.items()})
        return out

    def env(self):
        return {"name": self.name, "title": self.title, "T0": self.T0, "H0": self.H0, "version": self.version,
                "undo": len(self.undo_stack), "redo": len(self.redo_stack),
                "day": {"len": self.day_len, "hour": round(self.hour, 2), "light": round(self.light(), 3),
                        "T": round(self._T_amb(), 1), "activity": round(self.activity(), 2)}}
