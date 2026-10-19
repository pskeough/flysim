"""Auto-director: keeps something happening on camera. Every 20-40 sim seconds it introduces one event near the fly,
picked so the fly has something to react to and the mind something to think about:

  sugar   a drop of sugar 3-5 mm ahead, a little to one side (in view)
  water   a drop of water 4-6 mm ahead
  fire    a gentle fire (strength 0.6) 7-9 mm away to one side: the glow reaches it, the flame does not
  loom    a hand coming down 6-7 mm away: a loom that misses (the hand's radius is 3.2 mm)
  puff    a puff of air from 4 mm to one side

Events go through the same command queue as the editor (sim.cmds), so they look to the mind exactly like the user's
edits. It places at most MAX_ITEMS things over a session and then only looms and puffs. Seeded (the app's --seed).
Toggle: Director button / POST /api/cmd {"cmd": "director", "on": true, "every": [20, 40]}; FLYSIM_DIRECTOR=1 starts on.
"""
from __future__ import annotations

import math

import numpy as np

EVENTS = ("sugar", "fire", "loom", "water", "puff")
MAX_ITEMS = 6


class Director:
    def __init__(self, seed=0):
        self.on = False
        self.every = (20.0, 40.0)
        self.rng = np.random.default_rng(7_919 + int(seed))
        self.t_next = None
        self.n_items = 0
        self.history: list[dict] = []
        self.k = 0

    def configure(self, on=None, every=None):
        if on is not None:
            self.on = bool(on)
            self.t_next = None
        if every and len(every) == 2:
            lo, hi = float(every[0]), float(every[1])
            self.every = (max(3.0, min(lo, hi)), max(3.0, lo, hi))
            self.t_next = None

    def status(self):
        return {"on": self.on, "every": list(self.every), "next_in": None if self.t_next is None else round(self.t_next, 1),
                "last": self.history[-1] if self.history else None}

    def tick(self, sim):
        """Called from the sim loop with the lock held. Returns a command tuple for sim.cmds, or None."""
        if not self.on or sim.paused or sim.phys.dead:
            return None
        t = sim.sim_t
        if self.t_next is None:
            self.t_next = t + float(self.rng.uniform(4.0, 8.0))      # the first one comes soon
            return None
        if t < self.t_next:
            return None
        self.t_next = t + float(self.rng.uniform(*self.every))
        pool = list(EVENTS) if self.n_items < MAX_ITEMS else ["loom", "puff"]
        # not the same thing twice running
        if self.history and self.history[-1]["what"] in pool and len(pool) > 1:
            pool.remove(self.history[-1]["what"])
        what = pool[int(self.rng.integers(len(pool)))] if self.k else "sugar"
        self.k += 1
        p, h = np.asarray(sim.sens["pos"][:2], float), np.asarray(sim.sens["heading"][:2], float)
        n = float(np.hypot(*h)) or 1.0
        h = h / n
        left = np.array([-h[1], h[0]])
        side = 1.0 if self.rng.random() < 0.5 else -1.0

        from .cognition import _seg_seg_dist
        from .world import ARENA_R, footprint
        # scenery footprints as segments (a long wall was a disc round its centre before, which missed its ends)
        blockers = []
        for s in list(sim.world.sources.values()):
            if s.get("kind") in ("wall", "rock", "pillar", "log"):
                (x1, y1, x2, y2), r = footprint(s)
                blockers.append((np.array([x1, y1], float), np.array([x2, y2], float), float(r) + 1.2))

        def clear(q):
            """Inside the arena with a margin, and no scenery near the straight line from the fly to it (so a
            target is never behind a wall)."""
            if np.hypot(*q) > ARENA_R - 2.5:
                return False
            for a, b, r in blockers:
                if _seg_seg_dist(p, q, a, b) < r:
                    return False
            return True

        def at(ahead, lateral, los=True):
            # the wanted spot, else the mirror side, else nearer, else straight ahead nearer still.
            # los=False (loom, puff): keep the distance, only stay inside the arena
            if los:
                tries = ((ahead, lateral), (ahead, -lateral), (0.6 * ahead, lateral), (0.6 * ahead, -lateral), (2.5, 0.0))
            else:
                tries = ((ahead, lateral), (-ahead, -lateral))
            for a_, l_ in tries:
                q = p + a_ * h + l_ * left
                if (clear(q) if los else np.hypot(*q) < ARENA_R - 2.5):
                    return float(q[0]), float(q[1])
            if not los:
                q = p + ahead * h + lateral * left
                return float(q[0]), float(q[1])
            q = p + 2.5 * h
            r = float(np.hypot(*q))
            if r > ARENA_R - 2.5:                                  # never in or past the arena wall
                q = q * (ARENA_R - 2.5) / r
            return float(q[0]), float(q[1])

        if what == "sugar":
            x, y = at(self.rng.uniform(3.0, 5.0), side * self.rng.uniform(0.5, 1.5)); cmd = ("world_add", "sugar", x, y, {})
        elif what == "water":
            x, y = at(self.rng.uniform(4.0, 6.0), side * self.rng.uniform(0.5, 2.0)); cmd = ("world_add", "water", x, y, {})
        elif what == "fire":
            a = self.rng.uniform(0.6, 1.2) * side
            d = self.rng.uniform(7.0, 9.0)
            x, y = at(d * math.cos(a), d * math.sin(a)); cmd = ("world_add", "heat", x, y, {"strength": 0.6})
        elif what == "loom":
            a = self.rng.uniform(-1.0, 1.0)
            d = self.rng.uniform(6.0, 7.0)
            x, y = at(d * math.cos(a), d * math.sin(a), los=False); cmd = ("hand", x, y, 90.0)
        else:
            x, y = at(0.0, side * 4.0, los=False); cmd = ("puff", x, y)
        if cmd[0] == "world_add":
            self.n_items += 1
        self.history.append({"t": round(t, 2), "what": what, "x": round(x, 2), "y": round(y, 2)})
        del self.history[:-20]
        return cmd
