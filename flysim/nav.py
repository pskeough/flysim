"""Navigation lever for walled tasks (2026-10-13, ours; demo design, not a model of the fly's navigation circuits).

Why it exists: in the closed labyrinth (task maze4) the mind's NOW was nearly always "explore", which binds to "none"
(cognition.BIND), so the legs stayed with the connectome and the AI-on and AI-off flies walked identical paths.

What it adds, all from senses the fly has plus its own remembered experience (no map, no food coordinates):
- SMELL. A sugar odour that spreads along the corridors and fades with path distance (concentration exp(-d / LAMBDA),
  d = distance through open floor, a steady-state stand-in for diffusion with decay; it does not pass through walls).
  Sampled at the two antennae. Below THRESH the fly smells nothing. The mind hears: none / faint / clear / strong,
  which antenna has more, and whether it got stronger or weaker over the last ~1.5 s of walking.
- THE WAYS. Whether the way is open ahead, left, right and behind (sight and antenna touch, rays of RAY_MAX mm),
  relative to the fly's own heading.
- MEMORY OF THE WAYS. The fly's sense of heading and path integration (flies have both) let it remember forks it has
  stood at, which way it left each fork, and where a way ended in a dead end. At a fork it is told, per open way,
  "you came from there", "you tried that way: dead end", or "you went that way before".
- The lever: NOW choices "go forward", "take the left way", "take the right way", "turn back". The body carries the
  plan out until the next thought: it holds the chosen heading (the corridors are square, so headings are snapped to
  quarter turns), stays in the middle of the corridor, follows a bend when only one way goes on, and takes a planned
  turn at the next opening on that side. It stops at a fork it was not told about or at a dead end, and the mind is
  woken. The fly's own reflexes (backing off a touch) still run underneath and win.

Switched on per trial by Cognition.nav_on (conditions "nav"), never by default.
"""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np

LAMBDA = 14.0         # mm: odour fades e-fold per 14 mm of corridor
THRESH = 0.02         # below this the fly smells nothing (about 55 mm of corridor from the sugar)
RES = 0.25            # mm grid for the odour field
RAY_MAX = 9.0         # mm the fly can tell open from closed
OPEN_MM = 6.5         # a way is open if nothing is within this distance along it
BLOCK_MM = 2.6        # wall this close ahead: the body cannot go on that way
FORK_R = 4.5          # mm: same fork if within this
DIRS = ("ahead", "left", "right", "behind")
REL = {"ahead": 0, "left": 1, "right": -1, "behind": 2}     # quarter turns, counterclockwise


def walls_of(world):
    from .world import footprint
    out = []
    for s in world:
        if s.get("kind") == "wall":
            (x1, y1, x2, y2), r = footprint(s)
            out.append((x1, y1, x2, y2, float(r)))
    return out


def _seg_dist(P, W):
    """P (n,2) points, W (m,5) segments with radius -> (n,m) clearance."""
    a, b = W[None, :, :2], W[None, :, 2:4]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(-1), 1e-9)
    t = np.clip(((P[:, None, :] - a) * ab).sum(-1) / L2, 0, 1)
    q = a + t[..., None] * ab
    return np.linalg.norm(P[:, None, :] - q, axis=-1) - W[None, :, 4]


def ray(pos, ang, W, maxd=RAY_MAX, clear=0.35):
    if not len(W):
        return maxd
    s = np.arange(0.2, maxd + 1e-6, 0.2)
    P = np.asarray(pos)[None, :] + s[:, None] * np.array([np.cos(ang), np.sin(ang)])[None, :]
    hit = np.nonzero((_seg_dist(P, W) < clear).any(1))[0]
    return float(s[hit[0]]) if len(hit) else maxd


def snap_q(ang):
    return round(ang / (np.pi / 2)) % 4


def q_ang(q):
    return (q % 4) * np.pi / 2


class OdourField:
    def __init__(self):
        self.key, self.D, self.box = None, None, None

    def build(self, world):
        sug = [(round(s["x"], 2), round(s["y"], 2)) for s in world if s.get("kind") == "sugar"]
        W = walls_of(world)
        key = (tuple(sug), tuple(tuple(round(v, 2) for v in w) for w in W))
        if key == self.key:
            return
        self.key, self.sug, self.W = key, sug, np.array(W) if W else np.zeros((0, 5))
        if not sug or not W:
            self.D = None
            return
        xs = [w[0] for w in W] + [w[2] for w in W]
        ys = [w[1] for w in W] + [w[3] for w in W]
        x0, y0 = min(xs) - 1.0, min(ys) - 1.0
        nx, ny = int((max(xs) + 1.0 - x0) / RES) + 1, int((max(ys) + 1.0 - y0) / RES) + 1
        gx, gy = np.meshgrid(x0 + RES * np.arange(nx), y0 + RES * np.arange(ny), indexing="ij")
        P = np.stack([gx.ravel(), gy.ravel()], 1)
        free = (_seg_dist(P, self.W) > 0.05).all(1).reshape(nx, ny)
        D = np.full((nx, ny), np.inf)
        h = []
        for sx, sy in sug:
            i, j = int(round((sx - x0) / RES)), int(round((sy - y0) / RES))
            if 0 <= i < nx and 0 <= j < ny:
                D[i, j] = 0.0
                heapq.heappush(h, (0.0, i, j))
        st = [(di, dj, RES * np.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        while h:
            d, i, j = heapq.heappop(h)
            if d > D[i, j]:
                continue
            for di, dj, c in st:
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and free[a, b] and d + c < D[a, b]:
                    D[a, b] = d + c
                    heapq.heappush(h, (d + c, a, b))
        # blocked cells take the nearest free cell's value, so a sample beside a wall is defined
        from scipy.ndimage import distance_transform_edt
        _, (ii, jj) = distance_transform_edt(~np.isfinite(D), return_indices=True)
        self.D = D[ii, jj]
        self.box = (x0, y0, nx, ny)

    def conc(self, p):
        if self.D is None:
            return 0.0
        x0, y0, nx, ny = self.box
        fx, fy = (p[0] - x0) / RES, (p[1] - y0) / RES
        if not (0 <= fx < nx - 1 and 0 <= fy < ny - 1):
            d = min(np.hypot(p[0] - sx, p[1] - sy) for sx, sy in self.sug)
        else:
            i, j = int(fx), int(fy)
            u, v = fx - i, fy - j
            D = self.D
            d = (D[i, j] * (1 - u) * (1 - v) + D[i + 1, j] * u * (1 - v) + D[i, j + 1] * (1 - u) * v
                 + D[i + 1, j + 1] * u * v)
        c = float(np.exp(-d / LAMBDA))
        return c if c >= THRESH else 0.0


class Nav:
    """Per-fly navigation state: senses, the remembered forks and dead ends, and the plan the body carries out.

    Directions are quarter turns in the world frame (the fly's compass). "Ahead" is the fly's direction of TRAVEL
    (its displacement over the last few mm, snapped to a quarter), not its momentary heading, so a fly that slews
    sideways against a corridor wall still knows which way the corridor runs."""

    def __init__(self):
        self.odour = OdourField()
        self.reset()

    def reset(self):
        self.forks = []           # {"x", "y", "came": q, "tried": {q: "dead end" | "went"}}
        self.dead = []            # (x, y)
        self.cur_fork = None      # the fork the fly last stood at, and the way it left it (q)
        self.cur_exit = None
        self.odour_hist = deque(maxlen=80)
        self.trail = deque(maxlen=200)
        self.travel_q = None
        self.events = []          # strings for the mind, drained by Cognition
        self._stuck_pos = None

    # ------------------------------------------------------------------ senses
    def sense(self, snap, world_full):
        sens = snap["sens"]
        pos = np.asarray(sens["pos"][:2], float)
        h = np.asarray(sens["heading"][:2], float)
        h = h / (np.linalg.norm(h) + 1e-9)
        hd = float(np.arctan2(h[1], h[0]))
        self.odour.build(world_full)
        W = self.odour.W
        side = np.array([-h[1], h[0]])
        head = pos + 1.2 * h                      # antennae ~0.25 mm either side of the head
        cl, cr = self.odour.conc(head + 0.25 * side), self.odour.conc(head - 0.25 * side)
        t = snap["t"]
        self.odour_hist.append((t, 0.5 * (cl + cr), pos.copy()))
        if not self.trail or np.linalg.norm(pos - self.trail[-1]) > 0.3:
            self.trail.append(pos.copy())
        # direction of travel: displacement over the last ~2.5 mm of path
        for p in reversed(self.trail):
            d = pos - p
            if np.linalg.norm(d) > 2.5:
                self.travel_q = snap_q(float(np.arctan2(d[1], d[0])))
                break
        q0 = self.travel_q if self.travel_q is not None else snap_q(hd)
        openq = {q: ray(pos, q_ang(q), W) for q in range(4)}
        ways = {name: openq[(q0 + k) % 4] for name, k in REL.items()}
        return {"pos": pos, "hd": hd, "q0": q0, "openq": openq, "ways": ways, "cl": cl, "cr": cr, "W": W, "t": t}

    def is_open(self, s, name):
        return s["ways"][name] >= OPEN_MM

    def open_set(self, s):
        return {q for q, d in s["openq"].items() if d >= OPEN_MM}

    # ------------------------------------------------------------------ memory of forks and dead ends
    def _fork_at(self, pos):
        for i, f in enumerate(self.forks):
            if np.hypot(f["x"] - pos[0], f["y"] - pos[1]) < FORK_R:
                return i
        return None

    def update_memory(self, s):
        pos, q0 = s["pos"], s["q0"]
        O = self.open_set(s)
        i = self._fork_at(pos)
        if len(O) >= 3 and i is None:
            came = (q0 + 2) % 4
            self.forks.append({"x": float(pos[0]), "y": float(pos[1]), "came": came, "tried": {}})
            if self.cur_fork is not None and self.cur_exit is not None:
                self.forks[self.cur_fork]["tried"].setdefault(self.cur_exit, "went")
            i = len(self.forks) - 1
            self.events.append("you came to a fork: " + self._ways_text(s, i))
        if i is not None:
            if self.cur_fork != i:
                self.cur_fork, self.cur_exit = i, None
        elif self.cur_fork is not None and self.cur_exit is None:
            f = self.forks[self.cur_fork]
            if np.hypot(f["x"] - pos[0], f["y"] - pos[1]) > FORK_R + 1.0:
                d = pos - np.array([f["x"], f["y"]])
                self.cur_exit = snap_q(float(np.arctan2(d[1], d[0])))
                f["tried"].setdefault(self.cur_exit, "went")
        # a dead end: only the way back is open
        dead = O <= {(q0 + 2) % 4} and s["openq"][q0] < BLOCK_MM + 2.5
        if dead and not any(np.hypot(x - pos[0], y - pos[1]) < 5.0 for x, y in self.dead):
            self.dead.append((float(pos[0]), float(pos[1])))
            if self.cur_fork is not None and self.cur_exit is not None:
                self.forks[self.cur_fork]["tried"][self.cur_exit] = "dead end"
                self.events.append("a dead end: walls ahead, left and right. The way you took at the last fork leads nowhere")
            else:
                self.events.append("a dead end: walls ahead, left and right")

    def _ways_text(self, s, i=None):
        out = []
        f = self.forks[i] if i is not None else None
        for name, k in REL.items():
            if not self.is_open(s, name):
                continue
            q = (s["q0"] + k) % 4
            note = ""
            if f is not None:
                if q == f["came"]:
                    note = " (the way you came)"
                elif f["tried"].get(q) == "dead end":
                    note = " (you tried it: a dead end)"
                elif f["tried"].get(q) == "went":
                    note = " (you went that way before)"
            elif name == "behind":
                note = " (the way you came)"
            out.append(f"{name}{note}")
        return "open " + ", ".join(out) if out else "no way open"

    # ------------------------------------------------------------------ what the mind is told
    def smell_line(self, s):
        c = 0.5 * (s["cl"] + s["cr"])
        if c <= 0:
            return "no smell of food at all"
        lvl = "a faint" if c < 0.08 else ("a clear" if c < 0.35 else "a strong")
        out = f"{lvl} sweet smell"
        if s["cl"] > 0 and s["cr"] > 0 and abs(s["cl"] - s["cr"]) / c > 0.01:
            out += ", stronger on your " + ("left" if s["cl"] > s["cr"] else "right") + " antenna"
        t = self.odour_hist[-1][0]
        old = [x for x in self.odour_hist if t - x[0] >= 1.5]
        if old:
            c0, p0 = old[-1][1], old[-1][2]
            moved = float(np.linalg.norm(self.odour_hist[-1][2] - p0))
            if moved > 1.0 and c0 > 0:
                r = c / c0
                out += (", growing as you walk" if r > 1.03 else (", fading as you walk" if r < 0.97 else ""))
            elif moved > 1.0 and c0 == 0:
                out += ", which you only just caught"
        return out

    def describe(self, s):
        """-> (ways line, memory lines) for the prompt."""
        i = self._fork_at(s["pos"])
        ways = self._ways_text(s, i)
        if s["ways"]["ahead"] < BLOCK_MM + 1.5:
            ways += "; a wall right in front of you"
        mem = []
        n_dead_forks = sum(1 for j, f in enumerate(self.forks) if j != i and "dead end" in f["tried"].values())
        if n_dead_forks:
            mem.append(f"at {n_dead_forks} earlier fork{'s' if n_dead_forks > 1 else ''}, one way ended in a dead end")
        if self.dead:
            mem.append(f"you have hit {len(self.dead)} dead end{'s' if len(self.dead) > 1 else ''} so far")
        if self.forks:
            mem.append(f"you have passed {len(self.forks)} fork{'s' if len(self.forks) > 1 else ''}")
        return ways, mem

    def choices(self, s):
        """NOW choices: the ways the body can take from here. At a fork or a wall only the open ways are offered; in
        a plain corridor, "the next" left or right opening."""
        side_open = self.is_open(s, "left") or self.is_open(s, "right")
        blocked = s["ways"]["ahead"] < OPEN_MM
        out = []
        if not blocked:
            out.append(("go forward", {"act": "nav", "kind": "forward", "id": None}))
        for k in ("left", "right"):
            if self.is_open(s, k):
                out.append((f"take the {k} way", {"act": "nav", "kind": k, "id": None}))
            elif not side_open and not blocked:
                out.append((f"take the next {k} way", {"act": "nav", "kind": k, "id": None}))
        out.append(("turn back", {"act": "nav", "kind": "back", "id": None}))
        return out

    # ------------------------------------------------------------------ the body carries the plan out
    def bind(self, spec, s):
        """-> (catalog intent, note). spec["kind"]: forward | left | right | back; the spec dict holds the plan state
        ("q": the world quarter the body is heading for; "want": a turn still to take)."""
        q0 = s["q0"]
        O = self.open_set(s)
        if "q" not in spec:
            k = spec["kind"]
            spec["q"], spec["want"] = q0, None
            if k == "back":
                spec["q"] = (q0 + 2) % 4
            elif k in ("left", "right"):
                spec["want"] = (q0 + REL[k]) % 4
        want = spec.get("want")
        if want is not None and want in O and spec["q"] != want:
            # the wanted side is open here: take it (only once the fly is past the near corner of the opening)
            spec["q"], spec["want"] = want, None
        q = spec["q"]
        if s["openq"][q] < BLOCK_MM + 1.5:
            # the planned way ends here. The body follows a corridor bend on its own; at a fork or a dead end it stops
            back = (q + 2) % 4
            opts = O - {back, q}
            if want is not None and (want in opts):
                q = want
                spec["want"] = None
            elif len(opts) == 1:
                q = next(iter(opts))
            else:
                if self._stuck_pos is None or np.linalg.norm(s["pos"] - self._stuck_pos) > 3.0:
                    self._stuck_pos = s["pos"].copy()
                    self.events.append("your legs stop: the way you wanted goes no further (" +
                                       self._ways_text(s, self._fork_at(s["pos"])) + ")")
                return "stop", spec["kind"] + " blocked"
            spec["q"] = q
        note = spec["kind"] + (" (turn pending)" if spec.get("want") is not None else "")
        # steer to the plan heading, keeping to the middle of the corridor
        b = float(np.degrees((q_ang(q) - s["hd"] + np.pi) % (2 * np.pi) - np.pi))
        if abs(b) < 45:
            dl, dr = ray(s["pos"], q_ang(q + 1), s["W"], 5.0), ray(s["pos"], q_ang(q - 1), s["W"], 5.0)
            if dl < 5.0 and dr < 5.0:
                b += float(np.clip(12.0 * (dl - dr) / 2.0, -25, 25))
        a = abs(b)
        if a > 35:
            return ("rotate_left" if b > 0 else "rotate_right"), note
        if a > 7:
            return ("turn_left" if b > 0 else "turn_right"), note
        return "walk_forward", note
