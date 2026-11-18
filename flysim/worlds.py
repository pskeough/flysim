"""Prebuilt worlds for the editable arena (world.World.load) and the user's saved ones.

A world spec is plain JSON:
  {"name", "title", "desc", "ambient": {"T": deg C, "H": % RH}, "fly": {"x", "y", "heading"},
   "items": [{"kind", "x", "y", "strength"?, "r"?, "yaw"?, "size"?, "x2"?, "y2"?}, ...]}
Walls may be given as segments (x, y) -> (x2, y2). Units mm; the arena is a disc of radius 30.
Layouts, ambient values and strengths are demo design, not measurements. Saved worlds go to
sim/worlds/<name>.json.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

USER_DIR = Path(__file__).resolve().parents[1] / "worlds"


def _wall(x1, y1, x2, y2):
    return {"kind": "wall", "x": x1, "y": y1, "x2": x2, "y2": y2}


def _ring(r, n, gaps=(), overlap=0.3):
    """Walls along a circle of radius r as n chords; the chords listed in gaps are left out (doors)."""
    out = []
    for i in range(n):
        if i in gaps:
            continue
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        am = (a0 + a1) / 2
        half = r * math.sin(math.pi / n) + overlap
        out.append({"kind": "wall", "x": r * math.cos(am), "y": r * math.sin(am), "yaw": am + math.pi / 2,
                    "size": [half, 0.35, 1.6]})
    return out


def _classic():
    # water added 2026-10-15: without it the fly dries out at ~250 s in every session (WATER_S 300, start 0.7)
    return {"name": "classic", "title": "Classic", "desc": "The original arena: four props, a drop of sugar and a drop of water.",
            "ambient": {"T": 24.0, "H": 40.0}, "fly": {"x": 0.0, "y": 0.0, "heading": 0.0},
            "items": [{"kind": "wall", "x": 9.0, "y": 4.0, "size": [2.0, 1.2, 1.0]},
                      {"kind": "rock", "x": -7.0, "y": 9.0, "size": [2.6, 1.8, 1.3]},
                      {"kind": "log", "x": -4.0, "y": -10.0, "yaw": math.pi / 2, "size": [0.45, 7.0]},
                      {"kind": "rock", "x": 14.0, "y": -8.0, "size": [1.2, 1.2, 1.2]},
                      {"kind": "sugar", "x": 6.0, "y": -3.0, "r": 1.6},
                      {"kind": "water", "x": -5.0, "y": -3.5, "r": 1.6, "strength": 0.8}]}


def _labyrinth():
    items = _ring(10.0, 10, gaps=(2,)) + _ring(19.0, 20, gaps=(13,))
    for a in (0.9, 3.0, 4.9):            # spokes between the rings, so the way round is not free
        items.append(_wall(10.6 * math.cos(a), 10.6 * math.sin(a), 18.4 * math.cos(a), 18.4 * math.sin(a)))
    items += [{"kind": "sugar", "x": -1.5, "y": 1.0}, {"kind": "water", "x": 2.0, "y": -1.5, "strength": 0.8}]
    return {"name": "labyrinth", "title": "Labyrinth", "desc": "Two rings of wall, one door in each. Food and water at the centre.",
            "ambient": {"T": 24.0, "H": 35.0}, "fly": {"x": 0.0, "y": -25.0, "heading": math.pi / 2}, "items": items}


def _two_rooms():
    items = [_wall(-29.5, 0.0, -2.8, 0.0), _wall(2.8, 0.0, 29.5, 0.0),
             {"kind": "sugar", "x": -9.0, "y": 14.0}, {"kind": "sugar", "x": 11.0, "y": 17.0, "strength": 0.8},
             {"kind": "heat", "x": 0.0, "y": 5.5, "strength": 0.7},
             {"kind": "water", "x": 12.0, "y": -14.0}, {"kind": "rock", "x": -14.0, "y": -12.0},
             {"kind": "rock", "x": -18.0, "y": 8.0, "strength": 0.8}, {"kind": "pillar", "x": 5.0, "y": -8.0}]
    return {"name": "two_rooms", "title": "Two rooms", "desc": "Water on this side, food on the other. The door is warm.",
            "ambient": {"T": 24.0, "H": 40.0}, "fly": {"x": -6.0, "y": -16.0, "heading": math.pi / 2}, "items": items}


def _oasis():
    items = [{"kind": "water", "x": 0.0, "y": 0.0, "strength": 1.2}, {"kind": "water", "x": 2.6, "y": 1.5},
             {"kind": "water", "x": -1.8, "y": 2.2, "strength": 0.8}, {"kind": "sugar", "x": 4.5, "y": -3.0}]
    for i in range(5):
        a = 2 * math.pi * i / 5 + 0.3
        items.append({"kind": "pillar", "x": 7.5 * math.cos(a), "y": 7.5 * math.sin(a), "strength": 0.9 + 0.1 * (i % 3)})
    for i, (r, a) in enumerate([(19, 0.4), (22, 1.5), (18, 2.5), (23, 3.6), (20, 4.6), (22, 5.5)]):
        items.append({"kind": "rock", "x": r * math.cos(a), "y": r * math.sin(a), "yaw": a,
                      "size": [3.2 + 0.3 * (i % 2), 1.7, 1.0]})
    items += [{"kind": "heat", "x": 16.0, "y": 13.0, "strength": 0.6}, {"kind": "heat", "x": -15.0, "y": -16.0, "strength": 0.6}]
    return {"name": "oasis", "title": "Oasis", "desc": "Warm, dry air and dunes. A pool of water in the palms at the centre.",
            "ambient": {"T": 27.0, "H": 22.0}, "fly": {"x": -22.0, "y": 8.0, "heading": -0.3}, "items": items}


def _winter():
    items = [{"kind": "heat", "x": 10.0, "y": 6.0, "strength": 0.8}, {"kind": "sugar", "x": 7.0, "y": 3.0},
             {"kind": "water", "x": 13.5, "y": 9.0, "strength": 0.7}]
    for a in (0.3, 2.2, 4.2):
        items.append({"kind": "log", "x": 10.0 + 4.6 * math.cos(a), "y": 6.0 + 4.6 * math.sin(a), "yaw": a + math.pi / 2,
                      "size": [0.45, 2.2]})
    for x, y, s in [(-14, -4, 1.0), (-6, 14, 0.8), (-20, 10, 1.2), (4, -16, 1.0), (18, -12, 0.9), (-8, -20, 0.8)]:
        items.append({"kind": "cold", "x": x, "y": y, "strength": s})
    return {"name": "winter", "title": "Winter", "desc": "Cold air and ice. One small fire, with food beside it.",
            "ambient": {"T": 21.5, "H": 55.0}, "fly": {"x": -15.0, "y": -10.0, "heading": 0.5}, "items": items}


def _bitter_garden():
    items = []
    for i in range(4):
        for j in range(4):
            x, y = -10.5 + 7 * i, -8.5 + 7 * j
            items.append({"kind": "sugar" if (i + j) % 2 else "bitter", "x": x, "y": y})
    items += [{"kind": "pillar", "x": -14.0, "y": 14.0}, {"kind": "pillar", "x": 14.5, "y": -13.0},
              {"kind": "pillar", "x": 16.0, "y": 12.0, "strength": 1.2}, {"kind": "water", "x": -20.0, "y": -14.0}]
    return {"name": "bitter_garden", "title": "Bitter garden", "desc": "Sixteen drops in a grid. Half are sugar, half are bitter.",
            "ambient": {"T": 24.5, "H": 45.0}, "fly": {"x": 0.0, "y": -24.0, "heading": math.pi / 2}, "items": items}


def _forest():
    rng = np.random.default_rng(7)
    items = []
    while len([i for i in items if i["kind"] == "pillar"]) < 18:
        r, a = 26 * math.sqrt(rng.uniform(0.02, 1)), rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a), r * math.sin(a)
        if math.hypot(x + 12, y - 10) < 6 or math.hypot(x, y) < 4:   # a clearing, and room for the fly
            continue
        items.append({"kind": "pillar", "x": round(x, 2), "y": round(y, 2), "strength": round(float(rng.uniform(0.7, 1.4)), 2)})
    for _ in range(6):
        r, a = rng.uniform(6, 24), rng.uniform(0, 2 * math.pi)
        items.append({"kind": "log", "x": round(r * math.cos(a), 2), "y": round(r * math.sin(a), 2),
                      "yaw": round(float(rng.uniform(0, math.pi)), 2), "strength": round(float(rng.uniform(0.7, 1.1)), 2)})
    items += [{"kind": "rock", "x": 18.0, "y": -6.0}, {"kind": "rock", "x": -6.0, "y": -18.0, "strength": 1.3},
              {"kind": "sugar", "x": 15.0, "y": 15.0}, {"kind": "water", "x": -12.0, "y": 10.0}]
    return {"name": "forest", "title": "Forest", "desc": "Stems and fallen twigs. Water in a clearing, food somewhere in the trees.",
            "ambient": {"T": 23.5, "H": 60.0}, "fly": {"x": 0.0, "y": 0.0, "heading": 0.0}, "items": items}


def _ring_of_fire():
    items = []
    for i in range(12):
        if i in (8, 9):                                   # the way in
            continue
        a = 2 * math.pi * i / 12
        items.append({"kind": "heat", "x": 14 * math.cos(a), "y": 14 * math.sin(a), "strength": 0.8})
    items += [{"kind": "sugar", "x": 1.0, "y": 1.0}, {"kind": "water", "x": -2.5, "y": -1.0, "strength": 0.8}]
    return {"name": "ring_of_fire", "title": "Ring of fire", "desc": "Food and water inside a ring of fire, with one cool way in.",
            "ambient": {"T": 24.0, "H": 40.0}, "fly": {"x": 6.0, "y": -24.0, "heading": math.pi / 2}, "items": items}


def _empty():
    return {"name": "empty", "title": "Empty", "desc": "Nothing but the floor and the wall.",
            "ambient": {"T": 24.0, "H": 40.0}, "fly": {"x": 0.0, "y": 0.0, "heading": 0.0}, "items": []}


def _stage():
    # B-roll only: the empty world, meant for FLYSIM_ARENA_R=120 (the wall is set by the arena radius at start-up)
    return {"name": "stage", "title": "Stage", "desc": "An empty floor for filming (start the app with FLYSIM_ARENA_R=120).",
            "ambient": {"T": 24.0, "H": 40.0}, "fly": {"x": 0.0, "y": 0.0, "heading": 0.0}, "items": []}


def _showcase():
    # the Demo world: something in view to react to straight away (sugar ahead, water to one side), a fire at a
    # distance whose glow it can feel if it wanders, one rock to walk round; open floor so the camera has room
    return {"name": "showcase", "title": "Showcase", "desc": "Sugar in view, water to one side, a fire at a distance.",
            "ambient": {"T": 24.0, "H": 40.0}, "fly": {"x": 0.0, "y": 0.0, "heading": 0.0},
            "items": [{"kind": "sugar", "x": 4.5, "y": 0.8, "r": 1.6},
                      # water was at (5, -5): 2 mm from the sugar's edge, so the fly stood on both and see-sawed
                      # between feeding and drinking with every thought about "the water and the sweetness" (18 Nov)
                      {"kind": "water", "x": 8.0, "y": -12.0, "strength": 0.9},
                      {"kind": "heat", "x": -8.0, "y": 7.0, "strength": 0.7},
                      {"kind": "rock", "x": -6.0, "y": -8.0, "size": [1.8, 1.4, 1.1]},
                      {"kind": "pillar", "x": 11.0, "y": 6.0}]}


PRESETS = {w["name"]: w for w in (_classic(), _labyrinth(), _two_rooms(), _oasis(), _winter(), _bitter_garden(),
                                  _forest(), _ring_of_fire(), _empty(), _stage(), _showcase())}


def safe_name(name):
    return re.sub(r"[^a-z0-9_-]+", "_", str(name).strip().lower())[:40].strip("_") or "my_world"


def user_worlds():
    out = {}
    if USER_DIR.is_dir():
        for f in sorted(USER_DIR.glob("*.json")):
            try:
                w = json.loads(f.read_text(encoding="utf-8"))
                w["name"], w["user"] = f.stem, True
                out[f.stem] = w
            except (OSError, ValueError):
                continue
    return out


def save_user_world(name, spec):
    USER_DIR.mkdir(parents=True, exist_ok=True)
    n = safe_name(name)
    if n in PRESETS:                       # a preset keeps its name; the copy gets its own
        n = f"{n}_mine"
    spec ={**spec, "name": n, "title": str(name).strip()[:40] or n, "desc": spec.get("desc") or "saved from the editor"}
    (USER_DIR / f"{n}.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return n


def get(name):
    return PRESETS.get(name) or user_worlds().get(name)


def catalog():
    return [{k: w.get(k) for k in ("name", "title", "desc", "ambient", "fly", "items", "user")}
            for w in list(PRESETS.values()) + list(user_worlds().values())]
