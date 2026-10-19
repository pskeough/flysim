"""Memory: a second, concurrent system that keeps the world as the fly last perceived it and competes with
the fly's gated motor system for the body (MIND_DESIGN.md steps 2 and 4, done as mechanism, not as an LLM).

What it is (all of it demo design, ours; none of it is a model of the fly's mushroom body or central
complex):

- Perception with a horizon. Each tick it works out what the fly perceives right now: things within a
  sight range (per kind: a fire's glow carries further than a drop of sugar) inside a 300 deg field of
  view, not hidden behind a wall; what its feet taste; heat or cold its antennae feel. Only these update
  memory. With memory on, the motor layer (L1) and the mind (L2) also see only what is perceived, so
  anything else the fly knows, it knows from here.
- A belief map. For each thing it has perceived: where, when, how (saw / tasted / felt), and a confidence
  that halves every HALF_LIFE s of sim time unless it is perceived again. When the fly can see the spot
  where it believes something is and the thing is not there, the belief turns MISSING: a surprise, which
  wakes the mind. Things the user moves while the fly is away stay where the fly last saw them until it
  looks again.
- Episodes. Tastes (sugar, water, bitter), burning heat, freezing cold, surprises, blackouts and lines the
  mind asks to REMEMBER, each with a valence and the place it happened.
- A pull. From the needs and the belief map it proposes one action: toward the best remembered food or
  water (need x confidence / distance), or away from a remembered danger just ahead. The action is a
  catalog intent (llm_bridge.CATALOG), computed from the BELIEVED position, so a fly that trusts an old
  memory walks to where the water was.
- The gate. In "contend" mode the pull bids for the motor channel with claim = weight x urgency x
  confidence against the motor layer's fixed claim (cognition.L1_CLAIM, active while L1 holds an intent).
  The higher claim drives the descending neurons; an L2 override beats both, and the fly's own reflexes
  (autonomy.Policy) are never muted. In "shadow" mode it perceives, remembers and proposes but never
  drives. Either way the proposal is compared with what the gated system chose without it: ALIGNED
  (same direction), CONTENDING (different), or LEADS (the rest of the system had no preference).
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from .world import SCENERY_KINDS, TASTES, footprint, nearest_point

RANGE = {"sugar": 8.0, "water": 9.0, "bitter": 8.0, "heat": 18.0, "cold": 9.0, "block": 7.0,
         "wall": 7.0, "rock": 9.0, "pillar": 8.0, "log": 7.0}
FOV_DEG = 150.0              # half-angle; a blind wedge of 60 deg behind
HALF_LIFE = 90.0             # sim s
GONE_R = 2.5                 # mm: a thing further than this from where it was believed is not "there"
MISSING_KEEP = 25.0          # sim s a MISSING belief stays on the map
FORGET_CONF = 0.08
MIN_CLAIM = 0.12
NOTABLE = ("sugar", "water", "bitter", "heat", "cold")
FAMILY = {"walk_forward": "ahead", "turn_left": "left", "rotate_left": "left", "turn_right": "right",
          "rotate_right": "right", "back_up": "back", "stop": "stay", "feed": "stay", "none": "explore",
          "escape": "flee", "groom_antennae": "stay"}
NAME = {"sugar": "SUGAR", "water": "WATER", "bitter": "BITTER", "heat": "HEAT", "cold": "COLD", "block": "block",
        "wall": "wall", "rock": "rock", "pillar": "stem", "log": "twig"}


def _bearing(pos, h, q):
    to = np.asarray(q, float) - pos
    return float(np.linalg.norm(to)), float(np.degrees(np.arctan2(h[0] * to[1] - h[1] * to[0], h @ to)))


def _occluded(p, q, walls, skip=None):
    """Does the sight line p -> q cross any wall segment (other than `skip`)?"""
    if not walls:
        return False
    W = np.array([w for sid, w in walls if sid != skip])
    if not len(W):
        return False
    d = np.asarray(q, float) - p
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return False
    q = p + d * max(0.0, (L - 0.4) / L)          # stop just short: a thing leaning on a wall stays visible
    a, b = W[:, :2], W[:, 2:]

    def orient(u, v, w):
        return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])
    o1, o2 = orient(p[None], q[None], a), orient(p[None], q[None], b)
    o3, o4 = orient(a, b, p[None]), orient(a, b, q[None])
    return bool(np.any((o1 * o2 < 0) & (o3 * o4 < 0)))


def perceive(snap):
    """-> {sid: (source, how)} for what the fly perceives now, and the wall segments used for occlusion."""
    sens, world = snap["sens"], snap["world"]
    pos, h = np.array(sens["pos"][:2]), np.array(sens["heading"][:2])
    walls = [(s["id"], footprint(s)[0]) for s in world if s["kind"] == "wall"]
    T = sens.get("temp_antenna") or [24.0, 24.0]
    T0 = snap.get("T0", 24.0)
    # in the dark it sees less far (a fire's glow carries as far as ever); touch and taste are unchanged
    lt = snap.get("light")
    dim = 0.3 + 0.7 * (1.0 if lt is None else float(lt))
    seen = {}
    for s in world:
        k = s["kind"]
        q = nearest_point(s, pos)
        dist, b = _bearing(pos, h, q)
        how = None
        if k in TASTES and any(sens.get(f"{k}_legs", ())) and dist < s.get("r", 1.6) + 1.2:
            how = "tasted"
        elif k == "heat" and dist < 12.0 and max(T) > T0 + 1.5:
            how = "felt"
        elif k == "cold" and dist < 10.0 and min(T) < T0 - 1.5:
            how = "felt"
        if how is None and dist <= RANGE.get(k, 7.0) * (dim if k != "heat" else float(np.clip(s.get("strength", 1), 0.5, 1.5))) \
                and (abs(b) <= FOV_DEG or dist < 2.5) and not _occluded(pos, q, walls, skip=s["id"]):
            how = "saw"
        if how:
            seen[s["id"]] = (s, how)
    return seen, walls


def horizon_world(snap, seen):
    """The world list the motor layer and the mind get: only what is perceived, with scenery as an
    OBSTACLE at its nearest point (affordances() knows blocks, not walls)."""
    pos = snap["sens"]["pos"][:2]
    out = []
    for s in snap["world"]:
        if seen is not None and s["id"] not in seen:
            continue
        if s["kind"] in SCENERY_KINDS:
            x, y = nearest_point(s, pos)
            out.append({"id": s["id"], "kind": "block", "x": x, "y": y, "strength": 1.0, "r": 1.0})
        else:
            out.append(s)
    return out


class Memory:
    def __init__(self, cog, mode="contend", weight=0.5):
        from .cognition import Actuator
        self.cog, self.sim = cog, cog.sim
        self.mode = mode                      # off | shadow | contend
        self.weight = float(weight)
        self.act = Actuator(self.sim.table, "M")
        self.lock = threading.Lock()
        self.beliefs: dict[int, dict] = {}
        self.episodes: deque = deque(maxlen=60)
        self.seen: dict = {}
        self.proposal = None
        self.claim = 0.0
        self.gate = {"holder": "fly", "state": "quiet", "align": None, "m_intent": None, "other": None, "other_src": None}
        self._samples: deque = deque(maxlen=400)      # (t, aligned: bool) while aligned or contending
        self._last_taste = {}
        self._last_temp_ep = -1e9
        self._last_pull = (None, -1e9)
        self._state_since = ("quiet", 0.0)
        self._last_state_log = -1e9
        self._state_logged = "quiet"
        self.snap_t = 0.0
        self._primed = False
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------------ control
    def configure(self, mode=None, weight=None):
        if mode in ("off", "shadow", "contend"):
            self.mode = mode
            if mode != "contend":
                self.act.clear()
                self.claim = 0.0
        if weight is not None:
            self.weight = float(np.clip(float(weight), 0.0, 1.0))

    def reset(self):
        """A new world: forget everything (the belief map would only mislead)."""
        with self.lock:
            self.beliefs.clear()
            self.episodes.clear()
            self.proposal, self.claim = None, 0.0
            self.seen = {}
            self._samples.clear()
            self._primed = False
            self._last_pull = (None, -1e9)
        self.act.clear()

    def remember(self, t, text, valence=0.0, source="mind", where=None):
        pos = where if where is not None else self.sim.sens["pos"][:2]
        with self.lock:
            self.episodes.append({"t": round(float(t), 1), "text": str(text)[:140], "val": round(float(valence), 2),
                                  "x": round(float(pos[0]), 2), "y": round(float(pos[1]), 2), "src": source})

    # ------------------------------------------------------------------ loop (5 Hz, own thread)
    def _loop(self):
        while getattr(self.sim, "cog", None) is None:     # Sim is still building (cog_snapshot needs sim.cog)
            time.sleep(0.2)
        while not self._stop:
            t0 = time.time()
            try:
                if getattr(getattr(self.sim, "phys", None), "dead", False):
                    self.act.clear()
                    self.claim = 0.0
                elif self.mode != "off" and not self.sim.paused:
                    self.step()
                elif self.mode == "off":
                    self.act.clear()
                    self.claim = 0.0
            except Exception as e:                      # never take the sim down; show it instead
                self.gate = {**self.gate, "error": f"{type(e).__name__}: {e}"[:200]}
            time.sleep(max(0.02, 0.2 - (time.time() - t0)))

    def step(self):
        from .cognition import DETOUR, _approach, _avoid, detour_bearing
        snap = self.sim.cog_snapshot()
        t, sens = snap["t"], snap["sens"]
        pos, h = np.array(sens["pos"][:2]), np.array(sens["heading"][:2])
        seen, walls = perceive(snap)
        world = {s["id"]: s for s in snap["world"]}
        log = self.cog.log
        events = []
        with self.lock:
            # 1. what is perceived now updates the map
            for sid, (s, how) in seen.items():
                b = self.beliefs.get(sid)
                k = s["kind"]
                if b is None:
                    self.beliefs[sid] = {"id": sid, "kind": k, "x": s["x"], "y": s["y"], "t_seen": t, "conf": 1.0,
                                         "status": "seen", "how": how, "t_first": t}
                    if k in NOTABLE and self._primed:        # not everything in view at the first look
                        events.append(("notice", f"noticed {NAME[k]} ({how}), {self._where(pos, h, s)}", 0.0))
                    continue
                if b["status"] == "missing":
                    events.append(("found", f"found the {NAME[k]} again, {self._where(pos, h, s)}", 0.2))
                elif np.hypot(s["x"] - b["x"], s["y"] - b["y"]) > GONE_R and k in NOTABLE:
                    events.append(("moved", f"the {NAME[k]} is not where it was: it has moved", 0.0))
                b.update(x=s["x"], y=s["y"], t_seen=t, conf=1.0, status="seen", how=how)
            # 2. what is not perceived is remembered, fades, or is found missing where it should be visible
            for sid, b in list(self.beliefs.items()):
                if sid in seen:
                    continue
                b["conf"] = float(0.5 ** ((t - b["t_seen"]) / HALF_LIFE))
                if b["status"] == "missing":
                    if t - b.get("t_missing", t) > MISSING_KEEP:
                        del self.beliefs[sid]
                    continue
                b["status"] = "remembered"
                dist, bear = _bearing(pos, h, (b["x"], b["y"]))
                can_check = dist < 0.7 * RANGE.get(b["kind"], 7.0) and (abs(bear) <= FOV_DEG or dist < 2.5) \
                    and not _occluded(pos, np.array([b["x"], b["y"]]), walls, skip=sid)
                real = world.get(sid)
                if can_check and (real is None or np.hypot(real["x"] - b["x"], real["y"] - b["y"]) > GONE_R):
                    b.update(status="missing", t_missing=t)
                    if b["kind"] in NOTABLE:
                        age = t - b["t_seen"]
                        events.append(("surprise", f"the {NAME[b['kind']]} was right here a moment ago, and now it is gone" if age < 2.0
                                       else f"expected {NAME[b['kind']]} here ({b['how']} {age:.0f} s ago); it is gone", -0.3))
                elif b["conf"] < FORGET_CONF:
                    del self.beliefs[sid]
            # 3. episodes from the body
            for k in TASTES:
                if any(sens.get(f"{k}_legs", ())) and t - self._last_taste.get(k, -1e9) > 6.0:
                    events.append(("taste", f"tasted {NAME[k]} here", {"sugar": 0.6, "water": 0.5, "bitter": -0.8}[k]))
                if any(sens.get(f"{k}_legs", ())):
                    self._last_taste[k] = t
            T = sens.get("temp_antenna")
            if T and t - self._last_temp_ep > 8.0:
                if max(T) > 28.0:
                    events.append(("hot", f"burning heat here ({max(T):.0f} C at the antennae)", -0.6))
                    self._last_temp_ep = t
                elif min(T) < 20.5:
                    events.append(("cold", f"freezing cold here ({min(T):.0f} C)", -0.4))
                    self._last_temp_ep = t
            for kind, text, val in events:
                if kind not in ("notice", "moved"):
                    self.episodes.append({"t": round(t, 1), "text": text, "val": val, "x": round(float(pos[0]), 2),
                                          "y": round(float(pos[1]), 2), "src": "body"})
            self.seen = {sid: how for sid, (s, how) in seen.items()}
            # 4. the pull
            needs = self.cog.needs
            best = None
            obst, bumps = None, ()
            if DETOUR:
                obst, bumps = self.cog.detour_inputs(snap)
            for b in self.beliefs.values():
                if b["status"] == "missing":
                    continue
                dist, bear = _bearing(pos, h, (b["x"], b["y"]))
                if b["kind"] in ("sugar", "water"):
                    need = needs.hunger if b["kind"] == "sugar" else needs.thirst
                    u = float(np.clip((need - 0.3) / 0.5, 0.0, 1.0))
                    if u <= 0:
                        continue
                    score = u * b["conf"] / (1.0 + dist / 15.0)
                    on = any(sens.get(f"{b['kind']}_legs", ())) and dist < 2.5
                    intent = _approach(dist, bear, on, b["kind"])
                    if not on and obst is not None:                  # round the scenery, not through it
                        b2, how = detour_bearing(pos, h, (b["x"], b["y"]), obst, bumps)
                        if how == "blocked" or (how == "push" and abs(b2) > 40):
                            intent = "rotate_left" if b2 > 0 else "rotate_right"
                        elif how:
                            intent = _approach(max(dist, 6.0), b2, False, b["kind"])
                    why = f"{'hunger' if b['kind'] == 'sugar' else 'thirst'} {need:.2f}"
                    act = "toward"
                elif b["kind"] in ("heat", "bitter", "cold") and dist < 6.0 and abs(bear) < 60:
                    u = 0.9 * (1.0 - dist / 6.0)
                    score = u * b["conf"]
                    intent, why, act = _avoid(dist, bear), "danger ahead", "away from"
                else:
                    continue
                if best is None or score > best["score"]:
                    best = {"sid": b["id"], "kind": b["kind"], "x": b["x"], "y": b["y"], "dist": round(dist, 1),
                            "bearing": round(bear), "intent": intent, "urgency": round(u, 2), "conf": round(b["conf"], 2),
                            "score": score, "act": act, "why": why, "status": b["status"],
                            "age": round(t - b["t_seen"], 1)}
            self.proposal = best
            self.claim = float(self.weight * best["urgency"] * best["conf"]) if best else 0.0
        if self.mode == "contend" and best and self.claim >= MIN_CLAIM:
            self.act.set(best["intent"], 0.7, 0.45, t)
        else:
            self.act.clear()
        # 5. alignment with what the gated system chose without memory
        other, src = self._other_choice(t)
        m_int = best["intent"] if best else None
        if not best:
            state = "quiet"
        elif other is None:
            state = "leads"
        else:
            state = "aligned" if FAMILY.get(m_int, m_int) == FAMILY.get(other, other) else "contending"
        if state in ("aligned", "contending"):
            self._samples.append((t, state == "aligned"))
        while self._samples and t - self._samples[0][0] > 20.0:
            self._samples.popleft()
        n = len(self._samples)
        align = round(sum(a for _, a in self._samples) / n, 2) if n >= 5 else None
        self.gate = {"holder": self.cog.gate.get("holder", "fly"), "state": state, "align": align, "m_intent": m_int,
                     "other": other, "other_src": src, "claim": round(self.claim, 2), "c1": self.cog.gate.get("c1", 0.0)}
        self.snap_t = t
        self._primed = True
        # 6. log (the journal and the mind hear about surprises; pulls and contention are rate-limited)
        for kind, text, val in events:
            if kind in ("surprise", "found", "moved"):
                log.add(t, "M", "surprise" if kind != "found" else "found", text)
            elif kind == "notice":
                log.add(t, "M", "notice", text)
        pull_key = (best["sid"], best["act"]) if best and self.claim >= MIN_CLAIM * 0.5 else None
        if pull_key != self._last_pull[0] and t - self._last_pull[1] > 3.0:
            if best and pull_key:
                src_txt = "remembered" if best["status"] != "seen" else "seen"
                log.add(t, "M", "pull", f"pulls {best['act']} the {src_txt} {NAME[best['kind']]} ({best['dist']:.0f} mm, {best['why']}; claim {self.claim:.2f})")
            elif self._last_pull[0] is not None:
                log.add(t, "M", "pull", "no pull: nothing it remembers is needed now")
            self._last_pull = (pull_key, t)
        if state != self._state_since[0]:
            self._state_since = (state, t)
        elif state != self._state_logged and t - self._state_since[1] > 1.0 and t - self._last_state_log > 4.0 \
                and state in ("aligned", "contending"):
            holder = self.gate["holder"]
            txt = (f"agrees with the {src}: {m_int.replace('_', ' ')}" if state == "aligned" else
                   f"contends with the {src}: memory says {m_int.replace('_', ' ')}, {src} says {other.replace('_', ' ')} "
                   f"({'memory' if holder == 'memory' else holder} holds the body)")
            log.add(t, "M", state, txt)
            self._state_logged, self._last_state_log = state, t

    def _other_choice(self, t):
        """What the gated system (without memory) is choosing now, and who: the mind's override, a reflex,
        the motor layer, or None when only aimless exploration is running."""
        c = self.cog
        a2 = c.l2_act.current(t)
        if a2 is not None:
            return a2[0], "mind"
        pol = self.sim.policy
        if pol.enabled and (pol.state == "backup" or getattr(pol, "avoid_label", "")):
            return ("back_up" if pol.state == "backup" else f"turn_{pol.turn[0]}" if pol.turn else "back_up"), "reflex"
        if pol.enabled and pol.state == "feed":
            return "feed", "reflex"
        a1 = c.l1_act.current(t)
        if a1 is not None:
            return a1[0], "motor layer"
        return None, None

    @staticmethod
    def _where(pos, h, s):
        from .cognition import _fly_dir, _fly_dist
        dist, b = _bearing(pos, h, (s["x"], s["y"]))
        return f"{_fly_dist(dist)}, {_fly_dir(b)}"

    # ------------------------------------------------------------------ what the mind and the UI get
    def describe(self, snap):
        """Lines for the mind's prompt, in the fly's terms (no numbers): remembered things (not perceived now),
        recent episodes, the pull."""
        import re
        pos, h = np.array(snap["sens"]["pos"][:2]), np.array(snap["sens"]["heading"][:2])
        t = snap["t"]
        with self.lock:
            bel = [b for b in self.beliefs.values() if b["kind"] in NOTABLE and b["id"] not in self.seen]
            eps = list(self.episodes)[-8:]
            p = self.proposal
        bel.sort(key=lambda b: -b["conf"])
        when = lambda a: "a moment ago" if a < 8 else "a little while ago" if a < 40 else "a long while ago"
        lines = []
        for b in bel[:6]:
            if b["status"] == "missing":
                lines.append(f"{NAME[b['kind']]}: you expected it {self._where(pos, h, b)}, but it was not there")
                continue
            sure = "sure" if b["conf"] > 0.7 else "fairly sure" if b["conf"] > 0.4 else "unsure"
            lines.append(f"{NAME[b['kind']]}: {self._where(pos, h, b)}, {b['how']} {when(t - b['t_seen'])} ({sure})")
        ep = [f"{when(t - e['t'])}: " + re.sub(r" \([^)]*\d[^)]*\)", "", e["text"]) for e in eps]
        pull = None
        if p and self.claim >= MIN_CLAIM * 0.5:
            g = self.gate
            why = re.sub(r" [\d.]+", "", p["why"])
            pull = (f"away from the {NAME[p['kind']]} ({why})" if p["act"] == "away from" else
                    f"toward the {'remembered ' if p['status'] != 'seen' else ''}{NAME[p['kind']]} ({why})")
            if g.get("state") == "contending":
                pull += "; the rest of you wants something else"
            elif g.get("state") == "aligned":
                pull += "; the rest of you agrees"
        return lines, ep, pull

    def status(self):
        with self.lock:
            bel = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in b.items() if k in
                    ("id", "kind", "x", "y", "conf", "status", "how")} for b in self.beliefs.values()]
            eps = list(self.episodes)[-14:]
            p = dict(self.proposal) if self.proposal else None
        if p:
            p.pop("score", None)
        return {"mode": self.mode, "weight": self.weight, "claim": round(self.claim, 2), "proposal": p,
                "gate": dict(self.gate), "beliefs": bel, "episodes": eps, "seen": list(self.seen),
                "range": RANGE, "fov": FOV_DEG}
