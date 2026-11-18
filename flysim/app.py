"""flysim app: body + brain in one loop, served to a local web front end.

    python -m flysim.app [--backend warp|numba] [--device cuda:0] [--port 8765] [--no-browser]

Loop (one coupling window = 10 ms simulated):
  1. read body sensory state (end of the previous window)
  2. sensory layer + autonomy policy + user inputs -> brain inputs (Poisson / silence / current)
  3. launch brain window k on the GPU (async)
  4. body steps window k with the drive decoded from brain window k-1   (10 ms latency)
  5. collect brain window k spike counts -> bridge filter, neural observer, runaway watchdog
LLM layer (llm_bridge.py, off by default): a thread reads the neural observer (stimulus -> LLM)
and returns an intent that becomes Poisson drive / silencing through the same mixer (LLM -> fly).
Nothing is paced faster than real time; when slower, the ratio is shown and not hidden.

HTTP: /  (front end), /scene.json + /scene.bin, /neurons.json + /neurons.bin,
      /api/state (GET), /api/worlds (GET), /api/input (POST), /api/stimulus (POST), /api/cmd (POST), /api/llm (POST), /ws (websocket)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from aiohttp import web, WSMsgType

from . import neurons
from .autonomy import InputMixer, Policy, SensoryLayer
from .body import Body
from .brain import Brain
from .bridge import Bridge
from .cognition import Cognition
from .llm_bridge import CATALOG, LLMAgent, NeuralObserver, observation_text
from .nociception import MIND_MODES, Nociception, hand_damage, hand_hit
from .others import Others
from .director import Director
from .pharma import DRUGS, Pharmacology, Targets
from .physiology import Physiology
from .world import World

WEB = Path(__file__).resolve().parents[1] / "web"
WINDOW_MS = 10.0
# Runaway watchdog (GROUNDING.md item 6): the model has a self-sustaining attractor (~7-10% of
# neurons, antennal-lobe PN<->LN recurrence) that never ends on its own. If the filtered
# fraction of neurons spiking per window stays above RUNAWAY_FRAC for RUNAWAY_MS, reset membranes.
RUNAWAY_FRAC, RUNAWAY_MS = 0.02, 300.0
SEIZURE_MS = 3000.0   # under a drug the watchdog waits this long, so a seizure is a visible bout (then resets)
GROOM_DN_HZ = 15.0    # aDN1/aDN2 filtered rate that starts a grooming bout (basal 0 Hz; the groom_antennae intent gives
                      # ~40 Hz, an antenna touch ~20 Hz, eye bristles ~145 Hz: GROUNDING.md item 2) [ours]
NO_TAKEOFF = os.environ.get("FLYSIM_NO_TAKEOFF", "0") == "1"   # a shot can keep the fly on the ground (B-roll fly-cam)


class Sim:
    RUNAWAY_FRAC, RUNAWAY_MS, SEIZURE_MS = RUNAWAY_FRAC, RUNAWAY_MS, SEIZURE_MS      # others.py reads them from the sim

    def __init__(self, backend="warp", device="cuda:0", seed=0, physics="fast"):
        t0 = time.perf_counter()
        self.body = Body(seed=seed, physics=physics)
        self.director = Director(seed)
        self.director.on = os.environ.get("FLYSIM_DIRECTOR", "0") == "1"
        self.brain = Brain(backend=backend, device=device, seed=seed)
        self.table = neurons.load(self.brain.ids)
        self.bridge = Bridge(self.table)
        self.sensory_layer = SensoryLayer(self.table)
        self.policy = Policy(self.table, seed=seed)
        self.mixer = InputMixer(self.brain.n)
        self.user_inputs: dict[str, tuple] = {}     # name -> (idx, rate) Poisson
        self.user_silenced: dict[str, np.ndarray] = {}
        self.user_sil_mask = np.zeros(self.brain.n, bool)
        self.observer = NeuralObserver(self.table)
        self.llm = LLMAgent(self.table)
        self.llm.obs_fn = self._llm_observation
        self.runaway_ms = 0.0
        self.n_runaway = 0
        self.n_takeoff = 0
        self.last_runaway_t = -1e9
        self.stimuli: list[list] = []              # [name, idx, rate, ms_left]
        self.cmds: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.paused = False
        self.running = True
        self.spike_acc = np.zeros(self.brain.n, np.uint32)
        self.world = World(self.body)
        self.phys = Physiology()
        self.speed = 1.0                           # target sim s per wall s; 0 = as fast as it goes
        self.step_left = 0                         # windows to run while paused (the step button)
        self.hand = None                           # the hand coming down (see _hand_tick)
        self.ring = None                           # a closing ring of fire (cinema, _scene_tick)
        self.pinned = False                        # cinema: held down (cmd "pin")
        self.amb_ramp = None                       # the floor heating or cooling (cinema, _scene_tick)
        self.lives = 1
        self._light_ph = None
        self._ck_key, self._ck_t = None, 0.0
        self.sens = self.world.augment(self.body.sensory(), self.body)
        self.drive = np.zeros(2)
        self.sim_t = 0.0
        self.wall_body = self.wall_brain = self.wall_total = 0.0
        self.rtf = 0.0
        self.events: list[str] = []
        self.drug_targets = Targets(self.brain)    # drugs (pharma.py): who each drug acts on, resolved once
        self.pharma = Pharmacology(self.drug_targets, self.brain)
        self.drug_aware = "felt"                   # what the mind is told: hidden | felt | told
        self._fed_prev = 0.0
        self.noci = Nociception(self.table)        # injury -> stand-in ascending set -> PPL1 -> pain state
        self.cog = Cognition(self)
        self.others = Others(self)                 # more flies, each with its own brain (others.py)
        self._oth_touch = set()
        self.build_s = time.perf_counter() - t0
        self.brain.run(20.0)  # compile / capture graph

    # ------------------------------------------------------------------ API (thread-safe)
    def set_input(self, target: str, mode: str = "activate", value: float = 150.0, on: bool = True):
        """(a) Exogenous input to named neurons / cell types.
        mode: 'activate' (Poisson Hz), 'silence' (zero outgoing synapses), 'current' (mV)."""
        self.cmds.put(("input", target, mode, value, on))

    def inject_stimulus(self, kind: str, strength: float = 1.0, ms: float = 300.0):
        """(b) Sensory stimulus: 'loom_left' / 'loom_right' (LPLC2), 'sugar' (sugar GRNs),
        or 'drop' (a real object falls near the fly and is sensed by the body)."""
        self.cmds.put(("stimulus", kind, strength, ms))

    def read_state(self) -> dict:
        """(c) Sensory and descending readout."""
        with self.lock:
            return self._status()

    # ------------------------------------------------------------------ loop
    def _handle(self, c):
        k = c[0]
        if k == "input":
            _, target, mode, value, on = c
            idx = self.table.find(target)
            if not len(idx):
                self.events.append(f"no neurons match '{target}'")
                return
            key = f"{target}:{mode}"
            if mode == "activate":
                if on:
                    self.user_inputs[key] = (idx, float(value))
                else:
                    self.user_inputs.pop(key, None)
            elif mode == "silence":
                if on:
                    self.user_silenced[key] = idx
                else:
                    self.user_silenced.pop(key, None)
                self.user_sil_mask[:] = False
                for i in self.user_silenced.values():
                    self.user_sil_mask[i] = True
            elif mode == "current":
                self.mixer.i_ext[idx] = float(value) if on else 0.0
            self.events.append(f"{'+' if on else '-'} {mode} {target} ({len(idx)} neurons)"
                               + (f" {value:g}" if on and mode != "silence" else ""))
        elif k == "stimulus":
            _, kind, strength, ms = c
            if kind == "drop":
                self.body.drop_object()
                self.events.append("object dropped")
            elif kind in ("loom_left", "loom_right"):
                idx = self.sensory_layer.lplc2[kind.split("_")[1]]
                self.stimuli.append([kind, idx, 150.0 * strength, ms])
                self.events.append(f"stimulus {kind} -> LPLC2 {ms:g} ms")
            elif kind == "sugar":
                self.stimuli.append([kind, self.sensory_layer.sugar, 200.0 * strength, ms])
                self.events.append(f"stimulus sugar -> sugar GRNs {ms:g} ms")
        elif k == "drop":
            self.body.drop_object(*c[1:])
            self.events.append("object dropped")
        elif k == "push":
            self.body.push(*c[1:])
            self.events.append("push")
        elif k == "reset":
            self.body.reset(*c[1:])
            self.pharma.down = False             # the body stands up; a drug still at knock-down level fells it again
            if self.phys.dead:
                self.body.set_state("dead")
            self.world.reapply()
            self.policy.reset()
            self.bridge.reset()
            self.events.append("reset")
        elif k == "fly_place":
            _, x, y, heading = c
            x, y = self.world._inside(x, y, margin=2.5)
            self.body.reset((x, y), heading)
            self.pharma.down = False
            self.world.reapply()
            self.policy.reset()
            self.bridge.reset()
            if self.phys.dead:
                self.body.set_state("dead")
            self.cog.log.add(self.sim_t, "world", "fly", f"you picked the fly up and set it down at ({x:.1f}, {y:.1f}) mm")
        elif k == "puff":
            # a puff of air from (x, y): a push away from it (demo gain, 2x body weight for 25 ms) and the
            # wind-sensitive Johnston's organ neurons (JO-C/E; Yorozu et al. 2009, Nature,
            # doi:10.1038/nature07843) on the side it comes from, 150 Hz for 300 ms (rate and duration ours)
            _, x, y = c
            p, h = np.array(self.sens["pos"][:2]), np.array(self.sens["heading"][:2])
            away = p - np.array([x, y])
            n = float(np.linalg.norm(away)) or 1.0
            away /= n
            self.body.push(2.0 * away[0], 2.0 * away[1], 0.2, 25.0)
            frm = -away
            side = "left" if h[0] * frm[1] - h[1] * frm[0] > 0 else "right"
            ahead = float(h @ frm)
            self.stimuli.append([f"puff_{side}", self.table.find(f"JO_wind_{side}"), 150.0, 300.0])
            where = "in front" if ahead > 0.7 else "behind" if ahead < -0.7 else f"the {side}"
            self.cog.log.add(self.sim_t, "world", "puff", f"a gust of air hit you from {where}")
        elif k == "clear":
            self.user_inputs.clear(); self.user_silenced.clear(); self.stimuli.clear()
            self.user_sil_mask[:] = False; self.mixer.i_ext[:] = 0
            self.body.clear_objects()
            for sid in [i for i, w in self.world.sources.items() if w["kind"] == "block"]:
                self.world.sources.pop(sid)
            self.events.append("cleared inputs and objects")
        elif k == "pause":
            self.paused = bool(c[1])
            self.step_left = 0
            self.lockstep = False
        elif k == "speed":
            self.speed = float(np.clip(float(c[1]), 0.0, 4.0))
        elif k == "reserves":                 # set the fly's energy / water directly (scripted shots: a hungry fly now)
            if c[1] is not None:
                self.phys.energy = float(np.clip(float(c[1]), 0.0, 1.0))
            if c[2] is not None:
                self.phys.water = float(np.clip(float(c[2]), 0.0, 1.0))
        elif k == "life_rate":
            self.phys.rate = float(np.clip(float(c[1]), 0.25, 16.0))
            for o in self.others.flies.values():
                o.phys.rate = self.phys.rate
        elif k == "other_add":
            _, x, y, heading = c
            x, y = self.world._inside(x, y, margin=2.5)
            o = self.others.add(x, y, heading)
            if o is None:
                self.events.append(f"no room for another fly (at most {len(self.others.flies)})")
            else:
                self.cog.log.add(self.sim_t, "world", "other_add", f"you put another fly ({o.name}) down at ({x:.1f}, {y:.1f}) mm",
                                 x=float(x), y=float(y), who=o.k)
        elif k == "other_remove":
            o = self.others.flies.get(int(c[1]))
            if o is not None and self.others.remove(o.k):
                self.cog.log.add(self.sim_t, "world", "other_remove", f"you took {o.name} away", x=o.body.x, y=o.body.y, who=o.k)
        elif k == "other_move":
            o = self.others.flies.get(int(c[1]))
            if o is not None:
                x, y = self.world._inside(c[2], c[3], margin=2.5)
                if o.body.mode == "flight":
                    o.body.set_state("walk")
                o.body.place(x, y, c[4])
        elif k == "other_revive":
            o = self.others.flies.get(int(c[1]))
            if o is not None and o.phys.dead:
                o.revive()
                self.cog.log.add(self.sim_t, "world", "other_revive", f"you brought {o.name} back to life", x=o.body.x, y=o.body.y, who=o.k)
        elif k == "others_clear":
            if self.others.flies:
                self.others.clear()
                self.cog.log.add(self.sim_t, "world", "others_clear", "you took the other flies away")
        elif k == "step":
            self.paused, self.step_left = True, int(np.clip(int(c[1]), 1, 1000))
            self.lockstep = True                   # frame stepping (lockstep recording): the mind keeps thinking
        elif k == "revive":
            self.revive(bool(c[1]))
        elif k == "day":
            self.world.set_day(c[1], c[2])
            d = self.world.env()["day"]
            self.cog.log.add(self.sim_t, "world", "day", "day and night off: always day" if not d["len"] else
                             f"day and night: a day lasts {d['len']:g} s; it is {int(d['hour']):02d}:{int(d['hour'] % 1 * 60):02d}")
        elif k == "hand":
            if self.hand is None:
                x, y = self.world._inside(float(c[1]), float(c[2]), margin=1.0)
                v = float(c[3]) if len(c) > 3 and c[3] else float(self.body.rng.uniform(60.0, 160.0))   # swats vary
                kw = c[4] if len(c) > 4 and c[4] else {}
                if kw.get("rest"):               # the press settles on the fly where it is now
                    x, y = float(self.sens["pos"][0]), float(self.sens["pos"][1])
                elif kw.get("target") == "other" and self.others.flies:     # on the nearest other fly
                    p0 = self.sens["pos"]
                    o = min(self.others.flies.values(), key=lambda o: np.hypot(o.body.x - p0[0], o.body.y - p0[1]))
                    x, y = float(o.body.x), float(o.body.y)
                self.hand = {"x": x, "y": y, "z": float(kw.get("top", self.HAND_TOP)), "r": float(kw.get("r", self.HAND_R)),
                             "phase": "down", "v": v}
                if kw.get("crush"):              # a slow press (cinema): pins the fly, then crushes it by depth
                    self.hand.update(crush=True, vp=float(kw.get("vp", 0.4)), hold=float(kw.get("hold", 1.0)), rest=bool(kw.get("rest")))
                self.cog.log.add(self.sim_t, "world", "hand", f"you brought a hand down at ({x:.1f}, {y:.1f}) mm")
        elif k == "pin":                         # cinema: an invisible restraint (its legs cannot move; the brain runs on)
            self.pinned = bool(c[1])
            self.pin_others = bool(c[2]) if len(c) > 2 else False
            self.body.set_state("held" if self.pinned else "walk")
            self.cog.log.add(self.sim_t, "L0", "body", "PINNED: something holds it down, its legs cannot move" if self.pinned
                             else "released: it can move again")
        elif k == "fire_ring":
            self._fire_ring_start(c[1])
        elif k == "ambient":
            _, T, secs = c
            self.amb_ramp = {"T0": self.world.T0, "T1": float(T), "t0": self.sim_t, "s": max(0.01, float(secs))}
        elif k == "autonomy":
            self.policy.enabled = bool(c[1]); self.policy.reset()
        elif k == "sensory":
            self.sensory_layer.enabled = bool(c[1])
        elif k == "right":
            self.body.right()
        elif k == "llm_force":
            self.llm.force(*c[1:])
            self.events.append(f"intervention {c[1]} (manual)")
        elif k == "world_add":
            _, kind, x, y, kw = c
            self._checkpoint()
            sid = self.world.add(kind, x, y, **kw)
            if sid:
                w = self.world.sources[sid]
                laced = f", laced with {DRUGS[w['drug']].name.lower()}" if w.get("drug") in DRUGS else ""
                self.cog.log.add(self.sim_t, "world", "add", f"you placed {kind.upper()} at ({w['x']:.1f}, {w['y']:.1f}) mm{laced}",
                                 id=sid, vis=self._visible(sid))
            else:
                self.world.undo_stack.pop()
                self.cog.log.add(self.sim_t, "world", "full", f"no free {kind} left (the pool is full)")
        elif k == "world_move":
            _, sid, x, y, final = c
            if int(sid) in self.world.sources:
                self._checkpoint(("move", sid))
            if self.world.move(sid, x, y) and final:
                w = self.world.sources[int(sid)]
                self.cog.log.add(self.sim_t, "world", "move", f"you moved {w['kind'].upper()} to ({w['x']:.1f}, {w['y']:.1f}) mm",
                                 id=int(sid), vis=int(sid) in self.cog.memory.seen or self._visible(sid))
        elif k == "world_set":
            _, sid, kw = c
            if int(sid) in self.world.sources:
                self._checkpoint(("set", sid))
                self.world.set(sid, **kw)
        elif k == "world_remove":
            w = self.world.sources.get(int(c[1]))
            vis = w is not None and self._visible(c[1])
            if w:
                self._checkpoint()
            if self.world.remove(c[1]) and w:
                self.cog.log.add(self.sim_t, "world", "remove", f"you removed {w['kind'].upper()}", id=int(c[1]), vis=vis)
        elif k == "world_clear":
            self._checkpoint()
            self.world.clear()
            self.cog.log.add(self.sim_t, "world", "clear", "you cleared the arena")
        elif k in ("world_default", "world_load"):
            from . import worlds
            spec = worlds.get("classic") if k == "world_default" else (c[1] if isinstance(c[1], dict) else worlds.get(c[1]))
            if not spec:
                self.events.append(f"no world named {c[1]!r}")
                return
            self._checkpoint()
            self.world.load(spec)
            fly = spec.get("fly") or {}
            self.body.reset((float(fly.get("x", 0.0)), float(fly.get("y", 0.0))), float(fly.get("heading", 0.0)))
            self.world.reapply()
            self.policy.reset()
            self.bridge.reset()
            self.cog.memory.reset()
            self.pharma.down = False
            self.noci.reset()
            self.cog.log.add(self.sim_t, "world", "load", f"you loaded the world '{self.world.title}'")
            if self.phys.dead:
                self.revive(True)
        elif k in ("world_undo", "world_redo"):
            if (self.world.undo() if k == "world_undo" else self.world.redo()):
                self.cog.log.add(self.sim_t, "world", "undo", "you undid an edit" if k == "world_undo" else "you redid an edit")
        elif k == "drug":
            _, key, route, amount, who = c
            if key not in DRUGS:
                self.events.append(f"no drug named {key!r}")
                return
            d = DRUGS[key]
            if route not in d.routes:
                self.events.append(f"{d.name} cannot be given by {route}")
                return
            if who is None:
                ph, name = self.pharma, "the fly"
            else:
                o = self.others.flies.get(int(who))
                if o is None:
                    return
                ph, name = o.pharma, o.name
            ph.dose(key, route, amount)
            what = {"inject": f"you injected {name} with {d.name.lower()}",
                    "feed": f"you fed {name} {d.name.lower()}",
                    "vapour": (f"you filled the air around {name} with {d.name.lower()} vapour" if amount > 0
                               else f"you cleared the {d.name.lower()} vapour")}[route]
            self.cog.log.add(self.sim_t, "world", "drug", what, drug=key, route=route, who=who, on=amount > 0,
                             told=self.drug_aware == "told" and who is None)
        elif k == "drug_clear":
            who = c[1]
            ph = self.pharma if who is None else getattr(self.others.flies.get(int(who)), "pharma", None)
            if ph is not None and ph.any():
                ph.clear()
                self.cog.log.add(self.sim_t, "world", "drug", "you washed every drug out" +
                                 ("" if who is None else f" of {self.others.flies[int(who)].name}"), drug="", who=who,
                                 told=self.drug_aware == "told" and who is None)
        elif k == "drug_aware":
            self.drug_aware = c[1] if c[1] in ("hidden", "felt", "told") else "felt"
        # nociception: these never reach the mind as events (an opto drive must look like the circuit's own signal)
        elif k == "pain_opto":
            _, n, side, secs = c
            self.noci.set_opto(n, side if side in ("left", "right", "both") else "both", secs, self.sim_t)
            self.cog.log.add(self.sim_t, "user", "pain_opto", f"opto drive {n:.2f} on the {side} stand-in set for {secs:g} s")
        elif k == "pain_silence":
            self.noci.silence = c[1] if c[1] in ("entry", "readout") else ""
            self.cog.log.add(self.sim_t, "user", "pain_silence", f"nociceptive {c[1] or 'circuit'} " + ("silenced" if c[1] else "restored"))
        elif k == "pain_mind":
            self.noci.mind_mode = c[1] if c[1] in MIND_MODES else "text"
        elif k == "pain_steer":
            self.noci.steer = bool(c[1])
        elif k == "talk":
            self.cog.say(c[1], c[2])
        elif k == "talk_release":
            self.cog.release_order()
        elif k == "cog":
            self.cog.configure(c[1])
        elif k == "llm":
            self.llm.set(**c[1])
            self.events.append("LLM " + ("on" if self.llm.enabled else "off") + (f": {self.llm.goal}" if self.llm.enabled else ""))

    def _visible(self, sid):
        """Does the fly perceive this thing right now (memory.perceive)? For the mind's version of an edit."""
        if self.cog.memory.mode == "off":
            return True
        from .memory import perceive
        seen, _ = perceive({"sens": self.sens, "world": self.world.status(), "T0": self.world._T_amb(), "light": self.world.light() if self.world.day_len > 0 else None})
        return int(sid) in seen

    def _checkpoint(self, key=None):
        """Undo point before an edit. A drag or a wheel-resize sends many edits to one thing: one undo point
        for the gesture (same key within 1.5 s)."""
        now = time.time()
        if key is None or key != self._ck_key or now - self._ck_t > 1.5:
            self.world.checkpoint()
        self._ck_key, self._ck_t = key, now

    def drain_commands(self):
        """Apply queued commands (API calls, websocket, REST) on the sim thread. A command that raises is logged and
        dropped, so one bad command cannot stop the simulation thread."""
        while True:
            try:
                c = self.cmds.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle(c)
            except Exception as e:  # noqa: BLE001
                print(f"command dropped ({type(e).__name__}: {e}): {str(c)[:200]}", flush=True)

    def step_window(self):
        t0 = time.perf_counter()
        self.drain_commands()
        n_steps = int(round(WINDOW_MS * 1e-3 / self.body.dt))
        self.world.tick_day(self.sim_t)
        self._scene_tick()
        self.policy.activity = self.world.activity()
        if self.world.day_len > 0:                     # dusk and dawn reach the fly as a change in the light
            ph = 0 if self.world.light() < 0.2 else 2 if self.world.light() > 0.8 else 1
            if self._light_ph is not None and ph != self._light_ph and not self.phys.dead:
                self.cog.log.add(self.sim_t, "L0", "sense", {(2, 1): "the light is fading", (1, 0): "darkness falls",
                                                             (0, 1): "the dark is thinning: grey light", (1, 2): "full light again"}
                                 .get((self._light_ph, ph), "the light changes"))
            self._light_ph = ph
        else:
            self._light_ph = None
        if self.phys.dead:
            return self._dead_window(t0, n_steps)
        sens = self.sens
        self._drug_tick(sens)
        cog_src, cog_sil, held = self.cog.inputs(self.sim_t)
        self.policy.neural = self.observer.rate
        self.policy.yield_motor = held
        self.mixer.set_source("sensory", self.sensory_layer.update(sens, self.body))
        self.mixer.set_source("autonomy", self.policy.update(sens, WINDOW_MS * 1e-3, self.body))
        for name, src in cog_src.items():
            self.mixer.set_source(name, src)
        self.mixer.set_source("user", dict(self.user_inputs))
        stim = {}
        for s in self.stimuli:
            stim[s[0]] = (s[1], s[2])
            s[3] -= WINDOW_MS
        self.stimuli = [s for s in self.stimuli if s[3] > 0]
        self.mixer.set_source("stimulus", stim)
        self.mixer.set_source("noci", self.noci.inputs(self.phys, self.sim_t, sens))
        llm_act, llm_sil = self.llm.inputs()
        self.mixer.set_source("llm", llm_act)
        np.copyto(self.mixer.silenced, self.user_sil_mask)
        ns = self.noci.silenced_idx()
        if ns is not None:
            self.mixer.silenced[ns] = True
        if llm_sil is not None:
            self.mixer.silenced[llm_sil] = True
        for sl in cog_sil:
            self.mixer.silenced[sl] = True
        if self.runaway_ms >= (SEIZURE_MS if self.pharma.any() else RUNAWAY_MS):   # a drugged runaway is the seizure: let it show
            self.brain.reset_state()
            self.runaway_ms = 0.0
            self.n_runaway += 1
            self.last_runaway_t = self.sim_t
            self.observer.active_frac = 0.0
            self.events.append("runaway brain state -> membranes reset")
            self.cog.log.add(self.sim_t, "L0", "runaway", "brain overload: the simulation reset its neurons")
        self.mixer.apply(self.brain)
        self.brain.launch(WINDOW_MS)
        self._others_launch()
        # the body can refuse the brain: nothing reaches paralysed legs, a weak fly half-steps
        drive = self._drug_drive(self.bridge.drive() * self.phys.motor_gain())
        self._hand_tick()
        if self.bridge.takeoff() and self.body.mode == "walk" and self.phys.motor_gain() > 0 and not self.pharma.seizing and not NO_TAKEOFF:   # scripted: a convulsing fly cannot fly
            away = None
            if self.body.ext_loom:                     # jump away from what looms (straight ahead if it is overhead)
                p = np.asarray(self.sens["pos"][:2]); q = np.asarray(self.body.ext_loom[0][0][:2])
                d = p - q
                n = float(np.linalg.norm(d))
                away = d / n if n > 0.4 else np.asarray(self.sens["heading"][:2])
            self.body.takeoff(away=away)
            self.n_takeoff += 1
            self.events.append("giant fibre fired -> take-off")
        t1 = time.perf_counter()
        self.body.ext_loom_extra = self.others.flying()
        self.body.step(drive, n_steps)
        t2 = time.perf_counter()
        counts = self.brain.collect()
        t3 = time.perf_counter()
        self._others_finish()
        # a silenced neuron still spikes but its synapses are cut; for a DN those synapses
        # are its output to the body, so the bridge sees nothing from it
        seen = np.where(self.mixer.silenced, 0, counts)
        self.bridge.update(seen, WINDOW_MS)
        self.observer.update(seen, WINDOW_MS)
        self._anim_cues(drive)
        self.runaway_ms = self.runaway_ms + WINDOW_MS if self.observer.active_frac > RUNAWAY_FRAC else 0.0
        self._pain_tick()
        with self.lock:
            np.add(self.spike_acc, counts, out=self.spike_acc, casting="unsafe")
            self.sens = self.world.augment(self.body.sensory(), self.body)
            self.drive = drive
            self.sim_t += WINDOW_MS * 1e-3
            a = 0.05
            self.wall_body += a * ((t2 - t1) - self.wall_body)
            gpu = self.brain.last_gpu_ms
            bw = gpu * 1e-3 if gpu is not None else (t3 - t2)
            self.wall_brain += a * (bw - self.wall_brain)
        self._phys_tick()
        self._touch_others()
        self.cog.tick(WINDOW_MS * 1e-3)
        return time.perf_counter() - t0

    # ------------------------------------------------------------------ visible behaviour from read-outs (body.py, anim.py)
    def _anim_cues(self, drive, dead=False):
        """Read-outs the brain already produces, shown on the body:
        - antennal grooming: the aDN read-outs (aDN1 = DNg62, aDN2 = DNge078; Hampel et al. 2015) above GROOM_DN_HZ on
          a standing fly start or extend a foreleg grooming bout (body.groom_start); a walking command ends it;
        - proboscis: MN9 (anim.FaceAnim), head yaw from the steering asymmetry and the looming side, rest pumping.
        The cosmetic channels change only the streamed pose; the grooming legs are real position targets."""
        b = self.body
        if dead:
            b.anim.cues(dead=True)
            b.anim.tick(WINDOW_MS * 1e-3)
            return
        r = self.observer.rate
        g = max(r.get("aDN1", 0.0), r.get("aDN2", 0.0))
        if g > GROOM_DN_HZ and b.mode == "walk" and float(np.abs(drive).max()) < 0.25 and not self.pinned:
            if b.groom_start("anterior", 0.3) and b.groom["t"] == 0.0:
                self.events.append(f"grooming DNs {g:.0f} Hz -> antennal grooming")
        gr = b.grooming()
        s = self.sens
        loom = s.get("loom_side") if (s.get("loom_angle") or 0.0) > 0.35 else None
        still = abs(s.get("v_forward", 0.0)) < 0.3 and float(np.abs(drive).max()) < 0.05
        b.anim.cues(mn9=r.get("MN9", 0.0), groom_side="both" if (gr and gr[0] == "anterior" and gr[1] > 0.3) else None,
                    turn=float(np.clip(drive[1] - drive[0], -1.0, 1.0)), loom_side=loom, resting=still,
                    seizing=bool(self.pharma.seizing), flying=b.mode == "flight")
        b.anim.tick(WINDOW_MS * 1e-3)

    # ------------------------------------------------------------------ drugs (pharma.py)
    def _drug_tick(self, sens):
        """Kinetics, then the brain levers (resting shifts through the mixer, synapse gains on the GPU), then
        the scripted body channel (walking bouts, knock-down). Feeding doses only while the fly eats: legs on
        laced food and the proboscis motor neuron firing (physiology counts that as fed time)."""
        ph = self.pharma
        eating = sens.get("drug_legs", ()) if self.phys.fed_s > self._fed_prev else ()
        self._fed_prev = self.phys.fed_s
        if not ph.active(eating):
            return
        ph.tick(self.sim_t, WINDOW_MS * 1e-3, eating)
        np.copyto(self.mixer.drug_i, ph.rest)
        ph.apply_policy(self.policy)
        down = ph.knocked_down()
        if down and not ph.down:
            ph.down = True                       # set even while paralysed, so recovery does not stand it up first
            if not self.phys.paralysed:
                self.body.set_state("limp")
                self.cog.log.add(self.sim_t, "L0", "body", "KNOCKED DOWN: it falls and its legs stop answering")
        elif not down and ph.down:
            ph.down = False
            if not self.phys.paralysed and self.body.mode == "limp":
                self.body.set_state("walk")
                self.cog.log.add(self.sim_t, "L0", "body", "gets up: its legs answer again")

    @property
    def _drug_down(self):
        return self.pharma.down

    def _drug_drive(self, drive):
        """The scripted body channel on the legs' drive (pharma.Pharmacology.shape_drive)."""
        drive, onset = self.pharma.shape_drive(drive, self.body.rng, self.observer.active_frac > RUNAWAY_FRAC,
                                               self.sim_t, WINDOW_MS)
        if onset:
            self.cog.log.add(self.sim_t, "L0", "body", "SEIZURE: its brain activity runs away and its legs convulse")
        if self.pharma.seizing and self.body.mode == "flight":     # scripted: a convulsing fly drops out of the air
            self.body.set_state("limp")
            self.body.set_state("walk")
        return drive

    # ------------------------------------------------------------------ nociception (nociception.py)
    def _pain_tick(self, dead=False):
        """PPL1 read-out -> pain state; onsets are logged for the mind (cognition rate-limits the wakes) and leave
        an episode in memory, worded the way the mind is told it (nothing when hidden)."""
        nc = self.noci
        for kind, text, prio in nc.update(self.observer.rate, self.sim_t, WINDOW_MS * 1e-3, dead=dead):
            lines = {m: nc.sensor_line(m) for m in MIND_MODES} if kind == "pain" else {}
            self.cog.log.add(self.sim_t, "L0", kind, text, lines=lines, I=round(nc.I, 3), src=nc.src)
            line = lines.get(nc.mind_mode, "")
            if kind == "pain" and line and self.cog.memory.mode != "off":
                self.cog.memory.remember(self.sim_t, f"here: {line}", -nc.I, "body")

    # ------------------------------------------------------------------ the other flies
    def _others_launch(self):
        for o in self.others.flies.values():
            o.launch(self, WINDOW_MS)

    def _others_finish(self):
        for k, o in list(self.others.flies.items()):
            o.finish(self, WINDOW_MS, self.others.looms_for(exclude=k))

    def _touch_others(self):
        """Another fly against this one (a contact on its body, or its rays on ours) reaches the fly as touch."""
        now = set(self.sens.get("touch_others", ()))
        now |= {o.k for o in self.others.flies.values() if -1 in o.body.touch_others}
        new = now - self._oth_touch
        self._oth_touch = now
        for k in sorted(new):
            o = self.others.flies.get(k)
            if o is None:
                continue
            what = ("a dead fly" if o.phys.dead else "a fly that does not move" if o.body.mode == "limp"
                    else "another fly")
            self.cog.log.add(self.sim_t, "L0", "fly", f"{what} against you ({o.name})", who=k)

    # ------------------------------------------------------------------ life and death
    def _phys_tick(self):
        p = self.sens["pos"]
        ch = self.phys.update(self.sens, self.observer.rate, WINDOW_MS * 1e-3, self.world.temperature(p),
                              self.world.humidity(p), abs(self.sens["v_forward"]), self.body.mode == "flight")
        if ch is None:
            return
        tb, log = self.phys.tb, self.cog.log
        if ch in ("stupor", "coma"):
            self.body.set_state("limp")
            log.add(self.sim_t, "L0", "body", ("HEAT STUPOR: its body is at %.1f C and its legs stop answering" if ch == "stupor"
                                               else "CHILL COMA: its body is at %.1f C and it cannot move") % tb)
        elif ch == "dead":
            self._die()
        else:
            if self.body.mode == "limp" and not self._drug_down and not (self.hand and self.hand.get("zc")) and not self.pinned:   # still pinned
                self.body.set_state("walk")
                log.add(self.sim_t, "L0", "body", f"comes round: body at {tb:.1f} C, it can move again")
            if ch == "weak":
                log.add(self.sim_t, "L0", "body", "WEAK: " + ("no energy left" if self.phys.energy < 0.06 else
                                                              "almost dried out" if self.phys.water < 0.06 else "badly hurt"))

    def _die(self):
        ph = self.phys
        self.body.set_state("dead")
        self.stimuli.clear()
        self.cog.log.add(self.sim_t, "L0", "death", f"DIED: {ph.status()['cause_text']}, after {ph.alive_s:.0f} s alive",
                         cause=ph.cause)
        self.cog.on_death()

    def _dead_window(self, t0, n_steps):
        """After death the brain gets no input and is not run (the panel goes dark); the body is a
        physical object that can still be pushed, crushed or dropped on."""
        win = WINDOW_MS * 1e-3
        self._hand_tick()
        self._others_launch()
        self.body.ext_loom_extra = self.others.flying()
        self.body.step(np.zeros(2), n_steps)
        self._others_finish()
        zero = np.zeros(self.brain.n, np.uint32)
        self.bridge.update(zero, WINDOW_MS)
        self.observer.update(zero, WINDOW_MS)
        self._anim_cues(np.zeros(2), dead=True)
        self._pain_tick(dead=True)
        with self.lock:
            self.sens = self.world.augment(self.body.sensory(), self.body)
            self.drive = np.zeros(2)
            self.sim_t += win
        self.cog.tick(win)
        return time.perf_counter() - t0

    def revive(self, fresh):
        """fresh: a new fly at the world's start, with no memory; else the same fly, where it lies."""
        self.pharma.down = False
        self.noci.reset()
        self.phys.reset()
        if fresh:
            self.pharma.clear()
            from . import worlds
            spec = worlds.get(self.world.name) or {}
            fly = spec.get("fly") or {}
            self.body.reset((float(fly.get("x", 0.0)), float(fly.get("y", 0.0))), float(fly.get("heading", 0.0)))
            self.world.reapply()
            self.lives += 1
        else:
            self.body.set_state("walk")
        self.brain.reset_state()
        self.runaway_ms = 0.0
        self.policy.reset()
        self.bridge.reset()
        self.cog.on_revive(fresh)

    # ------------------------------------------------------------------ cinema: scripted ordeals (shots/5x_cinema_*)
    def _fire_ring_start(self, j):
        """n fires on a circle round (cx, cy) (default: the fly), closing from r0 to r1 mm over `seconds` sim s
        after `delay`. Each fire is an ordinary heat source (world.py), so the warmth, the burns and a death by heat
        are the model's own; only the closing is scripted."""
        p = self.sens["pos"]
        cx, cy = float(j.get("cx", p[0])), float(j.get("cy", p[1]))
        n, r0 = int(np.clip(int(j.get("n", 12)), 3, 40)), float(j.get("r0", 16.0))
        ids = []
        for i in range(n):
            a = 2 * np.pi * i / n
            sid = self.world.add("heat", cx + r0 * np.cos(a), cy + r0 * np.sin(a), strength=float(j.get("strength", 0.8)))
            if sid:
                ids.append((sid, a))
        nw = int(j.get("walls", 0))                # a pen of wall segments just outside the ring: nowhere to run
        rw = r0 + float(j.get("wall_gap", 2.5))
        walls = []
        for i in range(nw):                       # the pen closes with the fire (walls keep their angle, shrink in length)
            a0, a1 = 2 * np.pi * i / nw, 2 * np.pi * (i + 1.08) / nw
            sid = self.world.add("wall", cx + rw * np.cos(a0), cy + rw * np.sin(a0), x2=cx + rw * np.cos(a1), y2=cy + rw * np.sin(a1))
            if sid:
                w = self.world.sources[sid]
                w["size"][2] = float(j.get("wall_h", 7.0))     # tall: a fly that takes off cannot clear it
                self.world._place_scenery(w)
                walls.append((sid, (a0 + a1) / 2, w["size"][0]))
        self.ring = {"walls": walls, "gap": rw - r0, "ids": ids, "cx": cx, "cy": cy, "r0": r0, "r1": float(j.get("r1", 8.0)), "t0": self.sim_t,
                     "s": max(0.1, float(j.get("seconds", 10.0))), "delay": float(j.get("delay", 0.0))}
        self.cog.log.add(self.sim_t, "world", "add", f"you set a ring of {len(ids)} fires round the fly, {r0:.0f} mm out")

    def _scene_tick(self):
        if getattr(self, "pin_others", False):    # the other flies are held too (they stay where they were set down)
            for o in self.others.flies.values():
                if not o.phys.dead and o.body.mode != "held":
                    o.body.set_state("held")
        rg = self.ring
        if rg:
            u = float(np.clip((self.sim_t - rg["t0"] - rg["delay"]) / rg["s"], 0.0, 1.0))
            r = rg["r0"] + (rg["r1"] - rg["r0"]) * (u * u * (3 - 2 * u))      # smoothstep: eases in and out
            if u > 0 and abs(r - rg.get("r", -1)) > 0.02:
                for sid, a in rg["ids"]:
                    self.world.move(sid, rg["cx"] + r * np.cos(a), rg["cy"] + r * np.sin(a))
                rw, rw0 = r + rg["gap"], rg["r0"] + rg["gap"]
                for sid, a, L0 in rg["walls"]:
                    w = self.world.sources.get(sid)
                    if w:
                        w["size"][0] = max(0.3, L0 * rw / rw0)
                        self.world.move(sid, rg["cx"] + rw * np.cos(a) * np.cos(np.pi / len(rg["walls"])),
                                        rg["cy"] + rw * np.sin(a) * np.cos(np.pi / len(rg["walls"])))
                rg["r"] = r
            if u >= 1.0:
                self.ring = None
        ar = self.amb_ramp
        if ar:
            u = float(np.clip((self.sim_t - ar["t0"]) / ar["s"], 0.0, 1.0))
            self.world.T0 = ar["T0"] + (ar["T1"] - ar["T0"]) * u
            if int(u * 40) != ar.get("k"):
                self.world.version += 1
                ar["k"] = int(u * 40)
            if u >= 1.0:
                self.amb_ramp = None

    def _crush_tick(self, h, p):
        """The slow press: down at v until it meets the fly's back, which pins it (a scripted pin: the press is not a
        MuJoCo body), then on at vp mm/s to the floor. Compression c (0 at the touch, 1 flat) drives the mechanical
        nociceptors (the same entry as the hand's hit) and takes health in proportion, so it dies near c = 0.87."""
        dt = WINDOW_MS * 1e-3
        if h["phase"] == "down":
            under = np.hypot(p[0] - h["x"], p[1] - h["y"]) < h["r"] + 0.4
            top = p[2] + 0.4
            if under and (h.get("rest") or h["z"] - h["v"] * dt <= top) and not self.phys.dead:
                h.update(phase="crush", zc=top + 0.05, z=top + 0.05, c=0.0)
                self.body.set_state("limp")
                self.body.ext_loom = []
                self.cog.log.add(self.sim_t, "L0", "crushed", "the press came down on it (whole body) and pins it", q=0.5)
                self.cog.log.add(self.sim_t, "L0", "body", "PINNED: a weight on its back, its legs cannot move")
                return
            h["z"] = max(0.35, h["z"] - h["v"] * dt)
            self.body.ext_loom = [((h["x"], h["y"], h["z"]), h["r"], (0.0, 0.0, -h["v"]))]
            if h["z"] <= 0.35:
                self.body.ext_loom = []
                h["phase"], h["left"] = "press", h["hold"]
        elif h["phase"] == "crush":
            # compression builds at vp mm/s of nominal travel; the press is drawn resting on its back (it sinks only
            # 0.3 mm on screen: the body mesh cannot deform, so a full-depth press would pass through it)
            c = float(np.clip(h["c"] + h["vp"] * dt / max(0.2, h["zc"] - 0.35), 0.0, 1.0))
            h["z"] = h["zc"] - 0.3 * c
            if self.phys.dead:                  # dead: the press lifts off slowly and hangs over the body (which rolls)
                h["phase"] = "lift"
            if not self.phys.dead:
                self.phys.crush += 1.15 * (c - h["c"])
                if c > 0.02:
                    self.noci.hit(min(1.0, 0.35 + c), "both", "whole body", self.sim_t)
                if h["c"] < 0.5 <= c:
                    self.cog.log.add(self.sim_t, "L0", "body", "CRUSHING: the weight is pressing its body flat")
            h["c"] = c
            if c >= 1.0:
                h["phase"], h["left"] = "press", h["hold"]
        elif h["phase"] == "lift":
            h["z"] = min(h["zc"] + 3.0, h["z"] + 8.0 * dt)   # clear of the rolling body
        elif h["phase"] == "press":
            h["left"] -= dt
            if h["left"] <= 0:
                h["phase"] = "up"
        else:
            h["z"] += 20.0 * dt
            if h["z"] > self.HAND_TOP:
                self.hand = None

    HAND_R, HAND_TOP = 3.2, 26.0       # fingertip radius mm, start height mm; it comes down at 60-160 mm/s (all ours)

    def _hand_tick(self):
        """The hand: a fingertip column that comes down on a spot (60-160 mm/s), presses, and lifts. It looms
        (body.ext_loom -> LPLC2 / LC4 -> giant fibre, the real escape pathway: Ache et al. 2019, Curr Biol,
        doi:10.1016/j.cub.2019.01.079; von Reyn et al. 2014, Nat Neurosci, doi:10.1038/nn.3741), and a fly
        under it when it arrives, in the air or on the floor, is crushed."""
        h = self.hand
        if h is None:
            return
        if h.get("crush"):
            return self._crush_tick(h, self.body.d.xpos[self.body.thorax])
        dt = WINDOW_MS * 1e-3
        if h["phase"] == "down":
            h["z"] = max(0.3, h["z"] - h["v"] * dt)
            self.body.ext_loom = [((h["x"], h["y"], h["z"]), h["r"], (0.0, 0.0, -h["v"]))]
            if h["z"] <= 0.3:
                h["phase"], h["left"] = "press", 0.35
        elif h["phase"] == "press":
            self.body.ext_loom = []
            h["left"] -= dt
            if h["left"] <= 0:
                h["phase"] = "up"
        else:
            h["z"] += 60.0 * dt
            if h["z"] > self.HAND_TOP:
                self.hand = None
                return
        p = self.body.d.xpos[self.body.thorax]
        if h["phase"] != "up" and not h.get("hit") and np.hypot(p[0] - h["x"], p[1] - h["y"]) < h["r"] + 0.4 \
                and p[2] + 0.4 >= h["z"]:
            h["hit"] = True
            if not self.phys.dead:
                # graded (nociception.hand_hit): a glancing blow hurts without killing, a centred one is lethal
                q, side, part = hand_hit(p[:2], self.sens["heading"][:2], (h["x"], h["y"]), h["r"])
                self.phys.crush = hand_damage(q)
                self.noci.hit(q, side, part, self.sim_t)
                how = "full on it" if q > 0.85 else "on part of it" if q > 0.4 else "a glancing blow"
                self.cog.log.add(self.sim_t, "L0", "crushed", f"the hand came down {how}" + (" in the air" if p[2] > 2 else "")
                                 + f" ({part})", q=round(q, 2))
        for o in self.others.flies.values():
            hk = h.setdefault("hit_o", set())
            if h["phase"] != "up" and o.k not in hk and np.hypot(o.body.x - h["x"], o.body.y - h["y"]) < h["r"] + 0.4 \
                    and o.body.z + 1.3 >= h["z"]:
                hk.add(o.k)
                if not o.phys.dead:
                    o.phys.crush = 2.0           # brain flies have no nociception, so their swats stay all-or-nothing
                    self.cog.log.add(self.sim_t, "O", "crushed", f"the hand came down on {o.name}", who=o.k, x=o.body.x, y=o.body.y)
        if h["phase"] == "press" and not h.get("hit") and not h.get("missed"):
            h["missed"] = True
            if not self.phys.dead:
                self.cog.log.add(self.sim_t, "L0", "escaped", "the hand came down and missed it "
                                 f"({np.hypot(p[0] - h['x'], p[1] - h['y']):.1f} mm away)")

    def loop(self):
        last = time.perf_counter()
        while self.running:
            self.drain_commands()  # also while paused, so "unpause" gets through
            dc = self.director.tick(self)        # auto-director: an event every 20-40 sim s when it is on
            if dc is not None:
                self.cmds.put(dc)
            hold = self.cog.hold_for_mind and self.cog.l2_live.get("on")    # "wait for the mind"
            if (self.paused and self.step_left <= 0) or hold:
                time.sleep(0.02)
                last = time.perf_counter()
                continue
            if self.paused:
                self.step_left -= 1
            w = self.step_window()
            # never faster than the chosen speed (1 = real time; 0 = as fast as it goes)
            target = WINDOW_MS * 1e-3 / self.speed if self.speed > 0 else 0.0
            if w < target:
                time.sleep(target - w)
            now = time.perf_counter()
            self.wall_total += 0.05 * ((now - last) - self.wall_total)
            last = now

    def cog_snapshot(self):
        """Thread-safe copy of what the cognition layers read."""
        with self.lock:
            sens = dict(self.sens)
            t = self.sim_t
            world = self.world.status()
            drivers = self.cog._drivers
            T0 = self.world._T_amb()
            light = self.world.light() if self.world.day_len > 0 else None      # None: no day cycle, the mind is not told
            flies = [{"id": o.k, "x": o.body.x, "y": o.body.y, "mode": o.body.mode, "dead": o.phys.dead, "v": abs(o.body.v)}
                     for o in list(self.others.flies.values())]
            hand = None if self.hand is None else (float(self.hand["x"]), float(self.hand["y"]))
        obs = self.observer.observation(sens, {"t": t, "runaway": t - self.last_runaway_t < 3.0})
        return {"sens": sens, "t": t, "world": world, "obs": obs, "drivers": drivers, "T0": T0, "light": light,
                "flies": flies, "pain_mode": self.noci.mind_mode, "hand": hand}

    def _llm_observation(self):
        name, drv = self.llm.driven()
        own = [k for k, i in self.observer.idx.items() if len(drv) and np.isin(i, drv).mean() > 0.5]
        with self.lock:
            return self.observer.observation(self.sens, {"t": self.sim_t,
                                                         "runaway": self.sim_t - self.last_runaway_t < 3.0,
                                                         "active_intervention": name, "self_driven": own})

    def _status(self, world=None):
        win = WINDOW_MS * 1e-3
        s = dict(self.sens)
        s.update({
            "sim_t": self.sim_t,
            "rtf": win / self.wall_total if self.wall_total else 0.0,
            "body_wall_per_sim": self.wall_body / win,
            "brain_wall_per_sim": self.wall_brain / win,
            "brain_backend": self.brain.backend,
            "physics": f"{self.body.physics} dt={self.body.dt * 1e3:g} ms",
            "dn_rates": {k: round(v, 1) for k, v in self.bridge.rate.items()},
            "drive": [round(float(x), 3) for x in self.drive],
            "policy": self.policy.label, "autonomy": self.policy.enabled,
            "sensory_inputs": {k: v[1] for k, v in self.sensory_layer.out.items()},
            "user_inputs": {k: v[1] for k, v in self.user_inputs.items()},
            "silenced": list(self.user_silenced),
            "stimuli": [s[0] for s in self.stimuli],
            "paused": self.paused, "speed": self.speed, "lives": self.lives,
            "phys": self.phys.status(self.sens),
            "hand": ({k: round(float(v), 2) for k, v in self.hand.items() if k in ("x", "y", "z", "r")}
                     if self.hand else None),
            "events": self.events[-8:],
            "active_frac": round(self.observer.active_frac, 5),
            "n_runaway_resets": self.n_runaway, "n_takeoff": self.n_takeoff,
            "dng02_active": self.bridge.dng02_active,
            "neural_obs": {k: round(v, 1) for k, v in self.observer.rate.items()},
            "world": world if world is not None else self.world.status(),
            "world_env": self.world.env(),
            "cog": self.cog.status(),
            "others": self.others.status(),
            "director": self.director.status(),
            "drugs": {**self.pharma.status(), "aware": self.drug_aware, "down": self._drug_down,
                      "hist": list(self.pharma.hist)[-120:] if self.pharma.hist else []},
            "pain": self.noci.status(),
            "llm": {"enabled": self.llm.enabled, "goal": self.llm.goal, "model": self.llm.model,
                    "url": self.llm.url, "period": self.llm.period,
                    "active": self.llm.active[0] if self.llm.active else None, **self.llm.last},
        })
        return s


# ---------------------------------------------------------------------- server
def make_app(sim: Sim):
    app = web.Application(client_max_size=64 * 2**20)  # snapshots are several MB of PNG

    async def _no_cache(request, response):
        # every page file revalidates: the page's modules (llm_column.js, the panels) and CSS change with app.js, and a
        # browser that kept an old one against the new app.js draws a mix of the two
        if request.path.endswith((".js", ".css", ".html")) or request.path == "/":
            response.headers["Cache-Control"] = "no-cache"
    app.on_response_prepare.append(_no_cache)
    meta, blob = sim.body.scene()
    if sim.others.cal is not None:
        meta["others"] = sim.others.cal.frames()
    t = sim.table
    nmeta = {"n": int(sim.brain.n), "classes": [{"name": c, "color": col} for c, col in neurons.CLASSES],
             "groups": {k: v.tolist() for k, v in sim.bridge.groups.items()},
             "labels": _labels(t)}
    nmeta["nt"] = ["?"] + list(sim.drug_targets.nt)          # transmitter per neuron (brain view tint), byte after cls
    nmeta["pain"] = {"entry": sim.noci.all.tolist(), "ppl1": sim.noci.ppl1.tolist()}     # brain view markers
    nmeta["feel"] = _feel_groups(t)
    # the cells a drug acts on, labelled in the brain view while that drug is in the body
    dg = {}
    for key, label, idx in (("caffeine", "Dopamine cells", sim.drug_targets.resolve("type:PAM")),
                            ("ethanol", "Feeding neurons", np.concatenate([t.find("sugar_GRN"), t.find("MN9")]))):
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx):
            dg[key] = {"label": label, "idx": idx.tolist(), "pos": t.pos[idx].mean(0).tolist()}
    nmeta["drug_groups"] = dg
    nblob = t.pos.astype(np.float32).tobytes() + t.cls.astype(np.uint8).tobytes() + sim.drug_targets.nt_code.tobytes()
    clients: set = set()
    sim.viewers = 0

    async def index(_):
        return web.FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

    async def app_js(_):
        # no-cache: a browser that kept an old app.js against a new index.html shows a half-built page
        return web.FileResponse(WEB / "app.js", headers={"Cache-Control": "no-cache"})

    async def scene_json(_):
        return web.json_response(meta)

    async def scene_bin(_):
        return web.Response(body=blob, content_type="application/octet-stream")

    async def neurons_json(_):
        return web.json_response(nmeta)

    async def neurons_bin(_):
        return web.Response(body=nblob, content_type="application/octet-stream")

    async def api_state(_):
        return web.json_response(sim.read_state())

    async def api_drugs(_):
        from . import pharma
        return web.json_response(pharma.catalog(), headers={"Cache-Control": "no-cache"})

    async def api_worlds(_):
        from . import worlds
        return web.json_response(worlds.catalog(), headers={"Cache-Control": "no-cache"})

    async def api_input(req):
        j = await req.json()
        sim.set_input(j["target"], j.get("mode", "activate"), float(j.get("value", 150)), bool(j.get("on", True)))
        return web.json_response({"ok": True})

    async def api_cmd(req):
        # any websocket command over REST, e.g. {"cmd": "reset", "x": 20, "heading": 0}
        _command(sim, await req.json())
        return web.json_response({"ok": True})

    async def api_stim(req):
        j = await req.json()
        sim.inject_stimulus(j["kind"], float(j.get("strength", 1)), float(j.get("ms", 300)))
        return web.json_response({"ok": True})

    async def api_llm(req):
        # {"on": true, "goal": "...", "url": "http://127.0.0.1:8081/v1", "model": "...", "period": 1.5}
        _command(sim, {"cmd": "llm", **(await req.json())})
        return web.json_response({"ok": True, "intents": {k: v.what for k, v in CATALOG.items()}})

    async def api_llm_obs(_):
        return web.Response(text=observation_text(sim._llm_observation()))

    async def snapshot(req):
        # front end posts a PNG of its canvases (window.snap()); saved for docs/screenshots
        name = "".join(ch for ch in req.query.get("name", "snap") if ch.isalnum() or ch in "-_")[:60]
        out = WEB.parent / "docs" / "screenshots"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.png").write_bytes(await req.read())
        return web.json_response({"saved": str(out / f"{name}.png")})

    async def rec_start(req):
        j = await req.json() if req.can_read_body else {}
        c = sim.cog
        name = c.rec.start({"l1": {"mode": c.l1_mode, "model": c.l1_model}, "l2": {"mode": c.l2_mode, "model": c.l2_model},
                            "goal": c.goal, "world": sim.world.status(), "client": j}, sub=j.get("out"))
        c.log.sink = c.rec
        c.log.add(sim.sim_t, "user", "record", f"recording started ({name})")
        return web.json_response({"name": name})

    async def rec_chunk(req):
        ok = sim.cog.rec.video_chunk(req.query.get("name", ""), await req.read())
        return web.json_response({"ok": ok})

    async def rec_frame(req):
        try:
            i = int(req.query.get("i", "-1"))
        except ValueError:
            i = -1
        ok = sim.cog.rec.frame(req.query.get("name", ""), i, await req.read(), req.query.get("in", ""))
        return web.json_response({"ok": ok})

    async def rec_stop(_):
        c = sim.cog
        c.log.add(sim.sim_t, "user", "record", "recording stopped")
        c.log.sink = None
        d = c.rec.stop()
        sim.rec_done = str(d) if d else None
        return web.json_response({"dir": str(d) if d else None})

    async def replay_data(req):
        # ghost replays: trial paths prepared by experiments/ghost_replay.py into results/replay/<name>.json
        name = req.query.get("name", "")
        p = WEB.parent / "results" / "replay" / f"{name}.json"
        if not name.replace("_", "").replace("-", "").replace(".", "").isalnum() or not p.is_file():
            return web.json_response({"error": "no such replay"}, status=404)
        return web.Response(body=p.read_bytes(), content_type="application/json")

    async def rec_info(_):
        # for the shot driver (experiments/auto_shots.py): is a take running, which one finished last, is a page connected
        r = sim.cog.rec
        return web.json_response({"on": r.on, "name": r.last, "done": getattr(sim, "rec_done", None),
                                  "viewers": len(clients), "client": getattr(sim, "client_info", None)})

    async def api_ui(req):
        # page-side commands (cinematic camera, recorder), forwarded to every open page:
        # {"cine": {"on": true, "mode": "follow"}} | {"rec": "start", "out": "auto/x"} | {"rec": "stop"}
        j = await req.json()
        msg = json.dumps({"ui": j})
        n = 0
        for ws in list(clients):
            try:
                await ws.send_str(msg)
                n += 1
            except Exception:
                clients.discard(ws)
        return web.json_response({"ok": n > 0, "pages": n})

    async def ws_handler(req):
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(req)
        clients.add(ws)
        sim.viewers = len(clients)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    _command(sim, json.loads(msg.data))
        finally:
            clients.discard(ws)
            sim.viewers = len(clients)
        return ws

    async def broadcaster(app_):
        async def run():
            k, mind_last = 0, (None, None, None)
            while True:
                await asyncio.sleep(1 / 30)
                if not clients:
                    continue
                with sim.lock:
                    poses = sim.body.poses()
                    acc = sim.spike_acc
                    idx = np.flatnonzero(acc).astype(np.uint32)
                    cnt = np.minimum(acc[idx], 255).astype(np.uint8)
                    acc[:] = 0
                    t_now = sim.sim_t
                    oth = sim.others.poses()
                    world = sim.world.status() if k % 3 == 0 else None   # reads MuJoCo state: keep it under the lock
                try:
                    status = sim._status(world) if k % 3 == 0 else None
                except Exception as ex:          # one bad status must not stop the stream for every client
                    print(f"status failed: {ex!r}", flush=True)
                    status = None
                msgs = [np.array([1], np.uint32).tobytes() + np.array([t_now], np.float32).tobytes() + poses.tobytes(),
                        np.array([2, len(idx)], np.uint32).tobytes() + idx.tobytes() + cnt.tobytes(),
                        np.array([3, len(oth)], np.uint32).tobytes() + oth.tobytes()]
                if status is not None:
                    msgs.append(json.dumps(status))
                # the AI's brain at the frame rate, not the 10 Hz status: the newest activity frame and the doses in
                # force, sent only when one of them changed (so the panel lights within a frame of the token)
                cg = sim.cog
                af, ld, tf = getattr(cg, "act_frame", None), getattr(cg, "live_doses", None), getattr(cg, "twin_frame", None)
                if (af, ld, tf) != mind_last and (af is not None or ld is not None):
                    mind_last = (af, ld, tf)
                    msgs.append(json.dumps({"mind": {"act": af, "doses": ld, "twin": tf, "w": round(time.time(), 3)}}))
                for ws in list(clients):
                    try:
                        for m in msgs:
                            if isinstance(m, str):
                                await ws.send_str(m)
                            else:
                                await ws.send_bytes(m)
                    except Exception:
                        clients.discard(ws)
                        sim.viewers = len(clients)
                k += 1
        task = asyncio.create_task(run())
        yield
        task.cancel()

    app.router.add_get("/", index)
    app.router.add_get("/app.js", app_js)
    app.router.add_get("/scene.json", scene_json)
    app.router.add_get("/scene.bin", scene_bin)
    app.router.add_get("/neurons.json", neurons_json)
    app.router.add_get("/neurons.bin", neurons_bin)
    app.router.add_get("/api/state", api_state)
    app.router.add_get("/api/worlds", api_worlds)
    app.router.add_get("/api/drugs", api_drugs)
    app.router.add_post("/api/input", api_input)
    app.router.add_post("/api/stimulus", api_stim)
    app.router.add_post("/api/cmd", api_cmd)
    app.router.add_post("/api/snapshot", snapshot)
    app.router.add_post("/api/llm", api_llm)
    app.router.add_get("/api/llm/observation", api_llm_obs)
    app.router.add_post("/api/rec/start", rec_start)
    app.router.add_post("/api/rec/chunk", rec_chunk)
    app.router.add_post("/api/rec/frame", rec_frame)
    app.router.add_post("/api/rec/stop", rec_stop)
    app.router.add_get("/api/rec", rec_info)
    app.router.add_get("/api/replay", replay_data)
    app.router.add_post("/api/ui", api_ui)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/", WEB)
    app.cleanup_ctx.append(broadcaster)
    return app


def _feel_groups(t):
    """The neurons that feed each feeling (danger, pain, hunger, heat), for the brain view's feeling colours. Same
    groups as video_tools/ai_colour.py; heat is the antennal hot cells. Missing types are skipped."""
    want = {"threat": ["LPLC2", "LC4", "DNp01"], "pain": ["AN_noci", "PPL1"], "hunger": ["sugar_GRN", "MN9"],
            "heat": ["TRN_hot_left", "TRN_hot_right"]}
    out = {}
    for k, names in want.items():
        idx = []
        for nm in names:
            try:
                f = t.find(nm)
                if not len(f) and nm in ("LPLC2", "LC4", "DNp01"):
                    f = np.concatenate([t.type_side(nm, s) for s in ("left", "right")])
            except Exception:
                f = []
            idx += [int(i) for i in f]
        idx = sorted(set(idx))
        if idx:
            out[k] = {"idx": idx, "pos": t.pos[idx].mean(0).tolist()}
    return out


def _labels(t):
    out = []
    for name in ["MDN", "DNp09", "DNa02", "DNp01", "LPLC2"]:
        for s in ("left", "right"):
            idx = t.type_side(name, s)
            if len(idx):
                out.append({"name": f"{'GF' if name == 'DNp01' else name} {s[0].upper()}",
                            "pos": t.pos[idx].mean(0).tolist(), "idx": idx.tolist()})
    idx = t.find("sugar_GRN")
    out.append({"name": "sugar GRNs", "pos": t.pos[idx].mean(0).tolist(), "idx": idx.tolist()})
    for name, g in (("PPL1", "PPL1"), ("pain entry (stand-in)", "AN_noci")):
        idx = t.find(g)
        if len(idx):
            out.append({"name": name, "pos": t.pos[idx].mean(0).tolist(), "idx": idx.tolist()})
    return out


def _command(sim: Sim, j: dict):
    c = j.get("cmd")
    if c == "client_info":                 # the page reports its GL renderer and size once it draws (shot driver readiness)
        sim.client_info = {**(getattr(sim, "client_info", None) or {}),
                           **{k: j.get(k) for k in ("gl", "w", "h", "dpr", "t", "fps", "cine", "tabrec", "tabrec_t0",
                                                    "tabrec_error") if k in j}}
        return
    if c == "director":                    # auto-director (director.py): {"on": true, "every": [20, 40]}
        sim.director.configure(j.get("on"), j.get("every"))
        sim.events.append("auto-director " + ("on" if sim.director.on else "off"))
        return
    if c == "input":
        sim.set_input(j["target"], j.get("mode", "activate"), float(j.get("value", 150)), bool(j.get("on", True)))
    elif c == "stimulus":
        sim.inject_stimulus(j["kind"], float(j.get("strength", 1)), float(j.get("ms", 300)))
    elif c == "drop":
        sim.cmds.put(("drop", j.get("x"), j.get("y"), float(j.get("z", 8.0))))
    elif c == "push":
        sim.cmds.put(("push", float(j.get("fx", 0)), float(j.get("fy", 0)), float(j.get("fz", 0)),
                      float(j.get("ms", 20))))
    elif c == "reset":
        sim.cmds.put(("reset", (float(j.get("x", 0)), float(j.get("y", 0))), float(j.get("heading", 0))))
    elif c in ("clear", "right"):
        sim.cmds.put((c,))
    elif c in ("pause", "autonomy", "sensory"):
        sim.cmds.put((c, j.get("on", True)))
    elif c == "llm_force":
        sim.cmds.put(("llm_force", j["intent"], float(j.get("strength", 1.0)), float(j.get("duration_s", 1.0))))
    elif c == "llm":
        sim.cmds.put(("llm", {k: j.get(k) for k in ("on", "goal", "url", "model", "period")}))
    elif c == "world_add":
        kw = {k: float(j[k]) for k in ("strength", "yaw", "x2", "y2", "r") if j.get(k) is not None}
        if isinstance(j.get("size"), list):
            kw["size"] = [float(v) for v in j["size"][:3]]
        if j.get("drug") in DRUGS:
            kw["drug"] = j["drug"]
        sim.cmds.put(("world_add", str(j["kind"]), float(j["x"]), float(j["y"]), kw))
    elif c == "world_move":
        sim.cmds.put(("world_move", int(j["id"]), float(j["x"]), float(j["y"]), bool(j.get("final", False))))
    elif c == "world_set":
        kw = {k: float(j[k]) for k in ("strength", "r", "yaw", "scale") if j.get(k) is not None}
        sim.cmds.put(("world_set", int(j["id"]), kw))
    elif c == "world_remove":
        sim.cmds.put(("world_remove", int(j["id"])))
    elif c in ("world_clear", "world_default", "world_undo", "world_redo", "talk_release"):
        sim.cmds.put((c,))
    elif c == "world_load":
        sim.cmds.put(("world_load", j["spec"] if isinstance(j.get("spec"), dict) else str(j.get("name", "classic"))))
    elif c == "world_save":
        from . import worlds
        with sim.lock:
            spec = sim.world.snapshot(ids=False)
            p, h = sim.sens["pos"], sim.sens["heading"]
        spec["fly"] = {"x": round(float(p[0]), 2), "y": round(float(p[1]), 2),
                       "heading": round(float(np.arctan2(h[1], h[0])), 3)}
        name = worlds.save_user_world(j.get("name") or "my_world", spec)
        sim.events.append(f"world saved as {name}")
    elif c == "fly_place":
        sim.cmds.put(("fly_place", float(j["x"]), float(j["y"]), float(j.get("heading", 0.0))))
    elif c == "puff":
        sim.cmds.put(("puff", float(j["x"]), float(j["y"])))
    elif c == "other_add":
        h = j.get("heading")
        sim.cmds.put(("other_add", float(j["x"]), float(j["y"]), float(h) if h is not None else None))
    elif c in ("other_remove", "other_revive"):
        sim.cmds.put((c, int(j["id"])))
    elif c == "other_move":
        h = j.get("heading")
        sim.cmds.put(("other_move", int(j["id"]), float(j["x"]), float(j["y"]), float(h) if h is not None else None))
    elif c == "others_clear":
        sim.cmds.put(("others_clear",))
    elif c == "hand":
        v = j.get("v")
        kw = {k: j[k] for k in ("r", "top", "crush", "vp", "hold", "rest", "target") if j.get(k) is not None}
        lo = 0.5 if kw.get("crush") else 20.0
        sim.cmds.put(("hand", float(j["x"]), float(j["y"]), float(np.clip(float(v), lo, 400.0)) if v else None, kw))
    elif c == "pin":                       # cinema: {"on": true} holds the fly down (no visible restraint)
        sim.cmds.put(("pin", bool(j.get("on", True)), bool(j.get("others", False))))
    elif c == "fire_ring":                 # cinema: {"n", "r0", "r1", "seconds", "delay", "strength", "cx", "cy"}
        sim.cmds.put(("fire_ring", {k: j[k] for k in ("n", "r0", "r1", "seconds", "delay", "strength", "cx", "cy", "walls", "wall_gap", "wall_h") if k in j}))
    elif c == "ambient":                   # cinema: the floor heats (or cools) to temp C over seconds
        sim.cmds.put(("ambient", float(np.clip(float(j.get("temp", 24.0)), -10.0, 60.0)), float(j.get("seconds", 0.0))))
    elif c == "reserves":
        sim.cmds.put(("reserves", j.get("energy"), j.get("water")))
    elif c in ("speed", "life_rate"):
        sim.cmds.put((c, float(j.get("value", 1.0))))
    elif c == "day":
        g = lambda k: float(j[k]) if j.get(k) is not None else None
        sim.cmds.put(("day", g("len"), g("hour")))
    elif c == "step":
        sim.cmds.put(("step", int(j.get("n", 10))))
    elif c == "revive":
        sim.cmds.put(("revive", bool(j.get("fresh", False))))
    elif c == "drug":
        who = j.get("who")
        route = j.get("route", "inject")
        if route in ("inject", "feed", "vapour"):
            sim.cmds.put(("drug", str(j.get("drug", "")), route, float(np.clip(float(j.get("amount", 1.0)), 0.0, 4.0)),
                          int(who) if who is not None else None))
    elif c == "drug_clear":
        who = j.get("who")
        sim.cmds.put(("drug_clear", int(who) if who is not None else None))
    elif c == "drug_aware":
        sim.cmds.put(("drug_aware", str(j.get("value", "felt"))))
    elif c == "pain_opto":
        sim.cmds.put(("pain_opto", float(np.clip(float(j.get("level", 0.6)), 0.0, 1.0)), str(j.get("side", "both")),
                      float(np.clip(float(j.get("seconds", 2.0)), 0.0, 30.0))))
    elif c == "pain_silence":
        sim.cmds.put(("pain_silence", str(j.get("value", ""))))
    elif c == "pain_mind":
        sim.cmds.put(("pain_mind", str(j.get("value", "text"))))
    elif c == "pain_steer":
        sim.cmds.put(("pain_steer", bool(j.get("value", False))))
    elif c == "talk":
        sim.cmds.put(("talk", str(j.get("text", ""))[:240], "order" if j.get("mode") == "order" else "say"))
    elif c == "cog":
        sim.cmds.put(("cog", {k: v for k, v in j.items() if k != "cmd"}))


def open_window(url):
    edge = shutil.which("msedge") or next((p for p in [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"] if os.path.exists(p)), None)
    if edge:
        subprocess.Popen([edge, f"--app={url}", "--window-size=1680,960",
                          f"--user-data-dir={Path(os.environ.get('TEMP', '.')) / 'flysim_edge'}"])
    else:
        import webbrowser
        webbrowser.open(url)


def free_port(port, tries=20):
    """The asked port, or the next free one above it (another program on 8765 used to kill the app after the 30 s
    brain build, with the error lost in a closing window)."""
    import socket
    for p in range(port, port + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                print(f"port {p} is busy", flush=True)
                continue
        return p
    raise SystemExit(f"no free port in {port}..{port + tries - 1}")


def keep_up_priority():
    """Windows: the app's console sits behind the page while it runs or records, and Windows may run a background
    process at low QoS (EcoQoS) or give its threads a small share when the CPU is busy (a screen recorder, another
    job): measured 2026-10-24, the sim thread got 0.09 of a core and ran 0.08x real time while a fresh process on the
    same loaded machine stepped the same physics at full speed. Opt out of power throttling and run above normal
    priority. Scheduling only: the model is unchanged. FLYSIM_PRIORITY=0 leaves both alone."""
    if os.name != "nt" or os.environ.get("FLYSIM_PRIORITY", "1") == "0":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class _PPTS(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        h = k32.GetCurrentProcess()
        st = _PPTS(1, 0x1 | 0x4, 0)      # ProcessPowerThrottling: EXECUTION_SPEED + IGNORE_TIMER_RESOLUTION, both off
        ok = k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st))
        ok2 = k32.SetPriorityClass(h, 0x00008000)              # ABOVE_NORMAL_PRIORITY_CLASS
        print(f"priority above normal: {bool(ok2)}, power throttling off: {bool(ok)}", flush=True)
    except Exception as e:               # never stop the app over this
        print(f"priority / QoS not set: {e!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="warp", choices=["warp", "numba", "torch"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--physics", default="fast", choices=["fast", "flygym"])
    ap.add_argument("--llm", action="store_true", help="start with the LLM layer on")
    ap.add_argument("--llm-url", default=None, help="OpenAI-compatible base URL (default $FLYSIM_LLM_URL or http://127.0.0.1:8081/v1)")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--goal", default=None)
    ap.add_argument("--l1", default=None, choices=["off", "bind", "rule", "llm"], help="three-level mind: motor layer mode")
    ap.add_argument("--l2", default=None, choices=["off", "steer", "narrate", "blind"], help="three-level mind: conscious layer mode")
    ap.add_argument("--l2-model", default=None, help="OpenRouter model id for L2")
    ap.add_argument("--mind", default=None, choices=["local", "openrouter"],
                    help="who writes the thoughts: local = pain-steered Qwen3-4B on pain_server (default, $FLYSIM_MIND), openrouter = --l2-model")
    ap.add_argument("--l2-style", default=None, choices=["mind", "classic"],
                    help="mind: event-driven, streamed (default); classic: JSON every 4 s")
    ap.add_argument("--memory", default=None, choices=["off", "shadow", "contend"],
                    help="memory system (memory.py): contend (default) bids for the body, shadow only watches")
    ap.add_argument("--world", default=None, help="start in a prebuilt world (worlds.py), e.g. labyrinth")
    a = ap.parse_args()
    a.port = free_port(a.port)
    print("building body + brain ...", flush=True)
    try:
        sim = Sim(a.backend, a.device, a.seed, a.physics)
    except Exception as e:  # no CUDA -> CPU fallback, said out loud
        if a.backend != "warp":
            raise
        print(f"warp backend failed ({e}); falling back to numba CPU", flush=True)
        sim = Sim("numba", None, a.seed, a.physics)
    print(f"ready in {sim.build_s:.1f}s: {sim.brain.n} neurons, {sim.brain.n_syn} connections, "
          f"backend {sim.brain.backend}", flush=True)
    if a.llm or a.llm_url or a.goal:
        sim.cmds.put(("llm", {"on": a.llm, "goal": a.goal, "url": a.llm_url, "model": a.llm_model, "period": None}))
    cog = {k: v for k, v in (("mind_provider", a.mind), ("l1_mode", a.l1), ("l2_mode", a.l2), ("l2_model", a.l2_model), ("l2_style", a.l2_style),
                             ("mem_mode", a.memory)) if v}
    if cog:
        sim.cmds.put(("cog", cog))
    if a.world:
        sim.cmds.put(("world_load", a.world))
    keep_up_priority()
    th = threading.Thread(target=sim.loop, daemon=True)
    th.start()
    url = f"http://127.0.0.1:{a.port}/"
    if not a.no_browser:
        threading.Timer(1.0, open_window, args=(url,)).start()
    print(f"serving {url}", flush=True)
    web.run_app(make_app(sim), host="127.0.0.1", port=a.port, print=None)
    sim.running = False


if __name__ == "__main__":
    main()
