"""Cosmetic pose layer: motion for the fly's parts that have no joints in the physics build.

The body is built with FlyBodyJointPreset.LEGS_ONLY (body.py), so the head, antennae, proboscis (rostrum,
haustellum, labella), abdomen, wings and halteres are welded to their parents. Giving them joints and actuators
would change the compiled model, the walking calibration and the physics cost. Instead this layer rotates those
bodies at their own frame origins (where FlyBody places their joints) in the pose copy that is streamed to the
front end (Body.poses()). The physics never sees it: nothing here touches contacts, mass or the brain.

Each channel is a joint of the FlyBody model with its range from flygym's joints.yaml (Vaxenburg et al. 2025,
Nature, FlyBody). The targets come from read-outs the sim already has (see Sim._anim_cues in app.py):

  proboscis   MN9 rate (the rostrum protractor motor neuron; Gordon & Scott 2009, Neuron,
              doi:10.1016/j.neuron.2008.12.033; McKellar et al. 2020, eLife, doi:10.7554/eLife.54978). Retracted
              and extended poses are ours, by eye (results/anim_scout_2026-10-24/f06, f11).
  head        yaw toward the turning side (steering-DN asymmetry) and toward a looming object; pitch down while
              the proboscis is out; roll toward the groomed antenna. Directions ours, by eye; ranges FlyBody's.
  antennae    pressed down and out on the groomed side during an antennal grooming bout; ours, by eye.
  abdomen     a slow small pitch oscillation at rest (ours, by eye: no adult rate found); a ventral curl in a
              seizure and in death (ours, by eye).
  wings       spread / raised for scripted displays; held folded in flight because the front end flaps them there.

All time constants and amplitudes are ours.
"""
from __future__ import annotations

import mujoco as mj
import numpy as np

# joint targets (rad), FlyBody axis convention in each body's own frame: pitch about x, roll about y, yaw about z;
# wings (WingFlyBody): pitch about y, roll about x, yaw about z
PROBOSCIS_IN = {"rostrum": 0.18, "haustellum": 0.7}       # tucked under the head (ours, by eye; f06)
PROBOSCIS_OUT = {"rostrum": -1.1, "haustellum": -0.5}     # reaching down in front of the forelegs (ours; f11)
LABELLA_OPEN = 0.9
WING_FOLD = (1.5, 0.7, -1.0)                              # flygym's neutral wing angles (yaw, roll, pitch)
RANGES = {"head_yaw": (-0.2, 0.2), "head_pitch": (-0.5, 0.3), "head_roll": (-0.6, 0.6),
          "ant_pitch": (-0.2, 0.5), "ant_yaw": (-0.4, 0.8), "abd_pitch": (-0.15, 0.1)}
TAU = {"pe": 0.12, "lab": 0.08, "head": 0.08, "ant": 0.06, "abd": 0.25, "wing": 0.05}
MN9_ON, MN9_FULL = 10.0, 50.0       # Hz: the feeding gate used by physiology.py, and full extension (ours)


def _qmul(a, b):
    o = np.zeros(4)
    mj.mju_mulQuat(o, a, b)
    return o


def _qrot(q, v):
    o = np.zeros(3)
    mj.mju_rotVecQuat(o, v, q)
    return o


def _qaxis(axis, ang):
    return np.array([np.cos(ang / 2), *(np.sin(ang / 2) * np.asarray(axis, float))])


def _euler(pitch=0.0, roll=0.0, yaw=0.0):
    q = np.array([1.0, 0, 0, 0])
    for ax, a in (((0, 0, 1), yaw), ((0, 1, 0), roll), ((1, 0, 0), pitch)):
        if a:
            q = _qmul(q, _qaxis(ax, a))
    return q


def _wing_q(yaw, roll, pitch):
    return _qmul(_qmul(_qaxis((0, 0, 1), yaw), _qaxis((1, 0, 0), roll)), _qaxis((0, 1, 0), pitch))


def _inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


class FaceAnim:
    """State and targets of the cosmetic channels; apply() writes them into a pose array."""

    def __init__(self, body):
        m = body.m
        self.body = body
        ids = [int(i) for i in body.pose_body_ids()]
        self.row = {i: k for k, i in enumerate(ids)}
        name = {i: mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i).split("/")[-1] for i in ids}
        self.id_of = {v: k for k, v in name.items()}
        animated = ["c_head", "c_rostrum", "c_haustellum", "l_labrum", "r_labrum", "l_antenna", "r_antenna",
                    *(f"c_abdomen{k}" for k in range(1, 8)), "l_wing", "r_wing"]
        hit = {self.id_of[n] for n in animated if n in self.id_of}
        order = []                                       # the animated bodies and every descendant, tree order
        for i in ids:
            if i in hit or int(m.body_parentid[i]) in set(order):
                if int(m.body_jntnum[i]) == 0:           # only jointless bodies (none of the legs hang off these)
                    order.append(i)
        self.order = order
        self.parent = {i: int(m.body_parentid[i]) for i in order}
        self.lpos = {i: m.body_pos[i].astype(float).copy() for i in order}
        self.lquat = {i: m.body_quat[i].astype(float).copy() for i in order}
        # wings: the compiled quat already holds the folded angles; keep the frame they are measured from
        self.wing_base = {s: _qmul(self.lquat[self.id_of[f"{s}_wing"]], _inv(_wing_q(*WING_FOLD))) for s in "lr"}
        self.reset()

    def reset(self):
        self.s = {"pe": 0.0, "lab": 0.0, "head": np.zeros(3), "ant": np.zeros((2, 2)), "abd": 0.0,
                  "wing": np.array([WING_FOLD, WING_FOLD], float)}
        self.tgt = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.s.items()}
        self.t = 0.0
        self.phase = np.random.default_rng(0).uniform(0, 2 * np.pi, 4)
        self.enabled = True

    # ---------------------------------------------------------------- targets (sim thread, every window)
    def cues(self, mn9=0.0, groom_side=None, turn=0.0, loom_side=None, resting=False, seizing=False, dead=False,
             flying=False, display=None):
        """mn9: MN9 filtered rate (Hz). groom_side: 'left' | 'right' | 'both' | None (an antennal grooming bout).
        turn: steering asymmetry, + = left. loom_side: 'left' | 'right' | None. display: None | 'spread' | 'flick'."""
        T = self.tgt
        pe = float(np.clip((mn9 - MN9_ON) / (MN9_FULL - MN9_ON), 0.0, 1.0)) if not dead else 0.0
        T["pe"] = pe
        T["lab"] = 1.0 if pe > 0.8 else 0.0
        yaw = 0.15 * float(np.clip(turn, -1, 1))
        if loom_side is not None:
            yaw = 0.18 if loom_side == "left" else -0.18
        pitch = -0.25 * pe
        roll = 0.0
        ant = np.zeros((2, 2))                              # rows l, r; cols pitch, yaw
        if groom_side in ("left", "right", "both"):
            pitch -= 0.2
            if groom_side != "both":
                roll = 0.3 if groom_side == "left" else -0.3
            for i, s in enumerate(("left", "right")):
                if groom_side in (s, "both"):
                    ant[i] = (0.45, 0.6)                    # pressed down and out by the leg (ours, by eye)
        T["head"] = np.array([yaw, pitch, roll])
        T["ant"] = ant
        T["abd"] = -0.12 if (seizing or dead) else 0.0
        self._rest = bool(resting) and not dead
        w = np.array([WING_FOLD, WING_FOLD], float)
        if display == "spread" and not flying:              # both wings out and up at ~45 deg (ours)
            w[:, 0] = 0.75
            w[:, 1] = 0.2
        elif display == "flick" and not flying:
            w[:, 0] = 1.0
        if seizing and not flying:                          # wing buzz in a seizure: jitter round a part-open fold
            w[:, 0] = 1.1 + 0.25 * np.random.default_rng().standard_normal(2)
        T["wing"] = w
        self._flying = flying

    def tick(self, dt):
        self.t += dt
        S, T = self.s, self.tgt
        for k in S:
            a = 1.0 - np.exp(-dt / TAU["head" if k == "head" else k])
            if isinstance(S[k], np.ndarray):
                S[k] = S[k] + a * (T[k] - S[k])
            else:
                S[k] += a * (T[k] - S[k])

    # ---------------------------------------------------------------- write into a pose copy (web thread)
    def extras(self):
        S = self.s
        ex = {}
        pe = S["pe"]
        r = PROBOSCIS_IN["rostrum"] + pe * (PROBOSCIS_OUT["rostrum"] - PROBOSCIS_IN["rostrum"])
        h = PROBOSCIS_IN["haustellum"] + pe * (PROBOSCIS_OUT["haustellum"] - PROBOSCIS_IN["haustellum"])
        ex["c_rostrum"] = _euler(pitch=r)
        ex["c_haustellum"] = _euler(pitch=h)
        if S["lab"] > 0.02:
            ex["l_labrum"] = ex["r_labrum"] = _euler(pitch=LABELLA_OPEN * S["lab"])
        hy, hp, hr = S["head"]
        hy = np.clip(hy, *RANGES["head_yaw"]); hp = np.clip(hp, *RANGES["head_pitch"]); hr = np.clip(hr, *RANGES["head_roll"])
        if abs(hy) + abs(hp) + abs(hr) > 1e-3:
            ex["c_head"] = _euler(pitch=hp, roll=hr, yaw=hy)
        for i, s in enumerate("lr"):
            p, y = S["ant"][i]
            if abs(p) + abs(y) > 1e-3:
                ex[f"{s}_antenna"] = _euler(pitch=np.clip(p, *RANGES["ant_pitch"]), yaw=np.clip(y, *RANGES["ant_yaw"]))
        ab = S["abd"]
        if getattr(self, "_rest", False):
            ab += 0.012 * np.sin(2 * np.pi * 0.8 * self.t + self.phase[0])      # slow rest pumping (ours)
        if abs(ab) > 1e-4:
            q = _euler(pitch=float(np.clip(ab, *RANGES["abd_pitch"])))
            for k in range(1, 8):
                ex[f"c_abdomen{k}"] = q
        if not getattr(self, "_flying", False):
            for i, s in enumerate("lr"):
                y, ro, pi = S["wing"][i]
                if abs(y - WING_FOLD[0]) + abs(ro - WING_FOLD[1]) + abs(pi - WING_FOLD[2]) > 1e-3:
                    want = _qmul(self.wing_base[s], _wing_q(y, ro, pi))
                    ex[f"{s}_wing"] = _qmul(_inv(self.lquat[self.id_of[f"{s}_wing"]]), want)
        return ex

    def apply(self, poses):
        """poses: (n, 7) float32 = xpos, xquat (w first) per Body.pose_body_ids(). Returns a modified copy."""
        if not self.enabled:
            return poses
        ex = {self.id_of[k]: v for k, v in self.extras().items()}
        if not ex:
            return poses
        P = poses.copy()
        moved = set()
        for i in self.order:
            p = self.parent[i]
            if i not in ex and p not in moved:
                continue
            pr = self.row[p]
            pp, pq = P[pr, :3].astype(float), P[pr, 3:].astype(float)
            q = _qmul(pq, self.lquat[i])
            if i in ex:
                q = _qmul(q, ex[i])
            P[self.row[i], :3] = pp + _qrot(pq, self.lpos[i])
            P[self.row[i], 3:] = q
            moved.add(i)
        return P
