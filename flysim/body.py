"""Fly body: the FlyBody (Vaxenburg et al.) mesh model inside flygym 2.1, in a small arena.

What is real and what is not:
- Physics: MuJoCo 3.9 via flygym's FlyBody model (mm, g, s; timestep 0.1 ms; full mesh
  collision hulls). Real.
- Walking: flygym's HybridTurningController semantics (coupled CPG, tripod bias,
  preprogrammed FlyBody steps, leg retraction rule) reimplemented as one numba function
  so a control tick costs microseconds instead of ~2.7 ms. The descending drive is the
  same 2-vector socket: |drive| sets per-side stepping amplitude, a negative drive
  reverses that side's CPG (backward stepping). The stumbling rule is dropped (it needs
  per-segment contact forces every tick). Controller runs every `ctrl_substeps` physics
  steps (default 5 = 0.5 ms).
- Flight: PLACEHOLDER. The wings have no joints in this build and no aerodynamics are
  computed. In flight mode an external force/torque on the thorax holds altitude, gives
  thrust along the heading and yaw from the drive difference; the front end animates
  the wings. It exists so the giant-fibre -> take-off chain has somewhere to land.
- Objects: a pool of droppable free bodies, plus a pool of SCENERY (walls, rocks, pillars,
  logs): mocap bodies the world editor moves, turns and resizes at run time (world.py). The
  four props of the original arena are now scenery placed by the "classic" world. Fly
  body/legs collide with both via contype/conaffinity bits (ground uses bit 4, props bit 2).
"""
from __future__ import annotations

import os

import numba as nb
import numpy as np
import mujoco as mj

from flygym import Simulation
from flygym.anatomy import BodySegment, JointDOF, RotationAxis
from flygym.compose import ActuatorType, FlatGroundWorld, KinematicPosePreset
from flygym.compose.fly import FlyBody
from flygym.flybody import (FlyBodyActuatedDOFPreset, FlyBodyAxisOrder, FlyBodyBodySegment,
                            FlyBodyContactBodiesPreset, FlyBodyJointPreset, FlyBodySkeleton)
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import FlyBodyPreprogrammedSteps
from flygym_demo.complex_terrain.common import dof_spec_to_jointdof
from flygym_demo.complex_terrain.cpg_controller import get_cpg_biases

from .anim import FaceAnim

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
# grooming key poses: absolute joint angles per leg in pre.dofs_per_leg order (coxa pitch, roll, yaw, femur pitch,
# roll, tibia, tarsus), the same on both sides (FlyBody legs are mirror-symmetric in joint space). Solved by IK on this
# body to body-part targets (results/anim_scout_2026-10-24/s05_groom_ik.py; targets ours, by eye). The cycle runs at
# GROOM_HZ: anterior grooming sweeps and rubs at 5.7-6.6 Hz over 18-30 C (Ravbar, Zhang & Simpson 2021, eLife,
# doi:10.7554/eLife.71508). Antennal grooming is commanded by the aDN descending neurons (Hampel et al. 2015, eLife,
# doi:10.7554/eLife.08758); here the bout starts on their read-out (Sim._anim_cues in app.py).
GROOM_KEYS = {
    "front_up": (1.063, 0.8, -1.0, 0.743, -0.857, -0.349, -0.7),       # tibia-tarsus over the antenna
    "front_down": (0.292, 0.577, -0.667, 0.167, -0.925, -1.052, -0.435),  # swept down past the proboscis
    "hind_fwd": (1.146, -0.13, 0.194, -0.461, 0.94, -0.498, -0.745),   # beside abdomen segment 3, over the wing
    "hind_back": (0.067, -0.15, 0.25, -0.219, 1.0, 0.348, -0.722),     # back to segment 6
}
GROOM_HZ = 6.0
GROOM_BREAK = 0.25      # |drive| above this ends a bout (a walking command wins over grooming; ours)
# posterior grooming uses one hind leg at a time: both hind legs up tip the fly back onto its abdomen (s07)
GROOM_KINDS = {"anterior": ((0, 3), "front_up", "front_down"), "hind_left": ((2,), "hind_fwd", "hind_back"),
               "hind_right": ((5,), "hind_fwd", "hind_back")}
RIGHT_S = 0.55          # self-righting duration, sim s [ours]; the sequence follows Issa et al. 2019 (Curr Biol,
                        # doi:10.1016/j.cub.2019.06.082) loosely: legs search, the hind legs reach, the body tips over
# the curled death pose, added to the neutral standing pose per leg DOF (rad) [ours, 2026-10-12, by eye against dead
# flies: femur and tibia folded in, tarsi tucked under the ventral thorax near the midline]; clipped to the actuator ranges
DEATH_REST_Z = 0.62        # thorax height lying on its back, mm [ours, by eye]
DEATH_CURL = {"femur": 0.8, "tibia": -2.2, "tarsus": -1.2, "coxa_roll": 0.0}   # experiments/death_pose_search.py
N_LUT = 1024
GROUND_BIT, PROP_BIT = 4, 2
_CORR = {"f": [-0.03, 0.0, 0.0, -0.03, 0.0, 0.03, 0.03],
         "m": [-0.015, 0.001, 0.025, -0.02, 0.0, -0.02, 0.0],
         "h": [0.0, 0.0, 0.0, -0.02, 0.0, 0.01, -0.02]}

# Units mm; the fly is ~2.5 mm long.
# arena radius, mm. FLYSIM_ARENA_R changes it for a whole run (the B-roll 'stage' world uses 120 so a walking fly
# cannot reach the rim in a 10-20 s take); the wall is built from it, everything else reads it from here
ARENA_R = float(__import__('os').environ.get('FLYSIM_ARENA_R', '30'))
# scenery pool: mocap bodies, parked out of sight (collision off) until the world places them
SCENERY = {"box": 64, "ellipsoid": 16, "cylinder": 30}
_SCN_PARK = (300.0, 300.0, -30.0)
MAX_OTHERS = 8          # other flies (others.py): kinematic collision bodies the fly can bump into
_OTH_PARK = (300.0, 340.0, -30.0)
SUGAR = {"pos": (6.0, -3.0), "r": 1.6}  # sugar drop: sensed by tarsal contact, no collision
POOL = [("sphere", (0.6, 0, 0), 0.8), ("sphere", (1.0, 0, 0), 0.6), ("box", (0.8, 0.8, 0.8), 0.5),
        ("box", (1.5, 0.5, 0.5), 0.5), ("cylinder", (0.5, 1.0, 0), 0.7), ("sphere", (0.4, 0, 0), 1.0),
        ("ellipsoid", (1.2, 0.8, 0.6), 0.6), ("box", (0.5, 0.5, 0.5), 1.0)]  # (type, size, density g/cm^3)


@nb.njit(cache=True, fastmath=True)
def _ctrl_tick(dt, drive, phases, mags, base_freq, w, bias, alpha, lut, neutral, swing_end,
               corr, retr, persist, thorax_z, tars_z, perm, out, retract_on):
    n = 6
    # descending drive -> per-side amplitude and CPG direction (HybridTurningController)
    freq = np.empty(n)
    amp = np.empty(n)
    for i in range(n):
        s = drive[0] if i < 3 else drive[1]
        amp[i] = abs(s)
        freq[i] = base_freq[i] if s >= 0 else -base_freq[i]
    # retraction rule: leg hanging well below the others gets lifted
    sel = -1
    if retract_on:
        ez = np.empty(n)
        for i in range(n):
            ez[i] = thorax_z - tars_z[i]
        order = np.argsort(ez)
        if ez[order[n - 1]] > ez[order[n - 3]] + 0.05:
            sel = order[n - 1]
            if retr[sel] > 20.0:
                persist[sel] = 1
    for i in range(n):
        if persist[i] > 0:
            persist[i] += 1
            if persist[i] > 20:
                persist[i] = 0
    # CPG Euler step
    dth = np.empty(n)
    for i in range(n):
        c = 0.0
        for j in range(n):
            c += mags[j] * w[i, j] * np.sin(phases[j] - phases[i] - bias[i, j])
        dth[i] = 2 * np.pi * freq[i] + c
    for i in range(n):
        phases[i] += dth[i] * dt
        mags[i] += alpha * (amp[i] - mags[i]) * dt
    # joint targets
    angles = np.empty(n * 7)
    for i in range(n):
        if i == sel or persist[i] > 0:
            retr[i] += 800.0 * dt
        else:
            retr[i] = max(0.0, retr[i] - 700.0 * dt)
        ph = phases[i] % (2 * np.pi)
        x = ph / (2 * np.pi) * N_LUT
        k0 = int(x) % N_LUT
        k1 = (k0 + 1) % N_LUT
        f = x - int(x)
        se = swing_end[i]
        # step-phase gain (piecewise linear, as flygym's _step_phase_gain)
        p0, p1, p2, p3 = 0.0, 0.5 * se, se + np.pi / 4, 0.5 * (se + 2 * np.pi)
        if ph < p1:
            g = 0.8 * ph / p1
        elif ph < p2:
            g = 0.8 * (p2 - ph) / (p2 - p1)
        elif ph < p3:
            g = -0.1 * (ph - p2) / (p3 - p2)
        else:
            g = -0.1 * (2 * np.pi - ph) / (2 * np.pi - p3)
        nc = min(max(retr[i], 0.0), 80.0) * g
        for d in range(7):
            psi = lut[i, d, k0] * (1 - f) + lut[i, d, k1] * f
            angles[i * 7 + d] = neutral[i, d] + mags[i] * (psi - neutral[i, d]) + nc * corr[i, d]
    for k in range(perm.shape[0]):
        out[k] = angles[perm[k]]


class Body:
    PHYSICS = {  # (timestep s, solver iterations, noslip iterations)
        "flygym": (1e-4, 100, 3),   # flygym 2.1 FlyBody defaults
        "fast": (2e-4, 30, 0),      # ~2x faster; same walking speeds within 7% (bench/body_opts.json)
    }

    def __init__(self, ctrl_substeps: int = 5, retraction: bool = True, seed: int = 0,
                 physics: str = "fast"):
        self.physics = physics
        self.rng = np.random.default_rng(seed)
        self.ctrl_substeps = ctrl_substeps
        self.retraction = retraction
        fly = FlyBody(name="fly")
        sk = FlyBodySkeleton(axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
                             joint_preset=FlyBodyJointPreset.LEGS_ONLY)
        fly.add_joints(sk, KinematicPosePreset.FLYBODY_NEUTRAL)
        fly.add_actuators(fly.skeleton.get_actuated_dofs_from_preset(
            FlyBodyActuatedDOFPreset.LEGS_ACTIVE_ONLY), ActuatorType.POSITION, kp=100)
        fly.add_leg_adhesion()
        fly.colorize()
        world = FlatGroundWorld()
        wb = world.mjcf_root.worldbody
        # scenery pool (world.py places, turns and resizes these at run time)
        self.scn_type = []
        for typ, n in SCENERY.items():
            for _ in range(n):
                i = len(self.scn_type)
                b = wb.add_body(name=f"scn{i}", pos=[_SCN_PARK[0] + 3 * i, _SCN_PARK[1], _SCN_PARK[2]], mocap=True)
                b.add_geom(name=f"scn{i}_geom", type=getattr(mj.mjtGeom, f"mjGEOM_{typ.upper()}"),
                           size=[1.0, 1.0, 1.0] if typ != "cylinder" else [1.0, 1.0, 0], rgba=[0.5, 0.5, 0.5, 1],
                           friction=[0.3, 0.005, 1e-4], priority=1)
                self.scn_type.append(typ)
        # the other flies: a mocap thorax-abdomen capsule and a head sphere each, sized to this model's body
        # (others.py moves them; the fly touches them through ordinary contacts)
        for k in range(MAX_OTHERS):
            b = wb.add_body(name=f"oth{k}", pos=[_OTH_PARK[0] + 4 * k, _OTH_PARK[1], _OTH_PARK[2]], mocap=True)
            b.add_geom(name=f"oth{k}_body", type=mj.mjtGeom.mjGEOM_CAPSULE, size=[0.45, 0.85, 0], pos=[-0.4, 0, 1.1],
                       quat=[np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0], rgba=[0.55, 0.3, 0.2, 1],
                       friction=[0.3, 0.005, 1e-4], priority=1)
            b.add_geom(name=f"oth{k}_head", type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.36, 0, 0], pos=[0.62, 0, 1.18],
                       rgba=[0.55, 0.3, 0.2, 1], friction=[0.3, 0.005, 1e-4], priority=1)
        # a low circular wall
        n_wall = 48
        for k in range(n_wall):
            a = 2 * np.pi * k / n_wall
            wb.add_geom(name=f"wall{k}", type=mj.mjtGeom.mjGEOM_BOX,
                        size=[0.6, ARENA_R * np.pi / n_wall + 0.05, 2.5],
                        pos=[ARENA_R * np.cos(a), ARENA_R * np.sin(a), 2.5],
                        quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)], rgba=[0.2, 0.21, 0.23, 1],
                        friction=[0.2, 0.005, 1e-4], priority=1)
        # droppable pool (parked, collision-off, gravity-compensated until dropped)
        for k, (typ, size, dens) in enumerate(POOL):
            b = wb.add_body(name=f"obj{k}", pos=[200 + 5 * k, 200, 5])
            b.add_freejoint(name=f"obj{k}_free")
            b.add_geom(name=f"obj{k}_geom", type=getattr(mj.mjtGeom, f"mjGEOM_{typ.upper()}"),
                       size=list(size), density=dens * 1e-3, rgba=[0.8, 0.8, 0.8, 1])
        world.add_fly(fly, [0, 0, 0.7], Rotation3D("quat", [1, 0, 0, 0]),
                      bodysegs_with_ground_contact=FlyBodyContactBodiesPreset.TIBIA_TARSUS_ONLY,
                      add_ground_contact_sensors=False)
        # collision bits, set on the spec so the compiled midphase BVHs include these geoms:
        # ground=4, props/objects=6; fly torso/head/abdomen hit ground+props, legs hit props
        # only (their ground contact is flygym's explicit tibia/tarsus pairs); wings/halteres off
        leg_prefix = tuple(f"fly/{l}_" for l in LEGS)
        for g in world.mjcf_root.geoms:
            nm = g.name or ""
            if nm == "ground_plane":
                g.contype = g.conaffinity = GROUND_BIT
            elif nm.startswith(("prop", "wall", "obj", "scn", "oth")):
                g.contype = g.conaffinity = GROUND_BIT | PROP_BIT
            elif nm.startswith("fly/") and not any(x in nm for x in ("wing", "haltere")):
                g.contype = 0
                g.conaffinity = PROP_BIT if nm.startswith(leg_prefix) else PROP_BIT | GROUND_BIT
        self.fly, self.world = fly, world
        self.sim = Simulation(world)
        m, d = self.sim.mj_model, self.sim.mj_data
        self.m, self.d = m, d
        m.opt.timestep, m.opt.iterations, m.opt.noslip_iterations = self.PHYSICS[physics]
        self.dt = m.opt.timestep

        # ids
        self.fly_body_ids = np.asarray(self.sim._internal_bodyids_by_fly["fly"])
        name = lambda o, i: mj.mj_id2name(m, o, i)
        body_names = [name(mj.mjtObj.mjOBJ_BODY, int(i)).split("/")[-1] for i in self.fly_body_ids]
        self.fly_body_names = body_names
        self.thorax = int(self.fly_body_ids[body_names.index("c_thorax")])
        self.tarsus5 = np.array([self.fly_body_ids[body_names.index(f"{l}_tarsus5")] for l in LEGS])
        self.head = int(self.fly_body_ids[body_names.index("c_head")])
        # antennae (thermo/hygro receptors sit in the arista; used by world.py to sample fields per side)
        self.antennae = np.array([int(self.fly_body_ids[body_names.index(n)]) if n in body_names else self.head
                                  for n in ("l_antenna", "r_antenna")])
        self.fly_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in set(self.fly_body_ids)])
        self.ground_geom = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "ground_plane")
        self.prop_geoms = np.array([g for g in range(m.ngeom)
                                    if (name(mj.mjtObj.mjOBJ_GEOM, g) or "").startswith(("prop", "wall"))])
        self.arena_geoms = self.prop_geoms            # drawn from scene.json; scenery is drawn from the world status
        n_scn = len(self.scn_type)
        self.scn_bodies = np.array([mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"scn{k}") for k in range(n_scn)])
        self.scn_geoms = np.array([mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, f"scn{k}_geom") for k in range(n_scn)])
        self.scn_mocap = np.array([m.body_mocapid[b] for b in self.scn_bodies])
        self._scn_live = np.zeros(n_scn, bool)
        self._scn_pose = {}                           # k -> (x, y, size, yaw, lie): re-applied after a reset
        self.oth_bodies = np.array([mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"oth{k}") for k in range(MAX_OTHERS)])
        self.oth_mocap = np.array([m.body_mocapid[b] for b in self.oth_bodies])
        self.oth_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in set(self.oth_bodies.tolist())])
        self._oth_of_geom = {int(g): int(np.flatnonzero(self.oth_bodies == m.geom_bodyid[g])[0]) for g in self.oth_geoms}
        self._oth_pose = {}                           # k -> (x, y, z, yaw, roll): re-applied after a reset
        self.touch_geoms = np.concatenate([self.prop_geoms, self.scn_geoms, self.oth_geoms])
        for k in range(n_scn):
            self.scenery_park(k)
        self.obj_bodies = np.array([mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"obj{k}") for k in range(len(POOL))])
        self.obj_geoms = np.array([mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, f"obj{k}_geom") for k in range(len(POOL))])
        self.obj_qadr = np.array([m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"obj{k}_free")] for k in range(len(POOL))])
        self.obj_vadr = np.array([m.jnt_dofadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"obj{k}_free")] for k in range(len(POOL))])
        self.fly_qadr = m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "fly")]
        self.fly_vadr = m.jnt_dofadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "fly")]
        self.mass = float(m.body_subtreemass[self.thorax])
        self.weight = self.mass * -m.opt.gravity[2]  # g mm s^-2 (= uN)

        self._obj_live = np.zeros(len(POOL), bool)
        self._obj_placed = np.zeros(len(POOL), bool)   # set down by the world editor (not dropped)
        self._park_objects(initial=True)
        self._sync_body_bits()

        # controller tables
        pre = FlyBodyPreprogrammedSteps()
        grid = np.linspace(0, 2 * np.pi, N_LUT, endpoint=False)
        self.lut = np.stack([pre._psi_funcs[l](grid) for l in LEGS])  # (6,7,N)
        self.neutral = np.stack([pre.neutral_pos[l][:, 0] for l in LEGS])
        self.swing_end = np.array([pre.swing_period[l][1] for l in LEGS])
        # FlyBody legs are mirror-symmetric in joint space, so no right-leg sign flip
        self.corr = np.array([_CORR[l[1]] for l in LEGS])
        dofs = [JointDOF(BodySegment(x.parent.name), BodySegment(x.child.name), RotationAxis(x.axis.value))
                for x in fly.get_actuated_jointdofs_order(ActuatorType.POSITION)]
        lookup = {dof_spec_to_jointdof(l, spec): i * 7 + k
                  for i, l in enumerate(LEGS) for k, spec in enumerate(pre.dofs_per_leg)}
        self.perm = np.array([lookup[x] for x in dofs], dtype=np.int64)
        self.act_ids = np.asarray(self.sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION]["fly"])
        self.adh_ids = np.asarray(self.sim._intern_adhesionactuatorids_by_fly["fly"])
        bias = get_cpg_biases("tripod")
        self.cpg_w = (bias > 0) * 10.0
        self.cpg_bias = bias
        self.base_freq = np.full(6, 12.0)
        self.default_pose = self.neutral.reshape(-1)[self.perm].copy()
        # tucked leg pose for flight: tarsi folded up under the body
        tuck = self.neutral.copy()
        tuck[:, 3] -= 0.6
        tuck[:, 5] += 0.9
        self.tuck_pose = tuck.reshape(-1)[self.perm].copy()
        # death: legs folded hard under the body; limp (heat stupor, chill coma): half folded, no grip
        dead = self.neutral.copy()
        dead[:, 3] += DEATH_CURL["femur"]
        dead[:, 5] += DEATH_CURL["tibia"]
        dead[:, 6] += DEATH_CURL["tarsus"]
        dead[:, 1] += DEATH_CURL["coxa_roll"]
        self.dead_pose = self._clamp_ctrl(dead.reshape(-1)[self.perm].copy())
        limp = self.neutral.copy()
        limp[:, 3] -= 0.45
        limp[:, 5] += 0.6
        self.limp_pose = limp.reshape(-1)[self.perm].copy()
        self.ext_loom = []                  # [(pos xyz, radius, vel xyz)]: looming things that are not pool objects
        self.ext_loom_extra = []            # the same, for flying other flies (set by the sim each window)
        self._out = np.zeros(len(self.perm))
        self._leg_cols = np.zeros((6, 7), np.int64)        # actuator column of (leg, dof)
        for k, p in enumerate(self.perm):
            self._leg_cols[int(p) // 7, int(p) % 7] = k
        self.anim = FaceAnim(self)                          # cosmetic pose layer (anim.py): streamed poses only
        self._act_qadr = m.jnt_qposadr[m.actuator_trnid[self.act_ids, 0]]
        self.reset()
        self.stand_z = float(self.d.qpos[self.fly_qadr + 2])   # root height of the settled standing fly (righting)

    # ---------------------------------------------------------------- lifecycle
    def reset(self, pos=(0.0, 0.0), heading=0.0):
        self.sim.reset()
        d, m = self.d, self.m
        d.qpos[self.fly_qadr:self.fly_qadr + 2] = pos
        d.qpos[self.fly_qadr + 3:self.fly_qadr + 7] = [np.cos(heading / 2), 0, 0, np.sin(heading / 2)]
        self._park_objects()
        self.phases = self.rng.random(6) * 2 * np.pi
        self.mags = np.zeros(6)
        self.retr = np.zeros(6)
        self.persist = np.zeros(6, np.int64)
        self.mode = "walk"
        self.flight_t = 0.0
        self.flight_dur = 0.0
        self.push_left = 0
        self.push_f = np.zeros(6)
        self.t = 0.0
        self.last_drive = np.zeros(2)
        self.groom = None
        self.righting = None
        self.anim.reset()
        d.ctrl[self.act_ids] = self.default_pose
        d.ctrl[self.adh_ids] = 0
        for k, p in list(self._scn_pose.items()):    # a reset puts mocap bodies back at their compiled spots
            self.scenery_set(k, *p)
        for k, p in list(self._oth_pose.items()):
            self.other_set(k, *p)
        mj.mj_step(m, d, nstep=int(0.05 / self.dt))  # settle (flygym warmup)
        self._prev_obj_ang = np.zeros(len(POOL))

    def _sync_body_bits(self):
        # MuJoCo's broadphase filters on body-level bits (OR of the body's geoms),
        # computed at compile time, so they must follow runtime geom edits
        m = self.m
        m.body_contype[:] = 0
        m.body_conaffinity[:] = 0
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            m.body_contype[b] |= m.geom_contype[g]
            m.body_conaffinity[b] |= m.geom_conaffinity[g]

    def _park_objects(self, initial=False, only=None):
        m, d = self.m, self.d
        for k in (range(len(POOL)) if only is None else [only]):
            self._obj_live[k] = False
            self._obj_placed[k] = False
            g, b = self.obj_geoms[k], self.obj_bodies[k]
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
            m.body_gravcomp[b] = 1.0
            park = [200 + 5 * k, 200, 5, 1, 0, 0, 0]
            d.qpos[self.obj_qadr[k]:self.obj_qadr[k] + 7] = park
            d.qvel[self.obj_vadr[k]:self.obj_vadr[k] + 6] = 0
            if initial:  # the neutral keyframe was built before these joints existed
                m.key_qpos[self.sim._neutral_keyframe_id, self.obj_qadr[k]:self.obj_qadr[k] + 7] = park
        if not initial:
            self._sync_body_bits()

    def free_object(self, prefer=None):
        """Index of an unused pool object (prefer a shape type), or None when all are in use."""
        free = [k for k in range(len(POOL)) if not self._obj_live[k]]
        if prefer:
            free = sorted(free, key=lambda k: POOL[k][0] != prefer)
        return free[0] if free else None

    def place_object(self, k, x, y):
        """World editor: set pool object k down on the floor at (x, y), at rest, as an obstacle."""
        m, d = self.m, self.d
        g, b = self.obj_geoms[k], self.obj_bodies[k]
        m.geom_contype[g] = m.geom_conaffinity[g] = GROUND_BIT | PROP_BIT
        m.body_gravcomp[b] = 0.0
        z = float(m.geom_rbound[g]) + 0.05
        typ = POOL[k][0]
        if typ in ("box", "ellipsoid"):
            z = float(m.geom_size[g][2]) + 0.02
        elif typ == "cylinder":
            z = float(m.geom_size[g][1]) + 0.02
        elif typ == "sphere":
            z = float(m.geom_size[g][0]) + 0.02
        d.qpos[self.obj_qadr[k]:self.obj_qadr[k] + 7] = [x, y, z, 1, 0, 0, 0]
        d.qvel[self.obj_vadr[k]:self.obj_vadr[k] + 6] = 0
        self._obj_live[k] = True
        self._obj_placed[k] = True
        self._sync_body_bits()

    def remove_object(self, k):
        self._park_objects(only=int(k))

    def object_xy(self, k):
        return self.d.xpos[self.obj_bodies[k], :2].tolist()

    # ---------------------------------------------------------------- scenery (mocap bodies)
    def scenery_free(self, typ):
        free = [k for k, t in enumerate(self.scn_type) if t == typ and not self._scn_live[k]]
        return free[0] if free else None

    def scenery_set(self, k, x, y, size, yaw=0.0, lie=False):
        """Put scenery piece k at (x, y) on the floor, turned by yaw (rad), with geom size `size`
        (box: half-extents; ellipsoid: semi-axes; cylinder: radius, half-length). lie: a cylinder on
        its side (a log), its axis along yaw. The bounding volumes are updated with the size; the
        body's own box is a cube of the bounding radius, which is conservative at any rotation."""
        m, d = self.m, self.d
        g, b = self.scn_geoms[k], self.scn_bodies[k]
        typ = self.scn_type[k]
        sz = np.array([float(v) for v in size] + [0.0] * (3 - len(size)))[:3]
        if typ == "cylinder":
            sz[2] = 0.0
            ext = np.array([sz[0], sz[0], sz[1]])
        else:
            ext = sz.copy()
        m.geom_size[g] = sz
        rb = float(np.linalg.norm(ext))
        m.geom_rbound[g] = rb
        m.geom_aabb[g] = [0, 0, 0, *ext]
        if m.body_bvhnum[b]:
            m.bvh_aabb[m.body_bvhadr[b]] = [0, 0, 0, rb, rb, rb]
        m.geom_contype[g] = m.geom_conaffinity[g] = GROUND_BIT | PROP_BIT
        m.body_contype[b] = m.body_conaffinity[b] = GROUND_BIT | PROP_BIT
        qy = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        if typ == "cylinder" and lie:
            q = np.zeros(4)
            mj.mju_mulQuat(q, qy, np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]))   # axis z -> x, then yaw
            z = sz[0]
        else:
            q = qy
            z = {"box": sz[2], "ellipsoid": 0.45 * sz[2], "cylinder": sz[1]}[typ]
        i = self.scn_mocap[k]
        d.mocap_pos[i] = [x, y, z]
        d.mocap_quat[i] = q
        self._scn_live[k] = True
        self._scn_pose[k] = (float(x), float(y), [float(v) for v in size], float(yaw), bool(lie))

    def other_set(self, k, x, y, z=0.0, yaw=0.0, roll=0.0):
        """Place other fly k's collision body: ground frame at (x, y, z), heading yaw, rolled by roll (dead: pi)."""
        d = self.d
        i = self.oth_mocap[k]
        d.mocap_pos[i] = [x, y, z]
        cy, sy, cr, sr = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(roll / 2), np.sin(roll / 2)
        d.mocap_quat[i] = [cy * cr, cy * sr, sy * sr, sy * cr]      # yaw, then roll about the body axis
        self._oth_pose[k] = (x, y, z, yaw, roll)

    def other_park(self, k):
        i = self.oth_mocap[k]
        self.d.mocap_pos[i] = [_OTH_PARK[0] + 4 * k, _OTH_PARK[1], _OTH_PARK[2]]
        self.d.mocap_quat[i] = [1, 0, 0, 0]
        self._oth_pose.pop(k, None)

    def movers(self):
        """Moving pool objects as (pos, radius, vel): what could loom at an eye (others.py uses it too)."""
        m, d = self.m, self.d
        out = []
        for k in np.flatnonzero(self._obj_live):
            if self._obj_placed[k] and np.linalg.norm(d.qvel[self.obj_vadr[k]:self.obj_vadr[k] + 3]) < 5.0:
                continue
            out.append((d.xpos[self.obj_bodies[k]].copy(), float(m.geom_rbound[self.obj_geoms[k]]),
                        d.qvel[self.obj_vadr[k]:self.obj_vadr[k] + 3].copy()))
        return out

    def scenery_park(self, k):
        m, d = self.m, self.d
        g = self.scn_geoms[k]
        m.geom_contype[g] = m.geom_conaffinity[g] = 0
        # body bits too: the broadphase filters on them, and 90 parked bodies left in it cost ~25% of physics time
        m.body_contype[self.scn_bodies[k]] = m.body_conaffinity[self.scn_bodies[k]] = 0
        i = self.scn_mocap[k]
        d.mocap_pos[i] = [_SCN_PARK[0] + 3 * k, _SCN_PARK[1], _SCN_PARK[2]]
        d.mocap_quat[i] = [1, 0, 0, 0]
        self._scn_live[k] = False
        self._scn_pose.pop(k, None)

    # ---------------------------------------------------------------- stepping
    def step(self, drive, n_steps: int):
        """Advance `n_steps` physics steps with a constant 2-vector descending drive."""
        drive = np.asarray(drive, dtype=np.float64)
        self.last_drive = drive.copy()
        m, d = self.m, self.d
        sub = self.ctrl_substeps
        cdt = self.dt * sub
        done = 0
        while done < n_steps:
            k = min(sub, n_steps - done)
            if self.mode == "dead" and getattr(self, "death", None) is not None:
                self._death_tick(k * self.dt)
            elif self.mode == "righting":
                self._right_tick(k * self.dt)
            elif self.mode in ("limp", "dead", "held"):
                pass                                   # the pose set by set_state holds; no gait
            elif self.mode == "walk":
                xz = d.xpos[:, 2]
                _ctrl_tick(cdt * k / sub, drive, self.phases, self.mags, self.base_freq, self.cpg_w,
                           self.cpg_bias, 20.0, self.lut, self.neutral, self.swing_end, self.corr,
                           self.retr, self.persist, xz[self.thorax], xz[self.tarsus5], self.perm,
                           self._out, self.retraction)
                if self.groom is not None:
                    self._groom_overlay(cdt * k / sub, drive)
                d.ctrl[self.act_ids] = self._out
            else:
                self._flight_tick(drive, k * self.dt)
            if self.push_left > 0:
                d.xfrc_applied[self.thorax] += self.push_f
                mj.mj_step(m, d, nstep=k)
                d.xfrc_applied[self.thorax] -= self.push_f
                self.push_left -= k
            else:
                mj.mj_step(m, d, nstep=k)
            done += k
        self.t += n_steps * self.dt

    # ---------------------------------------------------------------- flight (placeholder)
    def takeoff(self, duration_s: float = 1.5, away=None):
        """away: unit xy vector away from a looming threat. Flies plan the jump away from the threat
        (Card & Dickinson 2008, Curr Biol, doi:10.1016/j.cub.2008.07.094); here that is a flatter jump in
        that direction and 0.2 s of low, ballistic flight before the placeholder's air drag applies (speeds ours)."""
        if self.mode != "walk":
            return
        self.mode = "flight"
        self.flight_t = 0.0
        self.flight_dur = duration_s
        self.d.ctrl[self.act_ids] = self.tuck_pose
        v = self.d.qvel[self.fly_vadr:self.fly_vadr + 3]
        if away is not None:
            self.escape_t = 0.2
            self.escape_v = np.array([away[0], away[1]]) * 110.0
            v += np.array([self.escape_v[0], self.escape_v[1], 60.0])
        else:
            self.escape_t = 0.0
            v += self.heading() * 60.0 + np.array([0, 0, 180.0])  # jump: ~ escape take-off speed

    def _flight_tick(self, drive, h):
        d = self.d
        self.flight_t += h
        z = d.xpos[self.thorax, 2]
        v = d.qvel[self.fly_vadr:self.fly_vadr + 3]
        w = d.qvel[self.fly_vadr + 3:self.fly_vadr + 6]  # body frame
        R = d.xmat[self.thorax].reshape(3, 3)
        landing = self.flight_t > self.flight_dur
        escaping = self.flight_t < getattr(self, "escape_t", 0.0)
        z_target = 0.3 if landing else (3.5 if escaping else 12.0)
        fwd = 0.5 * (drive[0] + drive[1])
        turn = drive[0] - drive[1]
        W = self.weight
        up = W * (1.0 + np.clip(0.8 * (z_target - z) - 0.25 * v[2], -0.9, 1.5))
        thrust = W * (0.15 + 0.5 * max(fwd, 0.0)) * (0.0 if landing else 1.0)
        hd = R[:, 0].copy()
        hd[2] = 0
        hd /= np.linalg.norm(hd) + 1e-9
        f = np.array([0, 0, up]) + thrust * hd - 0.02 * W * v * np.array([1, 1, 0])
        if escaping:                                   # the jump: hold its speed away from the threat
            f[:2] = 0.05 * W * (self.escape_v - v[:2])
        # soft arena boundary: near the wall, pull toward the centre and yaw to face it
        # (the wall is only 5 mm high and flight cruises at 12 mm)
        p = d.xpos[self.thorax, :2]
        r = np.linalg.norm(p)
        edge = ARENA_R - 8.0
        steer = 0.0
        if r > edge:
            inward = -p / r
            f[:2] += W * 1.5 * min((r - edge) / 8.0, 1.5) * inward - 0.1 * W * v[:2]
            steer = np.clip(hd[0] * inward[1] - hd[1] * inward[0], -1, 1)
        # attitude: keep roll/pitch level, yaw rate from drive asymmetry
        zb = R[:, 2]
        level = np.cross(zb, [0, 0, 1.0])
        wz_target = 4.0 * np.clip(turn, -1, 1) + 6.0 * steer
        wz = (R @ w)[2]
        tq = 2e-3 * W * level - 2e-4 * W * (R @ w) + np.array([0, 0, 3e-4 * W * (wz_target - wz)])
        d.xfrc_applied[self.thorax, :3] = f
        d.xfrc_applied[self.thorax, 3:] = tq
        # down: on its feet near the floor; or overturned on the floor (its thorax lies higher on its back, above
        # 1.6 mm, so it would hang there under the landing lift for ever); or landing has gone on too long
        if landing and ((z < 1.6 and v[2] > -5.0) or (z < 2.6 and R[2, 2] < 0.3)
                        or self.flight_t > self.flight_dur + 1.5):
            d.xfrc_applied[self.thorax] = 0
            self.mode = "walk"
            d.ctrl[self.act_ids] = self.default_pose

    # ---------------------------------------------------------------- manipulation
    def drop_object(self, x=None, y=None, z=8.0, kind=None, rgba=None):
        """Drop a pool object from height z above (x, y); default: near the fly."""
        free = np.flatnonzero(~self._obj_live)
        if not len(free) and self._obj_placed.any():   # pool full: reuse a dropped object, not a placed obstacle
            free = np.flatnonzero(~self._obj_placed)
        k = int(free[0]) if len(free) else int(self.rng.integers(len(POOL)))  # pool full: reuse one
        if kind is not None:
            cands = [i for i in range(len(POOL)) if POOL[i][0] == kind and not self._obj_live[i]]
            if cands:
                k = cands[0]
        if x is None:
            p = self.d.xpos[self.thorax]
            x, y = p[0] + self.rng.normal(0, 1.5), p[1] + self.rng.normal(0, 1.5)
        m, d = self.m, self.d
        g, b = self.obj_geoms[k], self.obj_bodies[k]
        m.geom_contype[g] = m.geom_conaffinity[g] = GROUND_BIT | PROP_BIT
        m.body_gravcomp[b] = 0.0
        if rgba is not None:
            m.geom_rgba[g] = rgba
        q = self.rng.normal(size=4)
        d.qpos[self.obj_qadr[k]:self.obj_qadr[k] + 7] = [x, y, z, *(q / np.linalg.norm(q))]
        d.qvel[self.obj_vadr[k]:self.obj_vadr[k] + 6] = 0
        self._obj_live[k] = True
        self._obj_placed[k] = False
        self._sync_body_bits()
        return k

    def clear_objects(self):
        self._park_objects()

    def push(self, fx, fy, fz=0.0, duration_ms=20.0):
        """Apply a force to the thorax, in multiples of the fly's body weight."""
        self.push_f = np.array([fx, fy, fz, 0, 0, 0]) * self.weight
        self.push_left = int(duration_ms * 1e-3 / self.dt)

    def place(self, x, y, heading=0.0):
        d = self.d
        d.qpos[self.fly_qadr:self.fly_qadr + 3] = [x, y, 0.7]
        d.qpos[self.fly_qadr + 3:self.fly_qadr + 7] = [np.cos(heading / 2), 0, 0, np.sin(heading / 2)]
        d.qvel[self.fly_vadr:self.fly_vadr + 6] = 0

    def right(self, instant=False):
        """Put an overturned fly back on its feet. Default: the self-righting animation (_right_tick, RIGHT_S sim s);
        instant=True: the old placement (still used when there is nothing to animate, e.g. it is already upright)."""
        if self.mode == "righting" and not instant:
            return                                    # already on its way up (autonomy.py asks every window)
        p = self.d.xpos[self.thorax]
        h = self.heading()
        up = float(self.d.xmat[self.thorax].reshape(3, 3)[2, 2])
        if instant or up > 0.5 or self.mode in ("flight", "dead"):
            self.place(p[0], p[1], float(np.arctan2(h[1], h[0])))
            return
        d = self.d
        yaw = float(np.arctan2(h[1], h[0]))
        self.groom = None
        self.mode = "righting"
        self.righting = {"t": 0.0, "T": RIGHT_S, "q0": d.qpos[self.fly_qadr + 3:self.fly_qadr + 7].copy(),
                         "q1": np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), "z0": float(d.qpos[self.fly_qadr + 2]),
                         "xy": d.qpos[self.fly_qadr:self.fly_qadr + 2].copy(), "pose0": d.ctrl[self.act_ids].copy(),
                         "ph": self.rng.uniform(0, 2 * np.pi, len(self.act_ids))}

    def _right_tick(self, dt):
        """Self-righting, kinematic like the death roll and its mirror: the legs search in the air (first ~45 %), the
        hind legs reach forward, and the body tips back over onto its feet (smoothstep slerp of the root from its
        overturned orientation to upright at the same heading, lifted while it turns). Amplitudes and timing ours."""
        D, d = self.righting, self.d
        D["t"] += dt
        u = min(1.0, D["t"] / D["T"])
        sm = lambda x: float(np.clip(x, 0.0, 1.0)) ** 2 * (3 - 2 * float(np.clip(x, 0.0, 1.0)))
        pose = D["pose0"] + (self.default_pose - D["pose0"]) * sm(u / 0.5)
        if u < 0.6:                                   # legs searching: fast, fading waves
            pose = pose + (1.0 - u / 0.6) * 0.4 * np.sin(2 * np.pi * 7.0 * D["t"] + D["ph"])
        if 0.25 < u < 0.8:                            # the hind legs swing forward to reach the floor
            reach = np.sin(np.pi * (u - 0.25) / 0.55)
            for li in (2, 5):
                pose[self._leg_cols[li, 0]] += 0.8 * reach
                pose[self._leg_cols[li, 5]] -= 0.5 * reach
        d.ctrl[self.act_ids] = self._clamp_ctrl(pose)
        s = sm((u - 0.3) / 0.55)                      # the tip-over, done at u = 0.85; then it stands still
        q0, q1 = D["q0"], D["q1"]
        c = float(np.dot(q0, q1))
        if c < 0:
            q1, c = -q1, -c
        if c > 0.9995:
            q = q0 + s * (q1 - q0)
        else:
            th = np.arccos(c)
            q = (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)
        d.qpos[self.fly_qadr + 3:self.fly_qadr + 7] = q / np.linalg.norm(q)
        d.qpos[self.fly_qadr:self.fly_qadr + 2] = D["xy"]
        d.qpos[self.fly_qadr + 2] = D["z0"] + (self.stand_z - D["z0"]) * s + 0.5 * np.sin(np.pi * s)
        d.qvel[self.fly_vadr:self.fly_vadr + 6] = 0
        if u >= 1.0:
            # release on its feet: legs exactly at the stance pose (a leg still inside the floor at release kicks the
            # body over), root at the settled standing height, at rest
            self.righting = None
            self.mode = "walk"
            d.ctrl[self.act_ids] = self.default_pose
            d.qpos[self._act_qadr] = self.default_pose
            d.qvel[:] = 0
            d.qpos[self.fly_qadr + 2] = self.stand_z
            self.mags[:] = 0.0

    def groom_start(self, kind="anterior", secs=0.5):
        """Start (or extend) a grooming bout: the grooming legs follow GROOM_KEYS at GROOM_HZ over the walking
        controller's output, blended in and out (~60 ms). A drive above GROOM_BREAK ends it. Walk mode only."""
        if self.mode != "walk" or kind not in GROOM_KINDS:
            return False
        if self.groom is not None and self.groom["kind"] == kind:
            self.groom["T"] = max(self.groom["T"], self.groom["t"] + float(secs))
            return True
        if self.groom is not None and self.groom["w"] > 0.05:
            return False                              # finish the bout in progress first
        self.groom = {"kind": kind, "t": 0.0, "T": float(secs), "w": 0.0}
        return True

    def _groom_overlay(self, dt, drive):
        G = self.groom
        G["t"] += dt
        want = 1.0 if (G["t"] < G["T"] and float(np.abs(drive).max()) < GROOM_BREAK) else 0.0
        G["w"] += (1.0 - np.exp(-dt / 0.06)) * (want - G["w"])
        if want == 0.0 and G["w"] < 0.02:
            self.groom = None
            return
        legs, ka, kb = GROOM_KINDS[G["kind"]]
        A, B = np.asarray(GROOM_KEYS[ka]), np.asarray(GROOM_KEYS[kb])
        ph = 2 * np.pi * GROOM_HZ * G["t"]
        w = G["w"]
        for i, li in enumerate(legs):
            a = 0.5 + 0.5 * np.sin(ph + np.pi * i)   # two legs in antiphase (the in-phase cycle rocks the body, s05)
            cols = self._leg_cols[li]
            self._out[cols] = (1.0 - w) * self._out[cols] + w * (B + a * (A - B))
        self._out[:] = self._clamp_ctrl(self._out)

    def grooming(self):
        """The bout in progress as (kind, weight), or None."""
        return None if self.groom is None else (self.groom["kind"], round(float(self.groom["w"]), 3))

    def _clamp_ctrl(self, c):
        """Clip position targets to the actuators' control ranges (the joint limits where they are set)."""
        m = self.m
        lim = m.actuator_ctrllimited[self.act_ids].astype(bool)
        lo, hi = m.actuator_ctrlrange[self.act_ids, 0], m.actuator_ctrlrange[self.act_ids, 1]
        return np.where(lim, np.clip(c, lo, hi), c)

    def _death_tick(self, dt):
        """Dying, then dead: twitches (first ~40 %), the legs curling toward dead_pose (smoothstep from 15 %), and
        the body rolled about its long axis onto its back over the whole span (kinematic: the root is set each chunk,
        lifted so the legs clear the floor while it turns). At the end the root is released at rest and holds."""
        D, d = self.death, self.d
        if D["done"]:                                 # dead and still: hold the root where it came to rest
            d.qpos[self.fly_qadr:self.fly_qadr + 7] = D["rest"]
            d.qvel[self.fly_vadr:self.fly_vadr + 6] = 0
            return
        D["t"] += dt
        u = min(1.0, D["t"] / D["T"])
        sm = lambda x: float(np.clip(x, 0.0, 1.0)) ** 2 * (3 - 2 * float(np.clip(x, 0.0, 1.0)))
        c = sm((u - 0.15) / 0.85)
        pose = D["pose0"] + (self.dead_pose - D["pose0"]) * c
        if u < 0.45:                                  # dying twitches: 3 jerks, fading
            env = (1.0 - u / 0.45) * 0.35
            pose = pose + env * np.sign(np.sin(2 * np.pi * 3.0 * u / 0.45 + D["ph"])) * (np.sin(D["ph"] * 3.1) ** 2)
        d.ctrl[self.act_ids] = self._clamp_ctrl(pose)
        ang = np.pi * sm(u)                           # onto its back
        out = np.zeros(4)
        mj.mju_mulQuat(out, D["q0"], np.array([np.cos(ang / 2), np.sin(ang / 2), 0.0, 0.0]))
        d.qpos[self.fly_qadr + 3:self.fly_qadr + 7] = out
        # lifted while it turns (the legs clear the floor), down to lying height on its back (DEATH_REST_Z) at the end
        d.qpos[self.fly_qadr + 2] = D["z0"] + (DEATH_REST_Z - D["z0"]) * sm(u) + 0.45 * np.sin(np.pi * sm(u))
        d.qvel[self.fly_vadr:self.fly_vadr + 6] = 0
        if u >= 1.0:
            D["done"] = True
            D["rest"] = d.qpos[self.fly_qadr:self.fly_qadr + 7].copy()

    def set_state(self, state):
        """walk | limp (paralysed: heat stupor, chill coma) | dead (legs curled, rolled onto its back)."""
        d = self.d
        if state == self.mode or (state == "walk" and self.mode == "flight"):
            return
        if self.mode == "flight":
            d.xfrc_applied[self.thorax] = 0              # it drops out of the air
        prev, self.mode = self.mode, state
        self.groom = None
        self.righting = None
        if state != "dead":
            self.death = None
        d.ctrl[self.adh_ids] = 0
        if state == "walk":
            d.ctrl[self.act_ids] = self.default_pose
            if prev == "dead" or self.d.xmat[self.thorax].reshape(3, 3)[2, 2] < 0.5:
                self.right()
        elif state == "held":                  # cinema: held still standing (pinned), legs in the normal stance
            d.ctrl[self.act_ids] = self.default_pose
        elif state == "limp":
            d.ctrl[self.act_ids] = self.limp_pose
        elif state == "dead":
            # the death animation (_death_tick): a few twitches, then the legs curl in while the body rolls onto its
            # back, over DEATH_S sim s (FLYSIM_DEATH_S; a slow-motion shot sets it so it takes ~1.3 s of video)
            q = d.qpos[self.fly_qadr + 3:self.fly_qadr + 7].copy()
            self.death = {"t": 0.0, "T": max(0.05, float(os.environ.get("FLYSIM_DEATH_S", "0.6"))), "q0": q,
                          "pose0": d.ctrl[self.act_ids].copy(), "z0": float(d.qpos[self.fly_qadr + 2]),
                          "ph": self.rng.uniform(0, 2 * np.pi, len(self.act_ids)), "done": False}

    # ---------------------------------------------------------------- readout
    def heading(self):
        h = self.d.xmat[self.thorax].reshape(3, 3)[:, 0].copy()
        h[2] = 0
        return h / (np.linalg.norm(h) + 1e-9)

    def sensory(self):
        """Body-side sensory state (the inputs a sensory->brain mapping would read)."""
        m, d = self.m, self.d
        con = d.contact.geom[:d.ncon] if d.ncon else np.zeros((0, 2), int)
        fly_set = self.fly_geoms
        is_fly = np.isin(con, fly_set)
        other_prop = np.isin(con, np.concatenate([self.touch_geoms, self.obj_geoms]))
        touch, t_oth = [], set()
        for (g1, g2), (f1, f2), (o1, o2) in zip(con, is_fly, other_prop):
            if f1 and o2:
                touch.append(int(m.geom_bodyid[g1]))
                if int(g2) in self._oth_of_geom:
                    t_oth.add(self._oth_of_geom[int(g2)])
            elif f2 and o1:
                touch.append(int(m.geom_bodyid[g2]))
                if int(g1) in self._oth_of_geom:
                    t_oth.add(self._oth_of_geom[int(g1)])
        touch_parts = sorted({self.fly_body_names[int(np.flatnonzero(self.fly_body_ids == b)[0])] for b in touch})
        tz = d.xpos[self.tarsus5, 2]
        stance = tz < 0.12
        txy = d.xpos[self.tarsus5, :2]
        sugar = np.linalg.norm(txy - np.array(SUGAR["pos"]), axis=1) < SUGAR["r"]
        # looming: angular size and expansion rate of the nearest moving object seen from the head
        hp = d.xpos[self.head]
        loom, loom_rate, loom_at = 0.0, 0.0, None
        for k in np.flatnonzero(self._obj_live):
            b = self.obj_bodies[k]
            if self._obj_placed[k] and np.linalg.norm(d.qvel[self.obj_vadr[k]:self.obj_vadr[k] + 3]) < 5.0:
                continue   # an obstacle at rest is not a looming object (its settling jitter would read as one)
            r = float(m.geom_rbound[self.obj_geoms[k]])
            dist = max(np.linalg.norm(d.xpos[b] - hp) - r, 0.05)
            ang = 2 * np.arctan(r / dist)
            vel = d.qvel[self.obj_vadr[k]:self.obj_vadr[k] + 3]
            rel = hp - d.xpos[b]
            closing = float(vel @ rel / (np.linalg.norm(rel) + 1e-9))
            rate = 2 * r * closing / (dist ** 2 + r ** 2)
            if ang > loom:
                loom, loom_rate, loom_at = ang, rate, d.xpos[b].copy()
        for p, r, vel in list(self.ext_loom) + list(self.ext_loom_extra):   # the hand (Sim.hand), flying flies
            rel = hp - np.asarray(p)
            dist = max(np.linalg.norm(rel) - r, 0.05)
            ang = 2 * np.arctan(r / dist)
            closing = float(np.asarray(vel) @ rel / (np.linalg.norm(rel) + 1e-9))
            if ang > loom:
                loom, loom_rate, loom_at = ang, 2 * r * closing / (dist ** 2 + r ** 2), np.asarray(p, float)
        loom_side = None
        if loom_at is not None:                  # +y in the fly frame = left
            loom_side = "left" if (d.xmat[self.thorax].reshape(3, 3).T @ (loom_at - hp))[1] >= 0 else "right"
        v = d.qvel[self.fly_vadr:self.fly_vadr + 3]
        h = self.heading()
        return {
            "t": self.t, "mode": self.mode,
            "pos": d.xpos[self.thorax].tolist(), "heading": h.tolist(),
            "v_forward": float(v[:2] @ h[:2]), "v_lateral": float(v[0] * -h[1] + v[1] * h[0]),
            "yaw_rate": float((d.xmat[self.thorax].reshape(3, 3) @ d.qvel[self.fly_vadr + 3:self.fly_vadr + 6])[2]),
            "leg_stance": stance.tolist(), "touch": touch_parts,
            "sugar_legs": sugar.tolist(), "loom_angle": float(loom), "loom_rate": float(loom_rate), "loom_side": loom_side,
            "touch_others": sorted(t_oth),
            "upright": float(d.xmat[self.thorax].reshape(3, 3)[2, 2]),
            "drive": self.last_drive.tolist(),
        }

    # ---------------------------------------------------------------- export for rendering
    def pose_body_ids(self):
        return np.concatenate([self.fly_body_ids, self.obj_bodies])

    def poses(self):
        """Streamed body poses: the physics pose with the cosmetic layer (anim.FaceAnim) on the jointless parts."""
        ids = self.pose_body_ids()
        P = np.concatenate([self.d.xpos[ids], self.d.xquat[ids]], axis=1).astype(np.float32)
        return self.anim.apply(P)

    def scene(self):
        """Meshes + primitives for the front end. Returns (json_dict, binary_blob)."""
        m = self.m
        pose_ids = list(self.pose_body_ids())
        blobs, off = [], 0
        meshes = {}
        geoms = []
        for g in self.fly_geoms:
            mid = int(m.geom_dataid[g])
            if mid not in meshes:
                va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
                v = m.mesh_vert[va:va + vn]
                f = m.mesh_face[fa:fa + fn]
                # weld duplicated vertices (MuJoCo stores them per face)
                key = np.round(v / 1e-5).astype(np.int64)
                _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
                v = v[first].astype(np.float32)
                f = inv.reshape(-1)[f].astype(np.uint32)
                meshes[mid] = {"v_off": off, "v_n": int(len(v))}
                blobs.append(v.tobytes()); off += v.nbytes
                meshes[mid].update(f_off=off, f_n=int(f.size))
                blobs.append(f.tobytes()); off += f.nbytes
            mat = int(m.geom_matid[g])
            rgba = (m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]).tolist()
            bname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])).split("/")[-1]
            geoms.append({"mesh": mid, "body": pose_ids.index(m.geom_bodyid[g]), "body_name": bname,
                          "pos": m.geom_pos[g].tolist(), "quat": m.geom_quat[g].tolist(), "rgba": rgba,
                          "name": mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, int(g)).split("/")[-1]})
        prims = []
        for g in list(self.arena_geoms):
            prims.append({"type": int(m.geom_type[g]), "size": m.geom_size[g].tolist(),
                          "pos": m.geom_pos[g].tolist(), "quat": m.geom_quat[g].tolist(),
                          "rgba": m.geom_rgba[g].tolist(), "body": -1,
                          "name": mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, int(g))})
        for k, g in enumerate(self.obj_geoms):
            prims.append({"type": int(m.geom_type[g]), "size": m.geom_size[g].tolist(),
                          "pos": [0, 0, 0], "quat": [1, 0, 0, 0], "rgba": m.geom_rgba[g].tolist(),
                          "body": pose_ids.index(self.obj_bodies[k]), "name": f"obj{k}"})
        meta = {"meshes": {str(k): v for k, v in meshes.items()}, "geoms": geoms, "prims": prims,
                "n_pose": len(pose_ids), "arena_r": ARENA_R, "sugar": SUGAR,
                "units": "mm", "geom_types": {"sphere": 2, "capsule": 3, "ellipsoid": 4,
                                              "cylinder": 5, "box": 6}}
        return meta, b"".join(blobs)
