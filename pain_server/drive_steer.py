"""Per-drive steering set (lever 1): from the fly's drives to one summed control vector (numpy only, no torch, so
the llama backend can import it).

Files (per weight variant, see model.py):
  vectors/<NAME>_drives.npz          drive_extract.py: directions per drive, variant and layer
  vectors/<NAME>_drives_config.json  which variant and layers are served, dose_max per drive, total cap
                                     (drive_calibrate.py writes dose_max and the cap)

Dose units: dose 1 adds 0.1 x the mean residual norm at the layer along the drive's unit direction, so it is the
same size as dose 1 of the old pain_s2 vector at layer 27 (|pain_s2| 19.0, residual 183). With a band of layers the
dose is split evenly over them. The summed vector  sum_k dose_k u_k  is scaled down as a whole when its norm (in
the same units) is over total_cap, so co-occurring drives cannot push the model past what calibration allows.

Sim -> steer mapping (from_sim): body["drives"] keys of flysim/embodiment.py. temp_stress has no sign, so heat and
cold are taken from the "heat" / "cold" keys the sim adds for lever 1 (embodiment.steer_drives); thirst and hunger
are separate drives; fatigue, intoxication and sedation have no direction here (they act through lever 2).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STEER_DRIVES = ("threat", "hunger", "thirst", "heat", "cold", "arousal", "pain")
UNIT_FRAC = 0.1
RANDOM_SEED = 20261008        # the random-vector control's directions (fixed across runs)
STEER_MODES = ("real", "sham", "random")


def from_sim(drives):
    """body["drives"] (0..1 floats) -> {steer drive: level 0..1}. Unknown keys are ignored."""
    if not drives:
        return {}
    g = lambda k: float(np.clip(float(drives.get(k) or 0.0), 0.0, 1.0))
    return {k: g(k) for k in STEER_DRIVES if g(k) > 0}


class DriveSet:
    def __init__(self, name, config=None):
        npz = HERE / "vectors" / f"{name}_drives.npz"
        cfgp = HERE / "vectors" / f"{name}_drives_config.json"
        if not npz.exists():
            raise FileNotFoundError(f"{npz.name}: run drive_extract.py")
        z = np.load(npz)
        self.cfg = config or (json.loads(cfgp.read_text()) if cfgp.exists() else {})
        self.variant = self.cfg.get("variant", "deshared")
        self.layers = [int(x) for x in self.cfg.get("layers", [27])]
        names = [str(x) for x in z["drives"]]
        M = z[self.variant].astype(np.float32)                      # drives x layers x d
        self.d = M.shape[-1]
        rn = z["resid_norm"].astype(np.float32)
        self.u = {k: {L: M[i, L] / (np.linalg.norm(M[i, L]) + 1e-9) for L in self.layers} for i, k in enumerate(names)}
        self.scale = {L: UNIT_FRAC * float(rn[L]) / len(self.layers) for L in self.layers}
        self.dose_max = {k: float(v) for k, v in self.cfg.get("dose_max", {}).items()}
        self.total_cap = float(self.cfg.get("total_cap", 1e9))
        self.names = names
        # random-vector control (experiments/behav): one fixed Gaussian direction per drive and layer (seed fixed, so
        # every run uses the same set), unit norm like self.u, so a "random" call gets the same doses, caps and norm
        # as a real one along directions that mean nothing.
        rng = np.random.default_rng(RANDOM_SEED)
        self.u_rand = {}
        for k in names:
            self.u_rand[k] = {}
            for L in self.layers:
                r = rng.standard_normal(self.d).astype(np.float32)
                self.u_rand[k][L] = r / (np.linalg.norm(r) + 1e-9)

    def doses_for(self, levels, mask=None):
        """{drive: level 0..1} -> {drive: dose} using dose_max (drives with no dose_max get 0). mask: drives allowed
        to steer (None = all); the others get dose 0 (they still act through lever 2 and the text)."""
        return {k: round(float(lv) * self.dose_max.get(k, 0.0), 4) for k, lv in levels.items()
                if k in self.u and (mask is None or k in mask)}

    def _basis(self, mode):
        return self.u_rand if mode == "random" else self.u

    def norm_of(self, doses, mode="real"):
        L0 = self.layers[len(self.layers) // 2]
        u = self._basis(mode)
        v = sum((d * u[k][L0] for k, d in doses.items() if d > 0), np.zeros(self.d, np.float32))
        return float(np.linalg.norm(v))

    def build(self, doses, mode="real"):
        """{drive: dose} -> ({layer: vector to add}, info). Per-drive doses clipped to dose_max (when known), the
        sum scaled to total_cap. mode: real (the drive directions), random (fixed random directions, same doses and
        cap), sham (the doses are computed and reported, nothing is added)."""
        dd = {}
        for k, d in (doses or {}).items():
            if k not in self.u or d <= 0:
                continue
            dd[k] = min(float(d), self.dose_max.get(k, float(d)))
        n = self.norm_of(dd, mode)
        f = min(1.0, self.total_cap / n) if n > 0 else 1.0
        dd = {k: d * f for k, d in dd.items()}
        add = {}
        u = self._basis(mode)
        if mode != "sham":
            for L in self.layers:
                v = np.zeros(self.d, np.float32)
                for k, d in dd.items():
                    v += d * u[k][L]
                if dd:
                    add[L] = v * self.scale[L]
        return add, {"doses": {k: round(d, 3) for k, d in dd.items()}, "norm": round(n * f, 3), "capped": f < 1.0,
                     "mode": mode}
