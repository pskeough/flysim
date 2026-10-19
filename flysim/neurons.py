"""Per-neuron positions, classes and named groups for the brain model's neuron order.

Positions: FlyWire v783 annotation table (Schlegel et al. 2024), soma position where
annotated, else the representative point pos_x/y/z. Both are in 4x4x40 nm voxels.
Output is in micrometres, centred on the brain, axes: +x = fly's left, +y = dorsal,
+z = anterior (FlyWire y points ventral and z runs posterior-to-anterior in section order;
we only flip y). Neurons missing from the table fall back to Codex coordinates.csv.gz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "flywire"
ANNOT = DATA / "Supplemental_file1_neuron_annotations.tsv"
VOX = np.array([4.0, 4.0, 40.0]) * 1e-3  # voxel -> um

# display classes (super_class), colour as hex
CLASSES = [
    ("optic", 0x3b4a8c), ("central", 0x2aa198), ("visual_projection", 0x4f7fe0),
    ("sensory", 0xe0a030), ("sensory_ascending", 0xd0c040), ("ascending", 0x5fc060),
    ("descending", 0xff4060), ("motor", 0xff8030), ("endocrine", 0xe060c0),
    ("visual_centrifugal", 0x8060d0), ("other", 0x707070),
]
CLASS_IDX = {c: i for i, (c, _) in enumerate(CLASSES)}

# Shiu et al. 2024 example notebook: right-side sugar GRNs (v630 IDs; 20/21 exist in v783)
SUGAR_GRN_630 = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
                 720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
                 720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
                 720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
                 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
                 720575940611875570]


@dataclass
class NeuronTable:
    ids: np.ndarray
    pos: np.ndarray            # (n,3) float32 um
    cls: np.ndarray            # (n,) uint8 index into CLASSES
    cell_type: np.ndarray      # (n,) object
    side: np.ndarray           # (n,) object
    groups: dict = field(default_factory=dict)  # name -> index array

    def type_side(self, cell_type: str, side: str | None = None) -> np.ndarray:
        m = self.cell_type == cell_type
        if side:
            m &= self.side == side
        return np.flatnonzero(m)

    def find(self, name: str) -> np.ndarray:
        """Resolve a group name ('MDN', 'MDN_left', 'sugar_GRN', a cell type, or a root id)."""
        if name in self.groups:
            return self.groups[name]
        for sfx in ("_left", "_right"):
            if name.endswith(sfx):
                return self.type_side(name[: -len(sfx)], sfx[1:])
        if name.isdigit():
            return np.flatnonzero(self.ids == int(name))
        return self.type_side(name)


def load(ids: np.ndarray) -> NeuronTable:
    a = pd.read_csv(ANNOT, sep="\t", low_memory=False,
                    usecols=["root_id", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z",
                             "super_class", "cell_type", "side", "cell_sub_class"])
    a = a.drop_duplicates("root_id").set_index("root_id").reindex(ids)
    p = a[["soma_x", "soma_y", "soma_z"]].to_numpy(float)
    q = a[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
    p = np.where(np.isnan(p).any(1, keepdims=True), q, p) * VOX
    miss = np.isnan(p).any(1)
    if miss.any():
        c = pd.read_csv(DATA / "coordinates.csv.gz").drop_duplicates("root_id").set_index("root_id")
        pts = c.reindex(ids[miss]).iloc[:, 0].fillna("[nan nan nan]")
        p[miss] = np.array([np.fromstring(s.strip("[]"), sep=" ") for s in pts]) * 1e-3
    ok = ~np.isnan(p).any(1)
    p[~ok] = np.nanmedian(p, 0)
    ctr = np.median(p[ok], 0)
    pos = (p - ctr) * np.array([-1.0, -1.0, 1.0])  # +x fly-left (FlyWire left has small x), +y dorsal
    sc = a["super_class"].fillna("other").to_numpy()
    cls = np.array([CLASS_IDX.get(s, CLASS_IDX["other"]) for s in sc], np.uint8)
    t = NeuronTable(ids=np.asarray(ids), pos=pos.astype(np.float32), cls=cls,
                    cell_type=a["cell_type"].fillna("").to_numpy(object),
                    side=a["side"].fillna("").to_numpy(object))
    idx = {int(f): i for i, f in enumerate(ids)}
    t.groups["sugar_GRN"] = np.array([idx[f] for f in SUGAR_GRN_630 if f in idx])
    t.groups["GF"] = t.type_side("DNp01")
    t.groups["giant_fiber"] = t.groups["GF"]
    # groups used by the sensory layer, the LLM layer and the readouts (see GROUNDING.md)
    for k, v in {"MN9_right": 720575940660219265, "MN9_left": 720575940618238523,
                 "aDN1": 720575940616185531, "aDN2": 720575940629806974, "aBN1": 720575940630907434}.items():
        if v in idx:
            t.groups[k] = np.array([idx[v]])
    t.groups["MN9"] = np.concatenate([t.groups.get("MN9_right", []), t.groups.get("MN9_left", [])]).astype(np.int64)
    import json
    sets = json.load(open(DATA.parent / "shiu_named_sets.json"))
    t.groups["JO_CE"] = np.array([idx[int(f)] for f in sets["neu_JON_CE"] if int(f) in idx])
    # taste sets from Shiu et al. 2024 (water: 18, bitter: 20; all present in v783)
    t.groups["water_GRN"] = np.array([idx[int(f)] for f in sets["neu_water"] if int(f) in idx])
    t.groups["bitter_GRN"] = np.array([idx[int(f)] for f in sets["neu_bitter"] if int(f) in idx])
    ct = pd.Series(t.cell_type)
    # antennal thermo- and hygrosensory receptor neurons, FlyWire cell types (Marin et al. 2020,
    # Curr Biol, doi:10.1016/j.cub.2020.06.028): VP2 = heating, VP3a/b = cold, VP5 = moist
    for s in ("left", "right"):
        on = t.side == s
        t.groups[f"TRN_hot_{s}"] = np.flatnonzero((ct == "TRN_VP2").to_numpy() & on)
        t.groups[f"TRN_cold_{s}"] = np.flatnonzero(ct.isin(["TRN_VP3a", "TRN_VP3b"]).to_numpy() & on)
        t.groups[f"HRN_moist_{s}"] = np.flatnonzero((ct == "HRN_VP5").to_numpy() & on)
    for k in ("TRN_hot", "TRN_cold", "HRN_moist"):
        t.groups[k] = np.concatenate([t.groups[f"{k}_left"], t.groups[f"{k}_right"]])
    sub = a["cell_sub_class"].fillna("").astype(str).to_numpy()
    for s in ("left", "right"):
        on = t.side == s
        t.groups[f"JO_wind_{s}"] = np.flatnonzero((sub == "wind_gravity") & on)  # JO-C/E class
        for bm in ("BM_InOm", "BM_Ant"):
            t.groups[f"{bm}_{s}"] = np.flatnonzero(ct.str.match(bm + r"(_|$)").to_numpy() & on)
        t.groups[f"DNg02_{s}"] = np.flatnonzero(ct.str.startswith("DNg02").to_numpy() & on)
    t.groups["DNg02"] = np.concatenate([t.groups["DNg02_left"], t.groups["DNg02_right"]])
    # nociception (nociception.py, research/PAIN_DESIGN.md): PPL1 aversive dopamine neurons as the read-out, and the
    # 20-neuron stand-in ascending set as the entry (frozen from results/pain_probe.json; not the real md targets)
    t.groups["PPL1"] = np.flatnonzero(ct.str.startswith("PPL10").to_numpy())
    noci = json.load(open(DATA.parent / "noci_stand_in.json"))["neurons"]
    for s in ("left", "right"):
        t.groups[f"AN_noci_{s}"] = np.array([idx[a["id"]] for a in noci if a["side"] == s and a["id"] in idx], np.int64)
    t.groups["AN_noci"] = np.concatenate([t.groups["AN_noci_left"], t.groups["AN_noci_right"]])
    t.n_missing_pos = int((~ok).sum())
    return t
