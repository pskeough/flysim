"""Brain -> body bridge: descending-neuron spike rates -> the body's 2-vector descending drive.

v2 (2026-09-23): DNg13 steering term added from the grounding work in GROUNDING.md (item 1).
Saturation rates, gains and the 50 ms filter are still ours. The neurons were chosen from the
literature on single identified DNs:
  - DNp09 ("P9"): forward walking when activated (Bidaye et al. 2020, Neuron).
  - MDN (moonwalker): backward walking when activated (Bidaye et al. 2014, Science).
  - DNa02: ipsilateral steering, its activity predicts turns (Rayshubskiy et al. 2025, eLife
    / bioRxiv 2020).
  - DNp01 (giant fibre): escape take-off (e.g. von Reyn et al. 2014, Nat Neurosci).
The numbers (saturation rates, gains, the 50 ms filter) are ours, not from those papers.

Per side s (left = index 0):
    f      = mean_s r(DNp09_s) / R_FWD    forward command, BOTH sides' legs (see below)
    b_s    = r(MDN_s)   / R_BACK          backward command
    turn_s = r(DNa02_s) / R_TURN          ipsilateral turn: slows the same-side legs
    g_s    = r(DNg13_s) / R_TURN          ipsiversive turn via the other side's legs
    drive_s = clip(f - b_s, -1.5, 1.5) * (1 - 0.6 * clip(turn_s, 0, 1) + 0.4 * clip(g_opp, 0, 1))
P9 is treated as a bilateral forward command because unilateral P9 activation produces forward
walking with IPSILATERAL turning (Bidaye et al. 2020), and in the connectome model DNp09_left
recruits DNa02_left and DNg13_left (screen_dn.json). Mapping DNp09_s onto same-side legs would
turn the fly the wrong way; with this mapping the ipsilateral turn comes from the recruited
steering DNs, i.e. from the connectome, not from the bridge.
DNg02 (flight power, a population code: Namiki et al. 2022, Curr Biol) is read out as the
number of active DNg02 cells for the status panel; the flight physics does not use it yet,
because nothing in the sim supplies the wide-field motion that drives its premotor inputs.
Rates are exponentially filtered with tau = 50 ms. Zero drive = standing still.
Take-off: any GF spike in the last coupling window with GF filtered rate > GF_THRESH.
"""
from __future__ import annotations

import os

import numpy as np

R_FWD, R_BACK, R_TURN = 50.0, 50.0, 50.0
TAU_MS = 50.0
GF_THRESH = 20.0
DN_TYPES = ["DNp09", "MDN", "DNa02", "DNa01", "DNg13", "DNb01", "DNp01", "DNg100", "DNb08", "DNb05"]

# v3 (2026-09-24): wider DN read-out and graded turning. Selected by FLYSIM_BRIDGE (v2 | v3).
#   f       = min(sum_t W_FWD[t] * mean_s r(t_s) / R_FWD, 1.5)          bilateral forward / stepping command
#   in_s    = sum_t W_IN[t] * r(t_s) / R_TURN3                          ipsiversive DNs that shorten side-s steps
#   out_s   = W_OUT * r(DNg13_s) / R_TURN3                              DNg13_s lengthens the OTHER side's steps
#   drive_s = clip(f - b_s, -1.5, 1.5) - K_TURN * clip(in_s - in_o, 0, 1) + K_TURN * clip(out_o - out_s, 0, 1)
# Only the left/right DIFFERENCE turns the fly (bilateral steering-DN activity cancels), and the
# turn terms are ADDED to the forward drive instead of scaling it. The turning rate therefore
# follows the L/R difference and exists without forward drive: with f = 0 the inner side's CPG
# runs backwards and the fly turns on the spot. In v2 the terms multiplied f, so a steering DN
# alone did nothing and the radius of a DNp09+steering turn was fixed (~4 mm).
# Literature (GROUNDING.md, "Bridge v3"): DNa02 shortens inside steps and DNg13 lengthens
# outside steps (Yang et al. 2024 Cell); DNa01 predicts lower-gain steering than DNa02
# (Rayshubskiy et al. 2025 eLife) and its activation triggers in-place turning (Braun et al.
# 2024 Nature); DNb05 correlates with ipsiversive steering (Yang 2024, correlational only);
# DNg100 (= BDN2) initiates forward walking (Sapkal et al. 2024 Nature; identity per Pugliese
# et al. 2025 bioRxiv); DNb08 drives rhythmic leg movement in decapitated flies that is not
# coordinated walking (Pugliese 2025), so it gets no direct weight. Weight magnitudes, R_TURN3 and K_TURN are ours.
BRIDGE_VERSION = os.environ.get("FLYSIM_BRIDGE", "v3").lower()   # v3 default since 2026-09-24 (turn_test.json, v3e closed loop)
W_FWD = {"DNp09": 1.0, "DNg100": 1.0, "DNb08": 0.0}   # DNb08: rhythmic leg movement that "differed qualitatively from the coordinated walking" (Pugliese 2025), so not a walking command; it still recruits DNg100 in the brain
W_IN = {"DNa02": 1.0, "DNa01": 0.5, "DNb05": 0.5}
W_OUT = 1.0
R_TURN3 = 200.0   # DN rates under activation span ~0-200 Hz; 50 Hz (v2) saturated the turn term
K_TURN = 1.0


class Bridge:
    def __init__(self, table):
        self.groups = {}
        for t in DN_TYPES:
            for s in ("left", "right"):
                self.groups[f"{t}_{s}"] = table.type_side(t, s)
        self.names = list(self.groups)
        self.dng02 = table.find("DNg02")
        self.dng02_active = 0
        self.rate = {k: 0.0 for k in self.names}
        self.gf_spiked = False

    def update(self, counts: np.ndarray, window_ms: float):
        a = 1.0 - np.exp(-window_ms / TAU_MS)
        for k, idx in self.groups.items():
            r = float(counts[idx].mean()) / (window_ms * 1e-3) if len(idx) else 0.0
            self.rate[k] += a * (r - self.rate[k])
        gf = np.concatenate([self.groups["DNp01_left"], self.groups["DNp01_right"]])
        self.gf_spiked = bool(counts[gf].sum() > 0)
        self.dng02_active = int((counts[self.dng02] > 0).sum()) if len(self.dng02) else 0

    def drive(self) -> np.ndarray:
        if BRIDGE_VERSION == "v3":
            return self.drive_v3()
        out = np.zeros(2)
        f = 0.5 * (self.rate["DNp09_left"] + self.rate["DNp09_right"]) / R_FWD
        for i, s in enumerate(("left", "right")):
            b = self.rate[f"MDN_{s}"] / R_BACK
            o = ("right", "left")[i]
            turn = np.clip(self.rate[f"DNa02_{s}"] / R_TURN, 0, 1)
            g_opp = np.clip(self.rate[f"DNg13_{o}"] / R_TURN, 0, 1)
            out[i] = np.clip(f - b, -1.5, 1.5) * (1 - 0.6 * turn + 0.4 * g_opp)
        return out

    def drive_v3(self) -> np.ndarray:
        r = self.rate
        f = min(sum(w * 0.5 * (r[f"{t}_left"] + r[f"{t}_right"]) for t, w in W_FWD.items()) / R_FWD, 1.5)
        ins = [sum(w * r[f"{t}_{s}"] for t, w in W_IN.items()) / R_TURN3 for s in ("left", "right")]
        outs = [W_OUT * r[f"DNg13_{s}"] / R_TURN3 for s in ("left", "right")]
        out = np.zeros(2)
        for i, s in enumerate(("left", "right")):
            o = 1 - i
            out[i] = (np.clip(f - r[f"MDN_{s}"] / R_BACK, -1.5, 1.5)
                      - K_TURN * np.clip(ins[i] - ins[o], 0, 1) + K_TURN * np.clip(outs[o] - outs[i], 0, 1))
        return np.clip(out, -1.5, 1.5)

    def takeoff(self) -> bool:
        gf = max(self.rate["DNp01_left"], self.rate["DNp01_right"])
        return bool(self.gf_spiked and gf > GF_THRESH)

    def reset(self):
        self.rate = {k: 0.0 for k in self.names}
        self.gf_spiked = False
