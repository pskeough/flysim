"""The fly's body economy and how it dies: energy and water reserves, body temperature, health.

Replaces cognition.Needs (hunger = 1 - energy, thirst = 1 - water, same status() fields), and adds
what it lacked: the body can run out, overheat, freeze, be crushed, and die.

Temperature limits (the only numbers here taken from measurements):
  heat knockdown (CTmax) 41 C: mean CTmax 40.98 +- 0.79 C across 34 DGRP lines, ramp 0.25 C/min
      (Rolandi et al. 2018, Ecology and Evolution, doi:10.1002/ece3.4409)
  chill coma onset (CTmin) 7 C: inside the -4.2 to 8.8 C range across rearing / acclimation /
      hardening treatments (Ransberry, MacMillan & Sinclair 2011, Physiol Biochem Zool,
      doi:10.1086/662642); 7 C is our pick for an unacclimated fly
Everything else is a demo choice, not a measurement: the body's thermal time constant, the damage
rates, and the reserve durations. Starvation and desiccation are TIME-COMPRESSED: a fly lasts days
without food and hours to a day without water; here the reserves last minutes of sim time
(ENERGY_S, WATER_S at metabolism 1x), so a session can show the whole arc. The metabolism rate
multiplies all of it.
"""
from __future__ import annotations

import numpy as np

CT_MAX, CT_MAX_OFF = 41.0, 39.0      # heat stupor on / recovered (hysteresis ours)
CT_MIN, CT_MIN_OFF = 7.0, 9.0        # chill coma on / recovered
HEAT_HARM, COLD_HARM = 38.0, CT_MIN  # damage starts above / below
TAU_BODY = 1.5                       # s: a ~1 mg body follows the air fast (demo value)
ENERGY_S, WATER_S = 420.0, 300.0     # sim s from full to empty at metabolism 1x, 24 C, 40% RH (compressed)
FEED_S = 6.0                         # sim s of feeding / drinking from empty to full
CCR_S = 3.0                          # sim s to come round after chill coma (real: 12-55 min, compressed)
CAUSES = {
    "starvation": "starved: its energy ran out",
    "dehydration": "dried out: its water ran out",
    "heat": "overheated",
    "cold": "froze",
    "crushed": "was crushed",
}


class Physiology:
    def __init__(self):
        self.rate = 1.0                  # metabolism multiplier (the UI's "life speed")
        self.reset()

    def reset(self, energy=0.6, water=0.7):
        self.energy, self.water = energy, water
        self.tb = None                   # body temperature, C (set from the air on the first update)
        self.health = 1.0
        self.state = "ok"                # ok | weak | stupor (heat) | coma (cold) | dead
        self.alive_s = 0.0
        self.dead_at = None
        self.cause = ""
        self.fed_s = self.drank_s = 0.0
        self.harm = {k: 0.0 for k in CAUSES}   # recent damage by cause (decays), for naming the death
        self._coma_left = 0.0
        self.crush = 0.0                 # pending crush damage (set by the sim)

    # ------------------------------------------------------------------ the old Needs interface
    @property
    def hunger(self):
        return 1.0 - self.energy

    @property
    def thirst(self):
        return 1.0 - self.water

    @property
    def dead(self):
        return self.state == "dead"

    @property
    def paralysed(self):
        return self.state in ("stupor", "coma", "dead")

    def update(self, sens, rates, dt, t_env, humid, moving=0.0, flying=False):
        """One window. t_env: air temperature at the body (C); humid: RH %; moving: speed, mm/s.
        Returns the new state when it changed, else None."""
        if self.dead:
            return None
        self.alive_s += dt
        r = self.rate
        # body temperature
        if self.tb is None:
            self.tb = float(t_env)
        self.tb += (t_env - self.tb) * (1.0 - np.exp(-dt / TAU_BODY))
        tb = self.tb
        # spending: energy with activity, water with dry air and heat
        act = 1.0 + 0.4 * min(1.0, moving / 10.0) + (2.0 if flying else 0.0)
        warm = 1.0 + max(0.0, tb - 25.0) / 10.0
        dry = (0.4 + 1.5 * (1.0 - min(humid, 100.0) / 100.0)) / 1.3      # 1.0 at 40% RH
        self.energy = max(0.0, self.energy - r * dt * act * warm / ENERGY_S)
        self.water = max(0.0, self.water - r * dt * dry * warm / WATER_S)
        # eating = legs on sugar AND the proboscis motor neuron MN9 firing (the brain's own readout)
        if not self.paralysed and any(sens.get("sugar_legs", ())) and rates.get("MN9", 0.0) > 10.0:
            self.energy = min(1.0, self.energy + dt / FEED_S)
            self.fed_s += dt
        if not self.paralysed and any(sens.get("water_legs", ())) and rates.get("water_GRN", 0.0) > 30.0:
            self.water = min(1.0, self.water + dt / FEED_S)
            self.drank_s += dt
        # damage
        harm = {
            "starvation": (1 / 60.0) * r if self.energy <= 0 else 0.0,
            "dehydration": (1 / 40.0) * r if self.water <= 0 else 0.0,
            "heat": 0.004 * (tb - HEAT_HARM) ** 2 if tb > HEAT_HARM else 0.0,
            "cold": 0.003 * (COLD_HARM - tb) + (0.05 * (-5.0 - tb) if tb < -5.0 else 0.0) if tb < COLD_HARM else 0.0,
            "crushed": 0.0,
        }
        dmg = sum(harm.values()) * dt
        if self.crush > 0:
            harm["crushed"] = self.crush / max(dt, 1e-6)
            dmg += self.crush
            self.crush = 0.0
        for k in self.harm:
            self.harm[k] = self.harm[k] * np.exp(-dt / 5.0) + harm[k] * dt
        well = self.energy > 0.15 and self.water > 0.15 and 15.0 < tb < 34.0
        self.health = float(np.clip(self.health - dmg + (dt / 120.0 if well and dmg == 0 else 0.0), 0.0, 1.0))
        # state
        prev = self.state
        if self.health <= 0.0:
            self.state = "dead"
            self.dead_at = self.alive_s
            self.cause = max(self.harm, key=self.harm.get)
        elif self.state == "stupor":
            if tb < CT_MAX_OFF:
                self.state = "ok"
        elif self.state == "coma":
            if tb > CT_MIN_OFF:
                self._coma_left -= dt
                if self._coma_left <= 0:
                    self.state = "ok"
            else:
                self._coma_left = CCR_S
        elif tb >= CT_MAX:
            self.state = "stupor"
        elif tb <= CT_MIN:
            self.state, self._coma_left = "coma", CCR_S
        else:
            self.state = "weak" if (self.energy < 0.06 or self.water < 0.06 or self.health < 0.25) else "ok"
        return self.state if self.state != prev else None

    def motor_gain(self):
        """How much of the brain's drive reaches the legs."""
        return {"ok": 1.0, "weak": 0.45}.get(self.state, 0.0)

    def status(self, sens=None):
        T = (sens or {}).get("temp_antenna") or [24.0, 24.0]
        t = max(T) if max(T) - 25 > 25 - min(T) else min(T)
        comfort = "too hot" if t > 28 else ("too cold" if t < 21 else "comfortable")
        return {"hunger": round(self.hunger, 3), "thirst": round(self.thirst, 3), "temp_c": round(float(t), 1),
                "comfort": comfort, "energy": round(self.energy, 3), "water": round(self.water, 3),
                "tb": round(float(self.tb if self.tb is not None else t), 1), "health": round(self.health, 3),
                "state": self.state, "alive_s": round(self.alive_s, 1), "cause": self.cause,
                "cause_text": CAUSES.get(self.cause, ""), "rate": self.rate,
                "harm": max(self.harm, key=self.harm.get) if max(self.harm.values()) > 1e-3 else ""}
