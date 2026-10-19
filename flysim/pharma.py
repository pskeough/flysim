"""Drugs: a dose becomes a concentration over time, and the concentration changes the brain.

Two channels, kept apart and labelled apart in the UI:

  BRAIN (simulated)  The drug changes parameters of the connectome model, and the 138,639-neuron
                     LIF network does the rest: which reflexes still fire, whether activity runs
                     away, how much of the brain is lit. Three levers exist in the model:
                       pre   x synaptic weight of every synapse a neuron SENDS, chosen by the sender's
                             predicted transmitter (Eckstein et al. 2024, Cell,
                             doi:10.1016/j.cell.2024.03.016): ACh, GABA, Glu, DA, 5-HT, OA
                       post  x synaptic weight of every synapse a neuron RECEIVES, chosen by region
                       rest  mV added to the resting potential (a tonic depolarisation or
                             hyperpolarisation), optionally weighted by how cholinergic a cell's
                             input is (an agonist acts where its receptor is)
                     In the Shiu et al. 2024 model ACh is excitatory and GABA and glutamate are
                     inhibitory; DA, 5-HT and OA neurons are ordinary signed spiking units. There is
                     no metabotropic signalling, no receptor subtypes, no glia, no plasticity, so a
                     drug whose action is mainly neuromodulatory is expressed only crudely.

  BODY (scripted)    Behavioural phenotypes reported in the fly literature that the brain channel
                     cannot produce on its own, because walking here is commanded by a placeholder
                     policy that drives the descending neurons directly (autonomy.py). These are
                     written by hand from the literature: more or less exploring, unsteady legs,
                     knock-down. The UI marks them "scripted".

Pharmacokinetics: one compartment with first-order absorption from the gut (fed) or the air
(vapour), first-order elimination. Concentration is in units of the drug's reference
concentration (1.0 = the concentration at which its main brain effect is half-maximal). Each
effect follows a Hill curve E = C^n / (C^n + EC50^n). Time constants are compressed for a demo:
seconds of sim time where a real fly takes minutes. All numbers in DRUGS are ours unless a
citation says otherwise; research/DRUGS_GROUNDING.md holds the literature per drug.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .neurons import DATA

NT_CODES = {"ACH": "ACh", "GABA": "GABA", "GLUT": "Glu", "DA": "DA", "SER": "5-HT", "OCT": "OA"}
REGIONS = {   # FlyWire primary neuropil (neurons.csv 'group', first token) -> a named region
    "MB": ("MB_CA", "MB_ML", "MB_VL", "MB_PED"),
    "CX": ("FB", "EB", "PB", "NO"),
    "AL": ("AL",),
    "optic": ("ME", "LO", "LA", "LOP", "AME"),
    "GNG": ("GNG", "PRW", "SAD", "FLA", "AMMC"),
    "LH": ("LH",),
}


@dataclass
class Effect:
    lever: str           # pre | post | rest | phen
    target: str          # pre: transmitter (ACh, GABA, ...); post/rest: region, 'all', 'central', 'ach_in'; phen: key
    full: float          # value at full effect: a gain for pre/post, mV for rest, a phenotype amount for phen
    ec50: float = 1.0    # concentration (reference units) at half effect
    hill: float = 2.0


@dataclass
class Drug:
    key: str
    name: str
    color: str
    what: str            # one line: target and action
    fly: str             # one line: what it does to real flies (literature)
    effects: list
    half_life_s: float = 40.0           # elimination, sim s (compressed)
    absorb_s: float = 6.0               # gut / airway uptake time constant, sim s
    routes: tuple = ("feed", "inject")
    feed_rate: float = 0.25             # reference units per s of feeding on laced food (into the gut)
    vapour_rate: float = 0.12           # reference units per s in the vapour (into the airway pool)
    bolus: float = 1.5                  # an injection, reference units straight into the body
    refs: list = field(default_factory=list)


E = Effect
# Literature per drug: research/DRUGS_GROUNDING.md. Every citation in `refs` was checked against its Europe PMC
# record on 2026-10-03. The brain levers and their sizes, the time constants and the scripted body effects are ours.
DRUGS = {d.key: d for d in [
    Drug("ethanol", "Ethanol", "#e8c25a",
         "a general depressant here: stronger inhibition, weaker excitation, slightly hyperpolarised cells "
         "(its molecular target in flies is unresolved)",
         "walking rises while the internal concentration rises, then gives way to incoordination and sedation",
         [E("pre", "GABA", 1.6, 1.0), E("pre", "ACh", 0.8, 1.2), E("rest", "all", -1.5, 1.4),
          E("phen", "explore", 0.9, 0.35, 3.0), E("phen", "explore", -1.6, 1.3, 3.0),
          E("phen", "ataxia", 1.5, 0.7), E("phen", "knockdown", 1.0, 2.2, 4.0)],
         half_life_s=60.0, absorb_s=8.0, routes=("vapour", "feed", "inject"), feed_rate=0.3, vapour_rate=0.03,   # vapour plateau ~2.6
         refs=["Wolf et al. 2002, J Neurosci, doi:10.1523/JNEUROSCI.22-24-11035.2002",
               "Bainton et al. 2000, Curr Biol, doi:10.1016/S0960-9822(00)00336-5"]),
    Drug("nicotine", "Nicotine", "#7fd36a",
         "nicotinic ACh receptor agonist: depolarises each cell in proportion to its cholinergic input; "
         "at high levels the receptors desensitise",
         "acute responses depend on dopamine; repeated exposure causes locomotor hyperactivity",
         [E("rest", "ach_in", 3.0, 1.0), E("pre", "ACh", 0.6, 2.5), E("phen", "explore", 0.8, 0.5)],
         half_life_s=45.0, absorb_s=5.0, routes=("feed", "inject", "vapour"),
         refs=["Bainton et al. 2000, Curr Biol, doi:10.1016/S0960-9822(00)00336-5",
               "Ren et al. 2012, PLoS One, doi:10.1371/journal.pone.0052521"]),
    Drug("caffeine", "Caffeine", "#c08a5a",
         "in flies it works through dopamine, not the adenosine receptor: excites the PAM dopamine neurons",
         "less night-time sleep, more time awake; the PAM dopamine cluster carries the effect",
         [E("rest", "type:PAM", 12.0, 0.8), E("phen", "explore", 1.1, 0.6), E("phen", "rest", -1.2, 0.6),
          E("phen", "tremor", 0.9, 0.6)],   # 12 mV: rest above the -45 mV threshold, the PAM cells fire on their own (our size); tremor scripted
         half_life_s=150.0, absorb_s=4.0,
         refs=["Wu et al. 2009, J Neurosci, doi:10.1523/JNEUROSCI.1653-09.2009",
               "Nall et al. 2016, Sci Rep, doi:10.1038/srep20938"]),
    Drug("cocaine", "Cocaine", "#f2f2f2",
         "modelled as stronger dopamine and octopamine synapses (the model has no transporters to block)",
         "volatilised cocaine causes dose-dependent stereotyped reflexive motor responses, and sensitisation to repeat doses",
         [E("pre", "DA", 2.5, 1.0), E("pre", "OA", 2.0, 1.0),
          E("phen", "explore", 0.9, 0.5), E("phen", "circling", 1.0, 1.0), E("phen", "ataxia", 0.7, 1.6)],
         half_life_s=35.0, absorb_s=3.0, routes=("vapour", "inject", "feed"),
         refs=["McClung & Hirsh 1998, Curr Biol, doi:10.1016/S0960-9822(98)70041-7",
               "Bainton et al. 2000, Curr Biol, doi:10.1016/S0960-9822(00)00336-5"]),
    Drug("picrotoxin", "Picrotoxin", "#ff6a6a",
         "blocks the fly's GABA-A receptor (Rdl): inhibition fails",
         "Rdl is the receptor picrotoxin and cyclodiene insecticides block; what the brain does without it is left to the model",
         [E("pre", "GABA", 0.12, 1.0, 2.5), E("rest", "gaba_in", 14.0, 1.2, 3.0)],   # + the tonic GABA-A current lost
         half_life_s=90.0, absorb_s=10.0, bolus=2.5,
         refs=["ffrench-Constant et al. 1993, Nature, doi:10.1038/363449a0"]),
    Drug("isoflurane", "Isoflurane", "#8fb8ff",
         "general anaesthetic: weaker synaptic transmission everywhere and hyperpolarised neurons",
         "abolishes behavioural responsiveness; sensitivity is set by a sleep/wake circuit; flies recover when it is removed",
         [E("pre", "all", 0.45, 1.0), E("rest", "all", -2.0, 1.0),
          E("phen", "explore", -1.5, 0.7), E("phen", "knockdown", 1.0, 1.4, 4.0)],
         half_life_s=8.0, absorb_s=3.0, routes=("vapour", "inject"), vapour_rate=0.4, bolus=2.5,   # down in seconds
         refs=["Kottler et al. 2013, Curr Biol, doi:10.1016/j.cub.2013.02.021",
               "Cohen et al. 2016, eNeuro, doi:10.1523/ENEURO.0116-16.2016"]),
    Drug("octopamine", "Octopamine", "#ffa45c",
         "the insect counterpart of noradrenaline: excites octopaminergic neurons and strengthens their synapses",
         "flies without it can fly but start and sustain flight abnormally",
         [E("rest", "OA_cells", 4.0, 1.0), E("pre", "OA", 2.0, 1.0),
          E("phen", "explore", 0.6, 0.8), E("phen", "takeoff", 1.0, 1.0)],
         half_life_s=40.0, absorb_s=6.0,
         refs=["Brembs et al. 2007, J Neurosci, doi:10.1523/JNEUROSCI.2704-07.2007"]),
    Drug("gaboxadol", "Gaboxadol", "#b48cff",
         "GABA-A agonist (THIP): more inhibition, quieter central brain",
         "used to increase sleep in flies",
         [E("pre", "GABA", 1.6, 1.0), E("rest", "central", -1.5, 1.0),
          E("phen", "rest", 1.2, 0.8), E("phen", "explore", -0.9, 1.0)],
         half_life_s=120.0, absorb_s=10.0,
         refs=["Dissel et al. 2015, Curr Biol, doi:10.1016/j.cub.2015.03.027"]),
]}


def hill(c, ec50, n):
    c = max(float(c), 0.0)
    return c ** n / (c ** n + ec50 ** n) if c > 0 else 0.0


class Targets:
    """Index sets the drugs act on, resolved once for the brain's neuron order."""

    def __init__(self, brain):
        n = brain.n
        df = pd.read_csv(DATA / "neurons.csv.gz", usecols=["root_id", "group", "nt_type"]).drop_duplicates("root_id")
        df = df.set_index("root_id").reindex(brain.ids)
        nt = df["nt_type"].map(NT_CODES).fillna("").to_numpy(object)
        grp = df["group"].fillna("").astype(str).str.split(".").str[0].to_numpy(object)
        self.n = n
        self.nt = {k: np.flatnonzero(nt == k) for k in NT_CODES.values()}
        self.nt_code = np.zeros(n, np.uint8)               # 0 unknown, 1.. in NT_CODES order (for the brain view)
        for i, k in enumerate(NT_CODES.values(), 1):
            self.nt_code[self.nt[k]] = i
        self.region = {r: np.flatnonzero(np.isin(grp, g)) for r, g in REGIONS.items()}
        cl = pd.read_csv(DATA / "classification.csv.gz", usecols=["root_id", "super_class"]).drop_duplicates("root_id")
        sc = cl.set_index("root_id").reindex(brain.ids)["super_class"].fillna("").to_numpy(object)
        self.region["central"] = np.flatnonzero(sc == "central")
        self.region["all"] = np.arange(n)
        for k in ("DA", "OA"):           # 5-HT left out: most 'SER' labels fall on olfactory and visual sensory cells
            self.region[f"{k}_cells"] = self.nt[k]
        ct = pd.read_csv(DATA / "Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False,
                         usecols=["root_id", "cell_type"]).drop_duplicates("root_id").set_index("root_id")
        self.cell_type = ct.reindex(brain.ids)["cell_type"].fillna("").astype(str).to_numpy(object)
        # share of each neuron's synaptic input that is excitatory (cholinergic in this model): where a
        # nicotinic agonist finds its receptors
        pos = np.zeros(n)
        tot = np.zeros(n)
        w = brain.wts
        np.add.at(pos, brain.col, np.maximum(w, 0.0))
        np.add.at(tot, brain.col, np.abs(w))
        self.ach_in = np.where(tot > 0, pos / np.maximum(tot, 1e-9), 0.0).astype(np.float32)
        # share of each neuron's input that comes from GABAergic senders: where blocking the GABA-A receptor removes
        # tonic inhibition (picrotoxin's resting depolarisation acts here)
        pre_idx = np.repeat(np.arange(n), np.diff(brain.row_ptr))
        is_gaba = np.zeros(n, bool)
        is_gaba[self.nt["GABA"]] = True
        gab = np.zeros(n)
        np.add.at(gab, brain.col, np.where(is_gaba[pre_idx], np.abs(w), 0.0))
        self.gaba_in = np.where(tot > 0, gab / np.maximum(tot, 1e-9), 0.0).astype(np.float32)

    def resolve(self, target):
        """A region name, 'all', 'central', 'DA_cells' / 'OA_cells', or 'type:PREFIX' (cell types starting with it)."""
        if target.startswith("type:"):
            if target not in self.region:
                self.region[target] = np.flatnonzero(np.char.startswith(self.cell_type.astype(str), target[5:]))
            return self.region[target]
        return self.region.get(target, np.zeros(0, np.int64))


class Pharmacology:
    """Per fly: what is in its gut and its body, and what that does to its brain and body now."""

    UPDATE_S = 0.1          # brain gains are pushed to the GPU at most every 100 ms of sim time

    def __init__(self, targets: Targets, brain):
        self.T, self.brain = targets, brain
        self.gut = {}       # drug -> amount not yet absorbed
        self.c = {}         # drug -> body concentration (reference units)
        self.vapour = {}    # drug -> vapour level 0..1 (the chamber is filled with it)
        self.hist = deque(maxlen=240)        # (t, {drug: c}) at 2 Hz: the last two minutes
        self.pre = np.ones(targets.n, np.float32)
        self.post = np.ones(targets.n, np.float32)
        self.rest = np.zeros(targets.n, np.float32)
        self.phen = {}
        self.levers = []                     # [(label, value)] for the UI
        self._t_push = -1e9
        self._t_hist = -1e9
        self._pushed = False
        # the scripted body channel's own state (per fly)
        self.down = False                    # knocked down
        self.seizing = False
        self._ataxia = np.zeros(2)           # smoothed noise on the legs' drive
        self._circle = 1.0
        self._seiz_t = -1e9
        self._jerk_t, self._jerk = -1e9, np.zeros(2)
        self._run_t = -1e9
        self._warp = getattr(brain, "backend", "") == "warp"     # synapse gains exist on the warp backend only
        if self._warp:
            brain.set_syn_gain(self.pre, self.post)          # sober: a brain reused from the pool may carry old gains
            self._pushed = True

    # ------------------------------------------------------------------ dosing (sim thread)
    def dose(self, key, route, amount=1.0):
        d = DRUGS[key]
        if route == "inject":
            self.c[key] = self.c.get(key, 0.0) + d.bolus * amount
        elif route == "feed":
            self.gut[key] = self.gut.get(key, 0.0) + d.feed_rate * amount
        elif route == "vapour":
            self.vapour[key] = float(np.clip(amount, 0.0, 1.0))
            if self.vapour[key] <= 0:
                self.vapour.pop(key)

    def clear(self):
        self.gut.clear(); self.c.clear(); self.vapour.clear()
        self._run_t = -1e9

    def active(self, eating=()):
        """Anything to do this window (a drug in the fly, or effects still to be undone)."""
        return bool(self.c or self.gut or self.vapour or self.phen or self.levers or eating)

    def any(self):
        return bool(self.c or self.gut or self.vapour)

    # ------------------------------------------------------------------ kinetics + effects
    def tick(self, t, dt, eating_drugs=()):
        """eating_drugs: drugs in the food the fly is eating this window."""
        for k in eating_drugs:
            self.gut[k] = self.gut.get(k, 0.0) + DRUGS[k].feed_rate * dt
        for k in set(self.gut) | set(self.c) | set(self.vapour):
            d = DRUGS[k]
            g = self.gut.get(k, 0.0)
            up = g * (1.0 - np.exp(-dt / d.absorb_s))
            inhaled = self.vapour.get(k, 0.0) * d.vapour_rate * dt
            c = self.c.get(k, 0.0) * np.exp(-dt * np.log(2) / d.half_life_s) + up + inhaled
            g -= up
            if g > 1e-4:
                self.gut[k] = g
            else:
                self.gut.pop(k, None)
            if c > 1e-3 or k in self.vapour or k in self.gut:
                self.c[k] = c
            else:
                self.c.pop(k, None)
        if t - self._t_hist >= 0.5:
            self._t_hist = t
            self.hist.append((round(t, 2), {k: round(v, 3) for k, v in self.c.items()}))
        if t - self._t_push >= self.UPDATE_S:
            self._t_push = t
            self._apply()

    def effect_levels(self, key):
        d, c = DRUGS[key], self.c.get(key, 0.0)
        return [(e, hill(c, e.ec50, e.hill)) for e in d.effects]

    def _apply(self):
        T = self.T
        pre = np.ones(T.n, np.float32)
        post = np.ones(T.n, np.float32)
        rest = np.zeros(T.n, np.float32)
        phen = {}
        levers = {}
        for k in self.c:
            for e, x in self.effect_levels(k):
                if x < 1e-3:
                    continue
                if e.lever == "pre":
                    g = 1.0 + (e.full - 1.0) * x
                    if e.target == "all":
                        pre *= g
                    else:
                        pre[T.nt.get(e.target, [])] *= g
                    levers[f"{e.target} synapses"] = levers.get(f"{e.target} synapses", 1.0) * g
                elif e.lever == "post":
                    g = 1.0 + (e.full - 1.0) * x
                    post[T.resolve(e.target)] *= g
                    levers[f"input to {e.target}"] = levers.get(f"input to {e.target}", 1.0) * g
                elif e.lever == "rest":
                    mv = e.full * x
                    if e.target == "ach_in":
                        rest += (mv * T.ach_in).astype(np.float32)
                        lab = "rest, cholinergic-input cells"
                    elif e.target == "gaba_in":
                        rest += (mv * T.gaba_in).astype(np.float32)
                        lab = "rest, GABA-input cells"
                    else:
                        rest[T.resolve(e.target)] += mv
                        lab = f"rest, {e.target.replace('_cells', ' cells').replace('type:', '') + (' cells' if e.target.startswith('type:') else '')}"
                    levers[lab] = levers.get(lab, 0.0) + mv
                else:
                    phen[e.target] = phen.get(e.target, 0.0) + e.full * x
        self.phen = phen
        self.levers = [(k, round(float(v), 3)) for k, v in levers.items()]
        changed = (not np.array_equal(pre, self.pre)) or (not np.array_equal(post, self.post))
        self.pre, self.post, self.rest = pre, post, rest
        if self._warp and (changed or not self._pushed):
            self.brain.set_syn_gain(pre, post)
            self._pushed = True

    # ------------------------------------------------------------------ the body channel (scripted)
    def explore_gain(self):
        """Multiplier on the placeholder policy's walking bouts (forward rate and bout length)."""
        return float(np.clip(np.exp(self.phen.get("explore", 0.0)), 0.15, 3.0))

    def rest_gain(self):
        """Multiplier on pause length (> 1 = longer, sleepier pauses)."""
        return float(np.clip(np.exp(self.phen.get("rest", 0.0)), 0.25, 5.0))

    def ataxia(self):
        return float(np.clip(self.phen.get("ataxia", 0.0) + 0.6 * self.phen.get("tremor", 0.0), 0.0, 1.5))

    def knocked_down(self):
        return self.phen.get("knockdown", 0.0) > 0.5

    def apply_policy(self, policy):
        """Walking bouts, pauses and spontaneous take-off of the placeholder policy (autonomy.Policy)."""
        policy.drug_explore = self.explore_gain()
        policy.drug_rest = self.rest_gain()
        policy.drug_takeoff = 1.0 + 6.0 * self.phen.get("takeoff", 0.0)
        if self.phen.get("circling", 0.0) < 0.01:
            self._circle = float(policy.rng.choice([-1.0, 1.0]))     # the next bout of circling picks a side

    def shape_drive(self, drive, rng, runaway, t, window_ms=10.0):
        """The legs' drive under the drugs: vigour (speed), unsteadiness, tremor, circling; nothing when knocked
        down. runaway: the brain is in its self-sustaining state now. A drugged brain in that state convulses the
        legs: the state comes from the connectome, the coupling to the legs is ours.
        Returns (drive, True when a seizure starts and has not been reported for 5 s)."""
        if self.down:
            return np.zeros(2), False
        if self.any() and runaway:
            self._run_t = t
        seiz = self.any() and t - self._run_t < 1.0      # held 1 s: the watchdog reset dips the brain briefly
        onset = seiz and not self.seizing and t - self._seiz_t > 5.0
        if onset:
            self._seiz_t = t
        self.seizing = seiz
        if seiz:
            # convulsion (scripted coupling): the legs jerk between hard opposite drives every ~80 ms
            if t - self._jerk_t > 0.08:
                self._jerk_t = t
                self._jerk = rng.choice([-1.5, 1.5], 2) * rng.uniform(0.6, 1.0, 2)
            return np.clip(0.2 * drive + self._jerk, -1.5, 1.5), onset
        if not self.phen:
            return np.clip(drive, -1.5, 1.5), onset
        drive = drive * float(np.clip(self.explore_gain() ** 0.5, 0.3, 1.4))
        a, tr = self.ataxia(), self.phen.get("tremor", 0.0)
        if a > 0.01:
            k = 1.0 - np.exp(-window_ms / 150.0)                 # noise smoothed over ~150 ms: lurches, not jitter
            self._ataxia += k * (rng.normal(0.0, 3.0, 2) - self._ataxia)
            drive = drive + 0.6 * a * self._ataxia * (0.3 + float(np.abs(drive).mean()))
        if tr > 0.02:
            drive = drive + 0.35 * tr * rng.normal(0.0, 1.0, 2)
        circ = self.phen.get("circling", 0.0)
        if circ > 0.01:
            drive = drive + np.array([0.35, -0.35]) * self._circle * circ
        return np.clip(drive, -1.5, 1.5), onset

    def status(self):
        # called from the web thread while the sim thread changes these: copy each dict in one C-level step first
        c, gut, phen = dict(self.c), dict(self.gut), dict(self.phen)
        out = {"c": {k: round(v, 3) for k, v in c.items()},
               "gut": {k: round(v, 3) for k, v in gut.items()},
               "vapour": dict(self.vapour),
               "levers": list(self.levers),
               "phen": {k: round(v, 3) for k, v in phen.items()},
               "knocked_down": phen.get("knockdown", 0.0) > 0.5, "seizing": bool(self.seizing)}
        if self.seizing:
            out["phen"]["convulsing"] = 1.0       # the scripted leg coupling to the brain's runaway, listed with the body effects
        return out


def _check():
    """results/drug_check.json -> {drug: {conc: {loom, sugar, touch: % of control, persist: bool}}}."""
    import json
    from pathlib import Path
    f = Path(__file__).resolve().parents[1] / "results" / "drug_check.json"
    if not f.exists():
        return {}
    r = json.loads(f.read_text())
    ctl = {k: float(np.mean([x["out_hz"] for x in v])) for k, v in r["control"].items()}
    out = {}
    for key, by_c in r["drugs"].items():
        out[key] = {}
        for c, v in by_c.items():
            row = {pk: (100.0 * float(np.mean([x["out_hz"] for x in runs])) / ctl[pk] if ctl.get(pk) else None)
                   for pk, runs in v["probes"].items()}
            row["persist"] = any(x["persists"] for runs in v["probes"].values() for x in runs)
            out[key][c] = row
    return out


def catalog():
    """For the front end: the drugs, their routes and effects (static)."""
    ck = _check()
    return [{"key": d.key, "check": ck.get(d.key), "name": d.name, "color": d.color, "what": d.what, "fly": d.fly, "routes": list(d.routes),
             "half_life_s": d.half_life_s, "refs": d.refs,
             "effects": [{"lever": e.lever, "target": e.target, "full": e.full, "ec50": e.ec50} for e in d.effects]}
            for d in DRUGS.values()]
