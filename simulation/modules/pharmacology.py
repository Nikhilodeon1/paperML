"""Pharmacology module — a DATA-DRIVEN, open substance framework.

Every drug/substance in `knowledge_base/substances.json` gets a one-compartment PK model
(gut -> blood absorption, first-order elimination by half-life) and a PD map of how its
blood level nudges shared BodyState variables. Because substances live in an open dict on
the state, adding a new drug is a JSON edit — the engine can then simulate "what does <X>
do to my body" for substances it was never explicitly coded for. This is the extensibility
that lets the simulator address stranger, unplanned questions.

PD effects are applied as RATE nudges on variables that have restoring dynamics (heart
rate, blood pressure, alertness, sleep pressure, cortisol): a sustained drug level holds a
sustained shift, which fades as the drug clears. Effects are graded weak/moderate and are
NOT medical advice.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from ..module import Module
from ..state import BodyState
from ..params import PhysioParams
from ..inputs import InputWindow, Dose

_PATH = Path(__file__).resolve().parents[2] / "knowledge_base" / "substances.json"
# Per-target restoring "gain" = the effective time-constant of each variable's own
# restoring dynamics (set by the module that governs it). Multiplying a rate push by this
# gives the steady-state level shift, so effects in substances.json can be written in the
# variable's REAL units (bpm, mmHg, ug/dL, 0-1 alertness) and converted here. Only
# variables with genuine restoring dynamics are drivable (no runaway accumulation).
_GAIN = {"heart_rate_bpm": 1.6, "sbp_mmhg": 1.5, "dbp_mmhg": 1.5,
         "alertness": 20.0, "cortisol_ug_dl": 50.0}


@lru_cache(maxsize=1)
def substance_defs() -> dict:
    return json.loads(_PATH.read_text(encoding="utf-8"))["substances"]


def known_substances() -> list[str]:
    return sorted(substance_defs())


class PharmacologyModule(Module):
    name = "pharmacology"
    writes = ("heart_rate_bpm", "sbp_mmhg", "dbp_mmhg", "alertness", "cortisol_ug_dl")
    reads = ()

    def on_impulse(self, s: BodyState, p: PhysioParams, event) -> None:
        if isinstance(event, Dose):
            defs = substance_defs()
            key = str(event.substance).strip().lower()
            if key not in defs:
                return
            mg = event.mg if event.mg and event.mg > 0 else defs[key].get("typical_dose_mg", 1.0)
            s.substances[key + "__gut"] = s.substances.get(key + "__gut", 0.0) + float(mg)

    def derivatives(self, s: BodyState, p: PhysioParams, w: InputWindow) -> dict[str, float]:
        if not s.substances:
            return {}
        defs = substance_defs()
        rates: dict[str, float] = {}
        for name, d in defs.items():
            gut = s.substances.get(name + "__gut", 0.0)
            blood = s.substances.get(name, 0.0)
            if gut <= 1e-9 and blood <= 1e-9:
                continue
            kabs = math.log(2.0) / max(1.0, d.get("absorption_min", 15))
            kelim = math.log(2.0) / max(1.0, d.get("half_life_min", 120))
            absorb = kabs * gut
            rates["substances." + name + "__gut"] = -absorb
            rates["substances." + name] = absorb - kelim * blood
            # PD: normalize blood level to 'standard doses'; each `coef` is the target's
            # steady-state shift in its own units at one standard dose (converted to a rate
            # push via the variable's restoring gain).
            norm = blood / max(1e-6, d.get("typical_dose_mg", 1.0))
            for target, coef in (d.get("effects") or {}).items():
                if target in _GAIN:
                    rates[target] = rates.get(target, 0.0) + (coef * norm) / _GAIN[target]
        return rates
