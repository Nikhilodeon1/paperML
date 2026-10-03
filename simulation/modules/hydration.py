"""Hydration module — body-water balance.

Tracks a running body-water DEFICIT (ml). Water is lost continuously (insensible +
respiratory) and much faster during exercise (sweat, scaled by intensity); drinking fluid
repays the deficit. Dehydration then propagates through the shared state: the autonomic
module raises heart rate and drops blood pressure as plasma volume falls, and the circadian
module lowers alertness. Grounded in exercise-physiology sweat rates (~0.5-2 L/h) and the
~3-5 bpm-per-%-body-mass cardiovascular drift of dehydration (Sawka 2007).
"""

from __future__ import annotations

from ..module import Module
from ..state import BodyState
from ..params import PhysioParams
from ..inputs import InputWindow, Water


class HydrationModule(Module):
    name = "hydration"
    writes = ("water_deficit_ml",)
    reads = ("core_temp_c",)

    def on_impulse(self, s: BodyState, p: PhysioParams, event) -> None:
        if isinstance(event, Water):
            s.water_deficit_ml = max(0.0, s.water_deficit_ml - event.ml)

    def derivatives(self, s: BodyState, p: PhysioParams, w: InputWindow) -> dict[str, float]:
        insensible = 0.5                                  # ml/min baseline loss (~0.7 L/day)
        sweat = 2.0 * w.exercise_mets if w.exercise_mets > 0 else 0.0   # up to ~1 L/h at 8 METs
        sweat += 8.0 * max(0.0, s.core_temp_c - 37.3)     # thermal sweating when hot
        return {"water_deficit_ml": insensible + sweat}


def pct_dehydration(s: BodyState, p: PhysioParams) -> float:
    """Body-water deficit as a percent of body mass (the physiologically meaningful unit)."""
    return 100.0 * (s.water_deficit_ml / 1000.0) / max(1.0, p.weight_kg)
