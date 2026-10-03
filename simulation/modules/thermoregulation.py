"""Thermoregulation — core body temperature.

Core temperature is the balance of heat produced (baseline metabolism + a lot during
exercise) against heat lost to the environment (bigger when it's cold, smaller when it's
hot) and by sweating. It relaxes toward a target set by exercise intensity, ambient
temperature, and hydration — dehydration blunts sweating, so a dehydrated body runs hotter.

Core temperature then feeds back into the shared state: hyperthermia drives heart rate up
(cardiovascular drift) and increases sweat loss (via the hydration module reading core
temp). Directions and magnitudes follow exercise thermophysiology (Sawka 2011); simplified
lumped model.
"""

from __future__ import annotations

from ..module import Module
from ..state import BodyState
from ..params import PhysioParams
from ..inputs import InputWindow


class ThermoregulationModule(Module):
    name = "thermoregulation"
    writes = ("core_temp_c",)
    reads = ("water_deficit_ml",)

    def derivatives(self, s: BodyState, p: PhysioParams, w: InputWindow) -> dict[str, float]:
        from .hydration import pct_dehydration
        dehy = pct_dehydration(s, p)                     # % body-mass water deficit
        # equilibrium core temp: exercise raises it, hot/cold environment shifts it,
        # dehydration adds (impaired evaporative cooling).
        target = (37.0 + p.luteal_temp_offset                # luteal-phase basal temp rise
                  + 0.17 * w.exercise_mets
                  + 0.05 * max(0.0, w.ambient_c - 27.0)     # heat load above ~27 C
                  - 0.04 * max(0.0, 14.0 - w.ambient_c)     # cold below ~14 C
                  + 0.18 * dehy)
        tau = 18.0 if w.exercise_mets > 0 else 30.0        # heats faster than it cools
        return {"core_temp_c": (target - s.core_temp_c) / tau}
