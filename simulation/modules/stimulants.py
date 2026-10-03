"""Caffeine module — one-compartment pharmacokinetics.

A coffee deposits caffeine; it clears first-order with a personal half-life (~5 h).
Plasma caffeine is read by the autonomic module (raises HR/BP), the metabolic module
(mild hepatic glucose output), and the circadian module (blocks sleep pressure / raises
alertness). Grounded in caffeine PK (half-life 3-6 h; Drake 2013 for the sleep link).
"""

from __future__ import annotations

import math

from ..module import Module
from ..state import BodyState
from ..params import PhysioParams
from ..inputs import InputWindow, Caffeine


class CaffeineModule(Module):
    name = "caffeine"
    writes = ("caffeine_mg",)
    reads = ()

    def on_impulse(self, s: BodyState, p: PhysioParams, event) -> None:
        if isinstance(event, Caffeine):
            s.caffeine_mg += event.mg

    def derivatives(self, s: BodyState, p: PhysioParams, w: InputWindow) -> dict[str, float]:
        k = math.log(2.0) / p.caffeine_halflife_min
        return {"caffeine_mg": -k * s.caffeine_mg}
