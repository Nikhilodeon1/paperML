"""Stress module — the cortisol axis.

An acute stressor pushes cortisol up; cortisol decays back to a circadian baseline with a
~1-2 h time constant. The output (cortisol) is read by the metabolic module (hepatic
glucose output) and the autonomic module (HR/BP), so 'stress' propagates through the body
rather than being a standalone number. Direction/magnitude grounded in HPA-axis physiology
(Segerstrom & Miller 2004); simplified single-compartment.
"""

from __future__ import annotations

import math

from ..module import Module
from ..state import BodyState
from ..params import PhysioParams
from ..inputs import InputWindow

_BASE = 11.0     # mean cortisol (ug/dL)


def diurnal_cortisol(hour: float) -> float:
    """Circadian cortisol baseline: peaks in the morning (~8am), troughs late evening —
    the well-known diurnal rhythm. Cosine approximation, floored at a physiological low."""
    return max(4.0, 9.5 + 5.0 * math.cos(2.0 * math.pi * (hour - 8.0) / 24.0))


class StressModule(Module):
    name = "stress"
    writes = ("cortisol_ug_dl",)
    reads = ()

    def derivatives(self, s: BodyState, p: PhysioParams, w: InputWindow) -> dict[str, float]:
        target = diurnal_cortisol(w.clock_hour)      # circadian baseline for the time of day
        drive = 0.16 * w.stress_level                # acute stressor raises cortisol on top
        decay = 0.020 * (s.cortisol_ug_dl - target)  # ~50 min return time constant
        return {"cortisol_ug_dl": drive - decay}
