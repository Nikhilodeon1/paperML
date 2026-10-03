"""Circadian / sleep-pressure module — a two-process-style homeostat.

Sleep pressure (Process S, an adenosine proxy) builds while awake and dissipates during
sleep; alertness tracks the inverse, blunted by high sleep pressure and lifted by caffeine
(which also, when high near a sleep window, keeps pressure from clearing as fast). This is
the hook the sleep/next-day-recovery predictions read. Simplified Borbely two-process
model (Borbely 1982); caffeine-alertness link per Drake 2013.
"""

from __future__ import annotations

from ..module import Module
from ..state import BodyState
from ..params import PhysioParams
from ..inputs import InputWindow


class CircadianModule(Module):
    name = "circadian"
    writes = ("sleep_pressure", "alertness")
    reads = ("caffeine_mg", "water_deficit_ml")

    def derivatives(self, s: BodyState, p: PhysioParams, w: InputWindow) -> dict[str, float]:
        from .hydration import pct_dehydration
        if w.asleep:
            # pressure dissipates; caffeine slows the clearance
            clear = 0.0022 / (1.0 + s.caffeine_mg / 150.0)
            dS = -clear * s.sleep_pressure
        else:
            dS = 0.00045 * (1.0 - s.sleep_pressure)     # builds across the waking day
        # alertness: high when rested + caffeinated, low under sleep pressure or dehydration
        alert_target = min(1.0, max(0.0, 1.0 - 0.8 * s.sleep_pressure
                                    + 0.25 * (s.caffeine_mg / 200.0)
                                    - 0.08 * pct_dehydration(s, p)))
        dA = (alert_target - s.alertness) / 20.0
        return {"sleep_pressure": dS, "alertness": dA}
