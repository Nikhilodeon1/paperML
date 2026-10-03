"""Substrate metabolism — fasting, glycogen depletion, ketosis, gluconeogenesis.

The metabolic module handles the fed state (a meal raising glucose). This module handles
the FASTED state — what happens over hours-to-days without carbohydrate:

  1. When insulin is low (no recent carbs), the liver drips glycogen out to hold blood
     glucose up, so glycogen slowly falls.
  2. Once glycogen is depleted and insulin stays low, the body shifts to fat oxidation and
     the liver makes KETONES (beta-hydroxybutyrate) — blood ketones climb from ~0.1 to
     several mmol/L over ~1-3 days (nutritional ketosis).
  3. GLUCONEOGENESIS defends blood glucose: as glucose drifts down while fasting, the liver
     makes new glucose from protein/glycerol, so glucose settles around ~70-80 rather than
     crashing.

Eating carbohydrate raises insulin, which shuts all of this off and refills glycogen — so
ketones fall back to baseline. Directions and rough magnitudes follow classic fasting
physiology (Cahill 2006, 'Fuel metabolism in starvation'); 'strong but simplified'.
"""

from __future__ import annotations

from ..module import Module
from ..state import BodyState
from ..params import PhysioParams
from ..inputs import InputWindow

_IB = 10.0        # basal insulin


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


class SubstrateModule(Module):
    name = "substrate"
    writes = ("ketones_mmol_l", "glycogen_g", "glucose_mg_dl")
    reads = ("insulin_uU_ml", "glycogen_g", "glucose_mg_dl")

    def derivatives(self, s: BodyState, p: PhysioParams, w: InputWindow) -> dict[str, float]:
        # fasted signal: 1 when insulin is at/below basal, 0 when fed (insulin elevated)
        fasted = _clamp01((_IB + 3.0 - s.insulin_uU_ml) / 6.0)

        # 1. resting hepatic glycogenolysis while fasted (liver glycogen depletes over ~a day)
        d_glycogen = -0.14 * fasted

        # 2. ketogenesis once glycogen is running low AND still fasted; clears toward baseline
        glycogen_low = _clamp01((350.0 - s.glycogen_g) / 150.0)     # ramps in as glycogen drops
        keto_drive = fasted * glycogen_low
        d_ketones = 0.12 * keto_drive - 0.020 * (s.ketones_mmol_l - 0.1)

        # 3. glucose in a fast: a mild drift down as ketosis deepens, with gluconeogenesis
        #    defending a lower fasting setpoint (~72) so it settles ~75-82, not crashing.
        drain = -0.6 * keto_drive
        gng = 0.08 * max(0.0, 72.0 - s.glucose_mg_dl) * fasted
        return {"ketones_mmol_l": d_ketones, "glycogen_g": d_glycogen,
                "glucose_mg_dl": drain + gng}
