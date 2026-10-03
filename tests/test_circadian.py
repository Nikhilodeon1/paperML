"""Circadian glucose tolerance — mechanism present, OFF by default.

Validation (evaluation/circadian_validation.py) found it does NOT reduce held-out iAUC error on
CGMacros, so the live default is amp=0. These tests lock in: default off (so the reference
calibrations are untouched), and the mechanism moves glucose the right direction when enabled
(so it's available for other data / absolute-glucose use).
"""

from __future__ import annotations

from simulation import Simulator, PhysioParams, Schedule, Meal
from simulation.modules.metabolic import _circadian_resistance


def _peak(hour, amp):
    p = PhysioParams.from_profile({"weight_kg": 80, "height_cm": 178, "age": 45, "sex": "male"})
    p.circadian_amp = amp
    s = Schedule(); s.add(Meal(30, carbs_g=75))
    tr = Simulator(p).run(s, duration_min=210, dt=1.0, record_every=5,
                          outputs=["glucose_mg_dl"], start_hour=hour - 0.5)
    return max(tr.series["glucose_mg_dl"])


def test_default_is_off():
    assert PhysioParams().circadian_amp == 0.0


def test_off_means_time_of_day_has_no_effect():
    """amp=0 -> a morning and an evening meal are identical (reference calibrations unchanged)."""
    assert _peak(8, 0.0) == _peak(20, 0.0)


def test_enabled_makes_evening_meals_hit_harder():
    assert _peak(20, 0.3) > _peak(8, 0.3)


def test_resistance_shape():
    assert _circadian_resistance(6) == 0.0            # morning = best
    assert _circadian_resistance(22) == 1.0           # late evening = worst
    assert _circadian_resistance(6) < _circadian_resistance(14) < _circadian_resistance(22)
