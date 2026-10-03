"""Integrator convergence + physiological ceilings — the class of bug the calibration
band was wide enough to hide.

Two failures lived here undetected and are locked out by these tests:

  1. NUMERICAL — the autonomic module relaxes HR with tau=0.8 min while the engine steps at
     dt=1.0, so the naive rate `(target-x)/tau` had dt/tau=1.25 and the explicit-Euler
     update factor (1 - dt/tau) went negative: HR overshot by 65 bpm for one step at
     exercise onset and reported a 225 bpm phantom peak. Fixed with an exact exponential
     step; these tests assert the trajectory is now dt-independent.

  2. STRUCTURAL — caffeine/cortisol/heat/hypo drivers were summed straight on top of
     `hr_rest + (hr_max-hr_rest)*ex_frac`, so a 40-year-old (HRmax 180) hit 220 bpm even
     with a perfect integrator. A better solver would only have made that wrong number more
     precise, which is why a dt-convergence test alone is not enough — we assert the
     physiological ceiling too.
"""

from __future__ import annotations

import pytest

from simulation import (Simulator, PhysioParams, Schedule, Meal, Exercise, Caffeine, Drink)


def _params():
    return PhysioParams.from_profile({"weight_kg": 80, "height_cm": 178, "age": 40, "sex": "male"})


def _scenario():
    s = Schedule()
    s.add(Meal(0, carbs_g=75))                        # OGTT-like glucose excursion
    s.add(Caffeine(60, mg=200))
    s.add(Drink(90, standard_drinks=2))
    s.add(Exercise(120, 150, intensity_mets=9))       # step change -> the stiff transition
    return s


# Euler is 1st-order, so the gap to a 10x-refined run bounds the error at dt=1.0.
# Tolerances are well inside physiological noise; HR was 65.8 bpm before the fix.
@pytest.mark.parametrize("var,tol", [
    ("heart_rate_bpm", 8.0),      # bpm
    ("hrv_rmssd_ms", 2.0),        # ms
    ("sbp_mmhg", 4.0),            # mmHg
    ("glucose_mg_dl", 4.0),       # mg/dL
    ("insulin_uU_ml", 4.0),       # uU/mL
    ("core_temp_c", 0.1),         # degC
])
def test_trajectory_is_step_size_independent(var, tol):
    p = _params()
    ref = Simulator(p).run(_scenario(), duration_min=240, dt=0.1, record_every=10)
    base = Simulator(p).run(_scenario(), duration_min=240, dt=1.0, record_every=1)
    err = max(abs(base.at(var, t) - ref.at(var, t)) for t in base.times_min)
    assert err < tol, f"{var}: dt=1.0 differs from dt=0.1 by {err:.3f} (> {tol})"


def test_heart_rate_never_exceeds_hr_max():
    """Sub-maximal drivers must approach HRmax asymptotically, never stack through it.
    (No drugs here: `pharmacology` also writes HR and is summed on top by design.)"""
    p = _params()
    for mets in (6, 9, 12, 20):                        # 20 METs is absurd on purpose
        s = Schedule()
        s.add(Exercise(30, 90, intensity_mets=mets))
        s.add(Caffeine(0, mg=400))                     # a large stimulant load on top
        tr = Simulator(p).run(s, duration_min=120, dt=0.1, record_every=10)
        peak = max(tr.series["heart_rate_bpm"])
        assert peak <= p.hr_max + 1e-6, f"{mets} METs -> {peak:.1f} bpm exceeds HRmax {p.hr_max}"


def test_resting_caffeine_response_still_calibrated():
    """The ceiling must not distort behaviour where there is plenty of headroom:
    ~+8-12 bpm per 200 mg caffeine at rest (Benowitz)."""
    p = _params()
    s = Schedule()
    s.add(Caffeine(0, mg=200))
    tr = Simulator(p).run(s, duration_min=180, dt=1.0)
    rise = max(tr.series["heart_rate_bpm"]) - p.hr_rest
    assert 5.0 <= rise <= 20.0, f"resting caffeine HR rise {rise:.1f} bpm out of range"


def test_no_overshoot_at_exercise_onset():
    """The old ringing produced a one-step spike ABOVE the eventual steady state."""
    p = _params()
    s = Schedule()
    s.add(Exercise(60, 120, intensity_mets=9))
    tr = Simulator(p).run(s, duration_min=180, dt=1.0)
    onset = max(tr.at("heart_rate_bpm", t) for t in (61, 62, 63))
    settled = tr.at("heart_rate_bpm", 115)
    assert onset <= settled + 2.0, (
        f"HR spiked to {onset:.1f} at onset vs {settled:.1f} settled — integrator ringing")
