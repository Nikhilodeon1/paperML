"""Calibration tests — the engine's quantified accuracy against reference physiology.

Fast guards that the engine's BAC matches the validated Widmark module and its
postprandial glucose tracks the healthy reference within tolerance. Full write-ups live in
evaluation/calibrate_engine_bac.py and calibrate_engine_glucose.py.
"""

from __future__ import annotations

import statistics

from simulation import Simulator, PhysioParams, Schedule, Meal, Drink
from modules.hepatic import Drink as HDrink, compute_bac


def test_engine_bac_matches_validated_widmark():
    errs = []
    for w, sex, n in [(80, "male", 2), (80, "male", 4), (60, "female", 2), (95, "male", 3)]:
        e = Simulator(PhysioParams(weight_kg=w, sex=sex)).run(
            Schedule().add(Drink(0, n)), 480, outputs=["bac_g_dl"])
        h = compute_bac([HDrink.standard(n)], weight_kg=w, sex=sex)
        errs.append(abs(max(e.series["bac_g_dl"]) - h.peak_bac))
    assert statistics.mean(errs) < 0.005            # < 0.005 g/dL peak MAE


def test_glucose_calibration_within_tolerance():
    ref = {0: 90, 30: 140, 45: 150, 60: 145, 90: 125, 120: 108, 180: 93}
    tr = Simulator(PhysioParams(weight_kg=75, sex="male", age=35)).run(
        Schedule().add(Meal(0, 75)), 200, outputs=["glucose_mg_dl"])
    mae = statistics.mean(abs(tr.at("glucose_mg_dl", t) - r) for t, r in ref.items())
    assert mae < 15.0                               # mean abs error vs reference


def test_engine_physiology_reference_ranges():
    from simulation import Caffeine, Exercise
    p = PhysioParams(weight_kg=75, sex="male", age=35, vo2max=42)
    f = Simulator(p).run(Schedule(), 60, outputs=["glucose_mg_dl", "insulin_uU_ml"]).final()
    assert 70 <= f["glucose_mg_dl"] <= 99 and 2 <= f["insulin_uU_ml"] <= 25   # fasting
    ins = max(Simulator(p).run(Schedule().add(Meal(0, 75)), 240, outputs=["insulin_uU_ml"]).series["insulin_uU_ml"])
    assert 40 <= ins <= 110                                                    # postprandial insulin
    cf = Simulator(p).run(Schedule().add(Caffeine(0, 200)), 600, outputs=["caffeine_mg"])
    assert 0.4 <= cf.at("caffeine_mg", 300) / 200 <= 0.6                       # ~5h half-life
    mx = Simulator(p).run(Schedule().add(Exercise(0, 20, p.vo2max / 3.5)), 40, outputs=["heart_rate_bpm"])
    assert 92 <= 100 * max(mx.series["heart_rate_bpm"]) / p.hr_max <= 112      # max effort ≈ HRmax


def test_ada_glucose_tolerance_behaviour():
    healthy = Simulator(PhysioParams(weight_kg=75, sex="male", age=35)).run(
        Schedule().add(Meal(0, 75)), 200, outputs=["glucose_mg_dl"]).at("glucose_mg_dl", 120)
    diabetic = Simulator(PhysioParams(weight_kg=75, sex="male", age=55,
                         insulin_sensitivity=0.35, insulin_secretion=0.08)).run(
        Schedule().add(Meal(0, 75)), 200, outputs=["glucose_mg_dl"]).at("glucose_mg_dl", 120)
    assert healthy < 140          # normal glucose tolerance
    assert diabetic >= 140        # impaired tolerance for the diabetic parameter set
