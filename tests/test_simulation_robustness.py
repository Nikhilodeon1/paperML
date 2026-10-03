"""Fast robustness tripwires for the simulation engine + parameter clamping.

The exhaustive 600-run fuzz lives in evaluation/backtest_sim_robustness.py; these keep a
quick guard that abusive inputs are absorbed (clamped, bounded, never crash/NaN).
"""

from __future__ import annotations

import math

import pytest

from simulation import Simulator, PhysioParams, Schedule, Meal, Drink, Caffeine, Exercise
from simulation.state import BodyState


@pytest.mark.parametrize("weight,age,sex", [
    (0, 35, "male"), (3, 35, "male"), (1e5, 35, "male"),
    (float("nan"), 35, "male"), (80, -5, "x"), (80, 500, "female"),
])
def test_bad_params_are_clamped_not_crashing(weight, age, sex):
    p = PhysioParams(weight_kg=weight, age=age, sex=sex)
    assert 30 <= p.weight_kg <= 400
    assert 5 <= p.age <= 110
    assert p.sex in ("male", "female")
    tr = Simulator(p).run(Schedule().add(Meal(0, 60)), 120)   # must not raise
    assert all(math.isfinite(v) for xs in tr.series.values() for v in xs)


@pytest.mark.parametrize("sch,dur", [
    (Schedule().add(Meal(0, 500)), 240),
    (Schedule().add(Exercise(0, 120, 50)), 180),
    (Schedule().add(Caffeine(0, 5000)), 300),
    (Schedule().add(Drink(0, 30)), 600),
    (Schedule().add(Meal(0, 60)), 100000),
])
def test_abusive_scenarios_stay_within_bounds(sch, dur):
    tr = Simulator(PhysioParams(weight_kg=80)).run(sch, dur)
    for var, xs in tr.series.items():
        assert all(math.isfinite(v) for v in xs)
        if var in BodyState._BOUNDS:
            lo, hi = BodyState._BOUNDS[var]
            assert all(lo - 1e-6 <= v <= hi + 1e-6 for v in xs), var


def test_duration_and_dt_are_capped():
    tr = Simulator(PhysioParams(weight_kg=80)).run(Schedule(), 10**9, dt=0.0001)
    assert tr.times_min[-1] <= 14 * 24 * 60 + 1     # duration capped to 14 days


def test_determinism():
    a = Simulator(PhysioParams(weight_kg=80)).run(Schedule().add(Meal(0, 60)), 180).series
    b = Simulator(PhysioParams(weight_kg=80)).run(Schedule().add(Meal(0, 60)), 180).series
    assert a == b


def test_hardening_preserves_normal_accuracy():
    tr = Simulator(PhysioParams(weight_kg=80, age=35)).run(Schedule().add(Meal(0, 60)), 240)
    g = tr.series["glucose_mg_dl"]
    assert 120 <= max(g) <= 190                     # unchanged by the clamps
    assert 40 <= tr.times_min[g.index(max(g))] <= 70
