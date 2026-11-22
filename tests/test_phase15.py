"""Phase 15 (Amendment 11): the pieces that can be tested without result files."""
from __future__ import annotations

import numpy as np

from evaluation import hypotheses, phase15_summary as p15, subject_source


def test_residual_gut_breaks_the_exchange_symmetry_and_restores_the_default():
    import equinox as eqx
    import jax.numpy as jnp
    from simulation import jax_engine as engine

    base = engine.JaxPhysioParams.from_population_defaults()

    def glucose(ke, ka):
        p = eqx.tree_at(lambda x: (x.gastric_emptying, x.carb_absorption), base, (jnp.float32(ke), jnp.float32(ka)))
        return np.asarray(engine.run_meal(p, 60.0, 0.0, 0.0, meal_time=30.0, duration_min=120.0, step_min=5.0)[1])

    assert engine.RESIDUAL_GUT_FRACTION == 0.0
    assert np.abs(glucose(0.030, 0.016) - glucose(0.016, 0.030)).max() < 0.05
    with engine.residual_gut(0.10):
        assert engine.RESIDUAL_GUT_FRACTION == 0.10
        assert np.abs(glucose(0.030, 0.016) - glucose(0.016, 0.030)).max() > 1.0
    assert engine.RESIDUAL_GUT_FRACTION == 0.0


def test_replica_arguments_separate_residual_and_perturbation():
    config = {"cohort": "cgmacros", "min_meals": 10, "limit": None}
    base = {"seed": 0, "cgm": True, "carb_cv": 0.0}
    a = subject_source._replica_args(config, base, "CGMacros-001")
    b = subject_source._replica_args(config, {**base, "residual_gut": 0.1}, "CGMacros-001")
    c = subject_source._replica_args(config, {**base, "perturb": {"x_gain": 1.3}}, "CGMacros-001")
    assert a != b and a != c and b != c
    assert a[-3:] == (0.0, "", 0.0) and b[-3] == 0.1 and "x_gain" in c[-2]


def _block(bounded_ke, cover_ke, bounded_ka, cover_ka) -> dict:
    def one(b, c):
        return {"bounded_interior": {"fraction": b, "bounded": 1, "n": 2}, "nominal_coverage": {"fraction": c}}
    return {"n": 45, "k_e": one(bounded_ke, cover_ke), "k_a": one(bounded_ka, cover_ka)}


def test_h29_needs_bounded_and_calibrated_for_both_rates():
    key = "seed0_0.1"
    assert p15.h29({key: _block(0.6, 0.95, 0.5, 0.9)})["status"] == "met"
    assert p15.h29({key: _block(0.6, 0.95, 0.4, 0.95)})["status"] == "not met"
    assert p15.h29({key: _block(0.9, 0.95, 0.9, 0.80)})["status"] == "not met"
    assert p15.h29({})["status"] == "not evaluated"


def test_h30_rule_is_ninety_percent_within_delta():
    assert p15.h30({"quarter": {"n": 45, "optimizer_gap_fraction": 0.93, "max_optimizer_gap": 3.0}})["status"] == "met"
    assert p15.h30({"quarter": {"n": 45, "optimizer_gap_fraction": 0.60, "max_optimizer_gap": 9.0}})["status"] == "not met"
    assert p15.h30({})["status"] == "not evaluated"


def test_covered_of_bounded_counts_only_bounded_profiles_with_a_truth():
    rows = [{"profiles": {"k": {"truth_covered": True}}}, {"profiles": {"k": {"truth_covered": False}}},
            {"profiles": {"k": {}}}]
    out = p15._covered_of_bounded(rows, "k")
    assert out == {"n": 2, "covered": 1, "fraction": 0.5}
    assert p15._covered_of_bounded([], "k")["n"] == 0


def test_h29_and_h30_are_not_evaluated_without_results():
    for fn in (hypotheses.h29, hypotheses.h30):
        assert fn({})["status"] == "not evaluated"
    met = hypotheses.h30({"phase15": {"h30": {"status": "met", "fraction": 0.95, "max_gap": 1.0}}})
    assert met["status"] == "met"
