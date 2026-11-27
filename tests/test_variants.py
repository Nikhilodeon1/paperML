"""Tests for the Stage 2 and Stage 3 variants: regression calibration, basal glucose, isolated meals, window length,
sensor lag, the Dalla Man generator, and the configuration keys that keep their results apart."""
import numpy as np
import pytest

from evaluation import profile_lik, variants
from evaluation.cohort_data import load_cohort


# --- regression calibration (decision rule R13) -------------------------------------------------

def test_calibration_without_input_error_changes_nothing():
    logged = np.array([20.0, 45.0, 90.0])
    np.testing.assert_allclose(variants.regression_calibrate(logged, 0.0), logged)


def test_calibration_is_closer_to_the_truth_than_the_log_it_corrects():
    rng = np.random.default_rng(0)
    truth = rng.lognormal(np.log(50.0), 0.4, 4000)
    s2 = np.log1p(0.75 ** 2)
    logged = truth * rng.lognormal(-0.5 * s2, np.sqrt(s2), truth.size)
    calibrated = variants.regression_calibrate(logged, 0.75)
    assert np.mean((calibrated - truth) ** 2) < 0.6 * np.mean((logged - truth) ** 2)
    assert abs(np.mean(calibrated) / np.mean(truth) - 1.0) < 0.05
    assert np.std(np.log(calibrated)) < np.std(np.log(logged))            # shrinks toward the subject's typical load


def test_calibration_leaves_meals_without_carbohydrate_at_zero():
    out = variants.regression_calibrate(np.array([0.0, 30.0, 60.0, 90.0]), 0.25)
    assert out[0] == 0.0 and np.all(out[1:] > 0) and np.all(np.isfinite(out))


# --- basal glucose (R15) -------------------------------------------------------------------------

def test_subject_basal_is_the_median_of_the_pre_meal_means():
    def record(level):
        return {"glucose": {"values": [level] * 6 + [500.0] * 37}}
    assert variants.subject_basal([record(100.0), record(110.0), record(170.0)]) == pytest.approx(110.0)


def test_the_default_basal_is_unchanged_and_a_new_one_moves_the_resting_level():
    import jax.numpy as jnp
    from personalization.subject_loss import base_params
    from simulation import jax_engine
    base = base_params(load_cohort("cgmacros", min_meals=10)[0].profile)
    assert float(base.basal_glucose) == 90.0
    _, resting = jax_engine.run_meal(base, 0.0)
    assert float(resting[0]) == pytest.approx(90.0)
    raised = variants.with_basal(base, 130.0)
    _, high = jax_engine.run_meal(raised, 0.0)
    assert float(high[0]) == pytest.approx(130.0) and abs(float(high[-1]) - 130.0) < 3.0
    del jnp


def test_the_canonical_engine_takes_the_same_basal():
    from personalization.subject_loss import base_params
    from simulation.jax_engine_canonical import CanonicalParams, run_meal
    base = variants.with_basal(base_params(load_cohort("cgmacros", min_meals=10)[0].profile), 120.0)
    glucose = run_meal(CanonicalParams.from_jax_params(base), 0.0)[1]
    assert float(glucose[0]) == pytest.approx(120.0) and abs(float(glucose[-1]) - 120.0) < 3.0


# --- isolated meals (R14) ------------------------------------------------------------------------

def test_meal_gaps_file_lines_up_with_the_cached_meals():
    gaps = variants.meal_gaps()
    for subject in load_cohort("cgmacros", min_meals=10):
        assert len(gaps[subject.subject_id]) == subject.n_meals, subject.subject_id


def test_isolation_keeps_only_meals_after_a_long_enough_gap():
    subject = load_cohort("cgmacros", min_meals=10)[0]
    gaps = variants.meal_gaps()[subject.subject_id]
    kept = variants.isolate(subject, 4.0)
    assert kept.n_meals == sum(g is None or g > 4.0 for g in gaps) < subject.n_meals


def test_units_drop_subjects_left_with_too_few_isolated_meals():
    base = {**profile_lik.default_config()}
    assert profile_lik.units({**base, "min_gap_h": 6.0}) == profile_lik.units(base)      # all 45 keep 10 or more
    every = profile_lik.units({**base, "min_meals": 14})
    isolated = profile_lik.units({**base, "min_meals": 14, "min_gap_h": 6.0})
    assert 0 < len(isolated) < len(every)
    for subject in load_cohort("cgmacros", min_meals=14):
        assert (variants.isolate(subject, 6.0).n_meals >= 14) == (subject.subject_id in isolated)


# --- window length -------------------------------------------------------------------------------

def test_a_window_override_reaches_the_observation_window():
    from evaluation.subject_source import get_subject, window_for
    config = {**profile_lik.default_config(), "window_min": 60.0}
    unit = profile_lik.units(config)[0]
    assert window_for(get_subject(config, unit)).post_min == 60.0


# --- sensor lag (R16) ----------------------------------------------------------------------------

def test_sensor_lag_leaves_a_constant_alone_and_delays_a_step():
    import jax.numpy as jnp
    flat = variants.sensor_lag(jnp.full(121, 100.0), 1.0, 10.0)
    np.testing.assert_allclose(flat, 100.0)
    step = jnp.concatenate([jnp.zeros(1), jnp.ones(120)])
    lagged = np.asarray(variants.sensor_lag(step, 1.0, 10.0))
    assert lagged[10] == pytest.approx(1.0 - np.exp(-1.0), abs=0.05) and lagged[-1] > 0.999


# --- Dalla Man generator (R11) -------------------------------------------------------------------

def test_dalla_man_replica_has_a_tau1_truth_only_and_exact_carbohydrate():
    from evaluation.subject_source import get_subject, replica_truth, truth_in_coordinates
    config = {**profile_lik.default_config(), "replica": {"seed": 0, "cgm": False, "carb_cv": 0.0,
                                                          "generator": "dalla_man"}}
    unit = profile_lik.units(config)[0]
    real = next(s for s in load_cohort("cgmacros", min_meals=10) if s.subject_id == unit)
    made = get_subject(config, unit)
    assert [r["carbs_g"] for r in made.records] == [r["carbs_g"] for r in real.records]
    assert all(r["fat_g"] == 0.0 and r["fiber_g"] == 0.0 for r in made.records)
    truth = truth_in_coordinates(replica_truth(config, unit)["theta_true"])
    assert set(truth) == {"log_tau1"} and 30.0 < truth["log_tau1"] < 200.0
    assert np.ptp(made.records[0]["glucose"]["values"]) > 5.0


# --- results stay apart --------------------------------------------------------------------------

@pytest.mark.parametrize("key,value", [("basal", "subject"), ("min_gap_h", 4.0), ("window_min", 60.0),
                                       ("sensor_lag_min", 10.0)])
def test_the_matcher_separates_every_new_key(key, value):
    config = profile_lik.default_config()
    payload = {"objective": "iauc", "bounds_scale": 1.0, "polish": None, "inner_steps": 100, "replica": None,
               "carb_scale": None}
    assert profile_lik._matches(payload, config)
    assert not profile_lik._matches(payload, {**config, key: value})
    assert profile_lik._matches({**payload, key: value}, {**config, key: value})
