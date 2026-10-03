"""SBI scaffold — summary stats, the simulator, and amortized point recovery (no torch)."""

from __future__ import annotations

import numpy as np
import pytest

from personalization.npe import (AmortizedEstimator, FEATURE_NAMES, PARAM_NAMES, PRIOR,
                                 SUMMARY_NAMES, generate_training_set, sample_prior,
                                 simulate_summary, summary_stats)


def test_summary_stats_on_a_known_curve():
    # baseline 90 for 6 samples, then a rectangle +20 for the 3h window
    vals = [90] * 6 + [110] * 37
    s = summary_stats(vals, 0.0, 5.0, meal_t_min=30.0)
    assert dict(zip(SUMMARY_NAMES, s))["iauc"] == pytest.approx(3600.0, abs=1)   # 20*180
    assert dict(zip(SUMMARY_NAMES, s))["peak"] == pytest.approx(110.0)
    assert dict(zip(SUMMARY_NAMES, s))["baseline"] == pytest.approx(90.0)


def test_simulator_returns_finite_stats_across_the_prior():
    rng = np.random.default_rng(0)
    for theta in sample_prior(8, rng):
        s = simulate_summary(theta, carbs=60.0)
        assert s.shape == (5,) and np.isfinite(s).all()


def test_lower_si_gives_bigger_iauc():
    """Sanity: a more insulin-resistant person has a larger glucose excursion for one meal."""
    resistant = simulate_summary(np.array([0.4, 0.03, 0.02]), 75.0)[0]
    sensitive = simulate_summary(np.array([1.4, 0.03, 0.02]), 75.0)[0]
    assert resistant > sensitive


def test_training_set_shape_and_prior_bounds():
    theta, x = generate_training_set(60, seed=2)
    assert theta.shape[1] == 3 and x.shape[1] == len(FEATURE_NAMES)   # stats + carbs + demog
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = PRIOR[name]
        assert theta[:, j].min() >= lo - 1e-9 and theta[:, j].max() <= hi + 1e-9


@pytest.mark.parametrize("param,min_r2", [("insulin_sensitivity", 0.6)])
def test_amortized_estimator_recovers_insulin_sensitivity(param, min_r2):
    """Si must be recoverable from the summary stats — the load-bearing claim of the
    summary-stat route. (The timing pair is a known ridge and is not asserted here.)"""
    theta, x = generate_training_set(1500, seed=3)   # more, since demographics add variance
    k = len(theta) * 4 // 5
    est = AmortizedEstimator().fit(theta[:k], x[:k])
    pred = est.predict(x[k:])
    j = PARAM_NAMES.index(param)
    yt, yp = theta[k:, j], pred[:, j]
    r2 = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
    assert r2 > min_r2, f"{param} R2={r2:.2f} below {min_r2}"


def test_infer_from_meals_returns_all_params():
    theta, x = generate_training_set(400, seed=4)
    est = AmortizedEstimator().fit(theta, x)
    # a fake logged meal with a plausible curve
    meal = {"carbs_g": 60, "glucose": {"values": [90] * 6 + [140] * 20 + [100] * 17,
                                       "t0_min": 0.0, "step_min": 5.0, "meal_t_min": 30.0}}
    out = est.infer_from_meals([meal])
    assert set(out) == set(PARAM_NAMES) and all(np.isfinite(v) for v in out.values())
