"""SMC joint parameter fit — recovery, posterior correlation, and teeth."""

from __future__ import annotations

import numpy as np
import pytest

from personalization.particle_fit import (PARAM_PRIORS, _sse, fit, synth_days_multi)

_TRUTH = {"insulin_sensitivity": 0.55, "gastric_emptying": 0.030, "carb_absorption": 0.020}


@pytest.fixture(scope="module")
def days():
    return synth_days_multi(n_days=4, truth=_TRUTH, seed=0)


@pytest.fixture(scope="module")
def posterior(days):
    return fit(days, n_particles=100, rounds=4, dt=2.0, seed=1)


def _rmse(theta, days):
    n = sum(len(d["observations"]["glucose"]["values"]) for d in days)
    return (_sse(theta, days, dt=2.0, step_min=5.0) / n) ** 0.5


def test_insulin_sensitivity_is_identifiable(posterior):
    """Si sets clearance depth and IS well-constrained by CGM — the most clinically important
    knob, and the one currently hardcoded (x0.45) in production. Fitted point lands near truth."""
    assert abs(posterior.mean()["insulin_sensitivity"] - 0.55) < 0.18


def test_timing_pair_is_predictively_recovered_even_if_not_point_identified(posterior, days):
    """gastric_emptying/carb_absorption both control curve TIMING and trade off, so neither is
    point-identified from CGM alone — an honest, correct result, not a failure. What the fit
    DOES recover is their combined effect: the posterior mean predicts glucose near the CGM
    noise floor (sigma=10 mg/dL)."""
    assert _rmse(posterior.mean(), days) < 15.0            # ~ noise floor + small model gap


def test_posterior_captures_the_timing_ridge_as_correlation(posterior):
    """The trade-off shows up as posterior correlation — exactly what a scalar/conjugate fit
    (fitting each param independently) is structurally incapable of representing."""
    c = posterior.corr("gastric_emptying", "carb_absorption")
    assert abs(c) > 0.2, f"ridge not captured as correlation (corr={c:.2f})"


def test_posterior_captures_parameter_correlation(posterior):
    """The whole point of a joint fit: it recovers that params co-vary. A scalar/conjugate
    fit reports each independently and can never produce this."""
    c = posterior.corr("insulin_sensitivity", "gastric_emptying")
    assert abs(c) > 0.15, f"no correlation structure recovered (corr={c:.2f})"


def test_posterior_is_a_cloud_not_a_point(posterior):
    """SMC yields a distribution with real spread — uncertainty, not a false point estimate."""
    s = posterior.sd()
    assert all(v > 0 for v in s.values())


def test_teeth_wrong_theta_has_higher_error_than_truth(days):
    """A validator that cannot fail is worthless: a clearly wrong theta must fit the CGM
    worse than the truth, otherwise recovery above means nothing."""
    good = _sse(_TRUTH, days, dt=2.0, step_min=5.0)
    bad = _sse({"insulin_sensitivity": 1.4, "gastric_emptying": 0.015,
                "carb_absorption": 0.032}, days, dt=2.0, step_min=5.0)
    assert bad > good, "wrong parameters fit as well as the truth — likelihood is flat/broken"


def test_fit_beats_the_uninformed_default(days, posterior):
    """The fitted posterior mean must predict the CGM better than the population default
    (all params at their engine defaults)."""
    fitted = _sse(posterior.mean(), days, dt=2.0, step_min=5.0)
    default = _sse({}, days, dt=2.0, step_min=5.0)         # {} -> from_profile defaults
    assert fitted < default


def test_synth_days_have_noise_and_shape():
    d = synth_days_multi(n_days=2, seed=3)
    vals = d[0]["observations"]["glucose"]["values"]
    assert len(vals) > 250 and max(vals) > min(vals)      # a real glucose excursion, noisy
