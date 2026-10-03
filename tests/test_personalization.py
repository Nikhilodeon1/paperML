"""Tests for Layer 3 Bayesian personalization."""

from __future__ import annotations

import numpy as np
import pytest

from knowledge_base import load_system
from modules.metabolic import _rmr_mifflin, _simulate_weight, project_weight
from personalization.gaussian_bayes import GaussianPosterior
from personalization.metabolic import (
    Weighin,
    estimate_rmr_multiplier,
    personal_params_from_logs,
)


# --- generic updater --------------------------------------------------------

def test_update_moves_toward_observation_and_shrinks_variance():
    post = GaussianPosterior(mean=1.0, var=0.01)
    new = post.update(0.8, obs_var=0.01)
    assert 0.8 < new.mean < 1.0      # pulled toward the observation
    assert new.var < post.var        # more certain


def test_repeated_consistent_observations_converge():
    post = GaussianPosterior(mean=1.0, var=0.01)
    for _ in range(50):
        post = post.update(0.85, obs_var=0.02)
    assert post.mean == pytest.approx(0.85, abs=0.02)
    assert post.sd < 0.05


def test_process_var_widens():
    post = GaussianPosterior(mean=1.0, var=0.01)
    assert post.predict(process_var=0.005).var == pytest.approx(0.015)


def test_rejects_nonpositive_obs_var():
    with pytest.raises(ValueError):
        GaussianPosterior(1.0, 0.01).update(0.9, obs_var=0.0)


# --- metabolic personalization ---------------------------------------------

def _synthetic_logs(true_m, height, age, sex, activity, intake, n_weighins,
                    spacing_days, noise_kg, seed):
    """Generate weigh-ins from a user whose real RMR multiplier is `true_m`."""
    kb = load_system("metabolic")
    pal = kb[f"pal_{activity}"].value
    kcal = kb["kcal_per_kg_fat"].value
    rng = np.random.default_rng(seed)
    days = np.arange(1, n_weighins * spacing_days + 1)
    true_traj = _simulate_weight(80.0, height, age, sex, intake, true_m, pal, kcal, days, kb)

    logs = [Weighin(day=0.0, weight_kg=80.0 + rng.normal(0, noise_kg),
                    mean_daily_intake_kcal=intake)]
    for i in range(1, n_weighins):
        idx = i * spacing_days - 1
        logs.append(Weighin(
            day=float(i * spacing_days),
            weight_kg=float(true_traj[idx] + rng.normal(0, noise_kg)),
            mean_daily_intake_kcal=intake,
        ))
    return logs


def test_recovers_known_multiplier():
    # Bi-weekly weigh-ins over ~9 months: enough spacing that weight noise (which
    # scales as 1/dt^2 in the observation variance) no longer dominates the prior.
    true_m = 0.88
    logs = _synthetic_logs(true_m, height=178, age=35, sex="male", activity="sedentary",
                           intake=2300, n_weighins=18, spacing_days=14, noise_kg=0.4, seed=1)
    post = estimate_rmr_multiplier(logs, height_cm=178, age=35, sex="male",
                                   activity="sedentary", weight_noise_kg=0.4)
    assert post.mean == pytest.approx(true_m, abs=0.05)


def test_more_data_tightens_posterior():
    kw = dict(true_m=0.92, height=178, age=35, sex="male", activity="sedentary",
              intake=2300, spacing_days=7, noise_kg=0.5, seed=2)
    few = estimate_rmr_multiplier(
        _synthetic_logs(n_weighins=3, **kw), 178, 35, "male", "sedentary", weight_noise_kg=0.5)
    many = estimate_rmr_multiplier(
        _synthetic_logs(n_weighins=20, **kw), 178, 35, "male", "sedentary", weight_noise_kg=0.5)
    assert many.sd < few.sd


def test_posterior_feeds_module_and_narrows_ci():
    """End-to-end: Layer 3 posterior SD drives a tighter module CI than the prior."""
    logs = _synthetic_logs(0.9, 178, 35, "male", "sedentary", 2300,
                           n_weighins=16, spacing_days=7, noise_kg=0.5, seed=3)
    pp = personal_params_from_logs(logs, 178, 35, "male", "sedentary")

    prior = project_weight(80, 178, 35, "male", daily_intake_kcal=2000, horizon_days=90)
    personalized = project_weight(80, 178, 35, "male", daily_intake_kcal=2000,
                                  horizon_days=90, personal=pp)
    wp = float(np.max(prior.weight_upper - prior.weight_lower))
    wpp = float(np.max(personalized.weight_upper - personalized.weight_lower))
    assert wpp < wp
    assert pp.rmr_multiplier_sd is not None and pp.rmr_multiplier_sd > 0
