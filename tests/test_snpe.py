"""Tests for the SNPE pipeline (Block 8). Uses the trained artifact if present, else a tiny fit."""
from __future__ import annotations

import numpy as np
import pytest

import paper_config as cfg


@pytest.fixture(scope="module")
def posterior():
    from personalization.snpe_trainer import SNPETrainer
    if cfg.SNPE_POSTERIOR.exists():
        return SNPETrainer.load().posterior
    tr = SNPETrainer(n_simulations=500)
    tr.train()
    return tr.posterior


def _synth_meals(true_si, n, seed=0):
    from evaluation.identifiability_analysis import _synth_meals as syn
    return syn(true_si, n, seed=seed)


def test_snpe_trainer_generates_training_data():
    from personalization.snpe_trainer import generate_training_data
    theta, x = generate_training_data(200, seed=0)
    assert theta.shape[1] == 3
    assert x.shape[1] == 10           # 5 summary stats + carbs + weight/height/age/sex
    assert len(theta) == len(x)
    assert len(theta) > 150           # few NaN rows dropped
    lo, hi = cfg.prior_bounds()
    assert (theta[:, 0] >= lo[0] - 1e-6).all() and (theta[:, 0] <= hi[0] + 1e-6).all()


@pytest.mark.slow
def test_snpe_trainer_trains_without_crash():
    from personalization.snpe_trainer import SNPETrainer
    tr = SNPETrainer(n_simulations=400)
    tr.train()
    assert tr.posterior is not None
    assert tr.meta["n_train"] > 300


def test_snpe_infer_returns_correct_schema(posterior):
    from personalization import snpe_infer
    meals = _synth_meals(0.6, 6, seed=1)
    post = snpe_infer.infer_posterior(meals, posterior)
    for k in ("Si_mean", "Si_std", "gastric_mean", "gastric_std", "carb_mean", "carb_std",
              "joint_samples", "Si_samples", "n_meals", "source"):
        assert k in post, k
    assert post["source"] == "snpe"
    assert post["n_meals"] == 6
    assert post["joint_samples"].shape[1] == 3
    assert np.isfinite(post["Si_mean"]) and post["Si_std"] > 0


def test_snpe_infer_empty_when_no_usable_meal(posterior):
    from personalization import snpe_infer
    assert snpe_infer.infer_posterior([], posterior) == {}
    assert snpe_infer.infer_posterior([{"carbs_g": 50}], posterior) == {}  # no glucose window


def test_snpe_point_estimate_in_prior_range(posterior):
    from personalization import snpe_infer
    meals = _synth_meals(0.7, 8, seed=2)
    pe = snpe_infer.point_estimate(meals, posterior)
    lo, hi = cfg.prior_bounds()
    assert lo[0] <= pe["insulin_sensitivity"] <= hi[0]


def test_snpe_recovers_synthetic_si(posterior):
    """Sanity: recovered Si should track a known true Si across the range (monotone-ish)."""
    from personalization import snpe_infer
    los = snpe_infer.point_estimate(_synth_meals(0.4, 8, seed=3), posterior)["insulin_sensitivity"]
    his = snpe_infer.point_estimate(_synth_meals(1.3, 8, seed=3), posterior)["insulin_sensitivity"]
    assert his > los  # a more insulin-sensitive body is recovered as higher Si
