"""Tests for the identifiability finding (Block 8), with thresholds set from the real 50k model.

Confirmed on CGMacros (2026-07-19): Si posterior std ~3% of prior width (sharp), gastric/carb
~13-14% (wide); and under more meals Si std shrinks monotonically while gastric/carb stay flat
(structural non-identifiability).
"""
from __future__ import annotations

import numpy as np
import pytest

import paper_config as cfg


@pytest.fixture(scope="module")
def posterior():
    from personalization.snpe_trainer import SNPETrainer
    if not cfg.SNPE_POSTERIOR.exists():
        pytest.skip("no trained SNPE artifact")
    return SNPETrainer.load().posterior


def test_posterior_width_ordering(posterior):
    """Si is far more identifiable (narrower, as a fraction of prior) than gastric/carb."""
    from evaluation import identifiability_analysis as ida
    rows = ida.posterior_width_analysis(posterior, limit=10)
    summ = ida.width_summary(rows)
    f = {k: summ["per_param"][k]["frac_of_prior"] for k in cfg.PARAM_NAMES}
    assert f["insulin_sensitivity"] < 0.10                      # Si sharp
    assert f["insulin_sensitivity"] < f["gastric_emptying"]     # Si sharper than gastric
    assert f["insulin_sensitivity"] < f["carb_absorption"]      # Si sharper than carb


def test_convergence_si_narrows_gastric_flat(posterior):
    """Si posterior std shrinks with more meals; gastric std stays ~flat (structural)."""
    from evaluation import identifiability_analysis as ida
    conv = ida.multi_meal_convergence(posterior)
    si = conv["curves"]["insulin_sensitivity"]
    ga = conv["curves"]["gastric_emptying"]
    assert si[-1] < 0.7 * si[0]                 # Si clearly narrows with data
    assert 0.5 * ga[0] < ga[-1] < 1.6 * ga[0]   # gastric does NOT collapse like Si
    # and Si narrows proportionally much more than gastric
    assert (si[-1] / si[0]) < (ga[-1] / ga[0])


@pytest.mark.slow
def test_rf_shrinkage_documented(posterior):
    """RF point estimator shrinks a high true Si toward the mean (bias clearly negative at 1.6)."""
    from evaluation import identifiability_analysis as ida
    rows = ida.rf_shrinkage_analysis(posterior, n_subjects=4, n_meals=8)
    top = [r for r in rows if abs(r["true_si"] - 1.6) < 1e-6][0]
    assert top["rf_bias"] < -0.10                # RF underestimates high Si
    # SNPE also reports a wider posterior where Si is less resolved (higher end)
    lows = [r for r in rows if abs(r["true_si"] - 0.3) < 1e-6][0]
    assert top["snpe_std"] >= lows["snpe_std"] * 0.8
