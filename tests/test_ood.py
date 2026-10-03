"""Tests for the simulator-reality gap analysis and the Spearman-primary bootstrap (Block 8)."""
from __future__ import annotations

import numpy as np
import pytest


def test_simulator_reality_gap_schema():
    from evaluation import ood_analysis as ood
    g = ood.simulator_reality_gap(n_sim=600, limit=8)
    assert 0.0 <= g["frac_real_ood_any_stat"] <= 1.0
    assert set(g["per_stat"]) == set(ood.SUMMARY_NAMES)
    for d in g["per_stat"].values():
        assert 0.0 <= d["frac_real_outside_central90"] <= 1.0
        assert np.isfinite(d["median_shift_z"])
    assert g["sim"].shape[1] == 5 and g["real"].shape[1] == 5


@pytest.mark.slow
def test_baseline_is_dominant_ood_axis():
    """Documented finding: real fasting baseline sits far above the simulated baseline."""
    from evaluation import ood_analysis as ood
    g = ood.simulator_reality_gap(n_sim=2000, limit=None)
    assert g["per_stat"]["baseline"]["median_shift_z"] > 2.0        # was ~+7.1 on the full set
    assert g["frac_real_ood_any_stat"] > 0.5                        # most real meals are OOD


def test_bootstrap_corr_spearman_and_pearson():
    from evaluation.clinical_recovery import bootstrap_corr
    x = list(range(20))
    y = [-v for v in x]                       # perfectly monotone-decreasing
    for method in ("spearman", "pearson"):
        d = bootstrap_corr(x, y, method)
        assert d["r"] < -0.95
        assert d["lo"] <= d["r"] <= d["hi"]
        assert d["n"] == 20
