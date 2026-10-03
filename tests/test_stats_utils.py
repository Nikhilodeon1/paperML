"""Statistics utilities checked against answers computed by hand or by a second route.

Each test either uses a case with a closed form (Wilson interval, Fisher z, Wilcoxon on a tiny
sample) or checks a property the function must have (a bootstrap interval covering a known mean, a
paired bootstrap narrower than the unpaired one on correlated arms).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from evaluation import stats_utils as su


def test_wilson_matches_closed_form():
    """Wilson interval for 5 successes in 45, computed independently here."""
    n, k, z = 45, 5, 1.959963984540054
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    ci = su.wilson_ci(k, n)
    assert ci.estimate == pytest.approx(phat)
    assert ci.low == pytest.approx(centre - half, abs=1e-12)
    assert ci.high == pytest.approx(centre + half, abs=1e-12)
    assert 0.0 <= ci.low < ci.high <= 1.0


def test_wilson_at_the_extremes_stays_in_the_unit_interval():
    """The reason Wald is not used: at 0 of 45 it would give a zero-width interval at zero."""
    for k in (0, 45):
        ci = su.wilson_ci(k, 45)
        assert 0.0 <= ci.low <= ci.high <= 1.0
        assert ci.high > ci.low


def test_binomial_ci_matches_scipy():
    ref = stats.binomtest(30, 45).proportion_ci(confidence_level=0.95, method="exact")
    ci = su.binomial_ci(30, 45)
    assert ci.low == pytest.approx(float(ref.low))
    assert ci.high == pytest.approx(float(ref.high))


def test_bootstrap_ci_covers_a_known_mean():
    rng = np.random.default_rng(7)
    sample = rng.normal(loc=10.0, scale=2.0, size=60)
    ci = su.bootstrap_ci(sample, seed=1)
    assert ci.low < 10.0 < ci.high
    assert ci.estimate == pytest.approx(float(np.mean(sample)))
    assert ci.method in ("BCa", "percentile")


def test_bootstrap_ci_is_reproducible():
    sample = list(range(1, 31))
    first = su.bootstrap_ci(sample, seed=3)
    second = su.bootstrap_ci(sample, seed=3)
    assert (first.low, first.high) == (second.low, second.high)


def test_bootstrap_degenerate_sample_falls_back_and_says_so():
    """A constant sample has no BCa acceleration; the fallback must be visible in `method`."""
    ci = su.bootstrap_ci([4.0] * 20, seed=0)
    assert ci.estimate == pytest.approx(4.0)
    assert ci.method == "percentile"


def test_bootstrap_with_one_value_reports_insufficient_n():
    ci = su.bootstrap_ci([1.0])
    assert ci.method == "insufficient-n"
    assert math.isnan(ci.low)


def test_paired_bootstrap_exploits_the_pairing():
    """On strongly correlated arms the paired interval must be much narrower than the unpaired one.

    This is the property that makes the comparison worth doing paired, and a regression here would
    mean the pairing had been lost.
    """
    rng = np.random.default_rng(11)
    shared = rng.normal(scale=50.0, size=40)
    a = shared + rng.normal(scale=1.0, size=40)
    b = shared + rng.normal(scale=1.0, size=40) + 3.0
    paired = su.paired_bootstrap_ci(a, b, seed=0)
    width_paired = paired.high - paired.low
    width_a = su.bootstrap_ci(a, seed=0).high - su.bootstrap_ci(a, seed=0).low
    assert width_paired < 0.2 * width_a
    assert paired.low < -3.0 < paired.high


def test_paired_bootstrap_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        su.paired_bootstrap_ci([1, 2, 3], [1, 2])


def test_wilcoxon_matches_scipy_on_a_small_sample():
    a = np.array([5.0, 3.0, 9.0, 1.0, 7.0, 2.0])
    b = np.array([4.0, 4.0, 6.0, 2.0, 3.0, 3.0])
    ref = stats.wilcoxon(a - b, zero_method="wilcox", alternative="two-sided")
    out = su.wilcoxon(a, b)
    assert out["statistic"] == pytest.approx(float(ref.statistic))
    assert out["p"] == pytest.approx(float(ref.pvalue))
    assert out["median_difference"] == pytest.approx(float(np.median(a - b)))


def test_wilcoxon_all_ties_does_not_raise():
    out = su.wilcoxon([1.0, 2.0], [1.0, 2.0])
    assert out["p"] == 1.0
    assert "note" in out


def test_tost_declares_equivalence_only_inside_the_margin():
    """A tiny constant difference is equivalent at margin 150 and not at margin 0.5."""
    a = np.arange(30.0) + 100.0
    b = a + 1.0
    assert su.tost(a, b, margin=150.0)["equivalent"] is True
    assert su.tost(a, b, margin=0.5)["equivalent"] is False


def test_tost_is_not_just_a_non_significant_difference():
    """Two noisy arms with a true difference of zero but a wide interval are NOT equivalent at a
    narrow margin. Conflating the two is the error the function exists to prevent."""
    rng = np.random.default_rng(5)
    a = rng.normal(scale=400.0, size=20)
    b = rng.normal(scale=400.0, size=20)
    assert su.wilcoxon(a, b)["p"] > 0.05
    assert su.tost(a, b, margin=20.0)["equivalent"] is False


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = [0.001, 0.013, 0.021, 0.04, 0.2]
    rej, adj, _, _ = multipletests(p, alpha=0.05, method="holm")
    out = su.holm(p)
    assert out["p_adjusted"] == pytest.approx(list(adj))
    assert out["reject"] == list(rej)


def test_holm_handles_an_empty_family():
    assert su.holm([])["n"] == 0


def test_holm_passes_through_nan_without_rejecting():
    out = su.holm([0.001, float("nan")])
    assert out["reject"][0] is True
    assert out["reject"][1] is False


def test_fisher_z_known_value():
    """Independent correlations 0.6 (n=45) and 0.2 (n=97), z computed by hand."""
    z1 = 0.5 * math.log(1.6 / 0.4)
    z2 = 0.5 * math.log(1.2 / 0.8)
    se = math.sqrt(1 / 42 + 1 / 94)
    out = su.fisher_z_difference(0.6, 45, 0.2, 97)
    assert out["z"] == pytest.approx((z1 - z2) / se, rel=1e-9)
    assert out["p"] < 0.01


def test_fisher_z_refuses_tiny_samples():
    out = su.fisher_z_difference(0.5, 3, 0.1, 50)
    assert math.isnan(out["z"])
    assert "note" in out


def test_kendall_and_spearman_match_scipy():
    x = [3.0, 1.0, 2.0, 5.0, 4.0]
    y = [2.0, 1.0, 4.0, 5.0, 3.0]
    assert su.kendall_tau(x, y)["tau"] == pytest.approx(float(stats.kendalltau(x, y).statistic))
    assert su.spearman(x, y)["rho"] == pytest.approx(float(stats.spearmanr(x, y).statistic))


def test_nan_values_are_dropped_not_propagated():
    a = [1.0, 2.0, float("nan"), 4.0]
    b = [1.0, 3.0, 5.0, 4.0]
    out = su.wilcoxon(a, b)
    assert out["n"] == 3
    assert su.spearman(a, b)["n"] == 3


def test_ci_within_and_excludes_zero():
    ci = su.CI(estimate=10.0, low=-5.0, high=25.0, level=0.9, n=20, method="percentile")
    assert ci.excludes_zero() is False
    assert ci.within(30.0) is True
    assert ci.within(20.0) is False
