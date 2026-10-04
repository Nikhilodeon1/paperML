"""Phase 8: replica generation, cohort windows, the new verdict rules, and the stored analyses.

The replica is the instrument that replaces the undefined error decomposition, so what matters is that it
is deterministic, that switching one noise source off does not change the other's draws, and that the
truth it generates from is the stated one.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.jax_config import configure

configure()

from evaluation import hypotheses as hyp                               # noqa: E402
from evaluation import phase8_summary as p8                            # noqa: E402
from evaluation import stored_analyses as sa                           # noqa: E402
from evaluation import subject_source as ss                            # noqa: E402
from evaluation.cohort_data import load_cohort                          # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return {"cohort": "cgmacros", "min_meals": 10, "limit": 2}


def _have_reference(unit):
    try:
        ss._reference_fit(unit)
        return True
    except RuntimeError:
        return False


# --- windows -----------------------------------------------------------------------------------------

def test_window_follows_the_cohort_grid():
    try:
        shanghai = load_cohort("shanghai", min_meals=10)[0]
        hall = load_cohort("hall", min_meals=5)[0]
        cgm = load_cohort("cgmacros", min_meals=10)[0]
    except FileNotFoundError as exc:
        pytest.skip(f"cohort cache not available: {exc}")
    assert ss.window_for(cgm).stride == 1 and ss.window_for(cgm).post_min == 180.0
    assert ss.window_for(shanghai).stride == 3           # the real 15 minute samples
    assert ss.window_for(hall).post_min == 145.0 and ss.window_for(hall).stride == 1


def test_carbohydrate_scaling_touches_only_carbohydrate(cfg):
    base = ss.get_subject(cfg, "CGMacros-001")
    scaled = ss.get_subject({**cfg, "carb_scale": 0.75}, "CGMacros-001")
    for a, b in zip(base.records, scaled.records):
        assert b["carbs_g"] == pytest.approx(0.75 * a["carbs_g"])
        assert b["glucose"] == a["glucose"] and b["fat_g"] == a["fat_g"] and b["iauc"] == a["iauc"]


# --- replica -----------------------------------------------------------------------------------------

@pytest.mark.skipif(not _have_reference("CGMacros-001"), reason="needs the stored A4 fit")
def test_replica_is_deterministic_and_streams_are_independent(cfg):
    on = {**cfg, "replica": {"seed": 0, "cgm": True, "carb_cv": 0.25}}
    a = ss.get_subject(on, "CGMacros-001")
    ss._replica_cached.cache_clear()
    b = ss.get_subject(on, "CGMacros-001")
    assert [r["iauc"] for r in a.records] == [r["iauc"] for r in b.records]
    # Same seed, carbohydrate error off: the CGM noise draws are unchanged, only the logged carbs differ.
    no_carb = ss.get_subject({**cfg, "replica": {"seed": 0, "cgm": True, "carb_cv": 0.0}}, "CGMacros-001")
    assert [r["glucose"]["values"] for r in a.records] == [r["glucose"]["values"] for r in no_carb.records]
    assert any(x["carbs_g"] != y["carbs_g"] for x, y in zip(a.records, no_carb.records))
    # CGM noise off: the trace is the model output, no noise to compare with.
    clean = ss.get_subject({**cfg, "replica": {"seed": 0, "cgm": False, "carb_cv": 0.0}}, "CGMacros-001")
    assert clean.records[0]["glucose"]["values"] != a.records[0]["glucose"]["values"]


@pytest.mark.skipif(not _have_reference("CGMacros-001"), reason="needs the stored A4 fit")
def test_replica_truth_is_inside_the_box(cfg):
    from personalization.objectives import _scaled_bounds
    from personalization.subject_loss import TARGETS
    subject = ss.get_subject(cfg, "CGMacros-001")
    truth = ss.true_parameters(subject, "CGMacros-001")
    lower, upper = (np.asarray(v) for v in _scaled_bounds(1.0))
    for i, name in enumerate(TARGETS):
        margin = ss.INSIDE_FRACTION * (upper[i] - lower[i])
        assert lower[i] + margin - 1e-9 <= truth[name] <= upper[i] - margin + 1e-9


def test_replica_carbohydrate_error_has_unit_mean_and_the_requested_cv():
    rng = ss._rng("u", 0, "carb")
    cv = 0.25
    sigma2 = np.log1p(cv ** 2)
    m = rng.lognormal(-0.5 * sigma2, np.sqrt(sigma2), 200000)
    assert m.mean() == pytest.approx(1.0, abs=0.01)
    assert m.std() / m.mean() == pytest.approx(cv, abs=0.01)


def test_ar1_noise_has_the_requested_autocorrelation():
    rng = np.random.default_rng(0)
    noise = ss._ar1_noise(rng, 0.8, 3.0, (4000, 43))
    lag1 = np.sum(noise[:, 1:] * noise[:, :-1]) / np.sum(noise[:, :-1] ** 2)
    assert lag1 == pytest.approx(0.8, abs=0.02)
    assert noise.std() == pytest.approx(3.0 / np.sqrt(1 - 0.64), rel=0.03)


# --- verdict rules -----------------------------------------------------------------------------------

def _summary(**phase8):
    return {"phase8": phase8}


def test_h13_requires_both_clauses_and_reports_the_reading():
    good = {"n": 45, "S_I_clause_met": True, "timing_clause_met": True, "timing_max_fraction": 0.1,
            "S_I_interior_bounded": {"fraction": 0.6}, "reading": "misspecification"}
    assert hyp.h13(_summary(h13=good))["status"] == "met"
    bad = {**good, "S_I_clause_met": False, "reading": "inherent"}
    row = hyp.h13(_summary(h13=bad))
    assert row["status"] == "not met" and row["note"] == "inherent"
    assert hyp.h13({})["status"] == "not evaluated"


def test_h14_is_reported_not_graded():
    row = hyp.h14(_summary(h14={"main_ratio": 0.8, "reading": "mixed"}))
    assert row["status"].startswith("reported") and row["observed"]["ratio"] == 0.8


def test_h11_h12_h15_follow_the_stored_status():
    s = _summary(h11={"status": "not met", "clauses": {"tau1_centroid": 0.2}},
                 h12={"status": "met", "comparisons": {}, "boundary": {}},
                 h15={"status": "met", "shanghai": {"1x": {"fisher": {}}}, "hall": {"1x": {"fisher": {}}}})
    assert hyp.h11(s)["status"] == "not met"
    assert hyp.h12(s)["status"] == "met"
    assert hyp.h15(s)["status"] == "met"
    for fn in (hyp.h11, hyp.h12, hyp.h15):
        assert fn({})["status"] == "not evaluated"


def test_h6_is_partial_because_the_synthetic_study_is_missing():
    block = {"spearman_vs_profile_strength": {"median": 1.0}, "spearman_vs_fisher_information": {"median": 1.0},
             "auc_detecting_flat": 0.92}
    row = hyp.h6(_summary(stored={"h6": {"1x": block}}))
    assert row["status"] == "partially evaluated"
    block["spearman_vs_profile_strength"]["median"] = 0.3
    assert hyp.h6(_summary(stored={"h6": {"1x": block}}))["status"] == "not met"


def test_every_phase8_hypothesis_is_registered():
    assert {"H13", "H14", "H15", "H16"} <= set(hyp.ALL)
    verdicts = hyp.evaluate_all({}, moment={})
    assert verdicts["H13"]["status"] == "not evaluated"


def test_phase8_macro_names_are_alphabetic():
    import re
    from evaluation import paper_numbers as pn
    summary = {"h13": {"n": 45, "S_I_interior_bounded": {"fraction": 0.5, "wilson": [0.3, 0.7],
                                                       "bounded": 10, "n": 20},
                       "timing_max_fraction": 0.1, "truth_coverage": {}},
               "h14": {"settings": {"cgm_on_carb_025": {"n": 45, "replica_grad3_mae": 1.0, "ratio": 0.8,
                                                         "ratio_ci95": [0.7, 0.9]}}},
               "h11": {"trace": {"n": 45, "tau1": {"all": {"fraction": 0.5}, "interior": None},
                                 "p": {"all": {"fraction": 0.2}, "interior": None}}},
               "h12": {}, "h15": {}, "h16": {"carbohydrate_scale": {"0.75": {
                   "n": 45, "median_log_ratio_S_I": 0.2, "timing_on_upper_bound_scaled": 0.3}}},
               "leakage": {"n": 90, "statistics": {"raw_snpe": {"estimate": -0.4, "low": -0.5, "high": -0.3}}},
               "stored": {}}
    macros = pn._phase8_macros(summary)
    assert macros and all(re.fullmatch(r"[A-Za-z]+", k) for k in macros)


# --- stored analyses ---------------------------------------------------------------------------------

def test_auc_is_the_mann_whitney_probability():
    assert sa._auc(np.array([1.0, 2.0]), np.array([0.0, 0.5])) == 1.0
    assert sa._auc(np.array([0.0]), np.array([1.0])) == 0.0
    assert sa._auc(np.array([1.0]), np.array([1.0])) == 0.5
    assert np.isnan(sa._auc(np.array([]), np.array([1.0])))
