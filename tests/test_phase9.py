"""Phase 9 (Amendment 5): random-truth replica, polished estimates, robustness sweeps, the new verdicts.

What matters here is that the instruments behave: the random truth is deterministic and inside the box, the
polished estimate can never be worse than the Adam estimate it started from, the H10 reading is the one
written in the amendment, and the verdict functions turn stored numbers into the statuses the plan defines.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from evaluation.jax_config import configure

configure()

import jax.numpy as jnp                                                  # noqa: E402

from evaluation import hypotheses as hyp                                # noqa: E402
from evaluation import robustness as rb                                 # noqa: E402
from evaluation import subject_source as ss                             # noqa: E402
from evaluation.gradient_diag import is_default                         # noqa: E402
from evaluation.profile_lik import polish_estimate, summarize_rows      # noqa: E402


# --- random truth --------------------------------------------------------------------------------------

def test_random_truth_is_deterministic_inside_the_box_and_varies_with_seed():
    from personalization.objectives import _scaled_bounds
    from personalization.subject_loss import TARGETS
    a = ss.random_parameters("CGMacros-001", 0)
    assert a == ss.random_parameters("CGMacros-001", 0)
    assert a != ss.random_parameters("CGMacros-001", 1)
    assert a != ss.random_parameters("CGMacros-002", 0)
    lower, upper = (np.asarray(v) for v in _scaled_bounds(1.0))
    for i, name in enumerate(TARGETS):
        margin = ss.INSIDE_FRACTION * (upper[i] - lower[i])
        assert lower[i] + margin - 1e-12 <= a[name] <= upper[i] - margin + 1e-12


def test_random_truth_is_spread_over_the_box_not_clustered():
    draws = np.array([ss.random_parameters(f"u{i}", 0)["gastric_emptying"] for i in range(400)])
    from personalization.objectives import _scaled_bounds
    lower, upper = (np.asarray(v) for v in _scaled_bounds(1.0))
    span = upper[1] - lower[1]
    hist, _ = np.histogram(draws, bins=4, range=(lower[1], upper[1]))
    assert hist.min() > 0.15 * len(draws) and span > 0


def test_truth_in_coordinates_matches_the_definition():
    theta = {"insulin_sensitivity": 0.8, "gastric_emptying": 0.05, "carb_absorption": 0.1}
    c = ss.truth_in_coordinates(theta)
    assert c["log_tau1"] == pytest.approx(1 / 0.05 + 1 / 0.1)
    assert c["log_p"] == pytest.approx(1 / (0.05 * 0.1))
    assert c["log_insulin_sensitivity"] == 0.8


# --- polishing -------------------------------------------------------------------------------------------

def test_polish_is_never_worse_and_finds_the_minimum_of_a_quadratic():
    target = jnp.asarray([0.3, -0.2, 0.5])

    def loss(phi):
        return jnp.sum((phi - target) ** 2) + 0.1 * jnp.sin(3 * phi[0])

    lower, upper = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    start = np.array([0.9, 0.9, -0.9])
    out = polish_estimate(loss, start, lower, upper, None, "test", starts=3, maxiter=100)
    assert out["nll_polished"] <= out["nll_adam"] + 1e-12 and out["gap"] >= 0
    assert out["gap"] > 0.5                         # a poor start is improved a lot
    assert np.all(out["phi"] >= lower - 1e-9) and np.all(out["phi"] <= upper + 1e-9)


def test_polish_respects_a_projection():
    def loss(phi):
        return jnp.sum((phi - jnp.asarray([2.0, 2.0])) ** 2)

    def project(phi):                               # a ceiling the unconstrained optimum violates
        return jnp.minimum(phi, 1.0)

    out = polish_estimate(loss, np.array([0.0, 0.0]), np.array([-3.0, -3.0]), np.array([3.0, 3.0]), project,
                          "ceiling", starts=2, maxiter=50)
    assert np.all(out["phi"] <= 1.0 + 1e-9) and out["nll_polished"] == pytest.approx(2.0, abs=1e-3)


# --- the robustness definition ---------------------------------------------------------------------------

def test_is_default_excludes_every_variant():
    base = {"bounds_scale": 1.0}
    assert is_default(base, 1.0) and not is_default(base, 2.0)
    assert not is_default({**base, "optimizer": "lbfgs"}, 1.0)
    assert not is_default({**base, "init_seed": 0}, 1.0)
    assert not is_default({**base, "log_param": True}, 1.0)
    assert not is_default({**base, "replica": {"seed": 0}}, 1.0)
    assert is_default({**base, "init_seed": None, "optimizer": "adam", "log_param": False, "replica": None})


def test_tau_is_one_for_identical_orderings_and_undefined_for_constants():
    a = np.array([5.0, 3.0, 1.0])
    assert rb._tau(a, np.array([50.0, 20.0, 2.0])) == pytest.approx(1.0)
    assert rb._tau(a, np.array([1.0, 3.0, 5.0])) == pytest.approx(-1.0)
    assert np.isnan(rb._tau(a, np.array([1.0, 1.0, 1.0])))


def test_variant_matcher_requires_every_field_to_match():
    row = {"bounds_scale": 1.0, "optimizer": "adam", "init_seed": 2, "log_param": False}
    assert rb._variant(row, init_seed=2) and not rb._variant(row, init_seed=3)
    assert not rb._variant(row, init_seed=2, optimizer="lbfgs")
    assert not rb._variant({**row, "replica": {"seed": 0}}, init_seed=2)


# --- verdicts --------------------------------------------------------------------------------------------

def _summary(**phase9):
    return {"phase9": phase9}


def test_new_hypotheses_are_registered_and_default_to_not_evaluated():
    assert {"H17", "H18", "H19", "H20", "H21", "H22"} <= set(hyp.ALL)
    verdicts = hyp.evaluate_all({}, moment={})
    for name in ("H10", "H17", "H18", "H19", "H20", "H21", "H22"):
        assert verdicts[name]["status"] == "not evaluated"


def test_h6_becomes_met_only_when_both_clauses_hold():
    real = {"spearman_vs_profile_strength": {"median": 1.0}, "spearman_vs_fisher_information": {"median": 1.0},
            "auc_detecting_flat": 0.92}
    base = {"phase8": {"stored": {"h6": {"1x": real}}}}
    assert hyp.h6(base)["status"] == "partially evaluated"
    synthetic = {"n": 90, "status": "met", "spearman_vs_profile_strength": {"median": 0.9},
                 "auc_detecting_flat": 0.9}
    assert hyp.h6({**base, "phase9": {"h6_synthetic": synthetic}})["status"] == "met"
    failed = {**synthetic, "status": "not met", "auc_detecting_flat": 0.6}
    assert hyp.h6({**base, "phase9": {"h6_synthetic": failed}})["status"] == "not met"


def test_h10_follows_the_stored_status():
    block = {"status": "not met", "factors": {"initialization": {"median_of_mean_tau": 1.0, "passes": True},
                                              "optimizer": {"median_of_mean_tau": 0.33, "passes": False}},
             "reference_S_I_first_interior": {"n": 20, "fraction": 1.0}}
    assert hyp.h10(_summary(h10=block))["status"] == "not met"
    assert hyp.h10(_summary(h10={**block, "status": "met"}))["status"] == "met"


def test_h20_and_h22_rows():
    h20 = {"status": "met", "equivalent_at_150": {"grad3-grid3": True, "grad3-grad1": True},
           "metrics": {"iauc": {"comparisons": {"grad3-grid3": {"mean_difference": 1.0},
                                                "grad3-grad1": {"mean_difference": 2.0}}}}}
    assert hyp.h20(_summary(h20=h20))["status"] == "met"
    h22 = {"status": "met", "parameters": {k: {"fraction": 0.0} for k in ("Vmx", "kabs", "kmax", "kmin")}}
    assert hyp.h22(_summary(h22=h22))["status"] == "met"


def test_h17_uses_the_pooled_clause():
    block = {"n": 225, "status": "not met", "pooled_S_I_interior_bounded": {"fraction": 0.27},
             "pooled_timing_max_fraction": 0.0, "seed_range_S_I_fraction": [0.2, 0.3],
             "S_I_truth_coverage": {"fraction": 0.8}, "reading": "mixed"}
    row = hyp.h17(_summary(h17=block))
    assert row["status"] == "not met" and row["note"] == "mixed"


def test_phase9_macro_names_are_alphabetic():
    from evaluation import paper_numbers as pn
    p9 = {"h17": {"n": 225, "n_seeds": 5, "seeds_meeting_S_I_clause": 0,
                  "pooled_S_I_interior_bounded": {"bounded": 40, "n": 150, "fraction": 0.27, "wilson": [0.2, 0.35]},
                  "pooled_timing_max_fraction": 0.0, "seed_range_S_I_fraction": [0.2, 0.3],
                  "S_I_truth_coverage": {"fraction": 0.8, "covered": 8, "n": 10}},
          "h6_synthetic": {"n": 90, "spearman_vs_profile_strength": {"median": 0.9}, "auc_detecting_flat": 0.9,
                           "recovery": {"insulin_sensitivity": {
                               "all_bounded": {"fraction": 0.3}, "interior": {"fraction": 0.4},
                               "median_abs_log_error_of_estimate": 0.2,
                               "coverage_of_bounded": {"fraction": 0.9},
                               "terciles": {"low": {"fraction": 0.1}, "middle": {"fraction": 0.3},
                                            "high": {"fraction": 0.6}}}}},
          "h22": {"n_subjects": 45, "parameters": {"kabs": {"fraction": 0.0, "wilson": [0.0, 0.08], "pinned": 40}},
                  "fisher": {"effective_rank_median": 2, "condition_number_median": 1e9},
                  "polish_gap_median": 0.3},
          "h10": {"factors": {"initialization": {"median_of_mean_tau": 1.0}},
                  "reference_S_I_first_interior": {"n": 20, "fraction": 1.0}}}
    macros = pn._phase9_macros(p9)
    assert macros and all(re.fullmatch(r"[A-Za-z]+", k) for k in macros)
    assert "resSynthTercileSIHigh" in macros


def test_summarize_rows_pools_rows_from_several_seeds():
    def row(bounded: bool, seed: int):
        profile = {"classification": {"verdict": "identifiable" if bounded else "one-sided", "max_rise": 3.0},
                   "unsettled": 0, "better_than_theta_hat": False, "profile_min_near_theta_hat": True}
        flat = {**profile, "classification": {"verdict": "flat", "max_rise": 0.1}}
        return {"profiles": {"insulin_sensitivity": profile, "gastric_emptying": flat, "carb_absorption": flat},
                "distance_to_bound": {"insulin_sensitivity": 0.3, "gastric_emptying": 0.0, "carb_absorption": 0.0}}
    rows = [row(True, 0), row(False, 0), row(False, 1), row(False, 1)]
    out = summarize_rows(rows, {"objective": "iauc", "bounds_scale": 1.0, "cohort": "cgmacros"})
    assert out["n_subjects"] == 4 and out["S_I_bounded_among_interior_for_S_I"]["bounded"] == 1
    assert out["parameters"]["gastric_emptying"]["bounded"] == 0
