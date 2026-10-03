"""Phase 2 and 4c modules: the pieces that interpret stored results, and one end-to-end profile.

The summary code is tested against hand-built result sets whose answers are known, because the way
these modules fail is by quietly computing something other than what their docstring says: a Holm
family counted wrongly, a sign flipped in a paired difference, a cosine taken against the wrong
direction.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.jax_config import configure

configure()

from evaluation import ladder, prediction_stats as ps, profile_lik       # noqa: E402


# --- profile grid ---------------------------------------------------------------------------------

def test_log_grid_extension_is_beyond_the_box_and_in_box_points_come_first():
    grid, n_inside = profile_lik.log_grid(0.01, 0.04, 15, extension=6)
    assert n_inside == 15 and len(grid) == 21
    assert grid[0] == pytest.approx(0.01) and grid[n_inside - 1] == pytest.approx(0.04)
    assert grid[-1] == pytest.approx(0.16)
    assert np.all(np.diff(grid) > 0)


def test_log_grid_without_extension_stays_in_the_box():
    grid, n_inside = profile_lik.log_grid(0.01, 0.04, 9)
    assert len(grid) == n_inside == 9 and grid.max() == pytest.approx(0.04)


# --- ladder ---------------------------------------------------------------------------------------

def test_trade_off_cosine_is_one_along_the_direction_and_zero_across_it():
    ke, ka = 0.03, 0.02
    along = np.array([1.0 / ka, -1.0 / ke])
    across = np.array([1.0 / ke, 1.0 / ka])
    assert ladder.trade_off_cosine(along, ke, ka) == pytest.approx(1.0)
    assert ladder.trade_off_cosine(-along, ke, ka) == pytest.approx(1.0)      # sign is arbitrary
    assert ladder.trade_off_cosine(across, ke, ka) == pytest.approx(0.0, abs=1e-12)


def test_ladder_h3_detects_a_decreasing_condition_number():
    rng = np.random.default_rng(0)
    subjects = [f"s{i}" for i in range(30)]
    base = 10 ** rng.uniform(5, 6, 30)
    table = {"iauc": dict(zip(subjects, base)),
             "iauc_centroid": dict(zip(subjects, base / 50 * 10 ** rng.uniform(-.1, .1, 30))),
             "trace": dict(zip(subjects, base / 2500 * 10 ** rng.uniform(-.1, .1, 30)))}
    out = ladder._stats(table)
    assert out["n"] == 30 and out["h3_met"] is True
    assert all(step["median_decreases"] for step in out["steps"])


def test_ladder_h3_not_met_when_a_step_does_not_decrease():
    subjects = [f"s{i}" for i in range(20)]
    flat = dict(zip(subjects, np.full(20, 1e4) * (1 + 0.01 * np.arange(20))))
    table = {"iauc": flat, "iauc_centroid": dict(flat),
             "trace": dict(zip(subjects, np.full(20, 1e2)))}
    out = ladder._stats(table)
    assert out["h3_met"] is False


def test_ladder_handles_infinite_condition_numbers():
    subjects = [f"s{i}" for i in range(10)]
    inf = {s: float("inf") for s in subjects}
    out = ladder._stats({"iauc": inf, "iauc_centroid": dict(inf), "trace": dict(inf)})
    assert out["n"] == 10          # no crash, no NaN poisoning


# --- prediction statistics ------------------------------------------------------------------------

def _units(cells: dict[str, float], n_subjects: int = 12, repeats: int = 2, seed: int = 0):
    """Synthetic A9 units where cell `c` has error `cells[c]` plus subject and repeat noise."""
    rng = np.random.default_rng(seed)
    out = []
    subject_offset = rng.normal(0, 50, n_subjects)
    for subject in range(n_subjects):
        for repeat in range(repeats):
            body = {}
            for cell, level in cells.items():
                value = level + subject_offset[subject] + rng.normal(0, 5)
                body[cell] = {"iauc_mae": value, "peak_mae": 30.0, "trace_rmse_mean": 20.0,
                              "pred_iauc": [1.0], "folds": []}
            out.append({"subject_id": f"S{subject}", "repeat": repeat, "cells": body,
                        "obs_iauc": [1.0], "carbs_g": [50.0]})
    return out


def test_paired_difference_sign_negative_means_first_cell_is_better():
    result = ps.analyse(_units({"grad3": 3000.0, "grid3": 3100.0, "personal_mean": 3400.0,
                                "grad1": 3000.0, "grid1": 3100.0}))
    row = next(c for c in result["comparisons"]
               if c["first"] == "grad3" and c["second"] == "grid3")
    assert row["mean_difference"] == pytest.approx(-100.0, abs=10)
    assert row["wins_first"] == row["n"]


def test_equivalence_verdict_follows_the_margin():
    units = _units({"grad3": 3000.0, "grid3": 3060.0, "grad1": 3000.0, "grid1": 3060.0,
                    "personal_mean": 3500.0})
    result = ps.analyse(units)
    verdict = result["h7_h8"]["grad3_vs_grid3"]["equivalent_at"]
    assert verdict["150"] is True and verdict["90"] is True
    far = ps.analyse(_units({"grad3": 3000.0, "grid3": 3400.0, "grad1": 3000.0, "grid1": 3400.0,
                             "personal_mean": 3500.0}))
    assert far["h7_h8"]["grad3_vs_grid3"]["equivalent_at"]["300"] is False


def test_h9_requires_the_whole_interval_below_minus_the_margin():
    better = ps.analyse(_units({"grad3": 3000.0, "grad3_trace": 2700.0, "grad1": 3000.0,
                                "grad1_trace": 2700.0, "personal_mean": 3500.0}))
    assert better["h9"]["grad3_trace_vs_grad3"]["lowers_error_by_more_than_margin"] is True
    small = ps.analyse(_units({"grad3": 3000.0, "grad3_trace": 2950.0, "grad1": 3000.0,
                               "grad1_trace": 2950.0, "personal_mean": 3500.0}))
    assert small["h9"]["grad3_trace_vs_grad3"]["lowers_error_by_more_than_margin"] is False


def test_holm_family_counts_every_testable_pair():
    result = ps.analyse(_units({"grad3": 3000.0, "grid3": 3050.0, "grad1": 3000.0,
                                "grid1": 3050.0, "personal_mean": 3400.0}))
    testable = [c for c in result["comparisons"] if "wilcoxon" in c]
    assert result["holm_family_size"] == len(testable) > 0
    assert all("p_holm" in c for c in testable)
    assert all(c["p_holm"] >= c["wilcoxon"]["p"] - 1e-12 for c in testable)


def test_spread_across_repeats_is_reported_separately():
    result = ps.analyse(_units({"grad3": 3000.0, "personal_mean": 3400.0}, repeats=4))
    spread = result["cells"]["grad3"]["spread_across_repeats"]
    assert spread["n"] == 12 and 1.0 < spread["median_within_subject_sd"] < 15.0


def test_subjects_are_the_resampling_unit_not_units():
    units = _units({"grad3": 3000.0, "personal_mean": 3400.0}, n_subjects=9, repeats=5)
    result = ps.analyse(units)
    assert result["n_units"] == 45 and result["n_subjects"] == 9
    assert all(c["n"] == 9 for c in result["comparisons"])


# --- end to end -----------------------------------------------------------------------------------

def test_profile_unit_runs_and_reports_every_point():
    from evaluation.cohort_data import load_cohort
    try:
        subject = load_cohort("cgmacros", min_meals=10, limit=1)[0]
    except FileNotFoundError as exc:
        pytest.skip(f"CGMacros not available: {exc}")
    config = profile_lik.default_config()
    config.update(limit=1, grid_points=5, extension_points=2, steps=8, fit_steps=20,
                  pilot_steps=10)
    result = profile_lik.run_unit(subject.subject_id, config)
    assert set(result["profiles"]) == {"insulin_sensitivity", "gastric_emptying",
                                       "carb_absorption"}
    for name, entry in result["profiles"].items():
        expected = 5 + (2 if name != "insulin_sensitivity" else 0)
        assert len(entry["profile"]) == len(entry["drift"]) == expected
        assert np.isfinite(entry["profile"]).all()
        assert entry["classification"]["verdict"] in {"identifiable", "one-sided", "flat"}
        assert ("extended" in entry) == (name != "insulin_sensitivity")
    assert result["delta"] == pytest.approx(1.9207295, rel=1e-6)


def test_inference_gap_reads_fold_losses():
    def unit(gl, hl):
        return {"cells": {"grad3": {"folds": [{"final_loss": gl}]},
                          "grid3": {"folds": [{"final_loss": hl}]},
                          "grad1": {"folds": [{"final_loss": gl}]},
                          "grid1": {"folds": [{"final_loss": hl}]}}}
    out = ps.inference_gap([unit(1.005, 1.0), unit(1.2, 1.0), unit(0.98, 1.0)])
    pair = out["pairs"]["grad3_vs_grid3"]
    assert pair["n_folds"] == 3
    assert pair["fraction_within_tolerance"] == pytest.approx(2 / 3)
    assert pair["fraction_gradient_better"] == pytest.approx(1 / 3)
    assert pair["max_relative_gap"] == pytest.approx(0.2)
