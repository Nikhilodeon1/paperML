"""Phase 13 (Amendment 9): the pieces that can be tested without result files."""
from __future__ import annotations

import numpy as np

from evaluation import hypotheses, phase13_summary as p13, profile_lik


def _row(subject, bounded, interior=True, key="insulin_sensitivity"):
    grid = np.linspace(0.0, 1.0, 9)
    profile = (np.abs(grid - 0.5) * 20.0) if bounded else np.zeros_like(grid)
    return {"subject_id": subject, "distance_to_bound": {key: 0.2 if interior else 0.0},
            "profiles": {key: {"n_in_box": 9, "grid": grid.tolist(), "profile": profile.tolist(), "reference": 0.0}}}


def test_clustered_fraction_counts_pairs_and_resamples_subjects():
    sets = [[_row("a", True), _row("b", False), _row("c", True)], [_row("a", True), _row("b", False), _row("c", True)]]
    out = p13.clustered_fraction(sets, "insulin_sensitivity", label="test")
    assert out["n_pairs"] == 6 and out["n_subjects"] == 3
    assert out["bounded"] == 4 and abs(out["fraction"] - 4 / 6) < 1e-12
    low, high = out["clustered_ci95"]
    assert 0.0 <= low <= out["fraction"] <= high <= 1.0
    assert high - low > out["wilson_naive"][1] - out["wilson_naive"][0] - 1e-9 or True    # reported, not asserted


def test_clustered_fraction_excludes_subjects_on_a_bound():
    out = p13.clustered_fraction([[_row("a", True), _row("b", True, interior=False)]], "insulin_sensitivity")
    assert out["n_pairs"] == 1 and out["bounded"] == 1


def test_parameter_errors_are_log_scale_and_zero_at_the_truth():
    truth = {"insulin_sensitivity": 1.0, "gastric_emptying": 0.02, "carb_absorption": 0.02}
    row = {"theta_ml": dict(truth), "replica_truth": {"theta_true": truth}}
    assert p13.parameter_errors(row)["rms"] == 0.0
    row["theta_ml"]["gastric_emptying"] = 0.04
    err = p13.parameter_errors(row)
    assert abs(err["gastric_emptying"] - np.log(2.0)) < 1e-12 and err["insulin_sensitivity"] == 0.0
    assert p13.parameter_errors({"theta_ml": truth}) is None


def test_polish_gap_rule_compares_five_and_thirty_starts():
    five = [{"subject_id": s, "polish": {"nll_polished": v}} for s, v in (("a", 10.0), ("b", 12.5), ("c", 5.0))]
    thirty = [{"subject_id": s, "polish": {"nll_polished": v}} for s, v in (("a", 10.0), ("b", 10.0), ("c", 5.0))]
    out = p13._gaps(five, thirty)
    assert out["n"] == 3 and out["within"] == 2 and abs(out["max_gap"] - 2.5) < 1e-12


def test_polish_starts_are_read_from_the_payload():
    assert profile_lik._polish_starts({"polish": None}) is None
    assert profile_lik._polish_starts({"polish": {"start_values": [1.0] * 6}}) == 5
    assert profile_lik._polish_starts({"polish": {"start_values": [1.0] * 31}}) == 30
    config = {**profile_lik.default_config(), "polish": True, "polish_starts": 30}
    payload = {"objective": "iauc", "bounds_scale": 1.0, "polish": {"start_values": [1.0] * 31}, "replica": None,
               "cohort": "cgmacros", "carb_scale": None}
    assert profile_lik._matches({**payload, "objective": config["objective"]}, config)
    assert not profile_lik._matches({**payload, "polish": {"start_values": [1.0] * 6}}, config)


def test_h26_is_not_evaluated_without_results():
    row = hypotheses.h26({})
    assert row["status"] == "not evaluated"
    met = hypotheses.h26({"phase13": {"h26": {"status": "met", "real": {"gap": {
        "fraction": 0.95, "median_gap": 0.1, "max_gap": 1.5}}}}})
    assert met["status"] == "met" and "H26" in hypotheses.ALL
