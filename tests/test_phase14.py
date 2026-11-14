"""Phase 14 (Amendment 10): the pieces that can be tested without result files."""
from __future__ import annotations

import numpy as np

from evaluation import hypotheses, phase14_summary as p14, profile_lik


def _entry(bounded: bool, unsettled: int = 0, better: bool = False) -> dict:
    grid = np.linspace(0.0, 1.0, 9)
    profile = (np.abs(grid - 0.5) * 20.0) if bounded else np.zeros_like(grid)
    return {"n_in_box": 9, "grid": grid.tolist(), "profile": profile.tolist(), "reference": 0.0,
            "unsettled": unsettled, "better_than_theta_hat": better}


def _row(subject, bounded_tau, bounded_p, **kw) -> dict:
    return {"subject_id": subject, "distance_to_bound": {"log_tau1": 0.2, "log_p": 0.2},
            "profiles": {"log_tau1": _entry(bounded_tau, **kw), "log_p": _entry(bounded_p, **kw)}}


def test_settling_counts_flagged_points_and_beaten_profiles():
    rows = [_row("a", True, True, unsettled=3, better=True), _row("b", False, False, unsettled=0)]
    out = p14.settling(rows)
    assert out["profiles"] == 4 and out["points"] == 36 and out["unsettled"] == 6
    assert abs(out["unsettled_fraction"] - 6 / 36) < 1e-12 and out["better_than_estimate"] == 2
    assert p14.settling([]) == {"n": 0}


def test_class_changes_counts_direction():
    short = [_row("a", True, True), _row("b", True, False)]
    long = [_row("a", True, False), _row("b", False, False)]
    out = p14.class_changes(short, long)
    assert out["n"] == 4 and out["changed"] == 2 and out["bounded_to_open"] == 2 and out["open_to_bounded"] == 0


def _block(frac_tau, frac_p, cover_tau=None, cover_p=None) -> dict:
    return {"n": 45, "tau1": {"bounded_interior": {"fraction": frac_tau}, "nominal_coverage": {"fraction": cover_tau}},
            "p": {"bounded_interior": {"fraction": frac_p}, "nominal_coverage": {"fraction": cover_p}}}


def test_h27_rule_is_ten_points_on_both_coordinates():
    met = p14.h27({"real": {"steps_100": _block(0.83, 0.88), "steps_500": _block(0.80, 0.85)}})
    assert met["status"] == "met"
    not_met = p14.h27({"real": {"steps_100": _block(0.83, 0.88), "steps_500": _block(0.60, 0.85)}})
    assert not_met["status"] == "not met" and abs(not_met["difference_points"]["tau1"] + 23.0) < 1e-9
    assert p14.h27({"real": {"steps_100": _block(0.83, 0.88), "steps_500": {"n": 0}}})["status"] == "not evaluated"


def test_h28_rule_is_ninety_percent_coverage_for_both():
    met = p14.h28({"quarter": {"steps_500": _block(0.5, 0.5, 0.93, 0.91)}})
    assert met["status"] == "met"
    assert p14.h28({"quarter": {"steps_500": _block(0.5, 0.5, 0.93, 0.80)}})["status"] == "not met"
    assert p14.h28({})["status"] == "not evaluated"


def test_inner_steps_separate_result_sets():
    config = {**profile_lik.default_config(), "objective": "trace", "parameterization": "coords", "polish": True}
    payload = {"objective": "trace", "bounds_scale": 1.0, "parameterization": "coords", "replica": None, "carb_scale": None,
               "polish": {"start_values": [1.0] * 6}, "cohort": "cgmacros"}
    assert profile_lik._matches(payload, config)                       # an old payload counts as 100 steps
    assert not profile_lik._matches(payload, {**config, "steps": 500})
    assert profile_lik._matches({**payload, "inner_steps": 500}, {**config, "steps": 500})
    assert not profile_lik._matches({**payload, "inner_steps": 500}, config)


def test_h27_and_h28_are_not_evaluated_without_results():
    for fn in (hypotheses.h27, hypotheses.h28):
        assert fn({})["status"] == "not evaluated"
    met = hypotheses.h27({"phase14": {"h27": {"status": "met", "difference_points": {"tau1": 1.0, "p": -2.0}}}})
    assert met["status"] == "met"
