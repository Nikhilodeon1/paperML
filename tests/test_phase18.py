"""Phase 18 (Amendment 14): the pieces that can be tested without result files."""
from __future__ import annotations

import numpy as np

from evaluation import hypotheses, phase18_summary as p18, profile_lik, subject_source
from evaluation.cv_utils import make_time_folds


def test_time_folds_are_contiguous_balanced_blocks():
    folds = make_time_folds(11, 5)
    assert folds.tolist() == sorted(folds.tolist())
    assert set(folds.tolist()) == {0, 1, 2, 3, 4}
    sizes = np.bincount(folds)
    assert sizes.max() - sizes.min() <= 1
    assert make_time_folds(0, 5).size == 0


def test_forward_scheme_trains_only_on_earlier_blocks():
    folds = make_time_folds(10, 5)
    for fold in range(5):
        train = np.flatnonzero(folds < fold)
        test = np.flatnonzero(folds == fold)
        assert (len(train) == 0) == (fold == 0)
        assert all(t < s for t in train for s in test)


def test_replica_arguments_carry_the_systematic_error():
    config = {"cohort": "cgmacros", "min_meals": 10, "limit": None}
    base = {"seed": 0, "cgm": True, "carb_cv": 0.0}
    plain = subject_source._replica_args(config, base, "CGMacros-001")
    biased = subject_source._replica_args(config, {**base, "carb_bias": 0.25}, "CGMacros-001")
    assert plain != biased and plain[-1] == 0.0 and biased[-1] == 0.25


def test_systematic_error_is_one_common_multiplier_per_subject():
    config = {"cohort": "cgmacros", "min_meals": 10, "limit": None}
    base = {"seed": 0, "cgm": False, "carb_cv": 0.0}
    plain, _, _, _ = subject_source._replica_cached(*subject_source._replica_args(config, base, "CGMacros-001"))
    biased, _, _, _ = subject_source._replica_cached(
        *subject_source._replica_args(config, {**base, "carb_bias": 0.25}, "CGMacros-001"))
    ratio = np.array([b["carbs_g"] / p["carbs_g"] for b, p in zip(biased, plain) if p["carbs_g"] > 0])
    assert ratio.size > 3 and np.ptp(ratio) < 1e-9 and abs(ratio[0] - 1.0) > 1e-6


def test_result_sets_are_separated_by_the_noise_source():
    base = {**profile_lik.default_config(), "objective": "trace", "parameterization": "coords", "polish": True}
    payload = {"objective": "trace", "bounds_scale": 1.0, "cohort": "cgmacros", "parameterization": "coords",
               "inner_steps": 100, "replica": None, "carb_scale": None, "noise_structure": "homoskedastic",
               "profile_only": None, "polish": {"start_values": [0.0] * 6}}
    assert profile_lik._matches(payload, base)                                    # an old payload counts as the pilot noise model
    assert not profile_lik._matches(payload, {**base, "noise_source": "final"})
    assert profile_lik._matches({**payload, "noise_source": "final"}, {**base, "noise_source": "final"})


def _block(cover_tau, cover_p, bound_tau=0.5, bound_p=0.5):
    return {"n": 45, "tau1": {"nominal_coverage": {"fraction": cover_tau}, "bounded_interior": {"fraction": bound_tau}},
            "p": {"nominal_coverage": {"fraction": cover_p}, "bounded_interior": {"fraction": bound_p}}}


def test_coverage_rules_need_both_coordinates():
    assert p18.h35(_block(0.93, 0.91))["status"] == "met"
    assert p18.h35(_block(0.93, 0.80))["status"] == "not met"
    assert p18.h37(_block(0.92, 0.95))["status"] == "met"
    assert p18.h35({})["status"] == "not evaluated"


def test_pattern_rule_is_thirty_points_both_ways():
    blocks = {"0": _block(0.93, 0.96, 0.17, 0.23), "0.75": _block(0.11, 0.31, 0.88, 0.78)}
    assert p18.pattern_rule(blocks, 30.0)["status"] == "met"
    weak = {"0": _block(0.93, 0.96, 0.17, 0.23), "0.75": _block(0.80, 0.31, 0.88, 0.78)}
    assert p18.pattern_rule(weak, 30.0)["status"] == "not met"
    assert p18.pattern_rule({}, 30.0)["status"] == "not evaluated"
    both = p18.h36({"seed1": blocks, "seed2": blocks})
    assert both["status"] == "met"
    assert p18.h36({"seed1": blocks})["status"] == "not evaluated"


def test_hypotheses_are_not_evaluated_without_results():
    for fn in (hypotheses.h35, hypotheses.h36, hypotheses.h37, hypotheses.h38):
        assert fn({})["status"] == "not evaluated"


def test_a_replica_of_the_shanghai_cohort_finds_its_stored_reference_fit():
    import pytest
    from evaluation.cohort_data import load_cohort
    from evaluation.ladder import a4_results
    if not a4_results("iauc", 1.0, cohort="shanghai"):
        pytest.skip("no stored Shanghai fits in this checkout")
    subject = load_cohort("shanghai", min_meals=10)[0]
    truth = subject_source.true_parameters(subject, subject.subject_id)
    assert set(truth) == {"insulin_sensitivity", "gastric_emptying", "carb_absorption"}
