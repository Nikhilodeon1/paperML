"""The decision rules of REPORTS/decision_rules_2026-11-26.md as code: each verdict follows from numbers alone."""
import math

import pytest

from evaluation import rules


def test_r1_stands_only_if_the_shift_grew_and_coverage_note_follows_the_counts():
    assert rules.r1(shift_075=0.19, covers_075=6, covers_exact=44)["account"] == "stands"
    out = rules.r1(shift_075=0.05, covers_075=44, covers_exact=44)
    assert out["account"] == "cut the explanation" and out["coverage_sentence"] == "replace with counts"


def test_r2_and_r3_need_lower_coverage_and_a_larger_shift_on_every_set():
    good = {"covers_0": 43, "covers_075": 10, "shift_0": 0.02, "shift_075": 0.2}
    bad = {**good, "shift_075": 0.01}
    assert rules.agrees(good) and not rules.agrees(bad)
    assert rules.r2([good, good]) == "held on every fitted-truth seed"
    assert rules.r2([good, bad]) == "report as seed dependence"
    assert rules.r3(good, exact_covers=75, n=87)["replicates"] is True
    assert rules.r3(good, exact_covers=60, n=87)["near_nominal_stated_for"] == "CGMacros only"


@pytest.mark.parametrize("counts,expected", [
    ({1.0: (0, 0), 2.0: (0, 0), 4.0: (0, 0)}, "keep: no interval bounded"),
    ({1.0: (0, 0), 2.0: (0, 0), 4.0: (3, 1)}, "say at most 3 of 45; keep zero for the primary box"),
    ({1.0: (1, 0), 2.0: (0, 0), 4.0: (0, 0)}, "state the count; remove the zero statement"),
])
def test_r5_abstract_statement(counts, expected):
    assert rules.r5(counts, n=45) == expected


def test_r6_box_dependence():
    assert rules.r6({1.0: 3, 2.0: 24, 4.0: 24}) == "box-dependent"
    assert rules.r6({1.0: 3, 2.0: 3, 4.0: 3}) == "remove box-dependent"


def test_r7_exact_input_and_pinning():
    out = rules.r7(shift_exact=-0.20, shift_noisy=-0.19, shift_interior=-0.12)
    assert out == {"present_with_exact_input": True, "attributed_to_pinning": False}
    assert rules.r7(-0.02, -0.19, -0.01) == {"present_with_exact_input": False, "attributed_to_pinning": True}


def test_r8_materiality_thresholds():
    base = {"coverage_pct": 87.0, "shift": -0.19}
    assert not rules.r8_material(base, {"coverage_pct": 73.0, "shift": -0.30})
    assert rules.r8_material(base, {"coverage_pct": 72.0, "shift": -0.19})
    assert rules.r8_material(base, {"coverage_pct": 87.0, "shift": 0.01})


def test_r9_and_r10_information_limit():
    assert rules.r9(bounded=10, interior=30) == "keep, with the new count"
    assert rules.r9(bounded=15, interior=30) == "remove and rewrite the information-limit reading"
    low = {"ke": {"bounded": 4, "interior": 31, "covers": 45, "n": 45},
           "ka": {"bounded": 2, "interior": 31, "covers": 45, "n": 45}}
    high = {k: {**v, "bounded": 20} for k, v in low.items()}
    assert rules.r10(low) == "keep, with the new counts" and rules.r10(high) == "rewrite the information-limit reading"


def test_r11_statements():
    out = rules.r11(covers_generator=0, covers_pseudo_bounded=26, bounded=44)
    assert out == {"zero_generator_statement": "keep", "short_of_nominal_statement": "keep"}
    out = rules.r11(covers_generator=2, covers_pseudo_bounded=41, bounded=44)
    assert out == {"zero_generator_statement": "give the count", "short_of_nominal_statement": "remove"}


def test_r13_needs_both_halves_on_both_seeds():
    good = {"shift_naive": 0.21, "shift_calibrated": 0.05, "covers_naive": 13, "covers_calibrated": 33,
            "covers_exact": 43}
    assert rules.r13([good, good]) == "supported"
    assert rules.r13([good, {**good, "covers_calibrated": 27}]) == "mechanism not established"
    assert rules.r13([good, {**good, "shift_calibrated": -0.12}]) == "mechanism not established"


def test_r14_r15_materiality():
    same = {"iauc_bounded_variant": 0, "iauc_bounded_full": 0, "trace_share_variant": 80.0, "trace_share_full": 70.0}
    assert not rules.material_variant(same)
    assert rules.material_variant({**same, "iauc_bounded_variant": 1})
    assert rules.material_variant({**same, "trace_share_full": 65.0})


def test_r19_statements():
    out = rules.r19(exact_random=[43, 40], covers_075_vs_exact=[(13, 43), (10, 40), (6, 44)],
                    widths_random=[(0.32, 0.27), (0.29, 0.31)], fit_bounded=(6, 22), random_bounded=[(37, 23), (35, 25)],
                    rank_075=[0.70, 0.55])
    assert out == {"near_nominal": False, "coverage_falls": True, "without_widening": False,
                   "opposite_directions": True, "ordering": False}


def test_interval_handles_rate_and_log_boxes():
    entry = {"truth": 0.02, "classification": {"verdict": "one-sided", "ci_low": 0.018, "ci_high": None}}
    rate_row = {"parameterization": "rates", "box": {"k": [0.015, 0.04]}, "profiles": {"k": entry}}
    log_row = {"parameterization": "coords", "box": {"k": [math.log(0.015), math.log(0.04)]}, "profiles": {"k": entry}}
    assert rules.clipped_interval(rate_row, "k") == pytest.approx((0.018, 0.04))
    assert rules.clipped_interval(log_row, "k") == pytest.approx((0.018, 0.04))
    assert rules.covers(rate_row, "k") is True


def test_a_rule_with_an_incomplete_sweep_is_withheld(monkeypatch):
    monkeypatch.setattr(rules, "complete_rows", lambda name: None)
    outcomes = rules.evaluate()
    assert outcomes and all(o["status"] == "waiting for data" for o in outcomes.values())
