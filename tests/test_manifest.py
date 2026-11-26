"""A summary must never be built from a partial sweep (reviewer plan, Stage 0)."""
import pytest

from evaluation import manifest


def _fake(monkeypatch, expected, found):
    monkeypatch.setattr(manifest, "expected_units", lambda config: list(expected))
    monkeypatch.setattr(manifest, "found_units", lambda config: set(found))


def test_check_names_the_missing_units(monkeypatch):
    _fake(monkeypatch, ["a", "b", "c"], ["a", "c"])
    report = manifest.check({"cohort": "cgmacros"})
    assert report == {"expected": 3, "found": 2, "missing": ["b"], "unexpected": []}


def test_equal_counts_with_different_units_are_incomplete(monkeypatch):
    _fake(monkeypatch, ["a", "b"], ["a", "z"])
    report = manifest.check({"cohort": "cgmacros"})
    assert report["missing"] == ["b"] and report["unexpected"] == ["z"]
    assert not manifest.is_complete(report)


def test_require_raises_and_lists_every_gap(monkeypatch):
    _fake(monkeypatch, ["a", "b"], ["a"])
    with pytest.raises(manifest.IncompleteSweep) as error:
        manifest.require({"first": {}, "second": {}})
    assert "first" in str(error.value) and "second" in str(error.value) and "b" in str(error.value)


def test_require_passes_when_complete(monkeypatch):
    _fake(monkeypatch, ["a"], ["a"])
    manifest.require({"only": {}})


def test_an_empty_expectation_is_never_complete(monkeypatch):
    _fake(monkeypatch, [], [])
    assert not manifest.is_complete(manifest.check({}))


def test_phase21_lists_every_planned_sweep():
    names = set(manifest.phase21_sweeps())
    assert {"real", "hall_real", "shanghai_real", "shanghai_rep0.0", "shanghai_rep0.75"} <= names
    assert {f"fit{cv}_s0" for cv in (0.0, 0.1, 0.25, 0.5, 0.75)} <= names
    assert {f"rand{cv}_s{seed}" for cv in (0.0, 0.25, 0.75) for seed in (0, 1)} <= names
    assert {f"fit{cv}_s{seed}" for cv in (0.0, 0.75) for seed in (1, 2)} <= names
    assert len(names) == 20


def _row(verdict, profile, threshold, n_in_box):
    return {"subject_id": "s1", "profiles": {"x": {"profile": profile, "n_in_box": n_in_box,
                                                   "classification": {"verdict": verdict, "threshold": threshold}}}}


def test_a_bounded_interval_with_no_point_below_the_threshold_is_flagged():
    rows = [_row("identifiable", [9.0, 5.0, 9.0, 0.0], 2.0, 3)]       # the only low point is beyond the box
    assert manifest.spurious_bounded(rows) == [("s1", "x")]


def test_ordinary_verdicts_are_not_flagged():
    rows = [_row("identifiable", [9.0, 1.0, 9.0], 2.0, 3), _row("one-sided", [9.0, 5.0, 9.0], 2.0, 3)]
    assert manifest.spurious_bounded(rows) == []
