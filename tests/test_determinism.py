"""The determinism checker itself, plus a cross-process check of the one thing that broke before.

A checker that always passes is worse than none, so these tests include cases it must REJECT.
"""
from __future__ import annotations

import pytest

from evaluation.determinism import assert_deterministic, max_abs_difference, run_module_twice


def test_identical_structures_have_zero_difference():
    payload = {"a": [1.0, 2.0], "b": {"c": 3.5}, "d": "text", "e": True}
    result = max_abs_difference(payload, payload)
    assert result["max_diff"] == 0.0
    assert result["mismatches"] == []


def test_small_numeric_difference_is_measured_and_located():
    a = {"x": {"y": [1.0, 2.0, 3.0]}}
    b = {"x": {"y": [1.0, 2.0, 3.0 + 4e-7]}}
    result = max_abs_difference(a, b)
    assert result["max_diff"] == pytest.approx(4e-7)
    assert result["worst_path"] == ".x.y[2]"
    assert_deterministic(a, b, tol=1e-6)


def test_difference_above_tolerance_is_rejected():
    with pytest.raises(AssertionError, match="differ by"):
        assert_deterministic({"x": 1.0}, {"x": 1.1}, tol=1e-6)


def test_structural_mismatch_is_rejected_regardless_of_tolerance():
    with pytest.raises(AssertionError, match="structural mismatch"):
        assert_deterministic({"x": 1.0}, {"y": 1.0}, tol=1e9)
    with pytest.raises(AssertionError, match="structural mismatch"):
        assert_deterministic({"x": [1.0, 2.0]}, {"x": [1.0]}, tol=1e9)


def test_string_change_is_a_mismatch_not_a_difference():
    result = max_abs_difference({"method": "BCa"}, {"method": "percentile"})
    assert result["mismatches"]
    assert result["max_diff"] == 0.0


def test_booleans_are_compared_as_booleans():
    """`True == 1` in Python; a flipped verdict must not pass as a zero numeric difference."""
    with pytest.raises(AssertionError, match="structural mismatch"):
        assert_deterministic({"equivalent": True}, {"equivalent": False})


def test_nan_in_both_places_is_agreement():
    nan = float("nan")
    assert_deterministic({"x": nan}, {"x": nan})


def test_nan_against_a_number_is_a_mismatch():
    with pytest.raises(AssertionError, match="structural mismatch"):
        assert_deterministic({"x": float("nan")}, {"x": 1.0})


def test_timing_fields_are_excluded():
    """Two runs legitimately take different amounts of time; that is not non-determinism."""
    a = {"fit_500_steps_s": 11.0, "value": 1.0, "timestamp": "A"}
    b = {"fit_500_steps_s": 12.5, "value": 1.0, "timestamp": "B"}
    assert_deterministic(a, b, tol=1e-9)


def test_fold_assignment_is_deterministic_across_processes():
    """The concrete regression: the old protocol seeded folds from a salted `hash`."""
    module = "evaluation.cv_utils"
    pytest.importorskip("numpy")
    # cv_utils has no CLI, so compare it through a tiny inline module-equivalent instead.
    from evaluation.cv_utils import make_folds
    import json
    import os
    import subprocess
    import sys
    from evaluation.determinism import ROOT

    script = ("import json;from evaluation.cv_utils import make_folds;"
              "print(json.dumps({'f': make_folds('CGMacros-003', 29, 5, 0).tolist()}))")
    runs = []
    for seed in ("0", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=180)
        assert proc.returncode == 0, proc.stderr
        runs.append(json.loads(proc.stdout))
    assert_deterministic(*runs, tol=0.0)
    assert runs[0]["f"] == make_folds("CGMacros-003", 29, 5, 0).tolist()
    assert module  # documents which module the regression lives in


def test_run_module_twice_reports_a_failing_module():
    with pytest.raises(RuntimeError, match="failed with hash seed"):
        run_module_twice("evaluation.does_not_exist")
