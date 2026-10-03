"""Fold assignment must be identical in every process. This is the test that says so.

The old protocol seeded from `hash(subject_id)`, which Python salts per process, so the held-out
comparison could not be reproduced exactly. The cross-process test below is the one that would have
caught it: it runs the assignment in a FRESH interpreter with a different `PYTHONHASHSEED` and
requires the same answer.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluation.cv_utils import fold_seed, make_folds, split_indices

ROOT = Path(__file__).resolve().parents[1]


def test_folds_are_balanced():
    folds = make_folds("CGMacros-001", 23, k=5, seed=0)
    counts = np.bincount(folds, minlength=5)
    assert counts.sum() == 23
    assert counts.max() - counts.min() <= 1, counts


def test_every_meal_gets_exactly_one_fold():
    folds = make_folds("CGMacros-007", 17, k=5, seed=3)
    assert folds.shape == (17,)
    assert set(np.unique(folds)).issubset(set(range(5)))
    for fold in range(5):
        train, test = split_indices(folds, fold)
        assert len(train) + len(test) == 17
        assert not set(train) & set(test)


def test_repeats_differ():
    """Different repeat seeds must give genuinely different assignments, or R repeats buy nothing."""
    a = make_folds("CGMacros-001", 40, k=5, seed=0)
    b = make_folds("CGMacros-001", 40, k=5, seed=1)
    assert not np.array_equal(a, b)


def test_subjects_differ():
    a = make_folds("CGMacros-001", 40, k=5, seed=0)
    b = make_folds("CGMacros-002", 40, k=5, seed=0)
    assert not np.array_equal(a, b)


def test_same_call_is_identical_in_process():
    for _ in range(5):
        assert np.array_equal(make_folds("S", 31, k=5, seed=2), make_folds("S", 31, k=5, seed=2))


def test_fold_seed_is_a_known_constant():
    """Pin the derivation. If this changes, every committed result file was produced under a
    different fold assignment and must be regenerated rather than silently compared."""
    assert fold_seed("CGMacros-001", 0) == int.from_bytes(
        __import__("hashlib").sha256(b"CGMacros-001|0").digest()[:8], "big")


def test_deterministic_across_processes():
    """Two fresh interpreters, two different hash seeds, one answer."""
    script = (
        "import json, numpy as np;"
        "from evaluation.cv_utils import make_folds;"
        "print(json.dumps({s: make_folds(s, n, 5, 0).tolist() "
        "for s, n in [('CGMacros-001', 23), ('CGMacros-042', 11), ('x y/z', 7)]}))"
    )
    outputs = []
    for hashseed in ("0", "12345"):
        env = {**__import__("os").environ, "PYTHONHASHSEED": hashseed,
               "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=180)
        assert proc.returncode == 0, proc.stderr
        outputs.append(json.loads(proc.stdout))
    assert outputs[0] == outputs[1], "fold assignment changed between processes"


def test_fewer_meals_than_folds_is_allowed_and_visible():
    """With 3 meals and 5 folds some folds are necessarily empty; that is reported, not hidden."""
    folds = make_folds("tiny", 3, k=5, seed=0)
    empty = [f for f in range(5) if (folds == f).sum() == 0]
    assert len(empty) == 2
    for fold in empty:
        _, test = split_indices(folds, fold)
        assert test.size == 0


def test_zero_meals():
    assert make_folds("none", 0, k=5, seed=0).shape == (0,)


@pytest.mark.parametrize("bad", [-1])
def test_rejects_negative_meal_count(bad):
    with pytest.raises(ValueError, match="non-negative"):
        make_folds("s", bad)


def test_rejects_k_below_two():
    with pytest.raises(ValueError, match="at least 2"):
        make_folds("s", 10, k=1)
