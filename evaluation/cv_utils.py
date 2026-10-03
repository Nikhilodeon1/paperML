"""Deterministic cross-validation fold assignment.

The previous protocol (`evaluation/unified_kfold.py`) seeded each subject's fold draw with
`abs(hash(subject_id))`. Python's `hash` for `str` is salted per process, so that assignment
changed between interpreter runs unless `PYTHONHASHSEED` happened to be set first. Two
consequences the reviewers were right to object to: the held-out comparison could not be
reproduced exactly, and it used a single fold assignment rather than repeats.

This module replaces it. The subject's fold vector is a pure function of
`(subject_id, n_meals, k, seed)`, derived through SHA-256, so it is identical in every process,
on every machine, and across Python versions. Folds are balanced by permutation rather than by
independent uniform draws, which also removes the chance of an empty fold on subjects with few
meals.
"""
from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["fold_seed", "make_folds", "split_indices"]


def fold_seed(subject_id: str, seed: int) -> int:
    """A 64-bit seed that depends only on the subject identifier and the repeat seed.

    SHA-256 rather than `hash`: stable across processes and interpreter versions, and spreads
    neighbouring identifiers (`CGMacros-001`, `CGMacros-002`) across the seed space.
    """
    digest = hashlib.sha256(f"{subject_id}|{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def make_folds(subject_id: str, n_meals: int, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index in `0..k-1` for each of a subject's `n_meals` meals.

    Balanced: fold sizes differ by at most one meal. When `n_meals < k` some folds are empty by
    necessity; callers skip a fold with an empty train or test set and the result file records
    how many such skips occurred, rather than silently dropping the subject.
    """
    if n_meals < 0:
        raise ValueError(f"n_meals must be non-negative, got {n_meals}")
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    rng = np.random.default_rng(fold_seed(subject_id, seed))
    order = rng.permutation(n_meals)
    folds = np.empty(n_meals, dtype=np.int64)
    folds[order] = np.arange(n_meals) % k
    return folds


def split_indices(folds: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """`(train_idx, test_idx)` for one fold of an assignment from :func:`make_folds`."""
    folds = np.asarray(folds)
    test = np.flatnonzero(folds == fold)
    train = np.flatnonzero(folds != fold)
    return train, test
