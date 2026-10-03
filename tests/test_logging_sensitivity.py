"""Event-log sensitivity — locks in the ordering that drives the logging-UX decision."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.logging_sensitivity import (_MEALS, perturb_carbs, perturb_timing,
                                            perturb_type, run_sensitivity)


@pytest.fixture(scope="module")
def result():
    return run_sensitivity(n_trials=20, seed=1)


def _by(result, label):
    return next(c for c in result["single"] if c.label == label)


def test_carb_TYPE_error_dominates(result):
    """The finding: confusing WHICH carb (GI) hurts far more than getting the AMOUNT wrong.
    If this flips, the whole logging-UX recommendation flips, so it is asserted directly."""
    typ = _by(result, "carb TYPE (GI) swap").iauc_pct
    amt = _by(result, "carb amount +/-50%").iauc_pct
    tim = _by(result, "meal timing +/-30min").iauc_pct
    assert typ > amt > tim, f"ordering changed: type={typ} amount={amt} timing={tim}"
    assert typ > 2 * amt, "carb type should dominate amount by a wide margin"


def test_timing_is_cheap_at_small_offsets(result):
    """+/-15 min barely moves iAUC — auto-timestamping is low-value-at-risk."""
    assert _by(result, "meal timing +/-15min").iauc_pct < 3.0


def test_error_grows_monotonically_with_carb_perturbation(result):
    p30 = _by(result, "carb amount +/-30%").iauc_pct
    p50 = _by(result, "carb amount +/-50%").iauc_pct
    p75 = _by(result, "carb amount +/-75%").iauc_pct
    assert p30 < p50 < p75


def test_perturbations_are_seeded_and_shaped():
    rng = np.random.default_rng(0)
    c = perturb_carbs(_MEALS, 0.5, rng)
    assert all(m["food"] == o["food"] and m["grams"] != o["grams"]  # only grams move
               for m, o in zip(_MEALS, c))
    rng = np.random.default_rng(0)
    t = perturb_timing(_MEALS, 30, rng)
    assert all(m["grams"] == o["grams"] and m["t_min"] != o["t_min"]  # only time moves
               for m, o in zip(_MEALS, t))
    rng = np.random.default_rng(0)
    ty = perturb_type(_MEALS, rng, p=1.0)
    assert all(m["grams"] == o["grams"] and m["food"] != o["food"]  # only food type moves
               for m, o in zip(_MEALS, ty))


def test_interaction_does_not_explode(result):
    """Timing+carb together should be near the larger single effect, not a blow-up —
    reassurance that the two errors don't compound catastrophically."""
    both = result["interaction"].iauc_pct
    assert both < 40.0
