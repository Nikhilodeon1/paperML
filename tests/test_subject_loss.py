"""The shared loss, and the proof that bucketed padding is free.

The padded width is a performance choice worth about a factor of 1.9 on the cross-validation cells.
It is only allowed to be a performance choice if the loss and its gradient do not depend on it, so
that is the first thing asserted here. Everything else in the revision rests on these fits, so the
masking, the regularization centre and the residual sigma are pinned too.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.jax_config import configure

configure()

import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402

from evaluation.cohort_data import load_cgmacros                 # noqa: E402
from personalization.subject_loss import (                        # noqa: E402
    LOWER, MAX_MEALS, MEAL_WIDTH_BUCKETS, RANGE, TARGETS, UPPER, base_params, bucket_width,
    build_params, iauc_loss, pad, sigma_from_residuals, subject_arrays, theta_of,
)


@pytest.fixture(scope="module")
def subject():
    try:
        subjects = load_cgmacros()
    except FileNotFoundError as exc:
        pytest.skip(f"CGMacros not available: {exc}")
    if not subjects:
        pytest.skip("CGMacros loaded but empty")
    return subjects[0]


# --- bucketing ----------------------------------------------------------------------------------

def test_bucket_width_picks_the_smallest_that_fits():
    assert bucket_width(1) == MEAL_WIDTH_BUCKETS[0]
    assert bucket_width(20) == 20
    assert bucket_width(21) == 28
    assert bucket_width(36) == 36
    assert bucket_width(37) == 44
    assert bucket_width(64) == 72
    assert bucket_width(72) == 72


def test_bucket_width_refuses_to_drop_meals():
    """A subject wider than every bucket must raise, not quietly lose meals."""
    with pytest.raises(ValueError, match="exceeds the widest padded width"):
        bucket_width(MEAL_WIDTH_BUCKETS[-1] + 1)


def test_widest_bucket_matches_max_meals():
    assert MEAL_WIDTH_BUCKETS[-1] == MAX_MEALS
    assert list(MEAL_WIDTH_BUCKETS) == sorted(MEAL_WIDTH_BUCKETS)


def test_pad_refuses_to_truncate():
    with pytest.raises(ValueError, match="cannot pad"):
        pad([1.0, 2.0, 3.0], 2)


@pytest.mark.slow
def test_loss_is_bit_identical_across_padded_widths(subject):
    """The claim that makes bucketing safe: width changes cost, not the answer."""
    records = list(subject.records)[:20]
    base = base_params(subject.profile)
    theta = theta_of(base)

    values = {}
    for width in (20, 28, 44, 72):
        loss = iauc_loss(base, subject_arrays(records, width=width), lam=0.01)
        values[width] = float(loss(theta))

    reference = values[20]
    for width, value in values.items():
        assert value == reference, (
            f"width {width} gave {value!r}, width 20 gave {reference!r}; padding is not neutral")


@pytest.mark.slow
def test_gradient_is_identical_across_padded_widths(subject):
    records = list(subject.records)[:20]
    base = base_params(subject.profile)
    theta = theta_of(base)

    grads = {}
    for width in (20, 44, 72):
        loss = iauc_loss(base, subject_arrays(records, width=width), lam=0.01)
        grads[width] = np.asarray(jax.grad(loss)(theta), dtype=float)

    for width, grad in grads.items():
        assert np.array_equal(grad, grads[20]), f"width {width} changed the gradient: {grad}"


@pytest.mark.slow
def test_a_padded_slot_contributes_nothing_despite_a_nonzero_prediction(subject):
    """A zero-carbohydrate slot does NOT predict zero iAUC; the mask is what makes it harmless.

    The smooth positive part of a flat trajectory integrates to a small positive number, so this is
    worth pinning: if the mask were ever dropped, the loss would pick up a constant offset per
    padded slot and every fit would shift.
    """
    from personalization.subject_loss import predict_iauc

    records = list(subject.records)[:5]
    base = base_params(subject.profile)
    arrays = subject_arrays(records, width=20)
    preds = np.asarray(predict_iauc(build_params(base, theta_of(base)), arrays["carbs"],
                                    arrays["fat"], arrays["fiber"]))
    padded = preds[5:]
    assert np.all(padded > 0.0), "expected a nonzero prediction for a zero-carbohydrate slot"
    assert float(np.asarray(arrays["mask"])[5:].sum()) == 0.0


# --- arrays and masking -------------------------------------------------------------------------

def test_subject_arrays_shapes_and_mask(subject):
    arrays = subject_arrays(list(subject.records)[:17])
    assert arrays["n"] == 17
    assert arrays["width"] == 20
    for key in ("carbs", "fat", "fiber", "obs_iauc", "mask"):
        assert arrays[key].shape == (20,)
    mask = np.asarray(arrays["mask"])
    assert mask[:17].sum() == 17.0
    assert mask[17:].sum() == 0.0


def test_subject_arrays_preserves_order(subject):
    records = list(subject.records)[:12]
    arrays = subject_arrays(records)
    assert np.allclose(np.asarray(arrays["carbs"])[:12], [r["carbs_g"] for r in records])
    assert np.allclose(np.asarray(arrays["obs_iauc"])[:12], [r["iauc"] for r in records])


def test_empty_fold_does_not_divide_by_zero(subject):
    """A fold with no training meals must give a finite loss rather than NaN."""
    base = base_params(subject.profile)
    loss = iauc_loss(base, subject_arrays([], width=20), lam=0.01)
    value = float(loss(theta_of(base)))
    assert np.isfinite(value)
    assert value == pytest.approx(0.0), "with no data the loss should be the penalty alone, zero at theta0"


# --- regularization and sigma -------------------------------------------------------------------

@pytest.mark.slow
def test_regularization_is_centred_on_the_population_parameters(subject):
    """At theta0 the penalty is zero; away from it, it grows as the squared scaled distance."""
    base = base_params(subject.profile)
    arrays = subject_arrays(list(subject.records)[:10])
    theta0 = theta_of(base)
    unpenalized = float(iauc_loss(base, arrays, lam=0.0)(theta0))
    penalized = float(iauc_loss(base, arrays, lam=1.0)(theta0))
    assert penalized == pytest.approx(unpenalized, rel=1e-12)

    moved = theta0 + 0.1 * RANGE
    expected = float(iauc_loss(base, arrays, lam=0.0)(moved)) + 1.0 * 3 * 0.1 ** 2
    assert float(iauc_loss(base, arrays, lam=1.0)(moved)) == pytest.approx(expected, rel=1e-10)


@pytest.mark.slow
def test_sigma_uses_n_minus_p(subject):
    base = base_params(subject.profile)
    records = list(subject.records)[:20]
    arrays = subject_arrays(records)
    theta = theta_of(base)
    sigma = sigma_from_residuals(base, arrays, theta)
    assert sigma > 0.0 and np.isfinite(sigma)

    from personalization.subject_loss import predict_iauc
    preds = np.asarray(predict_iauc(build_params(base, theta), arrays["carbs"], arrays["fat"],
                                    arrays["fiber"]))[:20]
    residual = preds - np.array([r["iauc"] for r in records])
    assert sigma == pytest.approx(float(np.sqrt(np.sum(residual ** 2) / (20 - 3))), rel=1e-10)


def test_bounds_are_ordered_and_match_the_targets():
    lo, hi = np.asarray(LOWER, dtype=float), np.asarray(UPPER, dtype=float)
    assert lo.shape == hi.shape == (len(TARGETS),)
    assert np.all(hi > lo)
    assert np.allclose(np.asarray(RANGE, dtype=float), hi - lo)


def test_theta_of_round_trips_through_build_params(subject):
    base = base_params(subject.profile)
    theta = jnp.asarray([0.8, 0.03, 0.02])
    assert np.allclose(np.asarray(theta_of(build_params(base, theta))), np.asarray(theta))
