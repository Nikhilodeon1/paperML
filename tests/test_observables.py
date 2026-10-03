"""The observable ladder: do the smooth forms agree with the hard ones, and do they measure what
their names claim?

Two kinds of test. First, agreement: the smooth forms exist only to make gradients possible, so if
they disagree with the hard forms the fit is optimizing something other than what is reported.
Second, behaviour: a timing observable must respond to timing and not to magnitude, and an area must
do the opposite. Those are the properties the whole ladder argument rests on, so they are asserted on
synthetic curves where the right answer is known rather than only on real data.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.jax_config import configure

configure()

import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402

from simulation.jax_observables import (                           # noqa: E402
    DEFAULT_BETA, OBSERVABLES, Window, baseline, centroid, iauc, observe, peak_time, peak_value,
    post_times, trace,
)

WINDOW = Window()
BASE = 90.0


def _curve(peak_at: float = 60.0, height: float = 60.0, width: float = 30.0,
           window: Window = WINDOW, duration_min: float = 210.0):
    """A Gaussian excursion on `window`'s grid: known peak time, height and area."""
    ts = jnp.arange(0.0, duration_min + 1e-9, window.step_min)
    minutes_since_meal = ts - window.meal_time_min
    glucose = BASE + height * jnp.exp(-0.5 * ((minutes_since_meal - peak_at) / width) ** 2)
    return ts, glucose


# --- the window -----------------------------------------------------------------------------------

def test_window_slices_are_the_documented_samples():
    assert WINDOW.meal_index == 6
    assert WINDOW.pre_slice == (0, 6)          # 30 min of baseline at 5 min steps
    assert WINDOW.post_slice == (6, 43)        # 0 to 180 min inclusive
    assert float(post_times(WINDOW)[0]) == 0.0
    assert float(post_times(WINDOW)[-1]) == 180.0


def test_window_requires_a_long_enough_trajectory():
    long_window = Window(post_min=360.0)
    with pytest.raises(ValueError, match="needs 79 samples"):
        long_window.require(43)
    long_window.require(79)


def test_a_baseline_longer_than_the_lead_in_is_refused():
    with pytest.raises(ValueError, match="before the meal"):
        _ = Window(meal_time_min=10.0, pre_min=30.0).pre_slice


def test_stride_selects_every_third_sample_for_a_coarse_cohort():
    coarse = Window(stride=3)
    times = np.asarray(post_times(coarse))
    assert times[0] == 0.0
    assert np.allclose(np.diff(times), 15.0)


def test_baseline_is_the_pre_meal_mean():
    ts, glucose = _curve()
    assert float(baseline(glucose, WINDOW)) == pytest.approx(
        float(jnp.mean(glucose[0:6])), rel=1e-12)


# --- smooth against hard --------------------------------------------------------------------------

def test_smooth_and_hard_iauc_agree_at_the_default_sharpness():
    ts, glucose = _curve()
    smooth = float(iauc(ts, glucose, WINDOW, beta=DEFAULT_BETA))
    hard = float(iauc(ts, glucose, WINDOW, beta=None))
    assert smooth == pytest.approx(hard, rel=0.01), (smooth, hard)


@pytest.mark.parametrize("beta", [1.0, 5.0, 20.0, 100.0])
def test_smooth_iauc_converges_to_hard_as_beta_grows(beta):
    """The approximation must be controlled: a larger beta is a closer approximation, always above."""
    ts, glucose = _curve()
    hard = float(iauc(ts, glucose, WINDOW, beta=None))
    smooth = float(iauc(ts, glucose, WINDOW, beta=beta))
    assert smooth >= hard - 1e-9, "the softplus positive part cannot understate the area"
    assert smooth - hard < 2000.0 / beta


def test_smooth_iauc_error_shrinks_monotonically_in_beta():
    ts, glucose = _curve()
    hard = float(iauc(ts, glucose, WINDOW, beta=None))
    errors = [abs(float(iauc(ts, glucose, WINDOW, beta=b)) - hard) for b in (1.0, 5.0, 20.0, 100.0)]
    assert all(a > b for a, b in zip(errors, errors[1:])), errors


def test_soft_and_hard_peak_time_agree_on_a_clear_peak():
    ts, glucose = _curve(peak_at=60.0)
    assert float(peak_time(ts, glucose, WINDOW, beta=None)) == pytest.approx(60.0)
    assert float(peak_time(ts, glucose, WINDOW, beta=DEFAULT_BETA)) == pytest.approx(60.0, abs=2.5)


def test_soft_and_hard_peak_value_agree_on_a_clear_peak():
    """Compared against the actual excursion, not against the Gaussian amplitude.

    The curve peaks 60 mg/dL above 90, but the pre-meal baseline is the mean of samples that already
    carry the leading tail of the excursion, so the measured height above baseline is slightly less
    than 60. That is correct behaviour for a baseline-subtracted observable and is what a real
    pre-meal window does too.
    """
    ts, glucose = _curve(height=60.0)
    expected = float(jnp.max(glucose[6:43] - baseline(glucose, WINDOW)))
    assert expected < 60.0
    assert float(peak_value(ts, glucose, WINDOW, beta=None)) == pytest.approx(expected, rel=1e-9)
    assert float(peak_value(ts, glucose, WINDOW, beta=DEFAULT_BETA)) == pytest.approx(
        expected, rel=0.01)


# --- do they measure what they claim? -------------------------------------------------------------

def test_iauc_scales_with_height_and_is_insensitive_to_delay():
    """Area doubles with height and barely moves when the whole excursion is delayed.

    The excursions are narrow (20 min) and peak at 60 and 90 min, so both lie entirely inside the
    0-180 min window. That matters: with a wider curve peaking at 60 min, part of the mass falls
    before the meal and is both excluded from the window and folded into the baseline, so a delayed
    curve would score HIGHER purely as a windowing artifact. The insensitivity being asserted here is
    the real one, and it is the confound the observable ladder exists to break.
    """
    ts, small = _curve(height=30.0, peak_at=60.0, width=20.0)
    _, large = _curve(height=60.0, peak_at=60.0, width=20.0)
    _, delayed = _curve(height=30.0, peak_at=90.0, width=20.0)
    a_small = float(iauc(ts, small, WINDOW, beta=None))
    a_large = float(iauc(ts, large, WINDOW, beta=None))
    a_delayed = float(iauc(ts, delayed, WINDOW, beta=None))
    assert a_large == pytest.approx(2.0 * a_small, rel=0.02)
    assert a_delayed == pytest.approx(a_small, rel=0.02), (a_small, a_delayed)

    # The same two curves are plainly distinguishable by their centroids: timing information the
    # area discards.
    c_small = float(centroid(ts, small, WINDOW, beta=None))
    c_delayed = float(centroid(ts, delayed, WINDOW, beta=None))
    assert c_delayed - c_small == pytest.approx(30.0, abs=4.0), (c_small, c_delayed)


def test_centroid_tracks_delay_and_ignores_magnitude():
    ts, early = _curve(peak_at=50.0, height=30.0)
    _, late = _curve(peak_at=90.0, height=30.0)
    _, taller = _curve(peak_at=50.0, height=90.0)
    c_early = float(centroid(ts, early, WINDOW, beta=None))
    c_late = float(centroid(ts, late, WINDOW, beta=None))
    c_taller = float(centroid(ts, taller, WINDOW, beta=None))
    assert c_late - c_early == pytest.approx(40.0, abs=8.0), (c_early, c_late)
    assert c_taller == pytest.approx(c_early, rel=0.02), (
        "the centroid must be a timing statistic, not a magnitude one")


def test_centroid_is_inside_the_window():
    ts, glucose = _curve(peak_at=60.0)
    value = float(centroid(ts, glucose, WINDOW, beta=None))
    assert 0.0 <= value <= 180.0


def test_centroid_of_a_flat_curve_is_finite():
    """No excursion means no timing information, which must read as a number rather than a NaN."""
    ts = jnp.arange(0.0, 210.0 + 1e-9, 5.0)
    flat = jnp.full_like(ts, BASE)
    value = float(centroid(ts, flat, WINDOW, beta=None))
    assert np.isfinite(value)


def test_trace_is_the_baseline_subtracted_curve():
    ts, glucose = _curve()
    values = np.asarray(trace(ts, glucose, WINDOW))
    assert values.shape == (37,)
    assert values[0] == pytest.approx(float(glucose[6] - baseline(glucose, WINDOW)), rel=1e-12)
    assert values.max() == pytest.approx(
        float(jnp.max(glucose[6:43] - baseline(glucose, WINDOW))), rel=1e-12)
    assert 55.0 < values.max() < 60.0, "height above a tail-inflated baseline, slightly under 60"


def test_trace_on_a_coarse_grid_scores_only_real_samples():
    ts, glucose = _curve()
    assert np.asarray(trace(ts, glucose, Window(stride=3))).shape == (13,)


# --- gradients ------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(OBSERVABLES))
def test_every_smooth_observable_has_a_finite_gradient(name):
    """A NaN here would surface as a failed fit with no explanation, so it is checked directly."""
    window = WINDOW

    def scalar(height):
        ts, glucose = _curve(height=height, window=window)
        value = observe(name, ts, glucose, window, beta=DEFAULT_BETA)
        return jnp.sum(value)

    grad = float(jax.grad(scalar)(40.0))
    assert np.isfinite(grad), f"{name} gradient is {grad}"
    assert grad != 0.0, f"{name} does not respond to the excursion height at all"


def test_peak_time_gradient_responds_to_delay():
    def scalar(delay):
        ts = jnp.arange(0.0, 210.0 + 1e-9, 5.0)
        minutes = ts - WINDOW.meal_time_min
        glucose = BASE + 60.0 * jnp.exp(-0.5 * ((minutes - delay) / 30.0) ** 2)
        return peak_time(ts, glucose, WINDOW, beta=DEFAULT_BETA)

    grad = float(jax.grad(scalar)(60.0))
    assert np.isfinite(grad)
    assert grad > 0.0, "delaying the peak must increase the reported peak time"


def test_hard_forms_are_not_differentiated_by_accident():
    """The hard argmax has a zero gradient; using it in a fit would silently learn nothing."""
    def scalar(delay):
        ts = jnp.arange(0.0, 210.0 + 1e-9, 5.0)
        minutes = ts - WINDOW.meal_time_min
        glucose = BASE + 60.0 * jnp.exp(-0.5 * ((minutes - delay) / 30.0) ** 2)
        return peak_time(ts, glucose, WINDOW, beta=None)

    assert float(jax.grad(scalar)(60.0)) == 0.0


def test_observe_rejects_an_unknown_name():
    ts, glucose = _curve()
    with pytest.raises(ValueError, match="unknown observable"):
        observe("auc", ts, glucose)


# --- on the real engine ---------------------------------------------------------------------------

@pytest.mark.slow
def test_observables_on_a_real_simulated_meal():
    from simulation.jax_engine import JaxPhysioParams, run_meal

    params = JaxPhysioParams.from_population_defaults()
    ts, glucose = run_meal(params, 60.0, meal_time=30.0, duration_min=210.0, step_min=5.0)

    area = float(iauc(ts, glucose, WINDOW, beta=None))
    centre = float(centroid(ts, glucose, WINDOW, beta=None))
    apex = float(peak_time(ts, glucose, WINDOW, beta=None))
    height = float(peak_value(ts, glucose, WINDOW, beta=None))

    # Physiological sanity for a 60 g meal in a population-default subject.
    assert 1000.0 < area < 12000.0, area
    assert 20.0 < apex < 120.0, apex
    assert 10.0 < height < 120.0, height
    assert apex < centre < 150.0, (apex, centre)

    smooth_area = float(iauc(ts, glucose, WINDOW, beta=DEFAULT_BETA))
    assert smooth_area == pytest.approx(area, rel=0.05)


@pytest.mark.slow
def test_a_longer_window_needs_a_longer_simulation():
    from simulation.jax_engine import run_meal, JaxPhysioParams

    params = JaxPhysioParams.from_population_defaults()
    window = Window(post_min=360.0)
    ts, glucose = run_meal(params, 60.0, meal_time=30.0, duration_min=390.0, step_min=5.0)
    window.require(len(ts))
    assert np.isfinite(float(iauc(ts, glucose, window, beta=None)))
