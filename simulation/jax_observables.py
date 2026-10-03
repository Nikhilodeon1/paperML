"""The observables the paper compares: area, timing, and the full trace.

The rejected submission fitted one summary statistic -- incremental area under the curve -- and
concluded that the timing parameters were not identifiable. A reviewer pointed out the obvious
confound: an area is almost blind to timing by construction, so the conclusion might be a property of
the chosen statistic rather than of the model or the data. Answering that requires the same model
fitted to a LADDER of observables, which is what this module provides.

Four observables, all baseline-subtracted against the mean of the 30 min before the meal, all over the
same 0-180 min post-meal window on the same 5 min grid:

* **iauc** -- the incremental area. One number. Sees how much glucose arrived, not when.
* **centroid** -- the time centre of mass of the excursion, `integral t [y]+ dt / integral [y]+ dt`.
  One number, in minutes, carrying timing and almost no magnitude.
* **peak_time** -- when the excursion peaked. One number, timing only, but a weaker statistic than the
  centroid because it depends on a single sample.
* **trace** -- the baseline-subtracted curve itself. Everything the data contain.

Each has a SMOOTH form for gradients and a HARD form for scoring, and the two must agree. The smooth
forms replace the two non-differentiable operations -- the positive part and the argmax -- with a
softplus and a softmax of sharpness `beta`. That approximation is a modelling choice that affects
fitted parameters, so it is stated, and tests pin the agreement between the forms as `beta` grows.

Nothing here modifies `simulation/jax_observation.py`. That module produced the submitted results and
stays exactly as it was, so the regression test in `personalization/fit_general.py` has something
fixed to reproduce.

A note on coarse cohorts. ShanghaiT2DM sampled every 15 min and its stored 5 min grid was filled by
interpolation. A trace objective there must score only the real samples, otherwise two thirds of the
residuals are a smoothness assumption and every interval shrinks accordingly. `Window.stride` selects
every third sample for exactly that reason.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = ["Window", "DEFAULT_BETA", "baseline", "iauc", "centroid", "peak_time", "peak_value",
           "trace", "observe", "OBSERVABLES", "post_indices", "post_times"]

# Sharpness of the smooth positive part and the soft argmax. 10.0 per mg/dL is the value the engine
# already uses for its own softplus relu, and at this sharpness the smooth and hard iAUC agree to
# better than 1% of the hard value on real excursions (tests/test_observables.py).
DEFAULT_BETA = 10.0


@dataclass(frozen=True)
class Window:
    """Which samples of a simulated trajectory an observable reads.

    Index arithmetic is done in Python on these fields, so every slice is a static shape and the
    jitted graph compiles once. `post_min` can be widened to 360 for the window sweep in H2 as long
    as the simulation was run that long.
    """
    meal_time_min: float = 30.0
    pre_min: float = 30.0
    post_min: float = 180.0
    step_min: float = 5.0
    stride: int = 1              # 3 selects the real samples of a 15 min cohort

    @property
    def meal_index(self) -> int:
        return int(round(self.meal_time_min / self.step_min))

    @property
    def pre_slice(self) -> tuple[int, int]:
        """Samples strictly before the meal, covering `pre_min` minutes."""
        n_pre = int(round(self.pre_min / self.step_min))
        start = self.meal_index - n_pre
        if start < 0:
            raise ValueError(
                f"a {self.pre_min:g} min baseline needs {n_pre} samples before the meal, but the "
                f"meal is at sample {self.meal_index}")
        return start, self.meal_index

    @property
    def post_slice(self) -> tuple[int, int]:
        """Samples from the meal to `post_min` minutes after it, inclusive."""
        n_post = int(round(self.post_min / self.step_min))
        return self.meal_index, self.meal_index + n_post + 1

    def require(self, n_samples: int) -> None:
        stop = self.post_slice[1]
        if n_samples < stop:
            raise ValueError(
                f"this window needs {stop} samples (a {self.post_min:g} min response after a meal "
                f"at {self.meal_time_min:g} min) but the trajectory has {n_samples}; run the "
                f"simulation for at least "
                f"{self.meal_time_min + self.post_min:g} min")


def post_indices(window: Window) -> jnp.ndarray:
    start, stop = window.post_slice
    return jnp.arange(start, stop, window.stride)


def post_times(window: Window) -> jnp.ndarray:
    """Minutes since the meal, for each scored sample. Starts at zero by construction."""
    return (post_indices(window) - window.meal_index) * window.step_min


def _positive(x, beta: float | None):
    """Positive part: `softplus(beta x) / beta` when smooth, `max(0, x)` when hard.

    The smooth form is strictly positive everywhere, including where the excursion is below baseline,
    which is why it slightly overstates an area; `beta` controls by how much.
    """
    if beta is None:
        return jnp.maximum(x, 0.0)
    return jax.nn.softplus(beta * x) / beta


def baseline(glucose, window: Window = Window()) -> jnp.ndarray:
    """Pre-meal fasting baseline: the mean of the samples in the `pre_min` window before the meal."""
    start, stop = window.pre_slice
    return jnp.mean(glucose[start:stop])


def _excursion(glucose, window: Window):
    base = baseline(glucose, window)
    return glucose[jnp.asarray(post_indices(window))] - base


def iauc(ts, glucose, window: Window = Window(), beta: float | None = DEFAULT_BETA):
    """Incremental area under the curve, in mg/dL*min. Trapezoid over the positive excursion."""
    del ts                      # the grid is defined by the window, not read from the trajectory
    excursion = _positive(_excursion(glucose, window), beta)
    return jnp.trapezoid(excursion, post_times(window))


def centroid(ts, glucose, window: Window = Window(), beta: float | None = DEFAULT_BETA,
             floor: float = 1e-6):
    """Time centre of mass of the excursion, in minutes since the meal.

    `integral t [y]+ dt / integral [y]+ dt`. Nearly pure timing information: scaling an excursion
    leaves it unchanged, while delaying one shifts it directly. That is what makes it the rung of the
    ladder between an area and a full trace.

    `floor` keeps the denominator away from zero for a meal with no detectable excursion. Without it
    the gradient is undefined exactly where the data are least informative, which would surface as a
    NaN in a fit rather than as the weak constraint it really is.
    """
    del ts
    weights = _positive(_excursion(glucose, window), beta)
    times = post_times(window)
    mass = jnp.trapezoid(weights, times)
    moment = jnp.trapezoid(weights * times, times)
    return moment / jnp.maximum(mass, floor)


def peak_time(ts, glucose, window: Window = Window(), beta: float | None = DEFAULT_BETA):
    """Time of the maximum, in minutes since the meal.

    Smooth form: a softmax over the excursion, so `beta` controls how sharply the weight concentrates
    on the maximum. Hard form: the argmax. The softmax is taken over the excursion rather than over
    raw glucose so that `beta` has the same units and meaning as in the positive part.
    """
    del ts
    excursion = _excursion(glucose, window)
    times = post_times(window)
    if beta is None:
        return times[jnp.argmax(excursion)]
    return jnp.sum(jax.nn.softmax(beta * excursion) * times)


def peak_value(ts, glucose, window: Window = Window(), beta: float | None = DEFAULT_BETA):
    """Height of the maximum above baseline, in mg/dL."""
    del ts
    excursion = _excursion(glucose, window)
    if beta is None:
        return jnp.max(excursion)
    return jnp.sum(jax.nn.softmax(beta * excursion) * excursion)


def trace(ts, glucose, window: Window = Window(), beta: float | None = None):
    """The baseline-subtracted excursion itself, one value per scored sample.

    Returns a vector, not a scalar: the objective built on it is a mean squared error over samples.
    Needs no smoothing -- subtraction and indexing are already differentiable -- so `beta` is accepted
    and ignored, which keeps the call signature uniform across observables.
    """
    del ts, beta
    return _excursion(glucose, window)


OBSERVABLES = {
    "iauc": iauc,
    "centroid": centroid,
    "peak_time": peak_time,
    "peak_value": peak_value,
    "trace": trace,
}


def observe(name: str, ts, glucose, window: Window = Window(),
            beta: float | None = DEFAULT_BETA):
    """Dispatch by name, so an analysis can take the observable as configuration."""
    if name not in OBSERVABLES:
        raise ValueError(f"unknown observable {name!r}; known: {', '.join(sorted(OBSERVABLES))}")
    return OBSERVABLES[name](ts, glucose, window, beta)
