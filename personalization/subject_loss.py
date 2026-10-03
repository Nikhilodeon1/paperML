"""One definition of a subject's data arrays, parameter vector, and iAUC loss.

Every analysis in the revision needs the same three things: a subject's meals as fixed-shape padded
arrays (so the jitted solve compiles once and is reused across subjects and folds), the parameter
vector in a fixed order, and the loss that the original submission minimized. Defining them in one
place rather than in each script is what keeps a figure and a table from disagreeing about what was
fitted.

The padding is not an optimization detail. `jax.jit` recompiles for every new input shape; a
subject with 36 meals and one with 41 would otherwise each pay tens of seconds of compilation, and
a five-fold repeat over 45 subjects would spend more time compiling than solving. Meals are padded
and masked, so a handful of shapes cover every subject and every fold.

PADDING WIDTH IS BUCKETED, which matters more than it sounds. The cost of a fit is close to linear
in the padded width, because a padded slot still solves a full ODE and only afterwards has its
residual multiplied by zero. Measured on this engine, 100 steps cost 1.09 s at width 8 and 4.51 s at
width 72. The cohort's median subject has 36 meals and a training fold of one has about 29, so
padding everything to `MAX_MEALS = 72` spends roughly half the run solving meals that do not exist:
on the real meal counts, bucketing is 1.6x faster on whole-subject fits and 1.9x faster on
cross-validation folds, which is where almost all of the compute goes.

Bucketing changes no result. A padded slot carries `carbs_g = 0`, whose simulated iAUC is a small
positive number rather than zero (the smooth positive part of a flat trajectory is not exactly
zero), but the residual is multiplied by `mask = 0`, so it contributes nothing to the loss and
nothing to the gradient. Adding exact zeros to a sum does not change it in IEEE arithmetic, so the
loss at a given theta is bit-identical across widths -- asserted in `tests/test_subject_loss.py`.

This module deliberately contains only the iAUC objective, which is the one the rejected submission
used and the one the revision has to reproduce exactly. The other observables (centroid, peak time,
full trace) are built on top of it in `personalization/objectives.py`.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from personalization.gradient_fit import MAX_MEALS, PARAM_RANGES, TARGETS
from simulation import PhysioParams
from simulation.jax_engine import JaxPhysioParams, run_meal
from simulation.jax_observation import iauc

__all__ = [
    "TARGETS", "MAX_MEALS", "MEAL_WIDTH_BUCKETS", "LOWER", "UPPER", "RANGE", "bucket_width",
    "pad", "subject_arrays", "base_params", "theta_of", "build_params", "predict_iauc",
    "iauc_loss", "sigma_from_residuals",
]

# Padded widths in use. Six shapes cover every subject and every fold of the cohort, so the jitted
# solve compiles six times per process at worst -- about 3 s each, and cached to disk across
# processes -- instead of once per distinct meal count. The spacing is tighter where the cohort is
# dense (the median subject has 36 meals, a training fold about 29) and loose in the tail.
MEAL_WIDTH_BUCKETS = (20, 28, 36, 44, 56, 72)

LOWER = jnp.array([PARAM_RANGES[name][0] for name in TARGETS])
UPPER = jnp.array([PARAM_RANGES[name][1] for name in TARGETS])
RANGE = UPPER - LOWER

# The simulation window every observable is defined on: meal at 30 min, 180 min of response, and a
# 30 min pre-meal baseline, sampled every 5 min. Matches the CGMacros loader exactly.
MEAL_TIME_MIN = 30.0
DURATION_MIN = 210.0
STEP_MIN = 5.0


def bucket_width(n_meals: int) -> int:
    """The padded width to use for `n_meals` real meals: the smallest bucket that fits them.

    Raises rather than truncating when a subject has more meals than the widest bucket. Silently
    dropping meals would change that subject's fit without changing anything visible in the result
    file, which is the worst possible failure mode here.
    """
    if n_meals < 0:
        raise ValueError(f"n_meals must be non-negative, got {n_meals}")
    for width in MEAL_WIDTH_BUCKETS:
        if n_meals <= width:
            return width
    raise ValueError(
        f"{n_meals} meals exceeds the widest padded width {MEAL_WIDTH_BUCKETS[-1]}; add a wider "
        f"bucket to MEAL_WIDTH_BUCKETS rather than dropping meals")


def pad(values, length: int) -> jnp.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size > length:
        raise ValueError(f"cannot pad {values.size} values into width {length}")
    out = np.zeros(length)
    out[:values.size] = values
    return jnp.asarray(out)


def subject_arrays(records, width: int | None = None) -> dict:
    """A subject's meals as padded, masked, fixed-shape arrays.

    `records` is a sequence of the dictionaries the cohort loader produces, or any subset of them
    (a training fold). `n` is the true count; everything past it is masked out. `width` defaults to
    the smallest bucket that fits, and can be forced to compare widths or to share one compiled
    shape across a batch.
    """
    records = list(records)
    width = bucket_width(len(records)) if width is None else width
    return {
        "carbs": pad([r["carbs_g"] for r in records], width),
        "fat": pad([r["fat_g"] for r in records], width),
        "fiber": pad([r["fiber_g"] for r in records], width),
        "obs_iauc": pad([r["iauc"] for r in records], width),
        "mask": pad([1.0] * len(records), width),
        "n": len(records),
        "width": width,
    }


def base_params(profile: dict) -> JaxPhysioParams:
    """The subject's population-default parameters, from biometrics alone.

    This is both the starting point of every fit and the centre of the regularization term, so a
    change here moves every result; it is read from the same `PhysioParams.from_profile` the
    product code uses rather than reimplemented.
    """
    return JaxPhysioParams.from_numpy(PhysioParams.from_profile(profile))


def theta_of(params: JaxPhysioParams) -> jnp.ndarray:
    """The free parameters of `params`, in `TARGETS` order."""
    return jnp.array([float(getattr(params, name)) for name in TARGETS])


def build_params(base: JaxPhysioParams, theta) -> JaxPhysioParams:
    """`base` with the free parameters replaced by `theta`, differentiably."""
    out = base
    for i, name in enumerate(TARGETS):
        out = eqx.tree_at(lambda m, n=name: getattr(m, n), out, theta[i])
    return out


def _meal_iauc(params, carbs, fat, fiber):
    ts, glucose = run_meal(params, carbs, fat, fiber, meal_time=MEAL_TIME_MIN,
                           duration_min=DURATION_MIN, step_min=STEP_MIN)
    return iauc(ts, glucose)


predict_iauc = jax.vmap(_meal_iauc, in_axes=(None, 0, 0, 0))


def iauc_loss(base: JaxPhysioParams, arrays: dict, lam: float = 0.0):
    """Masked mean squared iAUC error, optionally regularized toward the population parameters.

    `lam = 0` is required for profile likelihood: the chi-square threshold is a statement about a
    likelihood, and a penalty toward a prior would narrow every profile and turn a structurally
    flat direction into an apparently bounded one. `lam = 0.01` reproduces the original fit.
    """
    theta0 = theta_of(base)

    def loss(theta):
        params = build_params(base, theta)
        preds = predict_iauc(params, arrays["carbs"], arrays["fat"], arrays["fiber"])
        residual = arrays["mask"] * (preds - arrays["obs_iauc"])
        recon = jnp.sum(residual ** 2) / jnp.maximum(jnp.sum(arrays["mask"]), 1.0)
        return recon + lam * jnp.sum(((theta - theta0) / RANGE) ** 2)

    return loss


def sigma_from_residuals(base: JaxPhysioParams, arrays: dict, theta) -> float:
    """Residual standard deviation at `theta`, for the Fisher matrix and the profile threshold.

    Uses `n - p` in the denominator, `p` being the three free parameters: with a median of 36 meals
    the difference from `n` is a few percent of the variance, which is enough to matter for an
    interval width.
    """
    params = build_params(base, jnp.asarray(theta))
    preds = np.asarray(predict_iauc(params, arrays["carbs"], arrays["fat"], arrays["fiber"]))
    mask = np.asarray(arrays["mask"]) > 0.5
    residual = preds[mask] - np.asarray(arrays["obs_iauc"])[mask]
    dof = max(int(mask.sum()) - len(TARGETS), 1)
    return float(np.sqrt(np.sum(residual ** 2) / dof))
