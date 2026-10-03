"""Gradient fit to the PEAK-TIME observable (parallel to gradient_fit.py's iAUC fit).

Peak time (minutes to the postprandial glucose peak) is a TIMING observable, governed by the gastric
emptying / carb absorption parameters — the ones shown non-identifiable from iAUC. This module fits
Si + gastric + carb to peak time to test whether the timing parameters become identifiable (and
predictive) when the target rewards timing rather than area.

Same one-compile machinery as gradient_fit.py (padded meals + vmap + jitted value-and-grad + Adam +
range projection). Fit uses the differentiable soft-argmax peak time; held-out scoring uses the hard
argmax (the interpretable "minutes to peak"). New module — the committed iAUC path is untouched.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from simulation.jax_engine import run_meal
from simulation.jax_observation import peak_time as _soft_peak_time

MAX_MEALS = 72
_MEAL_T = 30.0

PARAM_RANGES = {
    "insulin_sensitivity": (0.30, 1.60),
    "gastric_emptying": (0.015, 0.040),
    "carb_absorption": (0.012, 0.032),
}
TARGETS = ("insulin_sensitivity", "gastric_emptying", "carb_absorption")
_LO = jnp.array([PARAM_RANGES[k][0] for k in TARGETS], dtype=jnp.float32)
_HI = jnp.array([PARAM_RANGES[k][1] for k in TARGETS], dtype=jnp.float32)
_RNG = _HI - _LO


def _meal_soft_peaktime(params, carbs):
    ts, g = run_meal(params, carbs, meal_time=_MEAL_T, duration_min=210.0, step_min=5.0)
    return _soft_peak_time(ts, g, temperature=5.0) - _MEAL_T   # minutes post-meal


_meals_peaktime = jax.vmap(_meal_soft_peaktime, in_axes=(None, 0))


def predict_peaktime(params, carbs_g) -> float:
    """Hard-argmax peak time (min post-meal) for held-out scoring."""
    ts, g = run_meal(params, jnp.asarray(carbs_g, jnp.float32), meal_time=_MEAL_T,
                     duration_min=210.0, step_min=5.0)
    g = np.asarray(g); ts = np.asarray(ts)
    post = ts >= _MEAL_T
    tp = ts[post]; gp = g[post]
    return float(tp[int(gp.argmax())] - _MEAL_T)


def _build(base, theta):
    p = base
    for i, name in enumerate(TARGETS):
        p = eqx.tree_at(lambda m, n=name: getattr(m, n), p, theta[i])
    return p


def _loss(theta, base, carbs, obs, mask, reg_coef):
    p = _build(base, theta)
    preds = _meals_peaktime(p, carbs)
    recon = jnp.sum(mask * (preds - obs) ** 2) / jnp.maximum(jnp.sum(mask), 1.0)
    theta0 = jnp.array([getattr(base, n) for n in TARGETS])
    reg = jnp.sum(((theta - theta0) / _RNG) ** 2)
    return recon + reg_coef * reg


_loss_grad = jax.jit(jax.value_and_grad(_loss))


def _pad(vals):
    a = np.zeros(MAX_MEALS, dtype=np.float32)
    a[:len(vals)] = vals
    return jnp.asarray(a)


def fit_parameters(meals: list[dict], n_steps: int = 150, learning_rate: float = 0.02,
                   regularization: float = 0.01, base=None) -> dict:
    """Fit (Si, gastric, carb) to peak time. meals: {carbs_g, observed_peaktime}."""
    from simulation.jax_engine import JaxPhysioParams
    base = base or JaxPhysioParams.from_population_defaults()
    n = len(meals)
    carbs = _pad([float(m["carbs_g"]) for m in meals])
    obs = _pad([float(m["observed_peaktime"]) for m in meals])
    mask = _pad([1.0] * n)
    reg_coef = jnp.asarray(regularization, jnp.float32)

    theta0 = jnp.array([float(getattr(base, k)) for k in TARGETS], dtype=jnp.float32)
    opt = optax.adam(learning_rate)
    theta = theta0
    st = opt.init(theta)
    grad_abs = []
    for _ in range(n_steps):
        _l, g = _loss_grad(theta, base, carbs, obs, mask, reg_coef)
        up, st = opt.update(g, st)
        theta = optax.apply_updates(theta, up)
        theta = jnp.clip(theta, _LO, _HI)
        grad_abs.append(np.abs(np.asarray(g)))

    rng_np = np.asarray(_RNG)
    grad_norms = {k: float(np.mean([ga[i] for ga in grad_abs]) * rng_np[i])
                  for i, k in enumerate(TARGETS)}
    return {"params": _build(base, theta),
            "theta": {k: float(theta[i]) for i, k in enumerate(TARGETS)},
            "gradient_norms": grad_norms}
