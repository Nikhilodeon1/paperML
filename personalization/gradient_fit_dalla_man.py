"""Gradient-based per-subject inference through the Dalla Man model (parallel to gradient_fit.py).

Same machinery as the Bergman gradient fit — padded meals + vmap + one-compile jitted
value-and-grad + Adam + projection to plausible ranges + scale-normalized gradient norms — but the
forward model is `simulation/dalla_man.py` and the observation is the SUBCUTANEOUS glucose
compartment (what CGM measures), not plasma.

Deliberately a NEW module rather than a `model=` switch inside `gradient_fit.py`: the brief requires
that existing committed results are not disturbed, and this keeps the Bergman path byte-identical.

Fitted per user (the plausibly identifiable subset from postprandial CGM):
  Vmx  insulin sensitivity (Bergman Si analogue)
  kabs intestinal absorption        kmax/kmin gastric emptying (Bergman gastric analogue)
  f    fraction of meal appearing   Td   subcutaneous/CGM lag
Everything else stays at population values. Basal glucose Gb is OBSERVED (pre-meal CGM baseline),
not fitted, so the model starts at the subject's own steady state.

Meal composition: the Dalla Man gut model carries the timing that Bergman approximates with a
fat/fibre blunting factor, so fat/fibre are not passed here; `f` absorbs per-user composition
effects. Noted as a modelling difference when comparing against Bergman.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from simulation.dalla_man import DallaManParams, run_meal
from simulation.jax_observation import iauc

MAX_MEALS = 72

PARAM_RANGES = {
    "Vmx": (0.005, 0.150),
    "kabs": (0.020, 0.200),
    "kmax": (0.020, 0.120),
    "kmin": (0.005, 0.080),
    "f": (0.60, 1.00),
    "Td": (5.0, 25.0),
}
TARGETS = ("Vmx", "kabs", "kmax", "kmin", "f", "Td")
_LO = jnp.array([PARAM_RANGES[k][0] for k in TARGETS], dtype=jnp.float32)
_HI = jnp.array([PARAM_RANGES[k][1] for k in TARGETS], dtype=jnp.float32)
_RNG = _HI - _LO


def _meal_iauc(params, carbs):
    # Looser solver tolerance than the validation gates (1e-4): inference runs this ODE
    # n_steps x n_meals times, and 1e-3 is well inside the CGM noise floor for iAUC.
    ts, gsc = run_meal(params, carbs, meal_time=30.0, duration_min=210.0, step_min=5.0,
                       rtol=1e-3, atol=1e-3, max_steps=4000)
    return iauc(ts, gsc)


_meals_iauc = jax.vmap(_meal_iauc, in_axes=(None, 0))


def _build(base: DallaManParams, theta: jnp.ndarray) -> DallaManParams:
    p = base
    for i, name in enumerate(TARGETS):
        p = eqx.tree_at(lambda m, n=name: getattr(m, n), p, theta[i])
    return p


def _loss(theta, base, carbs, obs, mask, reg_coef):
    p = _build(base, theta)
    preds = _meals_iauc(p, carbs)
    recon = jnp.sum(mask * (preds - obs) ** 2) / jnp.maximum(jnp.sum(mask), 1.0)
    theta0 = jnp.array([getattr(base, n) for n in TARGETS])
    reg = jnp.sum(((theta - theta0) / _RNG) ** 2)
    return recon + reg_coef * reg


_loss_grad = jax.jit(jax.value_and_grad(_loss))     # compiled once; reused across subjects/folds


def _pad(vals):
    a = np.zeros(MAX_MEALS, dtype=np.float32)
    a[:len(vals)] = vals
    return jnp.asarray(a)


def fit_parameters(meals: list[dict], n_steps: int = 150, learning_rate: float = 0.02,
                   regularization: float = 0.01, base: DallaManParams | None = None,
                   weight_kg: float = 78.0, Gb: float = 100.0) -> dict:
    """Fit the Dalla Man subset to a subject's meals ({carbs_g, observed_iAUC})."""
    base = base or DallaManParams.defaults(weight_kg, Gb)
    n = len(meals)
    carbs = _pad([float(m["carbs_g"]) for m in meals])
    obs = _pad([float(m["observed_iAUC"]) for m in meals])
    mask = _pad([1.0] * n)
    reg_coef = jnp.asarray(regularization, jnp.float32)

    theta0 = jnp.array([float(getattr(base, k)) for k in TARGETS], dtype=jnp.float32)
    opt = optax.adam(learning_rate)
    theta = theta0
    st = opt.init(theta)

    loss_curve, grad_abs = [], []
    for _ in range(n_steps):
        l, g = _loss_grad(theta, base, carbs, obs, mask, reg_coef)
        up, st = opt.update(g, st)
        theta = optax.apply_updates(theta, up)
        theta = jnp.clip(theta, _LO, _HI)
        loss_curve.append(float(l))
        grad_abs.append(np.abs(np.asarray(g)))

    rng_np = np.asarray(_RNG)
    grad_norms = {k: float(np.mean([ga[i] for ga in grad_abs]) * rng_np[i])
                  for i, k in enumerate(TARGETS)}
    grad_norm_history = {k: [float(ga[i]) * float(rng_np[i]) for ga in grad_abs]
                         for i, k in enumerate(TARGETS)}
    fitted = _build(base, theta)
    preds = np.asarray(_meals_iauc(fitted, carbs))[:n]
    return {
        "params": fitted,
        "theta": {k: float(theta[i]) for i, k in enumerate(TARGETS)},
        "Vmx": float(theta[0]),
        "gradient_norms": grad_norms,
        "grad_norm_history": grad_norm_history,
        "loss_curve": loss_curve,
        "iAUC_predicted": [float(v) for v in preds],
        "iAUC_observed": [float(m["observed_iAUC"]) for m in meals],
    }


def predict_iauc(params: DallaManParams, carbs_g: float) -> float:
    return float(_meal_iauc(params, jnp.asarray(carbs_g, jnp.float32)))
