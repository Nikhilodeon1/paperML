"""Gradient-based per-subject parameter inference (Phase 2 — the proposed method).

Fits physiological parameters to a user's meals+CGM by gradient descent through the differentiable
engine (`simulation/jax_engine.py`). No prior mismatch: the loss directly minimizes reconstruction
error (iAUC MSE) between simulated and observed postprandial responses, with light L2
regularization toward the subject's population-default parameters. Adam via optax; parameters
projected to plausible ranges each step (boundedness without clamping the ODE state — see
SMOOTHING_NOTES.md).

Performance: meals are batched with `vmap` (one diffrax solve, not a Python loop) AND padded to a
fixed length with a mask, so the jitted value-and-grad compiles ONCE and is reused across every
subject and every k-fold split — no per-fit recompilation (that was ~40s of compile per fit and
made k-fold intractable).

`gradient_norms` (scale-normalized mean |∂loss/∂θ| over the run) is the identifiability diagnostic:
a parameter the data cannot constrain has near-zero normalized gradient regardless of meal count —
the gradient-based sibling of the SNPE posterior-width finding.

Validated: recovers a synthetic subject's true Si=0.50 to ~0.51.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from simulation.jax_engine import JaxPhysioParams, run_meal
from simulation.jax_observation import iauc

MAX_MEALS = 72   # fixed padded length (CGMacros max ~62); keeps the jitted shape constant

PARAM_RANGES = {
    "insulin_sensitivity": (0.30, 1.60),
    "gastric_emptying": (0.015, 0.040),
    "carb_absorption": (0.012, 0.032),
}
TARGETS = ("insulin_sensitivity", "gastric_emptying", "carb_absorption")
_LO = jnp.array([PARAM_RANGES[n][0] for n in TARGETS], dtype=jnp.float32)
_HI = jnp.array([PARAM_RANGES[n][1] for n in TARGETS], dtype=jnp.float32)
_RNG = _HI - _LO


def _meal_iauc(params, carbs, fat, fiber):
    ts, g = run_meal(params, carbs, fat, fiber, meal_time=30.0, duration_min=210.0, step_min=5.0)
    return iauc(ts, g)


_meals_iauc = jax.vmap(_meal_iauc, in_axes=(None, 0, 0, 0))


def _build(base: JaxPhysioParams, theta: jnp.ndarray) -> JaxPhysioParams:
    p = base
    for i, name in enumerate(TARGETS):
        p = eqx.tree_at(lambda m, n=name: getattr(m, n), p, theta[i])
    return p


def _loss(theta, base, carbs, fat, fib, obs, mask, reg_coef):
    """Masked reconstruction MSE + L2 toward the subject's base params. Fixed-shape (jittable)."""
    p = _build(base, theta)
    preds = _meals_iauc(p, carbs, fat, fib)
    recon = jnp.sum(mask * (preds - obs) ** 2) / jnp.maximum(jnp.sum(mask), 1.0)
    theta0 = jnp.array([getattr(base, n) for n in TARGETS])
    reg = jnp.sum(((theta - theta0) / _RNG) ** 2)
    return recon + reg_coef * reg


# compiled ONCE (constant padded shapes); reused for every subject / fold.
_loss_grad = jax.jit(jax.value_and_grad(_loss))


def _pad(vals):
    a = np.zeros(MAX_MEALS, dtype=np.float32)
    a[:len(vals)] = vals
    return jnp.asarray(a)


def fit_parameters(meals: list[dict], targets=TARGETS, n_steps: int = 150,
                   learning_rate: float = 0.02, regularization: float = 0.01,
                   base: JaxPhysioParams | None = None,
                   free_mask=None) -> dict:
    """Fit (Si, gastric, carb) to a subject's meals. `targets` is accepted for API compatibility
    but the fitted set is fixed to TARGETS (needed for the one-compile fast path).

    `free_mask` (length-3, 1=trainable, 0=frozen at the base value) lets `adaptive_fit` fit only
    the parameters the gradient diagnostic flags as identifiable, freezing the rest at their prior.
    """
    if tuple(targets) != TARGETS:
        raise ValueError(f"this fast-path fit optimizes {TARGETS}; got {targets}")
    base = base or JaxPhysioParams.from_population_defaults()
    free = jnp.asarray([1.0, 1.0, 1.0] if free_mask is None else list(free_mask), jnp.float32)
    n = len(meals)
    carbs = _pad([float(m["carbs_g"]) for m in meals])
    fats = _pad([float(m.get("fat_g", 0.0)) for m in meals])
    fibs = _pad([float(m.get("fiber_g", 0.0)) for m in meals])
    obs = _pad([float(m["observed_iAUC"]) for m in meals])
    mask = _pad([1.0] * n)
    reg_coef = jnp.asarray(regularization, jnp.float32)

    theta0 = jnp.array([float(getattr(base, k)) for k in TARGETS], dtype=jnp.float32)
    opt = optax.adam(learning_rate)
    theta = theta0
    st = opt.init(theta)

    loss_curve, grad_abs = [], []
    for _ in range(n_steps):
        l, g = _loss_grad(theta, base, carbs, fats, fibs, obs, mask, reg_coef)
        g = g * free                              # frozen params get zero gradient -> stay at prior
        up, st = opt.update(g, st)
        theta = optax.apply_updates(theta, up)
        theta = jnp.clip(theta, _LO, _HI)
        loss_curve.append(float(l))
        grad_abs.append(np.abs(np.asarray(g)))

    rng_np = np.asarray(_RNG)
    grad_norms = {k: float(np.mean([ga[i] for ga in grad_abs]) * rng_np[i])
                  for i, k in enumerate(TARGETS)}
    grad_norms_raw = {k: float(np.mean([ga[i] for ga in grad_abs]))
                      for i, k in enumerate(TARGETS)}
    grad_norm_history = {k: [float(ga[i]) * float(rng_np[i]) for ga in grad_abs]
                         for i, k in enumerate(TARGETS)}

    fitted = _build(base, theta)
    preds = np.asarray(_meals_iauc(fitted, carbs, fats, fibs))[:n]
    return {
        "params": fitted,
        "theta": {k: float(theta[i]) for i, k in enumerate(TARGETS)},
        "Si": float(theta[0]),
        "gradient_norms": grad_norms,
        "gradient_norms_raw": grad_norms_raw,
        "grad_norm_history": grad_norm_history,
        "loss_curve": loss_curve,
        "iAUC_predicted": [float(v) for v in preds],
        "iAUC_observed": [float(m["observed_iAUC"]) for m in meals],
    }


def adaptive_fit(meals: list[dict], threshold: float = 0.1, probe_steps: int = 50,
                 n_steps: int = 150, base: JaxPhysioParams | None = None,
                 learning_rate: float = 0.02) -> dict:
    """Use the gradient diagnostic to pick which parameters to fit, then fit only those.

    Step 1: a short probe fit estimates the (scale-normalized) gradient norm per parameter.
    Step 2: keep only parameters whose norm is >= `threshold` * the max norm (the identifiable
            ones); freeze the rest at their prior.
    Step 3: full fit on the selected parameters.

    Rationale: fitting non-identifiable parameters (gastric/carb) can overfit noise on train and
    hurt held-out prediction; freezing them at the prior is the principled per-user choice the
    diagnostic enables. Records which parameters were selected in `free_params`.
    """
    base = base or JaxPhysioParams.from_population_defaults()
    probe = fit_parameters(meals, n_steps=probe_steps, base=base, learning_rate=learning_rate)
    gn = probe["gradient_norms"]
    mx = max(gn.values()) or 1.0
    free_mask = [1.0 if gn[k] >= threshold * mx else 0.0 for k in TARGETS]
    if sum(free_mask) == 0:              # degenerate: always keep at least Si
        free_mask[0] = 1.0
    res = fit_parameters(meals, n_steps=n_steps, base=base, free_mask=free_mask,
                         learning_rate=learning_rate)
    res["free_params"] = [k for k, f in zip(TARGETS, free_mask) if f]
    res["free_mask"] = free_mask
    return res
