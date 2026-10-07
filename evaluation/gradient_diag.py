"""The gradient-magnitude diagnostic of the submitted paper, with and without bound projection.

The submitted diagnostic ran the regularized iAUC fit (Adam, learning rate 0.02, 150 steps) and averaged
the absolute gradient of each parameter over the run, scaled by that parameter's interval width. A
parameter whose bound is active has a gradient that points through the wall: it is large, it is
irrelevant, and it inflates the diagnostic for exactly the parameters that are stuck. The projected
version zeroes the components that push through an active bound (within one percent of the interval,
the same rule the fit uses to call a parameter pinned) before averaging.

Both are computed in one run so the comparison is on the same trajectory. Phase 2.2 asks whether the
separation of S_I from the timing parameters in Figure 1 survives on the interior set; this supplies
the per-subject numbers and the summary compares them.

Phase 9 (Amendment 5) turns this into the H10 robustness sweep. The same diagnostic is recomputed under
a different optimizer (`optimizer`: Adam or L-BFGS), a random starting point (`init_seed`), the log
parameterization (`log_param`, gradients taken with respect to log theta and scaled by the log width of
the interval), and on a replica subject (`replica`), where the truth is known. The default configuration
is the one of the submitted paper; `is_default` identifies it so no summary mixes a variant into it.

Run:  python -m evaluation.runner evaluation.gradient_diag --workers 6
"""
from __future__ import annotations

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402
import optax                     # noqa: E402

from evaluation.cohort_data import load_cohort                          # noqa: E402
from evaluation.subject_source import _rng, get_subject                 # noqa: E402
from personalization.fit_general import BOUND_TOLERANCE, _optimizer     # noqa: E402
from personalization.objectives import (                                # noqa: E402
    ObjectiveSpec, build_objective, observed_values,
)
from personalization.subject_loss import TARGETS, base_params, subject_arrays  # noqa: E402

ANALYSIS_ID = "A8a_gradient_diag"


def default_config() -> dict:
    return {"cohort": "cgmacros", "min_meals": 10, "limit": None, "steps": 150,
            "learning_rate": 0.02, "lam": 0.01, "bounds_scale": 1.0, "beta": 10.0,
            "optimizer": "adam", "init_seed": None, "log_param": False,
            "replica": None, "carb_scale": None}


def is_default(payload: dict, scale: float | None = None) -> bool:
    """The configuration of the submitted diagnostic: real CGMacros data, Adam, default start, rate space."""
    return (payload.get("replica") is None and payload.get("carb_scale") in (None, 1, 1.0)
            and payload.get("optimizer", "adam") == "adam" and payload.get("init_seed") is None
            and not payload.get("log_param", False) and payload.get("cohort", "cgmacros") == "cgmacros"
            and (scale is None or abs(payload["bounds_scale"] - scale) < 1e-12))


def units(config: dict) -> list[str]:
    return [s.subject_id for s in load_cohort(config["cohort"], min_meals=config["min_meals"],
                                              limit=config["limit"])]


def run_unit(unit: str, config: dict) -> dict:
    subject = get_subject(config, unit)
    records = list(subject.records)
    spec = ObjectiveSpec(name="iauc", lam=config["lam"], beta=config["beta"], free=TARGETS,
                         bounds_scale=config["bounds_scale"])
    objective = build_objective(spec, base_params(subject.profile), subject_arrays(records),
                                observed_values(records, spec.window))
    log_param = bool(config.get("log_param"))
    if log_param:
        # Optimize phi = log theta. Gradients are with respect to phi and scaled by the log width of
        # the interval, so the three parameters stay comparable, exactly as in rate space.
        lower, upper = jnp.log(jnp.asarray(objective.lower)), jnp.log(jnp.asarray(objective.upper))

        def loss(phi):
            return objective.loss(jnp.exp(phi))

        def project(phi):
            return jnp.clip(phi, lower, upper)
    else:
        lower, upper = jnp.asarray(objective.lower), jnp.asarray(objective.upper)
        loss, project = objective.loss, objective.project
    span = upper - lower
    tolerance = BOUND_TOLERANCE * span
    optimizer = _optimizer(config.get("optimizer", "adam"), config["learning_rate"])
    if config.get("optimizer", "adam") == "lbfgs":
        # The line search evaluates trial points before the projection and the ODE does not survive
        # parameters far outside the box, so for L-BFGS the loss is evaluated at the projected point.
        raw_loss = loss

        def loss(theta):                                    # noqa: F811
            return raw_loss(project(theta))

    @jax.jit
    def run(theta0):
        def body(carry, _):
            theta, state = carry
            value, grad = jax.value_and_grad(loss)(theta)
            blocked = ((theta <= lower + tolerance) & (grad > 0)) | \
                      ((theta >= upper - tolerance) & (grad < 0))
            projected = jnp.where(blocked, 0.0, grad)
            updates, state = optimizer.update(grad, state, theta, value=value, grad=grad,
                                              value_fn=loss)
            theta = project(optax.apply_updates(theta, updates))
            return (theta, state), (jnp.abs(grad) * span, jnp.abs(projected) * span,
                                    blocked.astype(jnp.float64))

        (final, _), history = jax.lax.scan(body, (theta0, optimizer.init(theta0)), None,
                                           length=config["steps"])
        return final, history

    if config.get("init_seed") is not None:
        margin = BOUND_TOLERANCE * np.asarray(span)
        start = _rng("gradient_diag", unit, config["init_seed"]).uniform(
            np.asarray(lower) + margin, np.asarray(upper) - margin)
        theta0 = jnp.asarray(start)
    else:
        theta0 = project(jnp.log(objective.theta0) if log_param else objective.theta0)
    final, (raw, projected, blocked) = run(theta0)
    final = np.asarray(final, dtype=float)
    lo, hi = np.asarray(lower), np.asarray(upper)
    at_bound = {n: ("lower" if final[i] <= lo[i] + tolerance[i] else
                    "upper" if final[i] >= hi[i] - tolerance[i] else None)
                for i, n in enumerate(TARGETS)}
    raw_norm = np.asarray(raw).mean(axis=0)
    proj_norm = np.asarray(projected).mean(axis=0)
    return {
        "subject_id": unit, "bounds_scale": config["bounds_scale"], "n_meals": subject.n_meals,
        "theta": dict(zip(TARGETS, (np.exp(final) if log_param else final).tolist())),
        "cohort": config["cohort"], "optimizer": config.get("optimizer", "adam"),
        "init_seed": config.get("init_seed"), "log_param": log_param,
        "replica": config.get("replica"), "carb_scale": config.get("carb_scale"),
        "at_bound": at_bound,
        "raw_norm": dict(zip(TARGETS, raw_norm.tolist())),
        "projected_norm": dict(zip(TARGETS, proj_norm.tolist())),
        "fraction_of_steps_blocked": dict(zip(TARGETS, np.asarray(blocked).mean(axis=0).tolist())),
        "macros": {},
    }
