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
from personalization.fit_general import BOUND_TOLERANCE, _optimizer     # noqa: E402
from personalization.objectives import (                                # noqa: E402
    ObjectiveSpec, build_objective, observed_values,
)
from personalization.subject_loss import TARGETS, base_params, subject_arrays  # noqa: E402

ANALYSIS_ID = "A8a_gradient_diag"


def default_config() -> dict:
    return {"cohort": "cgmacros", "min_meals": 10, "limit": None, "steps": 150,
            "learning_rate": 0.02, "lam": 0.01, "bounds_scale": 1.0, "beta": 10.0}


def units(config: dict) -> list[str]:
    return [s.subject_id for s in load_cohort(config["cohort"], min_meals=config["min_meals"],
                                              limit=config["limit"])]


def run_unit(unit: str, config: dict) -> dict:
    subjects = load_cohort(config["cohort"], min_meals=config["min_meals"], limit=config["limit"])
    subject = next(s for s in subjects if s.subject_id == unit)
    records = list(subject.records)
    spec = ObjectiveSpec(name="iauc", lam=config["lam"], beta=config["beta"], free=TARGETS,
                         bounds_scale=config["bounds_scale"])
    objective = build_objective(spec, base_params(subject.profile), subject_arrays(records),
                                observed_values(records, spec.window))
    lower, upper = jnp.asarray(objective.lower), jnp.asarray(objective.upper)
    span = upper - lower
    tolerance = BOUND_TOLERANCE * span
    optimizer = _optimizer("adam", config["learning_rate"])
    loss = objective.loss

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
            theta = objective.project(optax.apply_updates(theta, updates))
            return (theta, state), (jnp.abs(grad) * span, jnp.abs(projected) * span,
                                    blocked.astype(jnp.float64))

        (final, _), history = jax.lax.scan(body, (theta0, optimizer.init(theta0)), None,
                                           length=config["steps"])
        return final, history

    theta0 = objective.project(objective.theta0)
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
        "theta": dict(zip(TARGETS, final.tolist())),
        "at_bound": at_bound,
        "raw_norm": dict(zip(TARGETS, raw_norm.tolist())),
        "projected_norm": dict(zip(TARGETS, proj_norm.tolist())),
        "fraction_of_steps_blocked": dict(zip(TARGETS, np.asarray(blocked).mean(axis=0).tolist())),
        "macros": {},
    }
