"""A small, fully deterministic fit used as the gate-check before every expensive job.

Prints one JSON document: for a handful of subjects, the fitted parameters, the final loss, the
projected-gradient norm, which parameters ended on a bound, and the residual sigma. Small enough to
run in a couple of minutes, and complete enough that running it twice proves the pipeline has no
unseeded source of randomness.

Used two ways:

* `python -m evaluation.smoke_fit --subjects 3` before a long run, to look at the numbers.
* `evaluation.determinism.run_module_twice("evaluation.smoke_fit", ["--subjects", "3"])`, which
  runs it in two processes with different hash seeds and requires agreement to 1e-6.

Nothing here is a paper result; it is the fit the later phases generalize.
"""
from __future__ import annotations

import argparse
import json

from evaluation.jax_config import configure

_JAX = configure()

import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402
import numpy as np              # noqa: E402
import optax                     # noqa: E402

from evaluation.cohort_data import load_cgmacros                 # noqa: E402
from personalization.subject_loss import (                        # noqa: E402
    LOWER, RANGE, TARGETS, UPPER, base_params, iauc_loss, sigma_from_residuals,
    subject_arrays, theta_of,
)

ANALYSIS_ID = "A0_smoke_fit"
BOUND_TOLERANCE = 0.01      # within 1% of the range of a bound counts as being at it


def fit(loss, theta0, steps: int = 500, learning_rate: float = 0.02):
    """Projected Adam under one `lax.scan`, so the whole fit is a single compiled call."""
    opt = optax.adam(learning_rate)

    @jax.jit
    def run(theta):
        def body(carry, _):
            th, state = carry
            value, grad = jax.value_and_grad(loss)(th)
            updates, state = opt.update(grad, state)
            th = jnp.clip(optax.apply_updates(th, updates), LOWER, UPPER)
            return (th, state), value

        (final, _), curve = jax.lax.scan(body, (theta, opt.init(theta)), None, length=steps)
        return final, curve

    return run(theta0)


def projected_gradient_norm(loss, theta) -> float:
    """Norm of the gradient after removing the components that push into an active bound.

    The plain gradient norm does not go to zero at a constrained optimum, so using it as a
    convergence criterion would mark every bound-limited fit as unconverged. Removing the
    outward-pointing components at active bounds is the standard projected-gradient criterion and
    is what the interior-converged subject set is defined by.
    """
    grad = np.asarray(jax.grad(loss)(theta), dtype=float)
    th = np.asarray(theta, dtype=float)
    lo, hi = np.asarray(LOWER, dtype=float), np.asarray(UPPER, dtype=float)
    tol = BOUND_TOLERANCE * np.asarray(RANGE, dtype=float)
    at_lower = th <= lo + tol
    at_upper = th >= hi - tol
    projected = grad.copy()
    projected[at_lower & (grad > 0)] = 0.0      # pushing further below the lower bound
    projected[at_upper & (grad < 0)] = 0.0      # pushing further above the upper bound
    return float(np.linalg.norm(projected))


def at_bound(theta) -> dict[str, str | None]:
    th = np.asarray(theta, dtype=float)
    lo, hi = np.asarray(LOWER, dtype=float), np.asarray(UPPER, dtype=float)
    tol = BOUND_TOLERANCE * np.asarray(RANGE, dtype=float)
    out: dict[str, str | None] = {}
    for i, name in enumerate(TARGETS):
        if th[i] <= lo[i] + tol[i]:
            out[name] = "lower"
        elif th[i] >= hi[i] - tol[i]:
            out[name] = "upper"
        else:
            out[name] = None
    return out


def run(n_subjects: int = 3, steps: int = 500, lam: float = 0.01) -> dict:
    subjects = load_cgmacros()[:n_subjects]
    out = {"config": {"n_subjects": n_subjects, "steps": steps, "lam": lam,
                      "x64": _JAX["x64"], "jax": _JAX["jax_version"]},
           "subjects": []}
    for subject in subjects:
        arrays = subject_arrays(subject.records)
        base = base_params(subject.profile)
        loss = iauc_loss(base, arrays, lam=lam)
        theta_hat, curve = fit(loss, theta_of(base), steps=steps)
        out["subjects"].append({
            "subject_id": subject.subject_id,
            "n_meals": arrays["n"],
            "theta": {name: float(theta_hat[i]) for i, name in enumerate(TARGETS)},
            "final_loss": float(curve[-1]),
            "projected_grad_norm": projected_gradient_norm(loss, theta_hat),
            "at_bound": at_bound(theta_hat),
            "sigma_iauc": sigma_from_residuals(base, arrays, theta_hat),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lam", type=float, default=0.01)
    args = ap.parse_args()
    print(json.dumps(run(args.subjects, args.steps, args.lam), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
