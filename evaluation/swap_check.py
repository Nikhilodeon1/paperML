"""Direct numerical check of the k_e <-> k_a swap symmetry in the engine (supports H1).

The gut is a two-compartment cascade, stomach -> intestine, with the absorption rate
`Ra = f * k_a * gut`. The transfer function from meal rate to Ra is `f k_e k_a / ((s + k_e)(s + k_a))`,
which is symmetric in the two rates, and the cascade starts empty, so exchanging `k_e` and `k_a` leaves
the rate of appearance, and therefore every observable, unchanged. This runs the engine both ways and
reports the largest difference in the glucose trace against the solver tolerance.

Run:  python -m evaluation.swap_check
"""
from __future__ import annotations

import json

from evaluation.jax_config import configure

configure()

import jax.numpy as jnp                      # noqa: E402
import numpy as np                           # noqa: E402

from evaluation.cohort_data import load_cohort                # noqa: E402
from personalization.subject_loss import base_params, build_params, theta_of  # noqa: E402
from personalization.objectives import simulate_meals         # noqa: E402

ANALYSIS_ID = "A2_swap_check"
PAIRS = ((0.015, 0.045), (0.02, 0.03), (0.04, 0.012), (0.03, 0.03))


def run() -> dict:
    subject = load_cohort("cgmacros", min_meals=10)[0]
    base = base_params(subject.profile)
    theta0 = np.asarray(theta_of(base), dtype=float)
    carbs = jnp.asarray([30.0, 60.0, 90.0])
    fat = jnp.asarray([10.0, 20.0, 5.0])
    fiber = jnp.asarray([2.0, 5.0, 8.0])
    rows = []
    for ke, ka in PAIRS:
        out = []
        for a, b in ((ke, ka), (ka, ke)):
            theta = theta0.copy()
            theta[1], theta[2] = a, b
            params = build_params(base, jnp.asarray(theta))
            out.append(np.asarray(simulate_meals(params, carbs, fat, fiber, 210.0, 5.0)))
        diff = float(np.max(np.abs(out[0] - out[1])))
        rows.append({"k_e": ke, "k_a": ka, "max_abs_glucose_difference_mg_dl": diff,
                     "max_glucose_swing_mg_dl": float(out[0].max() - out[0].min())})
    return {"pairs": rows, "solver_rtol": 1e-4, "solver_atol": 1e-4,
            "max_difference_over_pairs": max(r["max_abs_glucose_difference_mg_dl"] for r in rows)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
