"""How fast does the k_e <-> k_a exchange symmetry break when the gut does not start empty?

With an empty gut the transfer function from meal rate to the rate of appearance Ra is
`f k_e k_a / ((s + k_e)(s + k_a))`, exactly symmetric in the two rates. Residual intestinal content
`Q0` left over from an earlier meal adds a free response `f k_a Q0 / (s + k_a)`, which is not symmetric:
exchanging the rates then changes Ra and therefore glucose. Free-living data always has some residual
content, so the size of the swap-induced glucose change as a function of `Q0 / D` (D the meal mass) is
what decides whether the exchange symmetry matters at CGM noise levels.

Two parts:

* a symbolic check (sympy) of the statement above;
* a numerical sweep through the engine: the intestinal compartment starts at `fraction * D`, the meal is
  simulated at (k_e, k_a) and at (k_a, k_e), and the largest and median absolute glucose differences are
  reported for Q0/D in {0, 0.5, 1, 2, 5, 10} percent.

The stomach compartment is deliberately NOT given initial content: stomach content is a first-order
cascade input and is symmetric, so it cannot break the exchange. The symbolic check shows this too.

Run:  python -m evaluation.gut_sweep
"""
from __future__ import annotations

import json

import numpy as np
import sympy as sp

from evaluation.jax_config import configure

configure()

import jax.numpy as jnp                                                  # noqa: E402
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve  # noqa: E402

from evaluation.cohort_data import load_cohort                           # noqa: E402
from personalization.subject_loss import base_params, build_params, theta_of  # noqa: E402
from simulation.jax_engine import IDX, initial_state, vector_field       # noqa: E402

ANALYSIS_ID = "A2_gut_sweep"
FRACTIONS_PERCENT = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
PAIRS = ((0.015, 0.045), (0.02, 0.03), (0.04, 0.012), (0.035, 0.015), (0.012, 0.03))
MEALS = ((30.0, 10.0, 2.0), (60.0, 20.0, 5.0), (90.0, 5.0, 8.0))      # carbs, fat, fibre (g)


def symbolic_check() -> dict:
    """Verify, with sympy, the transfer functions with and without residual gut content."""
    s, ke, ka, f, q0, d0 = sp.symbols("s k_e k_a f Q0 D0", positive=True)
    u = sp.Symbol("U")
    # stomach s' = -ke*s + u (+ D0 initial), gut g' = ke*s - ka*g (+ Q0 initial), Ra = f*ka*g
    stomach = (u + d0) / (s + ke)
    gut = (ke * stomach + q0) / (s + ka)
    ra = sp.simplify(f * ka * gut)
    swapped = ra.subs({ke: ka, ka: ke}, simultaneous=True)
    empty = {q0: 0, d0: 0}
    empty_difference = sp.simplify(ra.subs(empty) - swapped.subs(empty))
    stomach_only = {q0: 0}
    stomach_difference = sp.simplify(ra.subs(stomach_only) - swapped.subs(stomach_only))
    gut_difference = sp.simplify(sp.simplify(ra - swapped).subs(d0, 0))
    return {
        "Ra_laplace": str(ra),
        "difference_empty_gut_with_meal_input": str(empty_difference),
        "difference_with_initial_stomach_content_only": str(stomach_difference),
        "difference_with_initial_gut_content": str(gut_difference),
        "exchange_symmetric_when_gut_empty": bool(empty_difference == 0),
        "exchange_symmetric_with_stomach_content_only": bool(stomach_difference == 0),
        "exchange_symmetric_with_gut_content": bool(gut_difference == 0),
    }


def _glucose(params, mass_mg: float, gut0_mg: float, meal_time=30.0, duration=210.0, step=5.0):
    y0 = initial_state().at[IDX["gut_glucose_mg"]].set(gut0_mg)
    ts = jnp.arange(0.0, duration + 1e-6, step)
    sol = diffeqsolve(ODETerm(vector_field), Tsit5(), t0=0.0, t1=float(duration), dt0=1.0, y0=y0,
                      args=(params, meal_time, mass_mg, 8.0), saveat=SaveAt(ts=ts),
                      stepsize_controller=PIDController(rtol=1e-4, atol=1e-4), max_steps=10000)
    return np.asarray(sol.ys[:, IDX["glucose_mg_dl"]])


def numerical_sweep() -> dict:
    subject = load_cohort("cgmacros", min_meals=10)[0]
    base = base_params(subject.profile)
    theta0 = np.asarray(theta_of(base), dtype=float)
    rows = []
    for percent in FRACTIONS_PERCENT:
        diffs = []
        for ke, ka in PAIRS:
            for carbs, fat, fibre in MEALS:
                blunt = 1.0 / (1.0 + 0.08 * fibre + 0.005 * fat)
                mass = carbs * 1000.0 * 0.90 * blunt
                gut0 = mass * percent / 100.0
                traces = []
                for a, b in ((ke, ka), (ka, ke)):
                    theta = theta0.copy()
                    theta[1], theta[2] = a, b
                    traces.append(_glucose(build_params(base, jnp.asarray(theta)), mass, gut0))
                diffs.append(float(np.max(np.abs(traces[0] - traces[1]))))
        rows.append({"gut_content_percent_of_meal": percent,
                     "max_abs_glucose_difference_mg_dl": float(np.max(diffs)),
                     "median_abs_glucose_difference_mg_dl": float(np.median(diffs)),
                     "n_cases": len(diffs)})
    return {"rows": rows, "pairs": [list(p) for p in PAIRS],
            "meals_g_carb_fat_fibre": [list(m) for m in MEALS],
            "note": ("The swap-induced change is the largest absolute difference in the simulated "
                     "glucose trace over 210 min. Compare it with CGM noise of roughly 5 to 10 mg/dL.")}


def run() -> dict:
    return {"symbolic": symbolic_check(), "sweep": numerical_sweep()}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
