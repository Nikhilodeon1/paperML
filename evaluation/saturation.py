"""H16(i): does the response to insulin sensitivity saturate, and is that why S_I intervals are one-sided?

A profile interval for `S_I` is one-sided when the likelihood rises on one side of the estimate and
stays flat on the other. If the simulated iAUC flattens as `S_I` grows, then at high `S_I` a change in
`S_I` barely changes the data, the sensitivity `d iAUC / d log S_I` falls toward zero, and the upper end
of the interval cannot be bounded however much data there is. This traces both curves, engine only:

* for each subject, the meal with the median carbohydrate content, at that subject's regularized
  prediction-fit timing parameters (A4 `theta_pilot`);
* `S_I` swept over TWICE the primary box (log-spaced, 25 points);
* at each point, iAUC and `|d iAUC / d log S_I|` by reverse-mode differentiation through the solver.

The summary is the median across subjects at each grid point, with the interquartile range. Nothing is
fitted. Exploratory (H16): no threshold.

Run:  python -m evaluation.runner evaluation.saturation --workers 8
"""
from __future__ import annotations

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import equinox as eqx            # noqa: E402
import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402

from evaluation.cohort_data import load_cohort                          # noqa: E402
from evaluation.ladder import a4_results                                # noqa: E402
from personalization.objectives import _scaled_bounds                   # noqa: E402
from personalization.subject_loss import TARGETS, base_params, build_params  # noqa: E402
from simulation.jax_engine import run_meal                              # noqa: E402
from simulation.jax_observables import Window, iauc                     # noqa: E402

ANALYSIS_ID = "A15_saturation"


def default_config() -> dict:
    return {"cohort": "cgmacros", "min_meals": 10, "limit": None, "points": 25, "box_scale": 2.0}


def units(config: dict) -> list[str]:
    return [s.subject_id for s in load_cohort(config["cohort"], min_meals=config["min_meals"],
                                              limit=config["limit"])]


def run_unit(unit: str, config: dict) -> dict:
    subject = next(s for s in load_cohort(config["cohort"], min_meals=config["min_meals"],
                                          limit=config["limit"]) if s.subject_id == unit)
    row = a4_results("iauc", 1.0).get(unit)
    if row is None:
        raise RuntimeError(f"no A4 iAUC result for {unit}")
    base = base_params(subject.profile)
    theta = np.array([row["theta_pilot"][n] for n in TARGETS], dtype=float)
    carbs = np.array([r["carbs_g"] for r in subject.records])
    meal = subject.records[int(np.argsort(carbs)[len(carbs) // 2])]
    lower, upper = (np.asarray(v, dtype=float) for v in _scaled_bounds(config["box_scale"]))
    grid = np.exp(np.linspace(np.log(lower[0]), np.log(upper[0]), config["points"]))
    window = Window()

    def area(log_si, si_value):
        params = build_params(base, jnp.asarray([si_value * jnp.exp(log_si), theta[1], theta[2]]))
        ts, glucose = run_meal(params, meal["carbs_g"], meal["fat_g"], meal["fiber_g"])
        return iauc(ts, glucose, window, beta=None)

    value_and_grad = jax.jit(jax.value_and_grad(area, argnums=0))
    areas, slopes = [], []
    for si in grid:
        a, g = value_and_grad(jnp.asarray(0.0), jnp.asarray(si))
        areas.append(float(a))
        slopes.append(abs(float(g)))
    return {"subject_id": unit, "carbs_g": meal["carbs_g"], "si_grid": grid.tolist(),
            "iauc": areas, "abs_d_iauc_d_log_si": slopes,
            "primary_box": [float(v) for v in _scaled_bounds(1.0)[0][:1]] +
                           [float(v) for v in _scaled_bounds(1.0)[1][:1]],
            "macros": {}}


def summarize() -> dict:
    from evaluation.results_io import largest_matching
    rows = list(largest_matching(ANALYSIS_ID, lambda p: "iauc" in p).values())
    if not rows:
        return {"n": 0}
    grid = np.array(rows[0]["si_grid"])
    area = np.array([r["iauc"] for r in rows])
    slope = np.array([r["abs_d_iauc_d_log_si"] for r in rows])
    relative = slope / np.maximum(area, 1e-9)

    def q(a):
        return {"median": np.median(a, axis=0).tolist(), "q1": np.percentile(a, 25, axis=0).tolist(),
                "q3": np.percentile(a, 75, axis=0).tolist()}

    primary_low, primary_high = rows[0]["primary_box"]
    inside = (grid >= primary_low) & (grid <= primary_high)
    med_slope = np.median(slope, axis=0)
    return {"n": len(rows), "si_grid": grid.tolist(), "iauc": q(area), "slope": q(slope),
            "relative_slope": q(relative), "primary_box": [primary_low, primary_high],
            "slope_ratio_top_to_bottom_of_primary_box": float(
                med_slope[inside][-1] / max(med_slope[inside][0], 1e-12)),
            "slope_ratio_top_of_double_box_to_top_of_primary": float(
                med_slope[-1] / max(med_slope[inside][-1], 1e-12))}
