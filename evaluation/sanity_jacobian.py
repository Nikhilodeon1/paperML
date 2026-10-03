"""Phase 1.5: the per-observable Jacobian for one subject, in log parameter coordinates.

A sanity check before any of Phase 2 runs, and the first place the paper's central claim becomes
visible as a number rather than an argument. For one subject at the maximum-likelihood estimate, this
prints `d observable / d log theta` for each of the four observables and each of the three parameters.

What to look for: if an area really is blind to timing, the iAUC row will be dominated by insulin
sensitivity and the two gut columns will be small; the centroid and peak-time rows should show the
reverse. If instead the iAUC row has comparable entries everywhere, the ladder argument is wrong and
the rest of Phase 2 would be measuring something else.

Log coordinates throughout, so the three columns are comparable despite insulin sensitivity being
order 1 and the rate constants order 0.02. For the trace, which is a vector, the reported entry is the
Euclidean norm over samples of the per-sample derivative, which is the quantity that enters the Fisher
matrix.

Run:  python -m evaluation.sanity_jacobian
      python -m evaluation.sanity_jacobian --subject CGMacros-003 --save
"""
from __future__ import annotations

import argparse
import json

from evaluation.jax_config import configure

_JAX = configure()

import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402
import numpy as np              # noqa: E402

from evaluation.cohort_data import load_cgmacros                      # noqa: E402
from personalization.fit_general import fit_ml                        # noqa: E402
from personalization.objectives import ObjectiveSpec                  # noqa: E402
from personalization.subject_loss import TARGETS, base_params, build_params, subject_arrays  # noqa: E402
from simulation.jax_observables import Window, centroid, iauc, peak_time, peak_value  # noqa: E402
from simulation.jax_engine import run_meal                            # noqa: E402

ANALYSIS_ID = "A1_sanity_jacobian"
OBSERVABLES = ("iauc", "centroid", "peak_time", "trace")


def _observable(name: str, glucose, window: Window, beta: float):
    if name == "iauc":
        return iauc(None, glucose, window, beta)
    if name == "centroid":
        return centroid(None, glucose, window, beta)
    if name == "peak_time":
        return peak_time(None, glucose, window, beta)
    if name == "peak_value":
        return peak_value(None, glucose, window, beta)
    if name == "trace":
        base = jnp.mean(glucose[window.pre_slice[0]:window.pre_slice[1]])
        start, stop = window.post_slice
        return glucose[start:stop] - base
    raise ValueError(f"unknown observable {name!r}")


def jacobian_table(subject, theta_hat: np.ndarray, meal_index: int = 0,
                   window: Window | None = None, beta: float = 10.0) -> dict:
    """`d observable / d log theta` at `theta_hat`, for one meal of one subject.

    One meal rather than the whole subject: the per-meal Jacobian is what stacks into the Fisher
    matrix, and a sum over meals would hide whether individual meals are informative. The meal chosen
    is reported.
    """
    window = window or Window()
    base = base_params(subject.profile)
    record = subject.records[meal_index]
    carbs = float(record["carbs_g"])
    fat = float(record["fat_g"])
    fiber = float(record["fiber_g"])
    theta_hat = jnp.asarray(theta_hat, dtype=jnp.float64)

    def simulate(log_theta):
        """Simulate at theta = exp(log_theta), so a derivative here is with respect to log theta."""
        params = build_params(base, jnp.exp(log_theta))
        _, glucose = run_meal(params, carbs, fat, fiber, meal_time=window.meal_time_min,
                              duration_min=window.meal_time_min + window.post_min,
                              step_min=window.step_min)
        return glucose

    log_theta = jnp.log(theta_hat)
    rows = {}
    for name in OBSERVABLES:
        derivative = jax.jacrev(lambda lt, n=name: _observable(n, simulate(lt), window, beta))(
            log_theta)
        derivative = np.asarray(derivative, dtype=float)
        if derivative.ndim == 1:
            rows[name] = {"per_parameter": derivative.tolist(), "is_vector": False}
        else:
            # Euclidean norm over samples: the column norm that enters J^T J.
            rows[name] = {"per_parameter": np.linalg.norm(derivative, axis=0).tolist(),
                          "is_vector": True, "n_samples": int(derivative.shape[0])}
        reference = abs(rows[name]["per_parameter"][0])
        rows[name]["ratio_to_si"] = [
            (abs(v) / reference if reference > 0 else float("inf"))
            for v in rows[name]["per_parameter"]]

    return {"subject_id": subject.subject_id, "meal_index": meal_index, "carbs_g": carbs,
            "theta_hat": {name: float(theta_hat[i]) for i, name in enumerate(TARGETS)},
            "parameters": ["log_" + n for n in TARGETS], "rows": rows,
            "window": {"post_min": window.post_min, "step_min": window.step_min}, "beta": beta}


def run(subject_id: str | None = None, meal_index: int = 0, steps: int = 400) -> dict:
    subjects = load_cgmacros()
    subject = next((s for s in subjects if s.subject_id == subject_id), None) if subject_id \
        else subjects[0]
    if subject is None:
        raise SystemExit(f"no subject {subject_id!r}; first few: "
                         f"{[s.subject_id for s in subjects[:5]]}")

    # theta_hat_ML, per amendment B1: the identifiability analyses never use the regularized estimate.
    ml = fit_ml(subject, ObjectiveSpec(name="iauc"), steps=steps, pilot_steps=steps // 2)
    theta_hat = np.array([ml.ml.theta_full[name] for name in TARGETS])

    table = jacobian_table(subject, theta_hat, meal_index=meal_index)
    table["fit"] = {"theta_ml": ml.ml.theta_full, "at_bound": ml.ml.at_bound,
                    "interior": ml.ml.interior, "converged": ml.ml.converged,
                    "noise": ml.noise.as_dict(), "n_meals": subject.n_meals}
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--meal", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    table = run(args.subject, args.meal, args.steps)

    print("=" * 86)
    print(f"PHASE 1.5 JACOBIAN  --  {table['subject_id']}, meal {table['meal_index']} "
          f"({table['carbs_g']:.0f} g carbohydrate)")
    theta = table["theta_hat"]
    print("  theta_hat_ML: " + "  ".join(f"{k} {v:.5f}" for k, v in theta.items()))
    flags = table["fit"]["at_bound"]
    print(f"  at a bound: {[k for k, v in flags.items() if v] or 'none'}   "
          f"interior {table['fit']['interior']}   converged {table['fit']['converged']}")
    print("-" * 86)
    print(f"  {'observable':12} {'d/dlog S_I':>14} {'d/dlog k_e':>14} {'d/dlog k_a':>14}"
          f"   {'|k_e|/|S_I|':>11} {'|k_a|/|S_I|':>11}")
    for name in OBSERVABLES:
        row = table["rows"][name]
        values = row["per_parameter"]
        ratios = row["ratio_to_si"]
        print(f"  {name:12} {values[0]:14.4f} {values[1]:14.4f} {values[2]:14.4f}"
              f"   {ratios[1]:11.4f} {ratios[2]:11.4f}")
    print("-" * 86)
    print("  (the trace row is the Euclidean norm over its "
          f"{table['rows']['trace'].get('n_samples')} samples)")
    print("=" * 86)

    if args.save:
        from evaluation.results_io import save_result
        print("saved", save_result(ANALYSIS_ID, table,
                                  {"subject": table["subject_id"], "meal": args.meal,
                                   "steps": args.steps}, unit=table["subject_id"], overwrite=True))
    else:
        print(json.dumps({k: table["rows"][k]["ratio_to_si"] for k in OBSERVABLES}, indent=2))


if __name__ == "__main__":
    main()
