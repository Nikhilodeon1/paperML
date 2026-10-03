"""Measure what the expensive operations actually cost, then project the plan from the measurement.

Nothing here is an analysis. The point is to replace guesses about wall time with numbers, so that
a job projected to take longer than the agreed budget is caught before it starts rather than after
twelve hours. Every projection in the printed plan is `measured unit cost x unit count / workers`.

Three things are measured separately because they scale differently:

* **Compilation.** Paid once per process per distinct shape. With the persistent compilation cache
  it is paid once per machine, which is why the cache is enabled.
* **Per-step cost** of the gradient fit, which sets the cost of fitting and of profiling.
* **Per-evaluation cost** of the forward solve, which sets the cost of the exhaustive grid.

Run:  python -m evaluation.benchmark_compute
      python -m evaluation.benchmark_compute --subjects 3 --save
"""
from __future__ import annotations

import argparse
import json
import os
import time

from evaluation.jax_config import configure

_JAX = configure()

import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402
import numpy as np              # noqa: E402
import optax                     # noqa: E402

from evaluation.cohort_data import load_cgmacros                     # noqa: E402
from evaluation.identifiability_tools import profile_likelihood      # noqa: E402
from personalization.subject_loss import (                            # noqa: E402
    LOWER as _LO, UPPER as _HI, TARGETS, base_params, iauc_loss, subject_arrays, theta_of,
)

ANALYSIS_ID = "A0_benchmark"

# The grid of Phase 4's `grid3` cell: 20 x 8 x 8 over the box.
GRID_SHAPE = (20, 8, 8)


def _time(fn, repeats: int = 1) -> tuple[float, float]:
    """`(first call seconds, median of the remaining calls)`. The first call includes compilation."""
    start = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    first = time.perf_counter() - start
    rest = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        rest.append(time.perf_counter() - start)
    return first, (float(np.median(rest)) if rest else float("nan"))


def fit_steps(loss, theta0, steps: int, learning_rate: float = 0.02):
    """`steps` Adam updates under a single `lax.scan`, so the whole fit is one compiled call."""
    opt = optax.adam(learning_rate)

    @jax.jit
    def run(theta):
        def body(carry, _):
            th, state = carry
            value, grad = jax.value_and_grad(loss)(th)
            updates, state = opt.update(grad, state)
            th = jnp.clip(optax.apply_updates(th, updates), _LO, _HI)
            return (th, state), value

        (final, _), curve = jax.lax.scan(body, (theta, opt.init(theta)), None, length=steps)
        return final, curve

    return run(theta0)


def benchmark(n_subjects: int = 3) -> dict:
    subjects = load_cgmacros()[:n_subjects]
    if not subjects:
        raise SystemExit("no subjects loaded; check the dataset path")

    cores = os.cpu_count() or 1
    workers = max(1, cores - 2)        # two cores left for the user's own work
    out: dict = {
        "jax": _JAX,
        "cores_logical": cores,
        "workers_planned": workers,
        "n_subjects_benchmarked": len(subjects),
        "subjects": [],
    }

    for subject in subjects:
        arrays = subject_arrays(subject.records)
        base = base_params(subject.profile)
        loss = iauc_loss(base, arrays, lam=0.0)
        theta0 = theta_of(base)

        compile_500, _ = _time(lambda: fit_steps(loss, theta0, 500))
        _, fit_500 = _time(lambda: fit_steps(loss, theta0, 500), repeats=2)
        _, refit_60 = _time(lambda: fit_steps(loss, theta0, 60), repeats=2)

        theta_hat = fit_steps(loss, theta0, 500)[0]

        grid = jnp.linspace(float(_LO[0]), float(_HI[0]), 15)
        profile_first, profile_rest = _time(
            lambda: profile_likelihood(loss, theta_hat, 0, grid, lower=_LO, upper=_HI,
                                       steps=60, optimizer="adam")["profile"])

        axes = [jnp.linspace(float(_LO[i]), float(_HI[i]), GRID_SHAPE[i]) for i in range(3)]
        mesh = jnp.stack([a.ravel() for a in jnp.meshgrid(*axes, indexing="ij")], axis=-1)
        grid_eval = jax.jit(jax.vmap(loss))
        grid_first, grid_rest = _time(lambda: grid_eval(mesh), repeats=1)

        jac_first, jac_rest = _time(lambda: jax.jacrev(loss)(theta_hat), repeats=2)

        out["subjects"].append({
            "subject_id": subject.subject_id,
            "n_meals": arrays["n"],
            "compile_plus_fit_500_s": compile_500,
            "fit_500_steps_s": fit_500,
            "refit_60_steps_s": refit_60,
            "profile_15_points_60_steps_s": profile_rest,
            "profile_first_call_s": profile_first,
            f"grid_{GRID_SHAPE[0]}x{GRID_SHAPE[1]}x{GRID_SHAPE[2]}_s": grid_rest,
            "grid_first_call_s": grid_first,
            "jacrev_loss_s": jac_rest,
            "theta_hat": [float(v) for v in theta_hat],
        })

    def median(key):
        values = [s[key] for s in out["subjects"] if np.isfinite(s.get(key, np.nan))]
        return float(np.median(values)) if values else float("nan")

    grid_key = f"grid_{GRID_SHAPE[0]}x{GRID_SHAPE[1]}x{GRID_SHAPE[2]}_s"
    unit = {
        "fit_500": median("fit_500_steps_s"),
        "refit_60": median("refit_60_steps_s"),
        "profile_one_parameter": median("profile_15_points_60_steps_s"),
        "grid3": median(grid_key),
        "jacrev": median("jacrev_loss_s"),
        "compile_once": median("compile_plus_fit_500_s") - median("fit_500_steps_s"),
    }
    out["unit_cost_seconds"] = unit
    out["plan"] = compute_plan(unit, workers=workers)
    return out


def compute_plan(unit: dict, workers: int, n_subjects: int = 45) -> list[dict]:
    """Projected wall time for each later job: unit cost times unit count over workers.

    Deliberately pessimistic in one respect and optimistic in another, both stated: compilation is
    counted once per worker, and no allowance is made for the machine being busy with the user's
    own work.
    """
    jobs = [
        # (name, units, seconds per unit)
        ("A4 full Fisher, iAUC", n_subjects, unit["jacrev"] * 3),
        ("A5 profile likelihood, iAUC (3 parameters)", n_subjects, unit["profile_one_parameter"] * 3),
        ("A6 ladder: fit + Fisher + profile, 2 further objectives", n_subjects,
         2 * (unit["fit_500"] + unit["jacrev"] * 3 + unit["profile_one_parameter"] * 3)),
        ("A9 grad3 CV, 5 folds x 5 repeats", n_subjects, 25 * unit["fit_500"]),
        ("A9 grad1 CV, 5 folds x 5 repeats", n_subjects, 25 * unit["fit_500"]),
        ("A9 grid3 CV, 5 folds x 5 repeats", n_subjects, 25 * unit["grid3"]),
        ("A9 multistart grad3, 10 inits x 5 folds x 5 repeats", n_subjects,
         250 * unit["fit_500"]),
        ("A9b trace-objective CV, 2 cells", n_subjects, 50 * unit["fit_500"]),
        ("A10 noise-floor replica, 3 repeats", n_subjects, 15 * unit["fit_500"]),
        ("A11 synthetic recovery, 60 subjects x 4 observables", 60,
         4 * (unit["fit_500"] + 3 * unit["profile_one_parameter"])),
        ("A12 replication on Hall and Shanghai", 121,
         unit["fit_500"] + 3 * unit["profile_one_parameter"]),
    ]
    plan = []
    for name, units, per_unit in jobs:
        serial = units * per_unit
        wall = serial / workers + unit["compile_once"]
        plan.append({
            "job": name,
            "units": units,
            "seconds_per_unit": round(per_unit, 2),
            "serial_hours": round(serial / 3600.0, 2),
            "projected_wall_hours": round(wall / 3600.0, 2),
            "over_12h_budget": bool(wall / 3600.0 > 12.0),
        })
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--save", action="store_true", help="write a result file with provenance")
    args = ap.parse_args()

    result = benchmark(args.subjects)

    print("=" * 84)
    print(f"COMPUTE BENCHMARK  --  {result['cores_logical']} logical cores, "
          f"{result['workers_planned']} workers planned, x64={result['jax']['x64']}")
    print("-" * 84)
    for s in result["subjects"]:
        print(f"  {s['subject_id']}  {s['n_meals']:>3} meals   "
              f"fit(500) {s['fit_500_steps_s']:6.2f}s   refit(60) {s['refit_60_steps_s']:5.2f}s   "
              f"profile {s['profile_15_points_60_steps_s']:6.2f}s")
    print("-" * 84)
    print("  median unit costs (seconds):")
    for key, value in result["unit_cost_seconds"].items():
        print(f"    {key:28} {value:8.2f}")
    print("-" * 84)
    print(f"  {'job':58} {'units':>6} {'serial h':>9} {'wall h':>7}")
    for job in result["plan"]:
        flag = "  OVER BUDGET" if job["over_12h_budget"] else ""
        print(f"  {job['job'][:58]:58} {job['units']:>6} {job['serial_hours']:>9.2f} "
              f"{job['projected_wall_hours']:>7.2f}{flag}")
    total = sum(j["projected_wall_hours"] for j in result["plan"])
    print(f"  {'TOTAL':58} {'':>6} {'':>9} {total:>7.2f}")
    print("=" * 84)

    if args.save:
        from evaluation.results_io import save_result
        path = save_result(ANALYSIS_ID, result, {"n_subjects": args.subjects,
                                                 "grid_shape": list(GRID_SHAPE)},
                           unit="benchmark", overwrite=True)
        print(f"saved {path}")
    else:
        print(json.dumps(result["unit_cost_seconds"], indent=2))


if __name__ == "__main__":
    main()
