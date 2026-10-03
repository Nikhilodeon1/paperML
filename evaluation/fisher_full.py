"""A4: the full Fisher information matrix per subject, with its eigenvectors.

This replaces the diagnostic the reviewers rejected. The submitted paper argued non-identifiability from
the magnitude of a gradient and from the DIAGONAL of a Fisher matrix. Neither can distinguish "this
parameter has little effect" from "this parameter has an effect another parameter can cancel", because
both of those statements are about a direction in parameter space and a diagonal has no directions in it.

What is computed, per subject, at `theta_hat_ML` (amendment B1 -- never the regularized estimate):

* the stacked design matrix `J = d r / d log theta`, where `r` is the weighted residual vector the
  likelihood is built from, so `F = J^T J` is that likelihood's Gauss-Newton information by
  construction rather than by a parallel derivation;
* every eigenvalue and eigenvector of `F`, in log coordinates so the three parameters are comparable;
* the **Schur complement** of the timing block after eliminating insulin sensitivity -- the information
  about `(k_e, k_a)` that survives re-optimizing `S_I`, which is what a profile likelihood over the
  timing pair would see. Taking the raw sub-block instead is precisely the misreading that makes a
  compensating pair look informative;
* an at-a-bound flag per parameter and the projected gradient norm, so 2.2 can split the cohort into
  interior-converged subjects and all subjects without re-fitting.

`J` is built as `vmap(jacrev(one meal))` rather than one reverse pass per residual: the residuals are
independent across meals, so a single batched gradient produces the whole matrix.

Run:  python -m evaluation.runner evaluation.fisher_full --gate 3
      python -m evaluation.runner evaluation.fisher_full --workers 8
"""
from __future__ import annotations

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402

from evaluation.cohort_data import load_cohort                        # noqa: E402
from evaluation.identifiability_tools import numerical_rank, timing_block  # noqa: E402
from personalization.fit_general import BOUND_TOLERANCE, fit_ml       # noqa: E402
from personalization.objectives import (                              # noqa: E402
    ObjectiveSpec, build_objective, observed_values,
)
from personalization.subject_loss import TARGETS, base_params, subject_arrays  # noqa: E402
from simulation.jax_observables import Window                         # noqa: E402

ANALYSIS_ID = "A4_fisher"


def default_config() -> dict:
    return {
        "cohort": "cgmacros",
        "objective": "iauc",
        "min_meals": 10,
        "limit": None,
        "steps": 600,
        "pilot_steps": 300,
        "learning_rate": 0.02,
        "optimizer": "adam",
        "rank_rtol": 1e-6,
        # H10 already pre-registers a bounds sweep at 0.5x, 1x and 2x, so a wider box is a planned
        # robustness analysis rather than a change of specification. 2.2 needs it: a parameter pinned at
        # a bound in the 1x box and interior in the 2x box was limited by the box, while one pinned in
        # both was limited by the data.
        "bounds_scale": 1.0,
    }


def units(config: dict) -> list[str]:
    subjects = load_cohort(config["cohort"], min_meals=config["min_meals"],
                           limit=config["limit"])
    return [s.subject_id for s in subjects]


def _subject(config: dict, unit: str):
    subjects = load_cohort(config["cohort"], min_meals=config["min_meals"], limit=config["limit"])
    subject = next((s for s in subjects if s.subject_id == unit), None)
    if subject is None:
        raise KeyError(f"no subject {unit!r} in cohort {config['cohort']!r}")
    return subject


def design_matrix(objective, theta, log_coords: bool = True) -> np.ndarray:
    """`d r / d theta` at `theta`, optionally in log parameter coordinates.

    Log coordinates are obtained by scaling each column by its parameter, which is the chain rule for
    `d/d log theta = theta d/d theta`. They are what makes an eigenvalue comparison meaningful when one
    parameter is of order 1 and two are of order 0.02.
    """
    jacobian = np.asarray(jax.jacrev(objective.weighted_residuals)(jnp.asarray(theta)), dtype=float)
    if jacobian.ndim == 1:
        jacobian = jacobian[:, None]
    if log_coords:
        jacobian = jacobian * np.asarray(theta, dtype=float)[None, :]
    return jacobian


def fisher_from_design(jacobian: np.ndarray) -> dict:
    """`F = J^T J`, symmetrized, with its full spectrum."""
    fisher = jacobian.T @ jacobian
    fisher = 0.5 * (fisher + fisher.T)
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    low, high = float(eigenvalues[0]), float(eigenvalues[-1])
    weak = np.asarray(eigenvectors[:, 0], dtype=float)
    if weak[np.argmax(np.abs(weak))] < 0:
        weak = -weak
    return {
        "fisher": fisher.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist(),
        "singular_values": singular.tolist(),
        "condition_number": (high / low) if low > 0 else float("inf"),
        "eigenvalue_ratio": (low / high) if high > 0 else float("nan"),
        "weak_direction": weak.tolist(),
    }


def run_unit(unit: str, config: dict) -> dict:
    subject = _subject(config, unit)
    spec = ObjectiveSpec(name=config["objective"], bounds_scale=config["bounds_scale"])

    # theta_hat_ML, with the frozen noise model the likelihood is weighted by.
    ml = fit_ml(subject, spec, steps=config["steps"], pilot_steps=config["pilot_steps"],
                learning_rate=config["learning_rate"], optimizer=config["optimizer"])
    theta = np.array([ml.ml.theta[name] for name in spec.theta_names], dtype=float)

    records = list(subject.records)
    objective = build_objective(
        ObjectiveSpec(name=config["objective"], lam=0.0, normalization="nll",
                      bounds_scale=config["bounds_scale"]),
        base_params(subject.profile), subject_arrays(records),
        observed_values(records, Window()), noise=ml.noise)

    jacobian = design_matrix(objective, theta, log_coords=True)
    spectrum = fisher_from_design(jacobian)
    fisher = np.asarray(spectrum["fisher"], dtype=float)

    names = list(spec.theta_names)
    block = timing_block(fisher, names)

    # Bound flags come from the ML fit, so 2.2 can split the cohort without refitting. The tolerance is
    # one percent of each interval, the same rule the fit itself used.
    lower = np.asarray(objective.lower, dtype=float)
    upper = np.asarray(objective.upper, dtype=float)
    distance = {name: float(min(theta[i] - lower[i], upper[i] - theta[i])
                            / max(upper[i] - lower[i], 1e-300))
                for i, name in enumerate(names)}

    return {
        "subject_id": unit,
        "cohort": config["cohort"],
        "objective": config["objective"],
        "bounds_scale": config["bounds_scale"],
        "n_meals": subject.n_meals,
        "n_residuals": objective.n_residuals,
        "theta_ml": ml.ml.theta,
        "theta_natural": ml.ml.theta_full,
        "theta_pilot": ml.pilot.theta,
        "log_theta": np.log(theta).tolist(),
        "noise": ml.noise.as_dict(),
        "fit": {
            "converged": ml.ml.converged,
            "still_moving": ml.ml.still_moving,
            "projected_grad_norm": ml.ml.projected_grad_norm,
            "initial_projected_grad_norm": ml.ml.initial_projected_grad_norm,
            "at_bound": ml.ml.at_bound,
            "n_at_bound": ml.ml.n_at_bound,
            "interior": ml.ml.interior,
            "final_loss": ml.ml.final_loss,
            "bound_tolerance": BOUND_TOLERANCE,
            "relative_distance_to_nearest_bound": distance,
        },
        "fisher": spectrum,
        "numerical_rank": numerical_rank(spectrum["singular_values"], rtol=config["rank_rtol"]),
        "timing_block": block,
        "parameters": ["log_" + n for n in names],
        "jacobian_shape": list(jacobian.shape),
        "macros": {},
    }


def summarize(config: dict | None = None) -> dict:
    """Cohort-level summary of whatever units have completed. Reads result files, computes nothing new."""
    from evaluation.results_io import largest_result_set

    config = config or default_config()
    config_hash, documents = largest_result_set(ANALYSIS_ID)
    rows = [doc["payload"] for key, doc in documents.items() if key != "_unreadable"]
    if not rows:
        return {"n_subjects": 0}

    def column(getter):
        return np.array([getter(r) for r in rows], dtype=float)

    interior = [r for r in rows if r["fit"]["interior"]]
    eigenvalues = np.array([r["fisher"]["eigenvalues"] for r in rows], dtype=float)
    condition = column(lambda r: r["timing_block"]["condition_number"])

    def quantiles(values):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"median": float("nan"), "q1": float("nan"), "q3": float("nan"), "n": 0}
        return {"median": float(np.median(finite)), "q1": float(np.percentile(finite, 25)),
                "q3": float(np.percentile(finite, 75)), "n": int(finite.size)}

    return {
        "config_hash": config_hash,
        "commit_of_results": rows[0].get("commit") if isinstance(rows[0], dict) else None,
        "n_subjects": len(rows),
        "n_interior_converged": len(interior),
        "fraction_interior": len(interior) / len(rows),
        "n_at_bound_any": sum(1 for r in rows if r["fit"]["n_at_bound"] > 0),
        "at_bound_by_parameter": {
            name: sum(1 for r in rows if r["fit"]["at_bound"].get(name)) for name in TARGETS},
        "eigenvalues": {f"lambda{i + 1}": quantiles(eigenvalues[:, i])
                        for i in range(eigenvalues.shape[1])},
        "full_condition_number": quantiles(column(lambda r: r["fisher"]["condition_number"])),
        "timing_block_condition_number": quantiles(condition),
        "numerical_rank_counts": {
            str(k): int(v) for k, v in
            zip(*np.unique(column(lambda r: r["numerical_rank"]).astype(int), return_counts=True))},
        "converged": sum(1 for r in rows if r["fit"]["converged"]),
        "still_moving": sum(1 for r in rows if r["fit"]["still_moving"]),
    }


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", action="store_true", help="summarize completed units and exit")
    ap.add_argument("--unit", default=None, help="compute one unit in the foreground")
    args = ap.parse_args()
    config = default_config()

    if args.summary:
        print(json.dumps(summarize(config), indent=2))
        return
    if args.unit:
        print(json.dumps(run_unit(args.unit, config), indent=2)[:4000])
        return
    print(__doc__)
    print(f"units: {len(units(config))} subjects")


if __name__ == "__main__":
    main()
