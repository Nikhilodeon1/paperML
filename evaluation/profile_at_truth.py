"""A19: the profile at the true value, computed four ways (Amendment 12, R13 and H31).

Phase 15 found that, at a quarter of the noise, the profile excess at the true value read off the stored coordinate
grid (median 5.3 for tau1) is far above the likelihood ratio at the true parameters (median 1.0). A profile value cannot
exceed the likelihood at a feasible point with the same coordinate, so the stored number is an overstatement. This
analysis separates the three ways a grid profile can overstate, for each subject and each of tau1 and p:

* `stored`      the excess interpolated at the true value from the stored 15-point grid (what H28 and R7 read);
* `adam_point`  the stored inner routine (Adam, the stored number of steps, from the estimate) run at the true value alone,
                so `stored - adam_point` is the interpolation across the grid;
* `lbfgsb_hat`  a converged L-BFGS-B inner solve from the estimate, so `adam_point - lbfgsb_hat` is the inner solver;
* `lbfgsb_truth` the same solve started from the true free coordinates, so `lbfgsb_hat - lbfgsb_truth` is the start.

All excesses are against the reference of the stored profile (the lower of the loss at the polished estimate and the
profile minimum). One unit is one subject; the inputs are the stored A5 coordinate-profile result of the same replica
and inner budget, as in A18.

Run:  python -m evaluation.runner evaluation.profile_at_truth --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"cgm_scale":0.25}'
"""
from __future__ import annotations

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import jax.numpy as jnp                                                      # noqa: E402

from evaluation import profile_lik                                           # noqa: E402
from evaluation.identifiability_tools import profile_likelihood             # noqa: E402
from evaluation.profile_solver import fixed_coordinate_minimum, reduced_value_and_grad  # noqa: E402
from evaluation.recalibration import excess_at_truth                         # noqa: E402
from evaluation.subject_source import get_subject, window_for                # noqa: E402
from evaluation.truth_calibration import stored_profile, units               # noqa: E402,F401
from personalization import noise_model as nm                                # noqa: E402
from personalization.objectives import ObjectiveSpec, build_objective, observed_values  # noqa: E402
from personalization.subject_loss import base_params, subject_arrays         # noqa: E402

ANALYSIS_ID = "A19_profile_at_truth"
COORDINATES = ("log_tau1", "log_p")


def default_config() -> dict:
    base = profile_lik.default_config()
    return {"cohort": "cgmacros", "min_meals": 10, "limit": None, "steps": 500, "maxiter": 200,
            "learning_rate": base["learning_rate"], "replica": {"seed": 0, "cgm": True, "carb_cv": 0.0},
            "bounds_scale": 1.0, "objective": "trace", "parameterization": "coords"}


def run_unit(unit: str, config: dict) -> dict:
    stored = stored_profile(config).get(unit)
    if stored is None:
        raise RuntimeError(f"no stored {config['parameterization']} profile result for {unit} at this replica and budget")
    subject = get_subject(config, unit)
    spec = ObjectiveSpec(name=config["objective"], lam=0.0, normalization="nll", bounds_scale=config["bounds_scale"],
                         window=window_for(subject), parameterization=config["parameterization"])
    records = list(subject.records)
    objective = build_objective(spec, base_params(subject.profile), subject_arrays(records),
                                observed_values(records, spec.window), noise=nm.NoiseModel(**stored["noise"]))
    names = list(spec.theta_names)
    truth = stored["replica_truth"]["theta_true"]
    phi_true = np.log(np.array([float(truth[n]) for n in names], dtype=float))
    phi_hat = np.array([stored["theta_ml"][n] for n in names], dtype=float)
    lower, upper = np.asarray(objective.lower, dtype=float), np.asarray(objective.upper, dtype=float)
    loss_fn, project = objective.loss, objective.project
    out = {"subject_id": unit, "replica": config["replica"], "steps": int(config["steps"]),
           "nll_truth": float(loss_fn(jnp.asarray(phi_true))), "coordinates": {}, "macros": {}}
    for name in COORDINATES:
        index = names.index(name)
        entry = stored["profiles"][name]
        value = float(phi_true[index])
        reference = float(entry["reference"])
        if not lower[index] <= value <= upper[index]:
            out["coordinates"][name] = {"truth_in_box": False}
            continue
        adam = profile_likelihood(loss_fn, phi_hat, index, np.array([value]), lower=lower, upper=upper,
                                  steps=int(config["steps"]), learning_rate=float(config["learning_rate"]),
                                  project=project)
        value_and_grad, free_index = reduced_value_and_grad(loss_fn, project, index, len(names))
        from_hat = fixed_coordinate_minimum(value_and_grad, free_index, phi_hat, index, value, lower, upper,
                                            int(config["maxiter"]))
        from_truth = fixed_coordinate_minimum(value_and_grad, free_index, phi_true, index, value, lower, upper,
                                              int(config["maxiter"]))
        stored_excess = excess_at_truth(entry)
        out["coordinates"][name] = {
            "truth_in_box": True, "reference": reference,
            "stored": stored_excess,
            "adam_point": float(adam["profile"][0]) - reference,
            "lbfgsb_hat": from_hat["loss"] - reference, "lbfgsb_truth": from_truth["loss"] - reference,
            "converged_hat": from_hat["converged"], "converged_truth": from_truth["converged"]}
    return out


def summarize(config: dict | None = None) -> dict:
    """Shares of subjects above delta and the median decomposition, per coordinate, for one replica."""
    from evaluation.identifiability_tools import CHI2_DELTA
    from evaluation.results_io import largest_matching
    config = {**default_config(), **(config or {})}
    rows = list(largest_matching(ANALYSIS_ID, lambda p, r=config["replica"], s=int(config["steps"]):
                                 p.get("replica") == r and int(p.get("steps", 0)) == s).values())
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for name in COORDINATES:
        cells = [r["coordinates"][name] for r in rows if r["coordinates"].get(name, {}).get("truth_in_box")]
        if not cells:
            out[name] = {"n": 0}
            continue
        columns = {k: np.array([c[k] if c[k] is not None else np.nan for c in cells], dtype=float)
                   for k in ("stored", "adam_point", "lbfgsb_hat", "lbfgsb_truth")}
        entry = {"n": len(cells), "off_grid": int(np.isnan(columns["stored"]).sum())}
        for k, v in columns.items():
            ok = ~np.isnan(v)
            entry[k] = {"share_above_delta": float((v[ok] > CHI2_DELTA).mean()) if ok.any() else None,
                        "share_within_delta": float((v[ok] <= CHI2_DELTA).mean()) if ok.any() else None,
                        "median": float(np.median(v[ok])) if ok.any() else None, "n": int(ok.sum())}
        both = ~(np.isnan(columns["stored"]) | np.isnan(columns["adam_point"]))
        entry["median_interpolation"] = float(np.median(columns["stored"][both] - columns["adam_point"][both])) if both.any() else None
        entry["median_inner_solver"] = float(np.median(columns["adam_point"] - columns["lbfgsb_hat"]))
        entry["median_start"] = float(np.median(columns["lbfgsb_hat"] - columns["lbfgsb_truth"]))
        entry["converged_hat"] = int(sum(bool(c["converged_hat"]) for c in cells))
        out[name] = entry
    return out


def main() -> None:
    import json
    print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
