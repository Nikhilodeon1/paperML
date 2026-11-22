"""A20: how the curvature of the trace likelihood moves with the carbohydrate logging error (Amendment 15, R21).

Replica intervals for tau1 and p become more often bounded and less often right as the logging error grows, while the
pilot noise standard deviation barely changes (Amendment 14). The candidate explanation is arithmetic: a logged load
that is the true load times a unit-mean multiplier has second moment 1 + CV^2 times the squared true load, the
sensitivity of the simulated trace to every parameter scales with the load, so the Fisher information about the
parameters, measured against an unchanged noise level, grows with the error and the profile narrows around a value that
is not the truth. This analysis measures it. For each subject of a stored coordinate-profile run (polished trace
estimates, primary box) it rebuilds the objective of the replica, evaluates the Fisher information J^T J of the whitened residuals at
the polished estimate in the coordinates (log S_I, log tau1, log p), and records its eigenvalues, the Schur complement of
the (tau1, p) block after profiling out S_I, the noise standard deviation of the stored noise model, and the ratio of the
mean squared logged load to the mean squared true load.

One unit is one subject. Run:
    python -m evaluation.runner evaluation.carb_curvature --set 'replica={"seed":0,"cgm":true,"carb_cv":0.25}'
"""
from __future__ import annotations

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import jax                                                                   # noqa: E402
import jax.numpy as jnp                                                      # noqa: E402

from evaluation import profile_lik                                           # noqa: E402
from evaluation.results_io import largest_matching                           # noqa: E402
from evaluation.subject_source import get_subject, window_for                # noqa: E402
from evaluation.truth_calibration import stored_profile, units               # noqa: E402,F401
from personalization import noise_model as nm                                # noqa: E402
from personalization.objectives import ObjectiveSpec, build_objective, observed_values  # noqa: E402
from personalization.subject_loss import base_params, subject_arrays         # noqa: E402

ANALYSIS_ID = "A20_carb_curvature"


def default_config() -> dict:
    return {"cohort": "cgmacros", "min_meals": 10, "limit": None, "steps": 100, "polish_maxiter": 300,
            "replica": {"seed": 0, "cgm": True, "carb_cv": 0.25}, "bounds_scale": 1.0,
            "objective": "trace", "parameterization": "coords"}


def run_unit(unit: str, config: dict) -> dict:
    stored = stored_profile(config).get(unit)
    if stored is None:
        raise RuntimeError(f"no stored {config['parameterization']} profile result for {unit} at this replica and budget")
    subject = get_subject(config, unit)
    true_subject = get_subject({**config, "replica": None}, unit)
    spec = ObjectiveSpec(name=config["objective"], lam=0.0, normalization="nll", bounds_scale=config["bounds_scale"],
                         window=window_for(subject), parameterization=config["parameterization"])
    records = list(subject.records)
    args = (base_params(subject.profile), subject_arrays(records), observed_values(records, spec.window))
    names = list(spec.theta_names)
    phi_hat = jnp.asarray([stored["theta_ml"][n] for n in names], dtype=float)
    # The stored profile result does not keep its pilot noise model. The noise standard deviation and autocorrelation are
    # estimated again from the residual at the stored polished estimate, which is what Amendment 14 shows the pilot
    # estimate to be within a fraction of a percent of.
    raw = build_objective(spec, *args, noise=None)
    noise = nm.build(raw.residuals(phi_hat), n_parameters=len(spec.free), whiten_trace=spec.whiten_trace)
    objective = build_objective(spec, *args, noise=noise)
    jacobian = np.asarray(jax.jacrev(objective.weighted_residuals)(phi_hat), dtype=float)   # d residual / d (log coordinate)
    hessian = jacobian.T @ jacobian                                                          # Fisher information, J^T J
    hessian = 0.5 * (hessian + hessian.T)
    eig = np.sort(np.linalg.eigvalsh(hessian))
    index_si = [i for i, n in enumerate(names) if n == "log_insulin_sensitivity"]
    timing = [i for i, n in enumerate(names) if n in ("log_tau1", "log_p")]
    schur = None
    if index_si and len(timing) == 2:
        a = index_si[0]
        h_aa = hessian[a, a]
        block = hessian[np.ix_(timing, timing)] - np.outer(hessian[timing, a], hessian[a, timing]) / h_aa
        schur = np.sort(np.linalg.eigvalsh(0.5 * (block + block.T))).tolist()
    logged = np.array([r["carbs_g"] for r in subject.records], dtype=float)
    true = np.array([r["carbs_g"] for r in true_subject.records], dtype=float)
    return {"subject_id": unit, "replica": config["replica"], "steps": int(config["steps"]),
            "names": names, "hessian_eigenvalues": eig.tolist(), "timing_schur_eigenvalues": schur,
            "hessian_diag": np.diag(hessian).tolist(),
            "noise_innovation_sd": float(list(noise.as_dict()["sigma"].values())[0]),
            "noise": noise.as_dict(), "n_meals": int(len(logged)),
            "load_ratio_squared": float(np.mean(logged ** 2) / np.mean(true ** 2)),
            "load_ratio_mean": float(np.mean(logged) / np.mean(true))}


def summarize(config: dict | None = None) -> dict:
    config = {**default_config(), **(config or {})}
    rows = list(largest_matching(ANALYSIS_ID, lambda p, r=config["replica"]: p.get("replica") == r).values())
    if not rows:
        return {"n": 0}
    pick = lambda key: np.array([r[key] for r in rows], dtype=float)
    out = {"n": len(rows), "load_ratio_squared": float(np.median(pick("load_ratio_squared"))),
           "load_ratio_mean": float(np.median(pick("load_ratio_mean")))}
    eig = np.array([r["hessian_eigenvalues"] for r in rows], dtype=float)
    out["eig_median"] = np.median(eig, axis=0).tolist()
    schur = [r["timing_schur_eigenvalues"] for r in rows if r["timing_schur_eigenvalues"]]
    if schur:
        out["schur_median"] = np.median(np.array(schur, dtype=float), axis=0).tolist()
    return out


def main() -> None:
    import json
    print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
