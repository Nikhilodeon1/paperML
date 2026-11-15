"""A18: where is the likelihood at the truth? (Amendment 11, R10 and H30)

On a replica the true parameters are known. For the trace coordinate profiles (polished estimates, primary box) this
analysis rebuilds the objective of a stored profile run, evaluates the negative log-likelihood at the TRUE parameters,
and starts an L-BFGS-B run from the truth. Two things follow per subject:

* `nll_truth - nll_polished` is the likelihood-ratio statistic at the truth (half of it): under a correct model and a
  global optimum it is about half of a chi-square with three degrees of freedom, so a value far above that means the
  likelihood puts the truth far from its optimum;
* `nll_polished - nll_truth_start` is positive when the stored polished estimate was NOT the best optimum and a run
  from the truth finds a better one, that is, when the optimizer missed it.

The first separates "the likelihood is miscalibrated" from the second, "the optimizer missed", which is the question
left open by the low-noise undercoverage of the coordinate profiles. One unit is one subject. The inputs are the stored
A5 result of the same replica and inner budget (the noise model, the polished estimate and the truth are taken from it).

Run:  python -m evaluation.runner evaluation.truth_calibration --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"cgm_scale":0.25}'
"""
from __future__ import annotations

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import jax.numpy as jnp                                                      # noqa: E402

from evaluation import profile_lik                                           # noqa: E402
from evaluation.results_io import largest_matching                           # noqa: E402
from evaluation.subject_source import get_subject, window_for                # noqa: E402
from personalization import noise_model as nm                                # noqa: E402
from personalization.objectives import ObjectiveSpec, build_objective, observed_values  # noqa: E402
from personalization.subject_loss import base_params, subject_arrays         # noqa: E402

ANALYSIS_ID = "A18_truth_calibration"


def default_config() -> dict:
    base = profile_lik.default_config()
    return {"cohort": "cgmacros", "min_meals": 10, "limit": None, "steps": 500, "polish_maxiter": 300,
            "replica": {"seed": 0, "cgm": True, "carb_cv": 0.0}, "bounds_scale": 1.0,
            "objective": "trace", "parameterization": "coords", "drift_tolerance": base["drift_tolerance"]}


def units(config: dict) -> list[str]:
    from evaluation.cohort_data import load_cohort
    return [s.subject_id for s in load_cohort(config["cohort"], min_meals=config["min_meals"], limit=config["limit"])]


def stored_profile(config: dict) -> dict:
    """The A5 coordinate-profile rows of this replica and inner budget, keyed by subject."""
    target = {**profile_lik.default_config(), "objective": config["objective"], "bounds_scale": config["bounds_scale"],
              "parameterization": config["parameterization"], "polish": True, "steps": config["steps"],
              "replica": config["replica"], "cohort": config["cohort"]}
    return largest_matching(profile_lik.ANALYSIS_ID, lambda p, c=target: profile_lik._matches(p, c))


def run_unit(unit: str, config: dict) -> dict:
    from simulation import jax_engine
    with jax_engine.residual_gut(float((config.get("replica") or {}).get("residual_gut", 0.0))):
        return _run_unit(unit, config)


def _run_unit(unit: str, config: dict) -> dict:
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
    loss_fn = objective.loss
    lower, upper = np.asarray(objective.lower, dtype=float), np.asarray(objective.upper, dtype=float)
    nll_truth = float(loss_fn(jnp.asarray(phi_true)))
    nll_polished = float(loss_fn(jnp.asarray(phi_hat)))
    start = profile_lik.polish_estimate(loss_fn, phi_true, lower, upper, objective.project, f"truth|{unit}", 0,
                                        int(config["polish_maxiter"]))
    return {"subject_id": unit, "replica": config["replica"], "steps": int(config["steps"]),
            "nll_truth": nll_truth, "nll_polished": nll_polished, "nll_truth_start": float(start["nll_polished"]),
            "stored_nll_polished": float(stored["polish"]["nll_polished"]) if stored.get("polish") else None,
            "likelihood_ratio_half": nll_truth - min(nll_polished, float(start["nll_polished"])),
            "optimizer_gap": nll_polished - float(start["nll_polished"]),
            "phi_true": phi_true.tolist(), "phi_polished": phi_hat.tolist(),
            "phi_truth_start": [float(v) for v in np.asarray(start["phi"])], "macros": {}}


def summarize(config: dict | None = None) -> dict:
    """Quantiles of the half likelihood ratio at the truth and the share of subjects where the optimizer missed."""
    from scipy.stats import chi2
    from evaluation.identifiability_tools import CHI2_DELTA
    from evaluation.stats_utils import wilson_ci
    config = config or default_config()
    rows = list(largest_matching(ANALYSIS_ID, lambda p, r=config["replica"], s=int(config["steps"]):
                                 p.get("replica") == r and int(p.get("steps", 0)) == s).values())
    if not rows:
        return {"n": 0}
    lr = np.array([r["likelihood_ratio_half"] for r in rows])
    gap = np.array([r["optimizer_gap"] for r in rows])
    within = int((gap < CHI2_DELTA).sum())
    ci = wilson_ci(within, len(rows))
    reference = chi2(3).ppf([0.5, 0.9, 0.95]) / 2.0
    return {"n": len(rows), "median_lr_half": float(np.median(lr)), "q90_lr_half": float(np.percentile(lr, 90)),
            "reference_half_chi2_3": [float(v) for v in reference],
            "share_lr_over_delta": float((lr > CHI2_DELTA).mean()),
            "optimizer_gap_within": within, "optimizer_gap_fraction": within / len(rows),
            "optimizer_gap_wilson": [ci.low, ci.high], "median_optimizer_gap": float(np.median(gap)),
            "max_optimizer_gap": float(np.max(gap))}


def main() -> None:
    import json
    print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
