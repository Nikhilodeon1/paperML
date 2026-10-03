"""A4b: is the Adam maximum-likelihood estimate actually a likelihood optimum?

The Fisher analysis and the profile likelihood are both evaluated at `theta_hat_ML`, and that estimate
comes from fixed-step Adam. Adam in rate coordinates takes steps that are large against a timing
interval 0.025 wide, and its gradient-norm convergence flag fails for most subjects, so this asks the
direct question instead: how much lower a negative log-likelihood can a different, better-conditioned
optimizer reach inside the same box?

L-BFGS-B in log coordinates, started from the stored estimate and from five seeded random points in the
box, taking the best. The gap is `nll(adam) - nll(best)`, in the units of the likelihood, so it can be
read against the 1.92 interval threshold: a gap far below 1.92 means the estimate is a likelihood
optimum to within the resolution that matters, even where the parameter values themselves differ along
a flat direction.

Run:  python -m evaluation.runner evaluation.optimizer_check --set bounds_scale=1.0
"""
from __future__ import annotations

import hashlib

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from evaluation.cohort_data import load_cohort                          # noqa: E402
from evaluation.profile_lik import stored_estimate                      # noqa: E402
from personalization import noise_model as nm                            # noqa: E402
from personalization.objectives import (                                # noqa: E402
    ObjectiveSpec, build_objective, observed_values,
)
from personalization.subject_loss import TARGETS, base_params, subject_arrays  # noqa: E402

ANALYSIS_ID = "A4b_optimizer_check"


def default_config() -> dict:
    return {"cohort": "cgmacros", "objective": "iauc", "min_meals": 10, "limit": None,
            "bounds_scale": 1.0, "n_random_starts": 5, "maxiter": 200, "seed": 0}


def units(config: dict) -> list[str]:
    return [s.subject_id for s in load_cohort(config["cohort"], min_meals=config["min_meals"],
                                              limit=config["limit"])]


def run_unit(unit: str, config: dict) -> dict:
    stored = stored_estimate(config, unit)
    if stored is None:
        raise RuntimeError(f"no A4 result for {unit} (objective {config['objective']}, "
                           f"box {config['bounds_scale']}); run A4 first")
    subject = next(s for s in load_cohort(config["cohort"], min_meals=config["min_meals"],
                                          limit=config["limit"]) if s.subject_id == unit)
    spec = ObjectiveSpec(name=config["objective"], lam=0.0, normalization="nll",
                         bounds_scale=config["bounds_scale"])
    records = list(subject.records)
    objective = build_objective(spec, base_params(subject.profile), subject_arrays(records),
                                observed_values(records, spec.window),
                                noise=nm.NoiseModel(**stored["noise"]))
    lower, upper = np.asarray(objective.lower), np.asarray(objective.upper)
    names = list(spec.theta_names)
    theta_adam = np.array([stored["theta_ml"][n] for n in names], dtype=float)
    nll_adam = float(objective.loss(jnp.asarray(theta_adam)))

    value_and_grad = jax.jit(jax.value_and_grad(lambda phi: objective.loss(jnp.exp(phi))))

    def fun(x):
        v, g = value_and_grad(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float)

    digest = hashlib.sha256(f"{unit}|{config['seed']}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    starts = [np.log(theta_adam)] + [np.log(rng.uniform(lower, upper))
                                     for _ in range(config["n_random_starts"])]
    bounds = list(zip(np.log(lower), np.log(upper)))
    results = []
    for x0 in starts:
        r = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": config["maxiter"]})
        results.append((float(r.fun), np.exp(r.x)))
    best_value, best_theta = min(results, key=lambda item: item[0])
    return {
        "subject_id": unit, "objective": config["objective"],
        "bounds_scale": config["bounds_scale"],
        "nll_adam": nll_adam, "nll_best": best_value, "gap": nll_adam - best_value,
        "theta_adam": dict(zip(names, theta_adam.tolist())),
        "theta_best": dict(zip(names, best_theta.tolist())),
        "start_values": [v for v, _ in results],
        "interval_threshold": 1.9207295, "macros": {},
    }


def summarize(config: dict | None = None) -> dict:
    from evaluation.results_io import load_all
    config = config or default_config()
    gaps = {}
    for documents in load_all(ANALYSIS_ID).values():
        for key, doc in documents.items():
            if key == "_unreadable":
                continue
            p = doc["payload"]
            if p["objective"] == config["objective"] and abs(
                    p["bounds_scale"] - config["bounds_scale"]) < 1e-12:
                gaps[p["subject_id"]] = p["gap"]
    if not gaps:
        return {"n": 0}
    g = np.array(list(gaps.values()))
    return {"n": int(g.size), "median_gap": float(np.median(g)), "max_gap": float(g.max()),
            "fraction_below_0.1": float(np.mean(g < 0.1)),
            "fraction_below_threshold": float(np.mean(g < 1.9207295)),
            "worst_subjects": sorted(gaps, key=gaps.get, reverse=True)[:5]}
