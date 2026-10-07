"""A5: profile-likelihood 95% intervals, per subject and per parameter.

At `theta_hat_ML` (amendment B1: lambda = 0, never the regularized estimate) and with the frozen noise
model of amendment B2 (and B3 for the trace), each of the three parameters is profiled in turn: a grid
over that parameter's interval, the other two re-optimized at every grid point, and the profile read
against `reference + 1.92` on the negative-log-likelihood scale.

Three choices that are about whether the answer can be trusted rather than about speed:

* **The inner problem is solved in log coordinates.** The timing rates live on intervals about 0.025
  wide, so a unit-scale Adam step in rate space crosses the whole interval and the inner optimizer
  bounces between walls. Log coordinates make a step a fixed relative change for every parameter. The
  likelihood is the same function; only the path the optimizer takes to its minimum changes, and a
  profile is invariant to reparameterization.
* **Every grid point reports whether its inner optimization had stopped improving.** A point that is
  still descending overstates the profile, and an overstated profile makes a flat parameter look
  bounded: the error runs in the direction that flatters identifiability. Such points are counted and
  reported, never dropped.
* **The reference is the lower of the loss at `theta_hat` and the profile minimum.** If a grid point
  finds a better fit than the stored estimate, the estimate was not the optimum and the subject is
  flagged rather than the interval silently shifted.

Amendment 2 adds, for `k_e` and `k_a` only, a diagnostic extension of the grid to four times the upper
bound. Those points are outside the physiological range and are reported separately: the classification
that enters H5 uses the in-box points alone.

Phase 9 (Amendment 5) adds `polish`: before profiling, the maximum-likelihood estimate is re-optimized with
L-BFGS-B in log coordinates from the Adam estimate and from five seeded random starting points, and the
best of the six replaces it. The Adam estimate is not discarded: its negative log-likelihood and the gap to
the polished one are stored, so "was the fixed-step Adam estimate a likelihood optimum" is a number per
subject (H21) and not an assumption.

Run:  python -m evaluation.runner evaluation.profile_lik --gate 3
      python -m evaluation.runner evaluation.profile_lik --workers 6 --set bounds_scale=2.0
"""
from __future__ import annotations

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import jax.numpy as jnp          # noqa: E402

from evaluation.cohort_data import load_cohort                              # noqa: E402
from evaluation.subject_source import get_subject, window_for               # noqa: E402
from evaluation.identifiability_tools import CHI2_DELTA, classify_profile, profile_likelihood  # noqa: E402
from personalization import noise_model as nm                                # noqa: E402
from personalization.fit_general import fit_ml                               # noqa: E402
from personalization.objectives import (                                     # noqa: E402
    ObjectiveSpec, build_objective, observed_values,
)
from personalization.subject_loss import TARGETS, base_params, subject_arrays  # noqa: E402

ANALYSIS_ID = "A5_profile"
EXTENSION_FACTOR = 4.0
EXTENDED = ("gastric_emptying", "carb_absorption")


def default_config() -> dict:
    return {
        "cohort": "cgmacros",
        "objective": "iauc",
        "min_meals": 10,
        "limit": None,
        "grid_points": 15,
        "extension_points": 6,
        "steps": 100,
        "learning_rate": 0.03,
        "bounds_scale": 1.0,
        "parameterization": "rates",
        "carb_scale": None,
        "replica": None,
        "fit_steps": 600,
        "pilot_steps": 300,
        "fit_learning_rate": 0.02,
        # Within this, a grid point whose inner loss is still falling is "unsettled". The unit is the
        # negative log likelihood, so 0.05 is small against the 1.92 interval threshold.
        "drift_tolerance": 0.05,
        "polish": False,
        "polish_starts": 5,
        "polish_maxiter": 300,
    }


def units(config: dict) -> list[str]:
    subjects = load_cohort(config["cohort"], min_meals=config["min_meals"], limit=config["limit"])
    return [s.subject_id for s in subjects]


def stored_estimate(config: dict, unit: str) -> dict | None:
    """The A4 maximum-likelihood fit for this subject, objective and box, if one exists.

    Using it keeps A4 and A5 about the same `theta_hat`. The most complete matching result set wins.
    """
    from evaluation.results_io import load_all
    transformed = (config.get("replica") or config.get("carb_scale") not in (None, 1, 1.0)
                   or config.get("parameterization", "rates") != "rates")
    if transformed or config["cohort"] != "cgmacros":
        return None          # the stored fits are for the real CGMacros subjects in rate coordinates
    best = None
    for documents in load_all("A4_fisher").values():
        doc = documents.get(unit)
        if doc is None:
            continue
        payload = doc["payload"]
        if (payload.get("objective") == config["objective"]
                and abs(float(payload.get("bounds_scale", 1.0)) - config["bounds_scale"]) < 1e-12
                and payload.get("cohort", "cgmacros") == "cgmacros"
                and payload.get("carb_scale") in (None, 1, 1.0) and payload.get("replica") is None):
            if best is None or len(documents) > best[0]:
                best = (len(documents), payload)
    return None if best is None else best[1]


def _noise_from(payload_noise: dict) -> nm.NoiseModel:
    return nm.NoiseModel(**payload_noise)


def polish_estimate(loss_fn, phi_hat, lower, upper, project, key: str, starts: int, maxiter: int) -> dict:
    """Best of an L-BFGS-B run from `phi_hat` and from `starts` seeded random points in the box.

    Works in the coordinates the profile uses (log coordinates). `project` enforces the real-pole
    constraint of the canonical parameterization; the rate parameterization has none. The result is never
    worse than the estimate it started from, because that estimate is one of the starting points.
    """
    import jax
    from scipy.optimize import minimize
    from evaluation.subject_source import _rng

    def f(phi):
        return loss_fn(project(phi) if project is not None else phi)

    value_and_grad = jax.jit(jax.value_and_grad(f))

    def fun(x):
        v, g = value_and_grad(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float)

    def feasible(x):
        return np.asarray(project(jnp.asarray(x)), dtype=float) if project is not None else np.asarray(x)

    rng = _rng("polish", key)
    points = [np.asarray(phi_hat, dtype=float)] + [rng.uniform(lower, upper) for _ in range(starts)]
    bounds = list(zip(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)))
    results = []
    for x0 in points:
        r = minimize(fun, feasible(x0), jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": maxiter})
        x = feasible(r.x)
        results.append((float(f(jnp.asarray(x))), x))
    best_value, best_x = min(results, key=lambda item: item[0])
    nll_adam = float(loss_fn(jnp.asarray(phi_hat)))
    if best_value > nll_adam:                       # cannot happen except through projection rounding
        best_value, best_x = nll_adam, np.asarray(phi_hat, dtype=float)
    return {"phi": best_x, "nll_adam": nll_adam, "nll_polished": best_value,
            "gap": nll_adam - best_value, "start_values": [v for v, _ in results]}


def log_grid(lower: float, upper: float, points: int, extension: int = 0) -> tuple[np.ndarray, int]:
    """Log-spaced grid over [lower, upper], optionally continued past `upper` to 4x it.

    Returns the grid and the number of in-box points, which are always the first ones.
    """
    inside = np.exp(np.linspace(np.log(lower), np.log(upper), points))
    if extension <= 0:
        return inside, len(inside)
    beyond = np.exp(np.linspace(np.log(upper), np.log(upper * EXTENSION_FACTOR), extension + 1))[1:]
    return np.concatenate([inside, beyond]), len(inside)


def run_unit(unit: str, config: dict) -> dict:
    subject = get_subject(config, unit)
    window = window_for(subject)
    scale = config["bounds_scale"]
    name = config["objective"]
    parameterization = config.get("parameterization", "rates")
    log_space = parameterization != "rates"        # the fitted vector is already in log coordinates
    spec = ObjectiveSpec(name=name, lam=0.0, normalization="nll", bounds_scale=scale, window=window,
                         parameterization=parameterization)
    names = list(spec.theta_names)

    stored = stored_estimate(config, unit)
    if stored is not None:
        theta_hat = np.array([stored["theta_ml"][n] for n in names], dtype=float)
        noise = _noise_from(stored["noise"])
        source = "A4"
    else:
        ml = fit_ml(subject, ObjectiveSpec(name=name, bounds_scale=scale, window=window,
                                           parameterization=parameterization),
                    steps=config["fit_steps"], pilot_steps=config["pilot_steps"],
                    learning_rate=config["fit_learning_rate"])
        theta_hat = np.array([ml.ml.theta[n] for n in names], dtype=float)
        noise = ml.noise
        source = "refit"

    records = list(subject.records)
    objective = build_objective(spec, base_params(subject.profile), subject_arrays(records),
                                observed_values(records, spec.window), noise=noise)
    lower = np.asarray(objective.lower, dtype=float)
    upper = np.asarray(objective.upper, dtype=float)

    if log_space:
        loss_fn = objective.loss
        phi_hat, phi_lower, phi_upper = theta_hat, lower, upper
        project = objective.project
    else:
        def loss_fn(phi):
            return objective.loss(jnp.exp(phi))
        phi_hat, phi_lower, phi_upper = np.log(theta_hat), np.log(lower), np.log(upper)
        project = None

    polish = None
    if config.get("polish"):
        polish = polish_estimate(loss_fn, phi_hat, phi_lower, phi_upper, project, f"{unit}|{name}|{parameterization}",
                                 int(config["polish_starts"]), int(config["polish_maxiter"]))
        phi_hat = polish["phi"]
        theta_hat = phi_hat if log_space else np.exp(phi_hat)
        polish = {k: v for k, v in polish.items() if k != "phi"}
    loss_at_hat = float(loss_fn(jnp.asarray(phi_hat)))
    tolerance = config["drift_tolerance"]
    truth = None
    if config.get("replica"):
        from evaluation.subject_source import replica_truth, truth_in_coordinates
        truth = replica_truth(config, unit)
        if log_space and parameterization == "coords":
            truth = {**truth, "theta_true": truth_in_coordinates(truth["theta_true"])}

    profiles = {}
    for index, parameter in enumerate(names):
        extension = (config["extension_points"] if (parameter in EXTENDED and not log_space) else 0)
        if log_space:
            phi_grid = np.linspace(phi_lower[index], phi_upper[index], config["grid_points"])
            grid, n_inside = np.exp(phi_grid), len(phi_grid)
        else:
            grid, n_inside = log_grid(lower[index], upper[index], config["grid_points"], extension)
            phi_grid = np.log(grid)
        raw = profile_likelihood(loss_fn, phi_hat, index, phi_grid, lower=phi_lower, upper=phi_upper,
                                 steps=config["steps"], learning_rate=config["learning_rate"],
                                 project=project)
        profile = np.asarray(raw["profile"], dtype=float)
        drift = np.asarray(raw["drift"], dtype=float)
        reference = min(loss_at_hat, float(profile.min()))
        inside = slice(0, n_inside)
        verdict = classify_profile(grid[inside], profile[inside], delta=CHI2_DELTA, reference=reference)
        theta_hat_natural = float(np.exp(phi_hat[index])) if log_space else float(theta_hat[index])
        entry = {
            "parameter": parameter,
            "grid": grid.tolist(),
            "n_in_box": int(n_inside),
            "profile": profile.tolist(),
            "drift": drift.tolist(),
            "unsettled": int(np.sum(np.abs(drift[inside]) > tolerance)),
            "inner_theta": np.asarray(raw["theta"], dtype=float).tolist(),
            "classification": verdict,
            "reference": reference,
            "loss_at_theta_hat": loss_at_hat,
            "better_than_theta_hat": bool(float(profile.min()) < loss_at_hat - 1e-6),
            "argmin_in_box": float(grid[inside][int(np.argmin(profile[inside]))]),
            "theta_hat": theta_hat_natural,
            "profile_min_near_theta_hat": bool(
                abs(np.log(grid[inside][int(np.argmin(profile[inside]))]) - phi_hat[index]
                    if not log_space else
                    np.log(grid[inside][int(np.argmin(profile[inside]))]) - phi_hat[index])
                <= (phi_upper[index] - phi_lower[index]) / (config["grid_points"] - 1) + 1e-9),
        }
        if extension:
            entry["extended"] = classify_profile(grid, profile, delta=CHI2_DELTA, reference=reference)
            entry["extended_note"] = ("points beyond the physiological upper bound; diagnostic "
                                      "only, not part of the pre-registered classification")
        if truth is not None and parameter in truth["theta_true"]:
            value = truth["theta_true"][parameter]
            entry["truth"] = value
            if verdict["bounded"]:
                entry["truth_covered"] = bool(verdict["ci_low"] <= value <= verdict["ci_high"])
        profiles[parameter] = entry

    distance = {n: float(min(phi_hat[i] - phi_lower[i], phi_upper[i] - phi_hat[i])
                         / (phi_upper[i] - phi_lower[i])) for i, n in enumerate(names)}
    return {
        "subject_id": unit, "cohort": config["cohort"], "objective": name,
        "parameterization": parameterization, "replica": config.get("replica"),
        "carb_scale": config.get("carb_scale"), "polish": polish,
        "bounds_scale": scale, "n_meals": subject.n_meals, "estimate_source": source,
        "sampling_min": subject.sampling_min,
        "theta_ml": dict(zip(names, theta_hat.tolist())),
        "box": {n: [float(lower[i]), float(upper[i])] for i, n in enumerate(names)},
        "distance_to_bound": distance, "delta": CHI2_DELTA, "profiles": profiles,
        "replica_truth": truth,
        "stored_fit": None if stored is None else {
            "converged": stored["fit"]["converged"], "interior": stored["fit"]["interior"]},
        "macros": {},
    }


def _matches(payload: dict, config: dict) -> bool:
    return (payload["objective"] == config["objective"]
            and abs(payload["bounds_scale"] - config["bounds_scale"]) < 1e-12
            and payload.get("cohort", "cgmacros") == config.get("cohort", "cgmacros")
            and payload.get("parameterization", "rates") == config.get("parameterization", "rates")
            and payload.get("replica") == config.get("replica")
            and payload.get("carb_scale") == config.get("carb_scale")
            and (payload.get("polish") is not None) == bool(config.get("polish")))


def summarize(config: dict | None = None) -> dict:
    """Per-parameter fractions of bounded intervals, with Wilson intervals, for one configuration."""
    from evaluation.results_io import largest_matching
    from evaluation.stats_utils import wilson_ci

    config = {**default_config(), **(config or {})}
    by_subject = largest_matching(ANALYSIS_ID, lambda p: _matches(p, config))
    if not by_subject:
        return {"n_subjects": 0}
    return summarize_rows(list(by_subject.values()), config)


def summarize_rows(rows: list[dict], config: dict) -> dict:
    """The per-parameter summary of any list of profile payloads (a pooled set of replica seeds, say)."""
    from evaluation.stats_utils import wilson_ci

    out = {"n_subjects": len(rows), "objective": config["objective"],
           "bounds_scale": config["bounds_scale"], "cohort": config["cohort"],
           "parameterization": config.get("parameterization", "rates"), "parameters": {}}
    for parameter in rows[0]["profiles"]:
        verdicts = [r["profiles"][parameter]["classification"]["verdict"] for r in rows]
        bounded = sum(v == "identifiable" for v in verdicts)
        ci = wilson_ci(bounded, len(rows))
        entry = {
            "bounded": bounded, "fraction": bounded / len(rows),
            "wilson": [ci.low, ci.high],
            "identifiable": verdicts.count("identifiable"),
            "one_sided": verdicts.count("one-sided"), "flat": verdicts.count("flat"),
            "unsettled_points": int(sum(r["profiles"][parameter]["unsettled"] for r in rows)),
            "subjects_with_better_fit_than_theta_hat": sum(
                r["profiles"][parameter]["better_than_theta_hat"] for r in rows),
            "profile_min_not_near_theta_hat": sum(
                not r["profiles"][parameter]["profile_min_near_theta_hat"] for r in rows),
        }
        covered = [r["profiles"][parameter]["truth_covered"] for r in rows
                   if "truth_covered" in r["profiles"][parameter]]
        if covered:
            entry["truth_coverage_of_bounded"] = {"n": len(covered), "covered": int(sum(covered)),
                                                  "fraction": float(sum(covered) / len(covered))}
        out["parameters"][parameter] = entry
    # Amendment 2: interior FOR a parameter, from the estimate's distance to its own bounds.
    key = "insulin_sensitivity" if "insulin_sensitivity" in rows[0]["profiles"] else "log_insulin_sensitivity"
    interior = [r for r in rows if r["distance_to_bound"][key] > 0.01]
    out["interior_for_S_I"] = len(interior)
    if interior:
        bounded = sum(r["profiles"][key]["classification"]["verdict"] == "identifiable" for r in interior)
        ci = wilson_ci(bounded, len(interior))
        out["S_I_bounded_among_interior_for_S_I"] = {
            "bounded": bounded, "n": len(interior), "fraction": bounded / len(interior),
            "wilson": [ci.low, ci.high]}
    out["interior_for_parameter"] = {
        p: sum(1 for r in rows if r["distance_to_bound"][p] > 0.01) for p in rows[0]["profiles"]}
    # The Amendment 2 reading for every parameter, not only S_I: bounded fraction among the subjects
    # whose estimate of THAT parameter is off its own bounds.
    out["bounded_among_interior"] = {}
    for p in rows[0]["profiles"]:
        inner = [r for r in rows if r["distance_to_bound"][p] > 0.01]
        if inner:
            b = sum(r["profiles"][p]["classification"]["verdict"] == "identifiable" for r in inner)
            ci = wilson_ci(b, len(inner))
            out["bounded_among_interior"][p] = {"bounded": b, "n": len(inner), "fraction": b / len(inner),
                                                "wilson": [ci.low, ci.high]}
    return out


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--unit", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    config = default_config()
    for item in args.set:
        key, _, value = item.partition("=")
        config[key] = type(config[key])(value) if key in config else value
    if args.summary:
        print(json.dumps(summarize(config), indent=2))
        return
    if args.unit:
        result = run_unit(args.unit, config)
        for parameter, entry in result["profiles"].items():
            c = entry["classification"]
            print(f"{parameter:20} {c['verdict']:13} ci=({c['ci_low']}, {c['ci_high']}) "
                  f"unsettled={entry['unsettled']} max_rise={c['max_rise']:.3f}")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
