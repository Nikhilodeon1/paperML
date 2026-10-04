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

Run:  python -m evaluation.runner evaluation.profile_lik --gate 3
      python -m evaluation.runner evaluation.profile_lik --workers 6 --set bounds_scale=2.0
"""
from __future__ import annotations

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import jax.numpy as jnp          # noqa: E402

from evaluation.cohort_data import load_cohort                              # noqa: E402
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
        "fit_steps": 600,
        "pilot_steps": 300,
        "fit_learning_rate": 0.02,
        # Within this, a grid point whose inner loss is still falling is "unsettled". The unit is the
        # negative log likelihood, so 0.05 is small against the 1.92 interval threshold.
        "drift_tolerance": 0.05,
    }


def units(config: dict) -> list[str]:
    subjects = load_cohort(config["cohort"], min_meals=config["min_meals"], limit=config["limit"])
    return [s.subject_id for s in subjects]


def _subject(config: dict, unit: str):
    subjects = load_cohort(config["cohort"], min_meals=config["min_meals"], limit=config["limit"])
    subject = next((s for s in subjects if s.subject_id == unit), None)
    if subject is None:
        raise KeyError(f"no subject {unit!r} in cohort {config['cohort']!r}")
    return subject


def stored_estimate(config: dict, unit: str) -> dict | None:
    """The A4 maximum-likelihood fit for this subject, objective and box, if one exists.

    Using it keeps A4 and A5 about the same `theta_hat`. The most complete matching result set wins.
    """
    from evaluation.results_io import load_all
    best = None
    for documents in load_all("A4_fisher").values():
        doc = documents.get(unit)
        if doc is None:
            continue
        payload = doc["payload"]
        if (payload.get("objective") == config["objective"]
                and abs(float(payload.get("bounds_scale", 1.0)) - config["bounds_scale"]) < 1e-12):
            if best is None or len(documents) > best[0]:
                best = (len(documents), payload)
    return None if best is None else best[1]


def _noise_from(payload_noise: dict) -> nm.NoiseModel:
    return nm.NoiseModel(**payload_noise)


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
    subject = _subject(config, unit)
    scale = config["bounds_scale"]
    name = config["objective"]
    spec = ObjectiveSpec(name=name, lam=0.0, normalization="nll", bounds_scale=scale)
    names = list(spec.theta_names)

    stored = stored_estimate(config, unit)
    if stored is not None:
        theta_hat = np.array([stored["theta_ml"][n] for n in names], dtype=float)
        noise = _noise_from(stored["noise"])
        source = "A4"
    else:
        ml = fit_ml(subject, ObjectiveSpec(name=name, bounds_scale=scale),
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

    def loss_log(phi):
        return objective.loss(jnp.exp(phi))

    phi_hat = np.log(theta_hat)
    loss_at_hat = float(loss_log(jnp.asarray(phi_hat)))
    phi_lower, phi_upper = np.log(lower), np.log(upper)
    tolerance = config["drift_tolerance"]

    profiles = {}
    for index, parameter in enumerate(names):
        extension = (config["extension_points"]
                     if parameter in EXTENDED else 0)
        grid, n_inside = log_grid(lower[index], upper[index], config["grid_points"], extension)
        raw = profile_likelihood(loss_log, phi_hat, index, np.log(grid), lower=phi_lower,
                                 upper=phi_upper, steps=config["steps"],
                                 learning_rate=config["learning_rate"])
        profile = np.asarray(raw["profile"], dtype=float)
        drift = np.asarray(raw["drift"], dtype=float)
        reference = min(loss_at_hat, float(profile.min()))
        inside = slice(0, n_inside)
        verdict = classify_profile(grid[inside], profile[inside], delta=CHI2_DELTA,
                                   reference=reference)
        entry = {
            "parameter": parameter,
            "grid": grid.tolist(),
            "n_in_box": int(n_inside),
            "profile": profile.tolist(),
            "drift": drift.tolist(),
            "unsettled": int(np.sum(np.abs(drift[inside]) > tolerance)),
            "inner_theta": np.exp(np.asarray(raw["theta"], dtype=float)).tolist(),
            "classification": verdict,
            "reference": reference,
            "loss_at_theta_hat": loss_at_hat,
            "better_than_theta_hat": bool(float(profile.min()) < loss_at_hat - 1e-6),
            "argmin_in_box": float(grid[inside][int(np.argmin(profile[inside]))]),
            "theta_hat": float(theta_hat[index]),
            "profile_min_near_theta_hat": bool(
                abs(np.log(grid[inside][int(np.argmin(profile[inside]))]) - phi_hat[index])
                <= (phi_upper[index] - phi_lower[index]) / (config["grid_points"] - 1) + 1e-9),
        }
        if extension:
            entry["extended"] = classify_profile(grid, profile, delta=CHI2_DELTA,
                                                 reference=reference)
            entry["extended_note"] = ("points beyond the physiological upper bound; diagnostic "
                                      "only, not part of the pre-registered classification")
        profiles[parameter] = entry

    return {
        "subject_id": unit, "cohort": config["cohort"], "objective": name,
        "bounds_scale": scale, "n_meals": subject.n_meals, "estimate_source": source,
        "theta_ml": dict(zip(names, theta_hat.tolist())),
        "box": {n: [float(lower[i]), float(upper[i])] for i, n in enumerate(names)},
        "distance_to_bound": {
            n: float(min(theta_hat[i] - lower[i], upper[i] - theta_hat[i])
                     / (upper[i] - lower[i])) for i, n in enumerate(names)},
        "delta": CHI2_DELTA, "profiles": profiles,
        "stored_fit": None if stored is None else {
            "converged": stored["fit"]["converged"], "interior": stored["fit"]["interior"]},
        "macros": {},
    }


def summarize(config: dict | None = None) -> dict:
    """Per-parameter fractions of bounded intervals, with Wilson intervals, for one box and objective."""
    from evaluation.results_io import largest_matching
    from evaluation.stats_utils import wilson_ci

    config = config or default_config()
    by_subject = largest_matching(
        ANALYSIS_ID, lambda p: p["objective"] == config["objective"]
        and abs(p["bounds_scale"] - config["bounds_scale"]) < 1e-12)
    if not by_subject:
        return {"n_subjects": 0}
    rows = list(by_subject.values())

    out = {"n_subjects": len(rows), "objective": config["objective"],
           "bounds_scale": config["bounds_scale"], "parameters": {}}
    for parameter in TARGETS:
        verdicts = [r["profiles"][parameter]["classification"]["verdict"] for r in rows]
        bounded = sum(v == "identifiable" for v in verdicts)
        ci = wilson_ci(bounded, len(rows))
        out["parameters"][parameter] = {
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
    # Amendment 2: interior FOR S_I, from the maximum-likelihood estimate's distance to its own bounds.
    interior = [r for r in rows if r["distance_to_bound"]["insulin_sensitivity"] > 0.01]
    out["interior_for_S_I"] = len(interior)
    if interior:
        bounded = sum(r["profiles"]["insulin_sensitivity"]["classification"]["verdict"]
                      == "identifiable" for r in interior)
        ci = wilson_ci(bounded, len(interior))
        out["S_I_bounded_among_interior_for_S_I"] = {
            "bounded": bounded, "n": len(interior), "fraction": bounded / len(interior),
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
