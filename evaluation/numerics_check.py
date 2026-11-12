"""Numerical checks (Amendment 9, R4): does the Fisher spectrum survive a tighter ODE tolerance?

The Fisher eigenvalues near 1e-8 are computed through a solver with relative and absolute tolerance 1e-4. A reader may
ask whether the weak direction is an artifact of solver error. For the first ten CGMacros subjects, in identifier
order, this recomputes the iAUC design matrix at the stored maximum-likelihood estimate (the A4 result) at the
default tolerance and at 1e-8, and compares the eigenvalues and the cosine between the weakest timing eigenvector and
the exchange direction. Nothing is refitted: the estimate and the frozen noise model are the stored ones.

(b) of the amendment, the profile classes under 500 inner Adam steps in place of 100, is a profile_lik run
(`--set steps=500 --set limit=10`) that `summarize_profile_steps` compares with the stored default run.

Run:  python -m evaluation.runner evaluation.numerics_check
"""
from __future__ import annotations

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

from evaluation import fisher_full, profile_lik                         # noqa: E402
from evaluation.cohort_data import load_cohort                           # noqa: E402
from evaluation.identifiability_tools import timing_block               # noqa: E402
from evaluation.ladder import trade_off_cosine                           # noqa: E402
from evaluation.results_io import largest_matching, load_all             # noqa: E402
from evaluation.subject_source import get_subject, window_for            # noqa: E402
from personalization.noise_model import NoiseModel                       # noqa: E402
from personalization.objectives import ObjectiveSpec, build_objective, observed_values  # noqa: E402
from personalization.subject_loss import base_params, subject_arrays    # noqa: E402
from simulation import jax_engine                                        # noqa: E402

ANALYSIS_ID = "A17_numerics"
TOLERANCES = (1e-4, 1e-8)
N_SUBJECTS = 10


def default_config() -> dict:
    return {"cohort": "cgmacros", "objective": "iauc", "min_meals": 10, "bounds_scale": 1.0, "n_subjects": N_SUBJECTS,
            "tolerances": list(TOLERANCES)}


def units(config: dict) -> list[str]:
    subjects = load_cohort(config["cohort"], min_meals=config["min_meals"])
    return [s.subject_id for s in subjects][:config["n_subjects"]]


def _is_primary_a4(p: dict, config: dict) -> bool:
    return (p.get("objective") == config["objective"] and p.get("cohort") == config["cohort"]
            and abs((p.get("bounds_scale") or 1.0) - config["bounds_scale"]) < 1e-12
            and not p.get("replica") and not p.get("carb_scale"))


_STORED: dict = {}


def _stored(unit: str, config: dict) -> dict:
    """The A4 result for the subject, from the one result set the paper uses (the most complete match)."""
    if "rows" not in _STORED:
        _STORED["rows"] = largest_matching(fisher_full.ANALYSIS_ID, lambda p: _is_primary_a4(p, config))
    if unit not in _STORED["rows"]:
        raise RuntimeError(f"no stored A4 result for {unit}")
    return _STORED["rows"][unit]


def _noise(stored: dict) -> NoiseModel:
    d = dict(stored["noise"])
    d["meal_scale"] = tuple(d.get("meal_scale") or ())
    return NoiseModel(**d)


def _fisher(subject, config: dict, stored: dict, rtol: float) -> dict:
    window = window_for(subject)
    jax_engine.SOLVER_RTOL, jax_engine.SOLVER_ATOL = rtol, rtol
    spec = ObjectiveSpec(name=config["objective"], lam=0.0, normalization="nll", bounds_scale=config["bounds_scale"],
                         window=window)
    records = list(subject.records)
    objective = build_objective(spec, base_params(subject.profile), subject_arrays(records),
                                observed_values(records, window), noise=_noise(stored))
    names = list(spec.theta_names)
    theta = np.array([stored["theta_ml"][n] for n in names], dtype=float)
    jacobian = fisher_full.design_matrix(objective, theta, log_coords=True)
    spectrum = fisher_full.fisher_from_design(jacobian)
    fisher = np.asarray(spectrum["fisher"], dtype=float)
    block = timing_block(fisher, names)
    return {"eigenvalues": spectrum["eigenvalues"], "weak_direction": block["weak_direction"],
            "cosine": trade_off_cosine(block["weak_direction"], stored["theta_ml"]["gastric_emptying"],
                                       stored["theta_ml"]["carb_absorption"]),
            "timing_condition": block["condition_number"]}


def run_unit(unit: str, config: dict) -> dict:
    stored = _stored(unit, config)
    subject = get_subject({**config, "replica": None, "carb_scale": None, "limit": None}, unit)
    out = {"subject_id": unit, "stored_eigenvalues": stored["fisher"]["eigenvalues"],
           "stored_cosine": trade_off_cosine(stored["timing_block"]["weak_direction"],
                                             stored["theta_ml"]["gastric_emptying"],
                                             stored["theta_ml"]["carb_absorption"]),
           "by_tolerance": {}, "macros": {}}
    try:
        for rtol in config["tolerances"]:
            out["by_tolerance"][f"{rtol:g}"] = _fisher(subject, config, stored, float(rtol))
    finally:
        jax_engine.SOLVER_RTOL = jax_engine.SOLVER_ATOL = 1e-4
    return out


def summarize() -> dict:
    rows = list(largest_matching(ANALYSIS_ID, lambda p: "by_tolerance" in p).values())
    if not rows:
        return {"n": 0}
    tight, loose = f"{TOLERANCES[1]:g}", f"{TOLERANCES[0]:g}"
    rel, rel_default, d_cos = [], [], []
    for r in rows:
        stored = np.asarray(r["stored_eigenvalues"], dtype=float)
        a = np.asarray(r["by_tolerance"][tight]["eigenvalues"], dtype=float)
        b = np.asarray(r["by_tolerance"][loose]["eigenvalues"], dtype=float)
        rel.append(float(np.max(np.abs(a - stored) / np.maximum(np.abs(stored), 1e-300))))
        rel_default.append(float(np.max(np.abs(b - stored) / np.maximum(np.abs(stored), 1e-300))))
        d_cos.append(float(abs(r["by_tolerance"][tight]["cosine"]) - abs(r["stored_cosine"])))
    return {"n": len(rows),
            "max_relative_eigenvalue_change_tight": float(np.max(rel)),
            "median_relative_eigenvalue_change_tight": float(np.median(rel)),
            "max_relative_eigenvalue_change_default_recompute": float(np.max(rel_default)),
            "min_abs_cosine_tight": float(np.min([abs(r["by_tolerance"][tight]["cosine"]) for r in rows])),
            "max_abs_cosine_change": float(np.max(np.abs(d_cos))),
            "profile_steps": summarize_profile_steps()}


def summarize_profile_steps() -> dict:
    """Profile classes of the first ten subjects at 500 inner steps against the stored 100-step run."""
    default = largest_matching(profile_lik.ANALYSIS_ID, lambda p: profile_lik._matches(
        p, {**profile_lik.default_config(), "objective": "iauc", "bounds_scale": 1.0}))
    long: dict = {}
    for units_ in load_all(profile_lik.ANALYSIS_ID).values():
        rows = {k: d["payload"] for k, d in units_.items() if k != "_unreadable"
                and d.get("config", {}).get("steps") == 500 and d.get("config", {}).get("objective") == "iauc"
                and not d.get("config", {}).get("replica") and d.get("config", {}).get("limit") == N_SUBJECTS}
        if len(rows) > len(long):
            long = rows
    ids = sorted(set(default) & set(long))
    if not ids:
        return {"n_subjects": 0}
    changed, detail = 0, []
    for sid in ids:
        for parameter in default[sid]["profiles"]:
            a = default[sid]["profiles"][parameter]["classification"]["verdict"]
            b = long[sid]["profiles"][parameter]["classification"]["verdict"]
            detail.append({"subject_id": sid, "parameter": parameter, "steps_100": a, "steps_500": b})
            changed += a != b
    return {"n_subjects": len(ids), "n_profiles": len(detail), "changed": int(changed), "detail": detail}
