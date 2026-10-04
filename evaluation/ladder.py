"""A6: the observable ladder, iAUC -> iAUC + centroid -> full trace.

H3 says the information about the timing parameters, measured by the condition number of the
Schur-complement timing block, increases along the ladder: the median condition number falls at each
step. This module does not fit anything new. It reads the A4 results (one Fisher analysis per objective
and box) and answers H3 and H4 two ways, because a flat likelihood makes the answer depend on WHERE the
Fisher matrix is evaluated:

* **at each objective's own `theta_hat_ML`**, the pre-registered comparison;
* **at one common reference point per subject**, the regularized iAUC prediction fit. Here every
  objective's information matrix is evaluated at the same parameters, so a difference between rungs is
  a difference in the observable and not in the location on a flat likelihood. The reference is a
  fitted point, not a parameter the paper chose, and it uses each objective's own frozen noise model.

Both are reported (amendment 2). Neither is promoted over the other.

Run:  python -m evaluation.runner evaluation.ladder --set bounds_scale=1.0
"""
from __future__ import annotations

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

from evaluation.cohort_data import load_cohort                          # noqa: E402
from evaluation.fisher_full import design_matrix, fisher_from_design    # noqa: E402
from evaluation.identifiability_tools import timing_block               # noqa: E402
from personalization import noise_model as nm                            # noqa: E402
from personalization.objectives import (                                # noqa: E402
    LADDER, ObjectiveSpec, build_objective, observed_values,
)
from personalization.subject_loss import TARGETS, base_params, subject_arrays  # noqa: E402

ANALYSIS_ID = "A6_ladder_reference"
INF_CAP = 1e300


def default_config() -> dict:
    return {"cohort": "cgmacros", "min_meals": 10, "limit": None, "bounds_scale": 1.0,
            "reference_objective": "iauc"}


def units(config: dict) -> list[str]:
    return [s.subject_id for s in load_cohort(config["cohort"], min_meals=config["min_meals"],
                                              limit=config["limit"])]


def a4_results(objective: str, bounds_scale: float) -> dict[str, dict]:
    """The most complete A4 result set for one objective and box, keyed by subject."""
    from evaluation.results_io import load_all
    best: dict[str, dict] = {}
    for documents in load_all("A4_fisher").values():
        rows = {k: d["payload"] for k, d in documents.items() if k != "_unreadable"}
        rows = {k: v for k, v in rows.items()
                if v.get("objective") == objective
                and abs(float(v.get("bounds_scale", 1.0)) - bounds_scale) < 1e-12}
        if len(rows) > len(best):
            best = rows
    return best


def run_unit(unit: str, config: dict) -> dict:
    """Each objective's timing-block information at one common reference point."""
    scale = config["bounds_scale"]
    reference_run = a4_results(config["reference_objective"], scale).get(unit)
    if reference_run is None:
        raise RuntimeError(f"no A4 {config['reference_objective']} result for {unit} at box {scale}"
                           f"; run A4 first")
    names = list(ObjectiveSpec(name=config["reference_objective"]).theta_names)
    theta_ref = np.array([reference_run["theta_pilot"][n] for n in names], dtype=float)

    subjects = load_cohort(config["cohort"], min_meals=config["min_meals"], limit=config["limit"])
    subject = next(s for s in subjects if s.subject_id == unit)
    records = list(subject.records)
    base = base_params(subject.profile)
    arrays = subject_arrays(records)

    out = {}
    for objective_name in LADDER:
        own = a4_results(objective_name, scale).get(unit)
        if own is None:
            out[objective_name] = {"missing": "no A4 result for this objective"}
            continue
        spec = ObjectiveSpec(name=objective_name, lam=0.0, normalization="nll", bounds_scale=scale)
        objective = build_objective(spec, base, arrays, observed_values(records, spec.window),
                                    noise=nm.NoiseModel(**own["noise"]))
        jacobian = design_matrix(objective, theta_ref, log_coords=True)
        spectrum = fisher_from_design(jacobian)
        fisher = np.asarray(spectrum["fisher"])
        block = timing_block(fisher, names)
        out[objective_name] = {
            "eigenvalues": spectrum["eigenvalues"], "weak_direction": spectrum["weak_direction"],
            "timing_block": block, "condition_number_full": spectrum["condition_number"],
            "theta_own_ml": own["theta_ml"],
        }
    return {"subject_id": unit, "bounds_scale": scale, "theta_reference": dict(zip(names, theta_ref)),
            "reference_objective": config["reference_objective"], "objectives": out, "macros": {}}


# --------------------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------------------

def trade_off_cosine(weak_direction, theta_ke: float, theta_ka: float) -> float:
    """|cos| between a weak timing eigenvector and (1/k_a, -1/k_e), in log coordinates."""
    weak = np.asarray(weak_direction, dtype=float)
    target = np.array([1.0 / theta_ka, -1.0 / theta_ke])
    return float(abs(weak @ target) / (np.linalg.norm(weak) * np.linalg.norm(target)))


def _finite_log10(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v = np.where(np.isfinite(v), v, INF_CAP)
    return np.log10(np.maximum(v, 1e-300))


def _stats(table: dict[str, dict[str, float]]) -> dict:
    """Medians, paired Wilcoxon per step with Holm, from `{objective: {subject: condition number}}`."""
    from evaluation.stats_utils import holm, wilcoxon
    subjects = sorted(set.intersection(*(set(table[o]) for o in LADDER if table.get(o))))
    if len(subjects) < 3:
        return {"n": len(subjects), "note": "too few subjects with all three objectives"}
    series = {o: _finite_log10([table[o][s] for s in subjects]) for o in LADDER}
    out = {"n": len(subjects), "median_condition_number": {
        o: float(10 ** np.median(series[o])) for o in LADDER}}
    steps = (("iauc", "iauc_centroid"), ("iauc_centroid", "trace"))
    tests = []
    for first, second in steps:
        w = wilcoxon(series[first], series[second])
        w["first"], w["second"] = first, second
        w["median_log10_difference"] = float(np.median(series[first] - series[second]))
        w["median_decreases"] = bool(np.median(series[second]) < np.median(series[first]))
        tests.append(w)
    adjusted = holm([t["p"] for t in tests])
    for t, p, r in zip(tests, adjusted["p_adjusted"], adjusted["reject"]):
        t["p_holm"], t["reject_holm"] = p, bool(r)
    out["steps"] = tests
    out["h3_met"] = bool(all(t["median_decreases"] and t["reject_holm"] for t in tests))
    return out


def summarize(bounds_scale: float = 1.0) -> dict:
    """H3 and H4 from stored results, at each objective's own estimate and at the common reference."""
    own, ref, cosines = {}, {}, {}
    for objective in LADDER:
        rows = a4_results(objective, bounds_scale)
        own[objective] = {s: r["timing_block"]["condition_number"] for s, r in rows.items()}
        cosines[objective] = {
            s: trade_off_cosine(r["timing_block"]["weak_direction"],
                                r["theta_ml"]["gastric_emptying"], r["theta_ml"]["carb_absorption"])
            for s, r in rows.items()}

    from evaluation.results_io import largest_matching
    reference_rows = {p["subject_id"]: p for p in largest_matching(
        ANALYSIS_ID, lambda p: abs(p["bounds_scale"] - bounds_scale) < 1e-12).values()}
    for objective in LADDER:
        ref[objective] = {s: r["objectives"][objective]["timing_block"]["condition_number"]
                          for s, r in reference_rows.items()
                          if "timing_block" in r["objectives"].get(objective, {})}

    def cosine_summary(values: dict) -> dict:
        v = np.array(list(values.values()), dtype=float)
        if v.size == 0:
            return {"n": 0}
        return {"n": int(v.size), "median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
                "q3": float(np.percentile(v, 75)), "fraction_at_least_0.8": float(np.mean(v >= 0.8))}

    return {
        "bounds_scale": bounds_scale,
        "h3_own_theta_hat": _stats(own) if all(own.values()) else {"note": "A4 incomplete"},
        "h3_common_reference": _stats(ref) if all(ref.values()) else {"note": "reference incomplete"},
        "h4_cosine_timing_block": {o: cosine_summary(cosines[o]) for o in LADDER},
        "h4_pre_registered_objective": "iauc_centroid",
    }
