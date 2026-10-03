"""2.2: how many subjects are on a bound, under which definition, and what that does to the findings.

Pure summary code over stored results (A4 Fisher, A5 profile, A8a gradient diagnostic, A9 predictions).
It computes nothing expensive.

Definitions, from `PREREG_AMENDMENT_2.md`:

* **interior (original, global)** -- converged and no parameter within 1% of its interval of a bound.
  Reported everywhere and recorded as unevaluable for the H5 S_I clause when its set is empty.
* **interior for parameter j** -- the maximum-likelihood estimate of `j` is more than 1% of its interval
  from both bounds, whatever the other parameters do. The H5 S_I clause, the H10 "S_I ranked first"
  clause and the projected-gradient check use interior-for-S_I.

The three boxes (0.5x, 1x, 2x, log-scaled about the geometric centre) are reported side by side. The 1x
box is the pre-registered primary one; none of the three is declared the winner.
"""
from __future__ import annotations

import numpy as np

from evaluation.ladder import a4_results
from personalization.subject_loss import TARGETS

BOXES = (0.5, 1.0, 2.0)
NEAR = 0.01


def box_summary(bounds_scale: float, objective: str = "iauc") -> dict:
    rows = a4_results(objective, bounds_scale)
    if not rows:
        return {"bounds_scale": bounds_scale, "n_subjects": 0}
    n = len(rows)

    def pinned(parameter, side=None):
        count = 0
        for r in rows.values():
            flag = r["fit"]["at_bound"].get(parameter)
            if flag is not None and (side is None or flag == side):
                count += 1
        return count

    distance = {p: [r["fit"]["relative_distance_to_nearest_bound"][p] for r in rows.values()]
                for p in TARGETS}
    interior_for = {p: int(sum(d > NEAR for d in distance[p])) for p in TARGETS}
    condition = np.array([r["timing_block"]["condition_number"] for r in rows.values()], dtype=float)
    condition = condition[np.isfinite(condition)]
    return {
        "bounds_scale": bounds_scale, "objective": objective, "n_subjects": n,
        "pinned": {p: {"n": pinned(p), "fraction": pinned(p) / n,
                       "lower": pinned(p, "lower"), "upper": pinned(p, "upper")} for p in TARGETS},
        "interior_for_parameter": {p: {"n": interior_for[p], "fraction": interior_for[p] / n}
                                   for p in TARGETS},
        "interior_converged_original": int(sum(r["fit"]["interior"] for r in rows.values())),
        "any_parameter_on_bound": int(sum(r["fit"]["n_at_bound"] > 0 for r in rows.values())),
        "converged_by_gradient_criterion": int(sum(r["fit"]["converged"] for r in rows.values())),
        "still_moving": int(sum(r["fit"]["still_moving"] for r in rows.values())),
        "timing_block_condition_number": {
            "median": float(np.median(condition)) if condition.size else float("nan"),
            "q1": float(np.percentile(condition, 25)) if condition.size else float("nan"),
            "q3": float(np.percentile(condition, 75)) if condition.size else float("nan")},
        "median_S_I": float(np.median([r["theta_ml"]["insulin_sensitivity"]
                                       for r in rows.values()])),
        "stop_rule_S_I_over_20_percent": bool(pinned("insulin_sensitivity") / n > 0.20),
    }


def sweep(objective: str = "iauc") -> dict:
    """The three boxes side by side: the sweep is reported as a result, not used to pick a box."""
    return {f"{scale:g}x": box_summary(scale, objective) for scale in BOXES}


def projected_gradient_check(bounds_scale: float = 1.0) -> dict:
    """Does the S_I-versus-timing separation of the gradient diagnostic survive on interior subjects?

    Compared on the same fits: the raw mean absolute gradient and the projected one, over all subjects
    and over the subjects interior for S_I at their maximum-likelihood estimate (A4, same box).
    """
    from evaluation.results_io import load_all

    diag = {}
    for documents in load_all("A8a_gradient_diag").values():
        for key, doc in documents.items():
            if key == "_unreadable":
                continue
            p = doc["payload"]
            if abs(p["bounds_scale"] - bounds_scale) < 1e-12:
                diag[p["subject_id"]] = p
    if not diag:
        return {"n": 0}
    a4 = a4_results("iauc", bounds_scale)

    def separation(subjects, kind):
        rows = [diag[s][kind] for s in subjects]
        if not rows:
            return {"n": 0}
        si = np.array([r["insulin_sensitivity"] for r in rows])
        ke = np.array([r["gastric_emptying"] for r in rows])
        ka = np.array([r["carb_absorption"] for r in rows])
        timing = np.maximum(ke, ka)
        ratio = timing / np.maximum(si, 1e-300)
        return {"n": len(rows), "S_I_largest_fraction": float(np.mean(si > timing)),
                "median_timing_over_S_I": float(np.median(ratio)),
                "median_S_I": float(np.median(si)), "median_k_e": float(np.median(ke)),
                "median_k_a": float(np.median(ka))}

    everyone = sorted(diag)
    interior = [s for s in everyone if s in a4
                and a4[s]["fit"]["relative_distance_to_nearest_bound"]["insulin_sensitivity"] > NEAR]
    return {"bounds_scale": bounds_scale,
            "all": {k: separation(everyone, k) for k in ("raw_norm", "projected_norm")},
            "interior_for_S_I": {k: separation(interior, k)
                                 for k in ("raw_norm", "projected_norm")},
            "n_interior_for_S_I": len(interior)}


def pinned_bias(bounds_scale: float = 1.0, cell: str = "grad3") -> dict:
    """Do subjects pinned at an upper bound look systematically different?

    A timing parameter at its upper bound is consistent with a flat likelihood in the instant-absorption
    limit. It is also what would happen if the timing parameters were soaking up an area deficit from
    under-reported carbohydrate or from the 180 min window cutting off a long tail. The two readings
    predict different things about the data: an area deficit shows up as a systematic signed prediction
    error and as unusual carbohydrate totals in the pinned group, a flat likelihood does not.

    Signed error is predicted minus observed iAUC from the held-out A9 predictions of `cell`.
    """
    from evaluation.prediction_stats import load_units
    from evaluation.stats_utils import bootstrap_ci

    a4 = a4_results("iauc", bounds_scale)
    units = load_units()
    if not a4 or not units:
        return {"n": 0, "note": "need A4 and A9 results"}

    per_subject: dict[str, dict] = {}
    for unit in units:
        body = unit["cells"].get(cell)
        if body is None:
            continue
        pred = np.asarray(body["pred_iauc"], dtype=float)
        obs = np.asarray(unit["obs_iauc"], dtype=float)
        carbs = np.asarray(unit["carbs_g"], dtype=float)
        rec = per_subject.setdefault(unit["subject_id"], {"signed": [], "carbs": [], "obs": []})
        rec["signed"].append(float(np.mean(pred - obs)))
        rec["carbs"].append(float(np.mean(carbs)))
        rec["obs"].append(float(np.mean(obs)))

    def group(parameter, side):
        members = [s for s in per_subject if s in a4
                   and a4[s]["fit"]["at_bound"].get(parameter) == side]
        rest = [s for s in per_subject if s in a4 and s not in members]
        return members, rest

    out = {"bounds_scale": bounds_scale, "cell": cell, "groups": {}}
    from scipy import stats
    for parameter in ("gastric_emptying", "carb_absorption"):
        members, rest = group(parameter, "upper")
        row = {"n_pinned_upper": len(members), "n_other": len(rest)}
        if len(members) >= 3 and len(rest) >= 3:
            for key in ("signed", "carbs", "obs"):
                a = np.array([np.mean(per_subject[s][key]) for s in members])
                b = np.array([np.mean(per_subject[s][key]) for s in rest])
                test = stats.mannwhitneyu(a, b, alternative="two-sided")
                row[key] = {"mean_pinned": float(a.mean()), "mean_other": float(b.mean()),
                            "p_mann_whitney": float(test.pvalue),
                            "signed_mean_ci95_pinned": (bootstrap_ci(a).as_dict()
                                                        if key == "signed" else None)}
        out["groups"][parameter] = row
    return out
