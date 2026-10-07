"""Collects every Phase 9 number (Amendment 5) into results/phase9_summary.json, from stored results.

Nothing is fitted here. Each section is computed from result files and reports a number whatever it is; a
section whose inputs have not been produced returns `{"n": 0}` and the verdict functions in
`evaluation/hypotheses.py` then report `not evaluated`. The Phase 8 summary is not touched: Phase 9 adds
sections and re-reads the same rules, it does not re-grade anything.

Run:  python -m evaluation.phase9_summary
"""
from __future__ import annotations

import json

import numpy as np

from evaluation import delta_sensitivity, phase8_summary as p8
from evaluation import prediction_stats as ps, profile_lik, robustness
from evaluation.identifiability_tools import CHI2_DELTA
from evaluation.results_io import RESULTS, largest_matching

REPLICA_SEEDS = (0, 1, 2, 3, 4)
H18_TAU_MIN, H18_P_MIN = 0.70, 0.50
H21_FRACTION_MIN = 0.90
H22_GUT_MAX = 0.25
EQUIVALENCE_MARGIN = 150.0


def _replica(seed: int) -> dict:
    return {"seed": seed, "cgm": True, "carb_cv": 0.25}


# --- H17: stability of the replica result over seeds ------------------------------------------------

def h17() -> dict:
    per_seed, pooled = {}, []
    for seed in REPLICA_SEEDS:
        config = {**profile_lik.default_config(), "objective": "iauc", "bounds_scale": 1.0,
                  "replica": _replica(seed)}
        rows = list(largest_matching(profile_lik.ANALYSIS_ID, lambda p, c=config: profile_lik._matches(p, c)).values())
        if not rows:
            per_seed[str(seed)] = {"n": 0}
            continue
        block = profile_lik.summarize_rows(rows, config)
        si = block.get("S_I_bounded_among_interior_for_S_I")
        per_seed[str(seed)] = {"n": len(rows), "S_I_interior_bounded": si,
                               "timing_max_fraction": max(block["parameters"]["gastric_emptying"]["fraction"],
                                                          block["parameters"]["carb_absorption"]["fraction"])}
        pooled += rows
    if len({s for s, v in per_seed.items() if v.get("n")}) < 2:
        return {"n": 0, "per_seed": per_seed}
    config = {**profile_lik.default_config(), "objective": "iauc", "bounds_scale": 1.0}
    block = profile_lik.summarize_rows(pooled, config)
    si = block.get("S_I_bounded_among_interior_for_S_I")
    timing = max(block["parameters"]["gastric_emptying"]["fraction"],
                 block["parameters"]["carb_absorption"]["fraction"])
    fractions = [v["S_I_interior_bounded"]["fraction"] for v in per_seed.values()
                 if v.get("n") and v.get("S_I_interior_bounded")]
    coverage = block["parameters"]["insulin_sensitivity"].get("truth_coverage_of_bounded")
    out = {"n": len(pooled), "n_seeds": sum(1 for v in per_seed.values() if v.get("n")),
           "per_seed": per_seed, "pooled_S_I_interior_bounded": si, "pooled_timing_max_fraction": timing,
           "seed_range_S_I_fraction": [float(min(fractions)), float(max(fractions))] if fractions else None,
           "seeds_meeting_S_I_clause": int(sum(f >= p8.H13_SI_MIN for f in fractions)),
           "S_I_truth_coverage": coverage,
           "S_I_clause_met": bool(si and si["fraction"] >= p8.H13_SI_MIN),
           "timing_clause_met": bool(timing <= p8.H13_TIMING_MAX)}
    if si is None:
        out["reading"] = "no subject interior for S_I"
    elif si["fraction"] >= p8.H13_SI_MIN:
        out["reading"] = "S_I is bounded at least as often as required when the model is true"
    elif si["fraction"] <= 0.25:
        out["reading"] = "the S_I weakness is inherent to the observable at this noise level"
    else:
        out["reading"] = "mixed"
    out["calibrated"] = bool(coverage and coverage["fraction"] >= 0.90)
    out["status"] = "met" if (out["S_I_clause_met"] and out["timing_clause_met"]) else "not met"
    return out


# --- H18 and H21: the coordinate profiles with polished estimates ------------------------------------

def h18() -> dict:
    out = {}
    for objective in ("iauc", "iauc_centroid", "trace"):
        block = p8._profile(objective, 1.0, parameterization="coords", polish=True, replica=_replica(0))
        if not block.get("n_subjects"):
            out[objective] = {"n": 0}
            continue
        inner = block.get("bounded_among_interior", {})
        out[objective] = {"n": block["n_subjects"], "tau1": inner.get("log_tau1"), "p": inner.get("log_p"),
                          "S_I": inner.get("log_insulin_sensitivity"),
                          "truth_coverage": {k: v.get("truth_coverage_of_bounded")
                                             for k, v in block["parameters"].items()}}
    t = out.get("trace", {})
    if not t.get("tau1") or not t.get("p"):
        out["status"] = "not evaluated"
    else:
        out["clauses"] = {"tau1_trace": t["tau1"]["fraction"], "p_trace": t["p"]["fraction"]}
        out["status"] = "met" if (t["tau1"]["fraction"] >= H18_TAU_MIN
                                  and t["p"]["fraction"] >= H18_P_MIN) else "not met"
    return out


def h21() -> dict:
    out = {"objectives": {}}
    for objective in ("iauc", "iauc_centroid", "trace"):
        config = {**profile_lik.default_config(), "objective": objective, "bounds_scale": 1.0,
                  "parameterization": "coords", "polish": True}
        rows = list(largest_matching(profile_lik.ANALYSIS_ID, lambda p, c=config: profile_lik._matches(p, c)).values())
        if not rows:
            out["objectives"][objective] = {"n": 0}
            continue
        gaps = np.array([r["polish"]["gap"] for r in rows], dtype=float)
        out["objectives"][objective] = {
            "n": len(rows), "median_gap": float(np.median(gaps)), "max_gap": float(gaps.max()),
            "fraction_below_threshold": float(np.mean(gaps < CHI2_DELTA)),
            "fraction_below_0.1": float(np.mean(gaps < 0.1)),
            "subjects_with_better_fit_than_polished_estimate": int(sum(
                any(e["better_than_theta_hat"] for e in r["profiles"].values()) for r in rows)),
            "worst_subjects": [rows[i]["subject_id"] for i in np.argsort(-gaps)[:3]]}
    graded = [out["objectives"].get(o, {}) for o in ("iauc_centroid", "trace")]
    if not all(g.get("n") for g in graded):
        out["status"] = "not evaluated"
    else:
        out["status"] = ("met" if all(g["fraction_below_threshold"] >= H21_FRACTION_MIN for g in graded)
                         else "not met")
    out["h11_on_polished_estimates"] = p8.h11(polish=True)
    return out


# --- H20: prediction on Shanghai ----------------------------------------------------------------------

def h20() -> dict:
    units = ps.load_units(match=lambda p: p.get("cohort") == "shanghai" and p.get("replica") is None
                          and p.get("carb_scale") in (None, 1, 1.0) and "grad3" in p["cells"])
    if not units:
        return {"n": 0, "status": "not evaluated"}
    out = {"n_units": len(units), "n_subjects": len({u["subject_id"] for u in units}),
           "repeats": sorted({u["repeat"] for u in units}), "metrics": {}}
    pairs = (("grad3", "grid3"), ("grad1", "grid1"), ("grad3", "grad1"), ("grad3", "personal_mean"),
             ("grid3", "personal_mean"), ("grad1", "personal_mean"), ("grad3", "population"),
             ("grad3", "persistence"))
    for metric in ("iauc", "peak", "trace"):
        scores = ps.subject_scores(units, metric)
        means = {c: float(np.mean(list(v.values()))) for c, v in scores["mean"].items()}
        block = {"cell_means": means, "comparisons": {}}
        for first, second in pairs:
            if first in scores["mean"] and second in scores["mean"]:
                c = ps.compare(scores, first, second)
                block["comparisons"][f"{first}-{second}"] = {
                    k: c.get(k) for k in ("n", "mean_first", "mean_second", "mean_difference", "ci95",
                                          "wins_first", "win_fraction", "tost")}
        out["metrics"][metric] = block
    iauc = out["metrics"]["iauc"]["comparisons"]
    keys = ("grad3-grid3", "grad3-grad1")
    if not all(k in iauc for k in keys):
        out["status"] = "not evaluated"
        return out
    out["equivalent_at_150"] = {k: bool(iauc[k]["tost"][str(int(EQUIVALENCE_MARGIN))]["equivalent"]) for k in keys}
    out["status"] = "met" if all(out["equivalent_at_150"].values()) else "not met"
    personal = out["metrics"]["iauc"]["cell_means"].get("personal_mean")
    if personal:
        out["margin_over_personal_mean_mae"] = EQUIVALENCE_MARGIN / personal
    return out


# --- H22: second model class ---------------------------------------------------------------------------

def h22() -> dict:
    from evaluation import dalla_man_identifiability as dm
    block = dm.summarize()
    if not block.get("n_subjects"):
        return {"n": 0, "status": "not evaluated"}
    fraction = block["gut_max_bounded_fraction"]
    block["status"] = "met" if (fraction is not None and fraction <= H22_GUT_MAX) else "not met"
    return block


# --- report-only items -----------------------------------------------------------------------------------

def hall_trace() -> dict:
    out = {}
    for box in (1.0,):
        block = p8._profile("trace", box, cohort="hall")
        out[f"{box:g}x"] = block if block.get("n_subjects") else {"n": 0}
    return out


def shanghai_complete() -> dict:
    return {f"{b:g}x": (p8._profile("iauc", b, cohort="shanghai").get("n_subjects") or 0) for b in (1.0, 2.0)}


def main() -> None:
    summary = {"h10": robustness.h10(), "h6_synthetic": robustness.h6_synthetic(), "h17": h17(),
               "h18": h18(), "h20": h20(), "h21": h21(), "h22": h22(), "hall_trace": hall_trace(),
               "shanghai_profile_subjects": shanghai_complete(),
               "delta_sensitivity": {f"{box:g}x": delta_sensitivity.summarize(box) for box in (0.5, 1.0, 2.0)},
               "carb_scale": p8.h16()["carbohydrate_scale"]}
    path = RESULTS / "phase9_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
