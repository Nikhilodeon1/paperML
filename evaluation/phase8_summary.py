"""Collects every Phase 8 number (Amendment 4) into results/phase8_summary.json, from stored results.

Nothing is fitted here. Each section is computed by the module named in its key, and each reports a
number whatever it is. Sections whose inputs have not been produced return `{"n": 0}` and the verdict
functions in `evaluation/hypotheses.py` then report `not evaluated`.

Run:  python -m evaluation.phase8_summary
"""
from __future__ import annotations

import json

import numpy as np

from evaluation import ladder, prediction_stats as ps, profile_lik
from evaluation.results_io import RESULTS, largest_matching
from evaluation.stats_utils import bootstrap_ci

REPLICA_MAIN = {"seed": 0, "cgm": True, "carb_cv": 0.25}
REPLICA_SETTINGS = {
    "cgm_off_carb_0": {"seed": 0, "cgm": False, "carb_cv": 0.0},
    "cgm_off_carb_025": {"seed": 0, "cgm": False, "carb_cv": 0.25},
    "cgm_on_carb_0": {"seed": 0, "cgm": True, "carb_cv": 0.0},
    "cgm_on_carb_025": REPLICA_MAIN,
    "cgm_on_carb_05": {"seed": 0, "cgm": True, "carb_cv": 0.5},
}
H13_SI_MIN, H13_TIMING_MAX = 0.50, 0.25
H14_LOW, H14_HIGH = 0.70, 0.90
H15_TIMING_MAX, H15_COSINE, H15_COSINE_FRACTION = 0.25, 0.8, 0.80
H11_TAU_MIN, H11_P_CENTROID_MAX, H11_P_TRACE_MIN = 0.70, 0.30, 0.50
EQUIVALENCE_MARGIN = 150.0


def _profile(objective="iauc", box=1.0, **extra) -> dict:
    return profile_lik.summarize({**profile_lik.default_config(), "objective": objective,
                                  "bounds_scale": box, **extra})


# --- H13 and H14: the well-specified replica -------------------------------------------------------

def h13() -> dict:
    block = _profile(replica=REPLICA_MAIN)
    if not block.get("n_subjects"):
        return {"n": 0}
    p = block["parameters"]
    si = block.get("S_I_bounded_among_interior_for_S_I")
    timing = max(p["gastric_emptying"]["fraction"], p["carb_absorption"]["fraction"])
    if si is None:
        reading = "no subject interior for S_I"
    elif si["fraction"] >= H13_SI_MIN:
        reading = ("the real-data shortfall is attributed to misspecification or unmodelled "
                   "variability: with the model true, S_I is bounded at least as often as required")
    elif si["fraction"] <= 0.25:
        reading = "the S_I weakness is inherent to the observable at this noise level"
    else:
        reading = "mixed"
    return {"n": block["n_subjects"], "S_I_interior_bounded": si, "timing_max_fraction": timing,
            "timing_fractions": {k: p[k]["fraction"] for k in ("gastric_emptying", "carb_absorption")},
            "S_I_all_bounded": p["insulin_sensitivity"], "interior_for_S_I": block["interior_for_S_I"],
            "truth_coverage": {k: v.get("truth_coverage_of_bounded") for k, v in p.items()},
            "S_I_clause_met": bool(si and si["fraction"] >= H13_SI_MIN),
            "timing_clause_met": bool(timing <= H13_TIMING_MAX), "reading": reading}


def _subject_means(units: list[dict], cell: str, repeats=None) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for u in units:
        if repeats is not None and u["repeat"] not in repeats:
            continue
        body = u["cells"].get(cell)
        if body and body.get("iauc_mae") is not None:
            per.setdefault(u["subject_id"], []).append(body["iauc_mae"])
    return {s: float(np.mean(v)) for s, v in per.items()}


def h14() -> dict:
    real_units = ps.load_units()
    real = _subject_means(real_units, "grad3", repeats={0, 1, 2})
    out = {"settings": {}}
    for name, setting in REPLICA_SETTINGS.items():
        units = ps.load_units(match=lambda p, s=setting: p.get("replica") == s and "grad3" in p["cells"])
        if not units:
            out["settings"][name] = {"n": 0}
            continue
        rep = {c: _subject_means(units, c) for c in ("grad3", "grid3", "personal_mean")}
        common = sorted(set(rep["grad3"]) & set(real))
        a = np.array([rep["grad3"][s] for s in common])
        b = np.array([real[s] for s in common])
        ci = bootstrap_ci(np.stack([a, b], axis=1), statistic=lambda x: float(np.mean(x[:, 0]) / np.mean(x[:, 1])))
        out["settings"][name] = {
            "n": len(common), "replica_grad3_mae": float(a.mean()), "real_grad3_mae": float(b.mean()),
            "ratio": float(a.mean() / b.mean()), "ratio_ci95": [ci.low, ci.high],
            "replica_grid3_mae": float(np.mean([rep["grid3"][s] for s in common if s in rep["grid3"]])),
            "replica_personal_mean_mae": float(np.mean(
                [rep["personal_mean"][s] for s in common if s in rep["personal_mean"]]))}
    main = out["settings"].get("cgm_on_carb_025", {})
    if main.get("n"):
        r = main["ratio"]
        out["main_ratio"] = r
        out["reading"] = ("a substantial part of the real held-out error is not explained by noise or "
                          "optimization" if r <= H14_LOW else
                          "noise accounts for almost all of the real held-out error" if r >= H14_HIGH
                          else "mixed")
    return out


# --- H11 and H12 ------------------------------------------------------------------------------------

def h11() -> dict:
    out = {}
    for objective in ("iauc", "iauc_centroid", "trace"):
        block = _profile(objective, 1.0, parameterization="coords")
        if not block.get("n_subjects"):
            out[objective] = {"n": 0}
            continue
        p, inner = block["parameters"], block.get("bounded_among_interior", {})
        out[objective] = {
            "n": block["n_subjects"],
            "tau1": {"all": p["log_tau1"], "interior": inner.get("log_tau1")},
            "p": {"all": p["log_p"], "interior": inner.get("log_p")},
            "S_I": {"all": p["log_insulin_sensitivity"], "interior": inner.get("log_insulin_sensitivity")}}

    def frac(objective, name):
        d = out.get(objective, {}).get(name)
        if not d or not d["interior"]:
            return None
        return d["interior"]["fraction"]

    tau_c, tau_t = frac("iauc_centroid", "tau1"), frac("trace", "tau1")
    p_c, p_t = frac("iauc_centroid", "p"), frac("trace", "p")
    if None in (tau_c, tau_t, p_c, p_t):
        out["status"] = "not evaluated"
    else:
        met = (tau_c >= H11_TAU_MIN and tau_t >= H11_TAU_MIN and p_c <= H11_P_CENTROID_MAX
               and p_t >= H11_P_TRACE_MIN)
        out["status"] = "met" if met else "not met"
        out["clauses"] = {"tau1_centroid": tau_c, "tau1_trace": tau_t, "p_centroid": p_c, "p_trace": p_t}
    return out


def h12() -> dict:
    h12_units = ps.load_units(match=lambda p: p.get("replica") is None and "grad2_tied" in p["cells"]
                              and p.get("cohort", "cgmacros") == "cgmacros")
    primary = {(u["subject_id"], u["repeat"]): u for u in ps.load_units()}
    merged = []
    for u in h12_units:
        base = primary.get((u["subject_id"], u["repeat"]))
        if base is None:
            continue
        cells = {**{k: v for k, v in base["cells"].items() if k in ("grad3", "grad3_trace", "personal_mean")},
                 **u["cells"]}
        merged.append({"subject_id": u["subject_id"], "repeat": u["repeat"], "cells": cells})
    if not merged:
        return {"n": 0, "status": "not evaluated"}
    out = {"n_units": len(merged), "comparisons": {}}
    for metric in ("iauc", "peak", "trace"):
        scores = ps.subject_scores(merged, metric)
        for first, second in (("grad2_tied", "grad3"), ("grad3_coords", "grad3"),
                              ("grad2_tied_trace", "grad3_trace"), ("grad3_coords_trace", "grad3_trace"),
                              ("grad2_tied", "personal_mean"), ("grad3_coords", "personal_mean"),
                              ("grad2_tied_trace", "personal_mean"), ("grad3_coords_trace", "personal_mean")):
            if first in scores["mean"] and second in scores["mean"]:
                c = ps.compare(scores, first, second)
                out["comparisons"][f"{metric}:{first}-{second}"] = {
                    k: c.get(k) for k in ("n", "mean_difference", "ci95", "wins_first", "tost")}
    boundary = {}
    for cell in ("grad3_coords", "grad3_coords_trace"):
        flags = [f.get("on_tied_boundary") for u in merged for f in u["cells"].get(cell, {}).get("folds", [])
                 if f.get("on_tied_boundary") is not None]
        if flags:
            boundary[cell] = {"folds": len(flags), "fraction_on_boundary": float(np.mean(flags))}
    out["boundary"] = boundary
    equivalence = out["comparisons"].get("iauc:grad2_tied-grad3")
    out["prediction_equivalent_at_150"] = bool(
        equivalence and equivalence["tost"][str(int(EQUIVALENCE_MARGIN))]["equivalent"])
    tied = _profile("iauc_centroid", 1.0, parameterization="tied")
    out["tau1_tied_iauc_centroid"] = (tied["parameters"].get("log_tau1") if tied.get("n_subjects") else None)
    inner = tied.get("bounded_among_interior", {}).get("log_tau1") if tied.get("n_subjects") else None
    out["tau1_tied_interior"] = inner
    if inner is None:
        out["status"] = "not evaluated"
    else:
        met = out["prediction_equivalent_at_150"] and inner["fraction"] >= H11_TAU_MIN
        out["status"] = "met" if met else "not met"
    return out


# --- H15: replication ---------------------------------------------------------------------------------

def _cohort_fisher(cohort: str, box: float) -> dict:
    rows = ladder.a4_results("iauc", box, cohort=cohort)
    if not rows:
        return {"n": 0}
    cos = [ladder.trade_off_cosine(r["timing_block"]["weak_direction"], r["theta_ml"]["gastric_emptying"],
                                   r["theta_ml"]["carb_absorption"]) for r in rows.values()]
    cos = np.array(cos)
    pinned = {p: sum(1 for r in rows.values() if r["fit"]["at_bound"].get(p) is not None)
              for p in ("insulin_sensitivity", "gastric_emptying", "carb_absorption")}
    return {"n": len(rows), "pinned": pinned,
            "pinned_upper": {p: sum(1 for r in rows.values() if r["fit"]["at_bound"].get(p) == "upper")
                             for p in pinned},
            "cosine_median": float(np.median(cos)), "cosine_fraction_at_least_0.8": float(np.mean(cos >= H15_COSINE)),
            "condition_number_median": float(np.median([r["timing_block"]["condition_number"]
                                                         for r in rows.values()])),
            "eigenvalue_medians": np.median([r["fisher"]["eigenvalues"] for r in rows.values()], axis=0).tolist()}


def h15() -> dict:
    out = {}
    for cohort in ("shanghai", "hall"):
        out[cohort] = {}
        for box in (1.0, 2.0):
            fisher = _cohort_fisher(cohort, box)
            profile = _profile("iauc", box, cohort=cohort)
            entry = {"fisher": fisher, "profile": profile if profile.get("n_subjects") else {"n": 0}}
            if fisher.get("n") and profile.get("n_subjects"):
                p = profile["parameters"]
                timing = max(p["gastric_emptying"]["fraction"], p["carb_absorption"]["fraction"])
                entry["timing_max_bounded_fraction"] = timing
                entry["met"] = bool(timing <= H15_TIMING_MAX
                                    and fisher["cosine_fraction_at_least_0.8"] >= H15_COSINE_FRACTION)
            out[cohort][f"{box:g}x"] = entry
    primary = [out[c].get("1x", {}).get("met") for c in ("shanghai", "hall")]
    out["status"] = ("not evaluated" if any(v is None for v in primary)
                     else "met" if all(primary) else "not met")
    return out


# --- H16 ----------------------------------------------------------------------------------------------

def h16() -> dict:
    from evaluation import saturation
    out = {"saturation": saturation.summarize()}
    base = ladder.a4_results("iauc", 1.0)
    scale_out = {}
    for scale in (0.75, 1.33):
        rows = ladder.a4_results("iauc", 1.0, carb_scale=scale)
        if not rows:
            scale_out[str(scale)] = {"n": 0}
            continue
        common = sorted(set(rows) & set(base))
        ratio = np.array([np.log(rows[s]["theta_ml"]["insulin_sensitivity"]
                                 / base[s]["theta_ml"]["insulin_sensitivity"]) for s in common])

        def upper(table, names=("gastric_emptying", "carb_absorption")):
            return float(np.mean([any(table[s]["fit"]["at_bound"].get(n) == "upper" for n in names)
                                  for s in common]))
        scale_out[str(scale)] = {
            "n": len(common), "median_log_ratio_S_I": float(np.median(ratio)),
            "q1": float(np.percentile(ratio, 25)), "q3": float(np.percentile(ratio, 75)),
            "timing_on_upper_bound_scaled": upper(rows), "timing_on_upper_bound_unscaled": upper(base),
            "expected_log_ratio_if_S_I_absorbs_the_scale": float(-np.log(scale))}
    out["carbohydrate_scale"] = scale_out
    out["heteroskedastic_noise"] = {"status": "not run", "reason": "lowest priority; cut for time"}
    return out


def leakage() -> dict:
    from evaluation import leakage_ci
    return leakage_ci.summarize()


def stored() -> dict:
    from evaluation import stored_analyses as sa
    return {"h6": {f"{b:g}x": sa.h6(b) for b in (1.0, 2.0)},
            "correlates": {f"{b:g}x": sa.correlates(b) for b in (1.0, 2.0)}}


def main() -> None:
    summary = {"h13": h13(), "h14": h14(), "h11": h11(), "h12": h12(), "h15": h15(), "h16": h16(),
               "leakage": leakage(), "stored": stored()}
    path = RESULTS / "phase8_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
