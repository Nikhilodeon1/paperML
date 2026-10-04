"""Turn the stored analyses into the scalars the paper quotes, as macro declarations.

Writes one result file (`P7_paper_numbers`) whose payload carries a `macros` block. `freeze_results`
collects it into `results/index.json` and `make_macros` turns it into `paper/results_macros.tex`, so a
number in the text is mechanically the number an analysis produced. Macro names are alphabetic only
(TeX does not accept digits in a command name), so the boxes are spelled Half, One and Two.

Run:  python -m evaluation.paper_numbers
"""
from __future__ import annotations

from evaluation import hypotheses
from evaluation.results_io import save_result

ANALYSIS_ID = "P7_paper_numbers"
BOX_WORD = {"0.5x": "Half", "1x": "One", "2x": "Two"}
PARAM_WORD = {"insulin_sensitivity": "SI", "gastric_emptying": "Ke", "carb_absorption": "Ka"}
NUMBER_WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven",
               8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen",
               14: "Fourteen", 15: "Fifteen", 16: "Sixteen"}
DIGIT_WORD = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
              "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def cell_word(cell: str) -> str:
    """`grad3_trace` -> `GradThreeTrace`: macro names may contain letters only."""
    out = "".join(part.capitalize() for part in cell.split("_"))
    return "".join(DIGIT_WORD.get(ch, ch) for ch in out)


OBJ_WORD = {"iauc": "Iauc", "iauc_centroid": "Centroid", "trace": "Trace"}


def sci(value: float, digits: int = 2) -> str:
    """A LaTeX scientific-notation string, because a plain 4e+05 does not typeset."""
    if value is None or value != value or value in (float("inf"), float("-inf")):
        return r"\infty" if value == float("inf") else "n/a"
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return "$" + mantissa + r"\times10^{" + str(int(exponent)) + "}$"


def _ci(est, low, high, fmt=".0f"):
    return {"value": est, "low": low, "high": high, "fmt": fmt}


def build(summary: dict, verdicts: dict) -> dict:
    m: dict = {}

    # --- hypotheses ------------------------------------------------------------------------------
    for name, row in verdicts.items():
        m[f"resVerdictH{NUMBER_WORD[int(name[1:])]}"] = {"value": row["status"]}

    # --- bounds sweep ---------------------------------------------------------------------------
    for box, block in summary.get("bounds_sweep_fisher_fit", {}).items():
        word = BOX_WORD[box]
        n = block["n_subjects"]
        for parameter, w in PARAM_WORD.items():
            pinned = block["pinned"][parameter]
            m[f"resPinned{w}{word}"] = {"value": pinned["n"], "fmt": ".0f"}
            m[f"resPinnedFrac{w}{word}"] = {"value": 100 * pinned["fraction"], "fmt": ".0f"}
        m[f"resInteriorForSI{word}"] = {"value": block["interior_for_parameter"]["insulin_sensitivity"]["n"],
                                        "fmt": ".0f"}
        m[f"resInteriorGlobal{word}"] = {"value": block["interior_converged_original"], "fmt": ".0f"}
        m[f"resSubjects{word}"] = {"value": n, "fmt": ".0f"}

    # --- profile likelihood ---------------------------------------------------------------------
    for key, obj in (("profile_likelihood", "Iauc"), ("profile_likelihood_iauc_centroid", "Centroid"),
                     ("profile_likelihood_trace", "Trace")):
        for box, block in summary.get(key, {}).items():
            if not block.get("n_subjects"):
                continue
            for parameter, w in PARAM_WORD.items():
                d = block["parameters"][parameter]
                m[f"resProfile{obj}{w}{BOX_WORD[box]}"] = _ci(100 * d["fraction"], 100 * d["wilson"][0],
                                                              100 * d["wilson"][1])
            si = block.get("S_I_bounded_among_interior_for_S_I")
            if si:
                m[f"resProfile{obj}SIInteriorBounded{BOX_WORD[box]}"] = {"value": si["bounded"], "fmt": ".0f"}
                m[f"resProfile{obj}SIInteriorN{BOX_WORD[box]}"] = {"value": si["n"], "fmt": ".0f"}
                m[f"resProfile{obj}SIInteriorFrac{BOX_WORD[box]}"] = _ci(
                    100 * si["fraction"], 100 * si["wilson"][0], 100 * si["wilson"][1])

    # --- ladder ---------------------------------------------------------------------------------
    for box, block in summary.get("ladder", {}).items():
        for tag, key in (("Own", "h3_own_theta_hat"), ("Ref", "h3_common_reference")):
            h = block.get(key, {})
            for obj, value in h.get("median_condition_number", {}).items():
                m[f"resCond{tag}{OBJ_WORD[obj]}{BOX_WORD[box]}"] = {"value": sci(value)}
        for obj, c in block.get("h4_cosine_timing_block", {}).items():
            if c.get("n"):
                m[f"resCosine{OBJ_WORD[obj]}{BOX_WORD[box]}"] = {"value": c["median"], "fmt": ".3f"}

    # --- prediction -----------------------------------------------------------------------------
    pred = summary.get("prediction", {})
    for cell, c in pred.get("cells", {}).items():
        word = cell_word(cell)
        m[f"resMae{word}"] = {"value": c["mean"], "fmt": ".0f"}
    for c in pred.get("comparisons", []):
        if "ci95" not in c:
            continue
        name = cell_word(c["first"]) + "Vs" + cell_word(c["second"])
        m[f"resDiff{name}"] = _ci(c["mean_difference"], c["ci95"]["low"], c["ci95"]["high"])
        m[f"resWins{name}"] = {"value": c["wins_first"], "fmt": ".0f"}
        if "p_holm" in c:
            m[f"resPHolm{name}"] = {"value": c["p_holm"], "fmt": ".3f"}
    gap = summary.get("inference_gap", {}).get("pairs", {}).get("grad3_vs_grid3")
    if gap and gap.get("n_folds"):
        m["resGapWithinTolerance"] = {"value": 100 * gap["fraction_within_tolerance"], "fmt": ".1f"}
        m["resGapGradientBetter"] = {"value": 100 * gap["fraction_gradient_better"], "fmt": ".0f"}
    # --- generic rank, optimizer check, gut sweep -----------------------------------------------
    for obs, d in summary.get("generic_rank", {}).items():
        word = {"iauc": "Iauc", "iauc_centroid": "Centroid", "iauc_peak": "Peak", "trace": "Trace"}[obs]
        m[f"resRankRatio{word}"] = {"value": sci(d["smallest_over_largest_median"])}
    for box, d in summary.get("optimizer_check", {}).items():
        if d.get("n"):
            m[f"resOptGapMax{BOX_WORD[box]}"] = {"value": d["max_gap"], "fmt": ".2f"}
    for row in summary.get("gut_sweep", {}).get("sweep", {}).get("rows", []):
        word = {0.0: "Zero", 0.5: "Half", 1.0: "One", 2.0: "Two", 5.0: "Five", 10.0: "Ten"}[
            row["gut_content_percent_of_meal"]]
        m[f"resGutSweepMax{word}"] = {"value": row["max_abs_glucose_difference_mg_dl"], "fmt": ".2f"}
    h2 = summary.get("moment_checks", {}).get("h2")
    if h2:
        for key, name in (("linear_360_ke_median", "resTwoLinKe"), ("linear_360_ka_median", "resTwoLinKa"),
                          ("nonlinear_180_ke_median", "resTwoNlKe"), ("nonlinear_180_ka_median", "resTwoNlKa")):
            if h2.get(key) is not None:
                m[name] = {"value": h2[key], "fmt": ".3f"}
    m.update(_phase8_macros(summary.get("phase8", {})))
    return m


def _phase8_macros(p8: dict) -> dict:
    """Macros for the Phase 8 results; absent sections simply contribute nothing."""
    m: dict = {}
    h13 = p8.get("h13", {})
    if h13.get("n"):
        si = h13["S_I_interior_bounded"] or {}
        if si:
            m["resReplicaSIBounded"] = _ci(100 * si["fraction"], 100 * si["wilson"][0], 100 * si["wilson"][1])
            m["resReplicaSIBoundedCount"] = {"value": si["bounded"], "fmt": ".0f"}
            m["resReplicaSIInterior"] = {"value": si["n"], "fmt": ".0f"}
        m["resReplicaTimingMax"] = {"value": 100 * h13["timing_max_fraction"], "fmt": ".0f"}
        for name, key in (("Ke", "gastric_emptying"), ("Ka", "carb_absorption")):
            cov = (h13.get("truth_coverage") or {}).get(key)
            if cov:
                m[f"resReplicaCover{name}"] = {"value": 100 * cov["fraction"], "fmt": ".0f"}
        cov = (h13.get("truth_coverage") or {}).get("insulin_sensitivity")
        if cov:
            m["resReplicaCoverSI"] = {"value": 100 * cov["fraction"], "fmt": ".0f"}
    for name, word in (("cgm_off_carb_0", "CgmOffCarbZero"), ("cgm_off_carb_025", "CgmOffCarbQuarter"),
                       ("cgm_on_carb_0", "CgmOnCarbZero"), ("cgm_on_carb_025", "CgmOnCarbQuarter"),
                       ("cgm_on_carb_05", "CgmOnCarbHalf")):
        d = p8.get("h14", {}).get("settings", {}).get(name, {})
        if d.get("n"):
            m[f"resReplicaMae{word}"] = {"value": d["replica_grad3_mae"], "fmt": ".0f"}
            m[f"resReplicaRatio{word}"] = _ci(d["ratio"], d["ratio_ci95"][0], d["ratio_ci95"][1], ".2f")
    for objective, word in (("iauc", "Iauc"), ("iauc_centroid", "Centroid"), ("trace", "Trace")):
        d = p8.get("h11", {}).get(objective, {})
        if d.get("n"):
            for key, tag in (("tau1", "Tau"), ("p", "P")):
                inner = d[key]["interior"] or d[key]["all"]
                frac = inner["fraction"] if "fraction" in inner else None
                if frac is not None:
                    m[f"resCoordsBounded{tag}{word}"] = {"value": 100 * frac, "fmt": ".0f"}
    h12 = p8.get("h12", {})
    comp = h12.get("comparisons", {}).get("iauc:grad2_tied-grad3")
    if comp and comp.get("ci95"):
        m["resTiedDiff"] = _ci(comp["mean_difference"], comp["ci95"]["low"], comp["ci95"]["high"])
    for cell, word in (("grad3_coords", "Coords"), ("grad3_coords_trace", "CoordsTrace")):
        b = h12.get("boundary", {}).get(cell)
        if b:
            m[f"resBoundary{word}"] = {"value": 100 * b["fraction_on_boundary"], "fmt": ".0f"}
    for cohort, word in (("shanghai", "Shanghai"), ("hall", "Hall")):
        for box, bword in (("1x", "One"), ("2x", "Two")):
            e = p8.get("h15", {}).get(cohort, {}).get(box, {})
            if e.get("fisher", {}).get("n"):
                f = e["fisher"]
                m[f"resRep{word}{bword}N"] = {"value": f["n"], "fmt": ".0f"}
                m[f"resRep{word}{bword}Cosine"] = {"value": 100 * f["cosine_fraction_at_least_0.8"], "fmt": ".0f"}
                m[f"resRep{word}{bword}CosineMedian"] = {"value": f["cosine_median"], "fmt": ".3f"}
            if e.get("profile", {}).get("n_subjects"):
                for pk, pw in (("insulin_sensitivity", "SI"), ("gastric_emptying", "Ke"),
                               ("carb_absorption", "Ka")):
                    d = e["profile"]["parameters"][pk]
                    m[f"resRep{word}{bword}Bounded{pw}"] = _ci(100 * d["fraction"], 100 * d["wilson"][0],
                                                               100 * d["wilson"][1])
    for scale, word in (("0.75", "Low"), ("1.33", "High")):
        d = p8.get("h16", {}).get("carbohydrate_scale", {}).get(scale, {})
        if d.get("n"):
            m[f"resCarbScale{word}LogRatio"] = {"value": d["median_log_ratio_S_I"], "fmt": ".2f"}
            m[f"resCarbScale{word}Upper"] = {"value": 100 * d["timing_on_upper_bound_scaled"], "fmt": ".0f"}
    sat = p8.get("h16", {}).get("saturation", {})
    if sat.get("n"):
        m["resSatTopToBottom"] = {"value": sat["slope_ratio_top_to_bottom_of_primary_box"], "fmt": ".2f"}
        m["resSatDoubleTop"] = {"value": sat["slope_ratio_top_of_double_box_to_top_of_primary"], "fmt": ".2f"}
    leak = p8.get("leakage", {})
    for key, word in (("raw_snpe", "RawSnpe"), ("raw_gradient", "RawGradient"), ("partial_snpe", "PartialSnpe"),
                      ("partial_gradient", "PartialGradient"), ("snpe_vs_own_baseline_feature", "SnpeFeature"),
                      ("raw_difference", "RawDiff"), ("partial_difference", "PartialDiff")):
        d = leak.get("statistics", {}).get(key)
        if d:
            m[f"resLeak{word}"] = _ci(d["estimate"], d["low"], d["high"], ".2f")
    if leak.get("n"):
        m["resLeakN"] = {"value": leak["n"], "fmt": ".0f"}
    h6 = p8.get("stored", {}).get("h6", {}).get("1x")
    if h6:
        m["resSixMedianSpearman"] = {"value": h6["spearman_vs_profile_strength"].get("median"), "fmt": ".2f"}
        m["resSixAuc"] = {"value": h6["auc_detecting_flat"], "fmt": ".2f"}
    return m


def main() -> None:
    summary = hypotheses.load_summary()
    verdicts = hypotheses.evaluate_all(summary)
    macros = build(summary, verdicts)
    payload = {"macros": macros, "hypotheses": verdicts, "n_macros": len(macros)}
    path = save_result(ANALYSIS_ID, payload, {"source": "results/phase2_summary.json"},
                       unit="paper_numbers", overwrite=True)
    print(f"{len(macros)} macros -> {path}")
    for name, row in verdicts.items():
        print(f"  {name:4} {'*' if row['primary'] else ' '} {row['status']}")


if __name__ == "__main__":
    main()
