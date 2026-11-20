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
               14: "Fourteen", 15: "Fifteen", 16: "Sixteen", 17: "Seventeen", 18: "Eighteen",
               19: "Nineteen", 20: "Twenty", 21: "TwentyOne", 22: "TwentyTwo", 23: "TwentyThree", 24: "TwentyFour", 25: "TwentyFive", 26: "TwentySix", 27: "TwentySeven", 28: "TwentyEight", 29: "TwentyNine", 30: "Thirty", 31: "ThirtyOne", 32: "ThirtyTwo", 33: "ThirtyThree", 34: "ThirtyFour", 35: "ThirtyFive", 36: "ThirtySix"}
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
    m.update(_phase9_macros(summary.get("phase9", {})))
    m.update(_phase10_macros(summary.get("phase10", {})))
    m.update(_phase11_macros(summary.get("phase11", {})))
    m.update(_phase13_macros(summary.get("phase13", {})))
    m.update(_phase14_macros(summary.get("phase14", {})))
    m.update(_phase15_macros(summary.get("phase15", {})))
    m.update(_phase16_macros(summary.get("phase16", {})))
    m.update(_phase17_macros(summary.get("phase17", {})))
    m.update(_cohort_macros())
    m.update(_extra_macros(summary, verdicts))
    m.update(_stored_macros(summary.get("phase8", {})))
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
                    interior = e["profile"].get("interior_for_parameter", {}).get(pk)
                    if interior is not None:
                        m[f"resRep{word}{bword}Interior{pw}"] = {"value": interior, "fmt": ".0f"}
    for scale, word in (("0.75", "Low"), ("1.33", "High")):
        d = p8.get("h16", {}).get("carbohydrate_scale", {}).get(scale, {})
        if d.get("n"):
            m[f"resCarbScale{word}LogRatio"] = {"value": d["median_log_ratio_S_I"], "fmt": ".2f"}
            m[f"resCarbScale{word}Upper"] = {"value": 100 * d["timing_on_upper_bound_scaled"], "fmt": ".0f"}
            if d.get("timing_on_upper_bound_unscaled") is not None:
                m["resCarbScaleUnscaledUpper"] = {"value": 100 * d["timing_on_upper_bound_unscaled"], "fmt": ".0f"}
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


def _pct(d, key="fraction"):
    return None if d is None or d.get(key) is None else 100 * d[key]


def _phase11_macros(p11: dict) -> dict:
    """Amendment 7: the noise-scaled replica (H24) and the pooled coordinate calibration."""
    m: dict = {}
    h24 = p11.get("h24", {})
    words = {"1": "One", "0.5": "Half", "0.25": "Quarter"}
    for scale, content in h24.get("scales", {}).items():
        word = words.get(scale)
        trace = content.get("trace", {})
        if word and trace.get("n"):
            for name, tag in (("tau1", "Tau"), ("p", "P"), ("S_I", "SI")):
                block = trace[name]
                if block["bounded_interior"].get("fraction") is not None:
                    m[f"resNoise{word}Trace{tag}"] = {"value": 100 * block["bounded_interior"]["fraction"],
                                                      "fmt": ".0f"}
                if block["nominal_coverage"].get("fraction") is not None:
                    m[f"resNoise{word}Cover{tag}"] = {"value": 100 * block["nominal_coverage"]["fraction"],
                                                      "fmt": ".0f"}
                cut = block.get("cutoff", {})
                if cut.get("n") and name != "S_I":
                    m[f"resNoise{word}Cutoff{tag}"] = {"value": cut["cutoff"], "fmt": ".1f"}
        iauc = content.get("iauc", {})
        if word and iauc.get("n") and iauc.get("S_I_interior"):
            m[f"resNoise{word}IaucSI"] = {"value": 100 * iauc["S_I_interior"]["fraction"], "fmt": ".0f"}
            m[f"resNoise{word}IaucSICount"] = {"value": iauc["S_I_interior"]["bounded"], "fmt": ".0f"}
            m[f"resNoise{word}IaucSIN"] = {"value": iauc["S_I_interior"]["n"], "fmt": ".0f"}
            m[f"resNoise{word}IaucTimingMax"] = {"value": 100 * max(iauc["timing_bounded"].values()),
                                                 "fmt": ".0f"}
            m[f"resNoise{word}IaucCoverSI"] = {"value": 100 * iauc["S_I_nominal_coverage"]["fraction"],
                                               "fmt": ".0f"}
    h25 = p11.get("h25", {})
    for scale, word in (("1", "One"), ("0.25", "Quarter")):
        content = h25.get("scales", {}).get(scale)
        if not content:
            continue
        trace = content.get("trace", {})
        if trace.get("n"):
            for name, tag in (("tau1", "Tau"), ("p", "P"), ("S_I", "SI")):
                block = trace[name]
                if block["bounded_interior"].get("fraction") is not None:
                    m[f"resExact{word}TraceBounded{tag}"] = {"value": 100 * block["bounded_interior"]["fraction"],
                                                             "fmt": ".0f"}
                if block["nominal_coverage"].get("fraction") is not None:
                    m[f"resExact{word}Cover{tag}"] = {"value": 100 * block["nominal_coverage"]["fraction"],
                                                      "fmt": ".0f"}
                cut = block.get("cutoff", {})
                if cut.get("n") and name != "S_I":
                    m[f"resExact{word}Cutoff{tag}"] = {"value": cut["cutoff"], "fmt": ".1f"}
        iauc = content.get("iauc", {})
        if iauc.get("n") and iauc.get("S_I_interior"):
            m[f"resExact{word}IaucSI"] = {"value": 100 * iauc["S_I_interior"]["fraction"], "fmt": ".0f"}
            m[f"resExact{word}IaucSICount"] = {"value": iauc["S_I_interior"]["bounded"], "fmt": ".0f"}
            m[f"resExact{word}IaucSIN"] = {"value": iauc["S_I_interior"]["n"], "fmt": ".0f"}
            m[f"resExact{word}IaucCoverSI"] = {"value": 100 * iauc["S_I_nominal_coverage"]["fraction"],
                                               "fmt": ".0f"}
    pooled = p11.get("pooled_calibration", {})
    if pooled.get("n"):
        m["resPooledN"] = {"value": pooled["n"], "fmt": ".0f"}
        for name, tag in (("tau1", "Tau"), ("p", "P")):
            block = pooled.get(name)
            if not block:
                continue
            cut = block["cutoff"]
            m[f"resPooledCutoff{tag}"] = {"value": cut["cutoff"], "low": cut["ci95"][0], "high": cut["ci95"][1],
                                          "fmt": ".1f"}
            if block["real_calibrated"].get("fraction") is not None:
                m[f"resPooledReal{tag}"] = {"value": 100 * block["real_calibrated"]["fraction"], "fmt": ".0f"}
            if block["replica_nominal_coverage"].get("fraction") is not None:
                m[f"resPooledCover{tag}"] = {"value": 100 * block["replica_nominal_coverage"]["fraction"],
                                             "fmt": ".0f"}
    return m


def _phase10_macros(p10: dict) -> dict:
    """Amendment 6: the recalibrated cutoff (H23) and the heteroskedastic noise sensitivity (H16)."""
    m: dict = {}
    h23 = p10.get("h23", {})
    parts = (("SI", h23.get("S_I_iauc", {})),
             ("Tau", h23.get("trace_coordinates", {}).get("log_tau1", {})),
             ("P", h23.get("trace_coordinates", {}).get("log_p", {})))
    for word, part in parts:
        if not part.get("n"):
            continue
        cut = part["cutoff"]
        m[f"resRecalCutoff{word}"] = {"value": cut["cutoff"], "low": cut["ci95"][0], "high": cut["ci95"][1],
                                      "fmt": ".1f"}
        m[f"resRecalN{word}"] = {"value": cut["n"], "fmt": ".0f"}
        m[f"resRecalExcluded{word}"] = {"value": cut["excluded"], "fmt": ".0f"}
        m[f"resRecalWithinNominal{word}"] = {"value": 100 * cut["fraction_within_nominal"], "fmt": ".0f"}
        for level in ("nominal", "calibrated"):
            real = part["real"][level]
            if real.get("fraction") is not None:
                tag = level.capitalize()
                m[f"resRecalReal{tag}{word}"] = {"value": 100 * real["fraction"], "fmt": ".0f"}
                m[f"resRecalReal{tag}Count{word}"] = {"value": real["bounded"], "fmt": ".0f"}
                m[f"resRecalReal{tag}N{word}"] = {"value": real["n"], "fmt": ".0f"}
            rep_block = part["replica"][level]
            if rep_block.get("fraction") is not None:
                m[f"resRecalReplica{level.capitalize()}{word}"] = {"value": 100 * rep_block["fraction"], "fmt": ".0f"}
        cover = part["replica"]["coverage_nominal"]
        if cover.get("fraction") is not None:
            m[f"resRecalCoverNominal{word}"] = {"value": 100 * cover["fraction"], "fmt": ".0f"}
    het = p10.get("heteroskedastic", {})
    if het.get("baseline"):
        base = het["baseline"].get("S_I_interior") or {}
        if base:
            m["resHeteroBaseline"] = {"value": 100 * base["fraction"], "fmt": ".0f"}
            m["resHeteroBaselineCount"] = {"value": base["bounded"], "fmt": ".0f"}
            m["resHeteroBaselineN"] = {"value": base["n"], "fmt": ".0f"}
        if het["baseline"].get("all_bounded") is not None:
            m["resHeteroBaselineAll"] = {"value": het["baseline"]["all_bounded"], "fmt": ".0f"}
    for structure, word in (("root", "Root"), ("proportional", "Prop")):
        block = het.get("structures", {}).get(structure, {})
        interior = block.get("S_I_interior")
        if block.get("n") and interior:
            m[f"resHetero{word}"] = {"value": 100 * interior["fraction"], "fmt": ".0f"}
            m[f"resHetero{word}Count"] = {"value": interior["bounded"], "fmt": ".0f"}
            m[f"resHetero{word}N"] = {"value": interior["n"], "fmt": ".0f"}
            m[f"resHetero{word}Low"] = {"value": 100 * interior["wilson"][0], "fmt": ".0f"}
            m[f"resHetero{word}High"] = {"value": 100 * interior["wilson"][1], "fmt": ".0f"}
            ch = block["changes_vs_baseline"]
            m[f"resHetero{word}More"] = {"value": ch["more_bounded"], "fmt": ".0f"}
            m[f"resHetero{word}Less"] = {"value": ch["less_bounded"], "fmt": ".0f"}
            m[f"resHetero{word}Same"] = {"value": ch["same"], "fmt": ".0f"}
            m[f"resHetero{word}All"] = {"value": block["all_bounded"], "fmt": ".0f"}
            m[f"resHetero{word}Sigma"] = {"value": block["median_sigma"], "fmt": ".1f"}
    return m


def _phase9_macros(p9: dict) -> dict:
    """Macros for the Phase 9 results (Amendment 5); absent sections contribute nothing."""
    m: dict = {}
    # H17: replica seeds pooled
    h17 = p9.get("h17", {})
    if h17.get("n"):
        si = h17["pooled_S_I_interior_bounded"] or {}
        if si:
            m["resSeedsSIBounded"] = _ci(100 * si["fraction"], 100 * si["wilson"][0], 100 * si["wilson"][1])
            m["resSeedsSIBoundedCount"] = {"value": si["bounded"], "fmt": ".0f"}
            m["resSeedsSIInterior"] = {"value": si["n"], "fmt": ".0f"}
        m["resSeedsTimingMax"] = {"value": 100 * h17["pooled_timing_max_fraction"], "fmt": ".0f"}
        m["resSeedsCount"] = {"value": h17["n_seeds"], "fmt": ".0f"}
        m["resSeedsMeetingClause"] = {"value": h17["seeds_meeting_S_I_clause"], "fmt": ".0f"}
        if h17.get("seed_range_S_I_fraction"):
            lo, hi = h17["seed_range_S_I_fraction"]
            m["resSeedsRangeLow"] = {"value": 100 * lo, "fmt": ".0f"}
            m["resSeedsRangeHigh"] = {"value": 100 * hi, "fmt": ".0f"}
        cov = h17.get("S_I_truth_coverage")
        if cov:
            m["resSeedsCoverSI"] = {"value": 100 * cov["fraction"], "fmt": ".0f"}
            m["resSeedsCoverCount"] = {"value": cov["covered"], "fmt": ".0f"}
            m["resSeedsCoverN"] = {"value": cov["n"], "fmt": ".0f"}
    # H18: coordinate profiles on the true-model replica (polished)
    for objective, word in (("iauc", "Iauc"), ("iauc_centroid", "Centroid"), ("trace", "Trace")):
        d = p9.get("h18", {}).get(objective, {})
        if d.get("n"):
            for key, tag in (("tau1", "Tau"), ("p", "P"), ("S_I", "SI")):
                v = _pct(d.get(key))
                if v is not None:
                    m[f"resReplicaCoords{tag}{word}"] = {"value": v, "fmt": ".0f"}
                    c = d.get(key) or {}
                    if "bounded" in c and "n" in c and "wilson" in c:
                        m[f"resReplicaCoords{tag}{word}Count"] = {"value": c["bounded"], "fmt": ".0f"}
                        m[f"resReplicaCoords{tag}{word}N"] = {"value": c["n"], "fmt": ".0f"}
                        m[f"resReplicaCoords{tag}{word}Low"] = {"value": 100 * c["wilson"][0], "fmt": ".0f"}
                        m[f"resReplicaCoords{tag}{word}High"] = {"value": 100 * c["wilson"][1], "fmt": ".0f"}
            if objective == "trace":
                for key, tag in (("log_tau1", "Tau"), ("log_p", "P"), ("log_insulin_sensitivity", "SI")):
                    cov = (d.get("truth_coverage") or {}).get(key)
                    if cov:
                        m[f"resReplicaCoordsCover{tag}Trace"] = {"value": 100 * cov["fraction"], "fmt": ".0f"}
    # H21: polished estimates
    h21 = p9.get("h21", {})
    for objective, word in (("iauc", "Iauc"), ("iauc_centroid", "Centroid"), ("trace", "Trace")):
        d = h21.get("objectives", {}).get(objective, {})
        if d.get("n"):
            m[f"resPolishMaxGap{word}"] = {"value": d["max_gap"], "fmt": ".2f"}
            m[f"resPolishMedianGap{word}"] = {"value": d["median_gap"], "fmt": ".3f"}
            m[f"resPolishWithin{word}"] = {"value": 100 * d["fraction_below_threshold"], "fmt": ".0f"}
            m[f"resPolishNotWithin{word}"] = {"value": 100 * (1 - d["fraction_below_threshold"]), "fmt": ".0f"}
        e = h21.get("h11_on_polished_estimates", {}).get(objective, {})
        if e.get("n"):
            for key, tag in (("tau1", "Tau"), ("p", "P")):
                inner = e[key]["interior"] or e[key]["all"]
                if inner and "fraction" in inner:
                    m[f"resPolishedBounded{tag}{word}"] = {"value": 100 * inner["fraction"], "fmt": ".0f"}
                    if "bounded" in inner and "n" in inner and "wilson" in inner:
                        m[f"resPolishedBounded{tag}{word}Count"] = {"value": inner["bounded"], "fmt": ".0f"}
                        m[f"resPolishedBounded{tag}{word}N"] = {"value": inner["n"], "fmt": ".0f"}
                        m[f"resPolishedBounded{tag}{word}Low"] = {"value": 100 * inner["wilson"][0], "fmt": ".0f"}
                        m[f"resPolishedBounded{tag}{word}High"] = {"value": 100 * inner["wilson"][1], "fmt": ".0f"}
    # H20: prediction on Shanghai
    h20 = p9.get("h20", {})
    if h20.get("n_units"):
        m["resShCvSubjects"] = {"value": h20["n_subjects"], "fmt": ".0f"}
        m["resShCvRepeats"] = {"value": len(h20["repeats"]), "fmt": ".0f"}
        for cell, value in h20["metrics"]["iauc"]["cell_means"].items():
            m[f"resShCvMae{cell_word(cell)}"] = {"value": value, "fmt": ".0f"}
        for key, c in h20["metrics"]["iauc"]["comparisons"].items():
            first, second = key.split("-")
            name = cell_word(first) + "Vs" + cell_word(second)
            if c.get("ci95"):
                m[f"resShCvDiff{name}"] = _ci(c["mean_difference"], c["ci95"]["low"], c["ci95"]["high"])
                m[f"resShCvWins{name}"] = {"value": c["wins_first"], "fmt": ".0f"}
        for key, c in h20["metrics"]["peak"]["comparisons"].items():
            first, second = key.split("-")
            if c.get("ci95") and second == "personal_mean":
                m[f"resShCvPeakDiff{cell_word(first)}"] = _ci(c["mean_difference"], c["ci95"]["low"],
                                                                c["ci95"]["high"], ".1f")
        if h20.get("margin_over_personal_mean_mae") is not None:
            m["resShCvMarginShare"] = {"value": 100 * h20["margin_over_personal_mean_mae"], "fmt": ".1f"}
    # H6 synthetic (H19) and the recovery curve
    syn = p9.get("h6_synthetic", {})
    if syn.get("n"):
        m["resSynthN"] = {"value": syn["n"], "fmt": ".0f"}
        m["resSynthSpearman"] = {"value": syn["spearman_vs_profile_strength"]["median"], "fmt": ".2f"}
        m["resSynthAuc"] = {"value": syn["auc_detecting_flat"], "fmt": ".2f"}
        for parameter, word in (("insulin_sensitivity", "SI"), ("gastric_emptying", "Ke"),
                                ("carb_absorption", "Ka")):
            r = syn.get("recovery", {}).get(parameter)
            if not r:
                continue
            m[f"resSynthBounded{word}"] = {"value": 100 * r["all_bounded"]["fraction"], "fmt": ".0f"}
            m[f"resSynthInteriorBounded{word}"] = {"value": 100 * r["interior"]["fraction"], "fmt": ".0f"}
            m[f"resSynthErrorLog{word}"] = {"value": r["median_abs_log_error_of_estimate"], "fmt": ".2f"}
            cov = r["coverage_of_bounded"]
            if cov.get("fraction") is not None:
                m[f"resSynthCover{word}"] = {"value": 100 * cov["fraction"], "fmt": ".0f"}
            for tercile, tw in (("low", "Low"), ("middle", "Mid"), ("high", "High")):
                t = r["terciles"][tercile]
                if t["fraction"] is not None:
                    m[f"resSynthTercile{word}{tw}"] = {"value": 100 * t["fraction"], "fmt": ".0f"}
    # H10
    h10 = p9.get("h10", {})
    if h10.get("factors"):
        for factor, word in (("initialization", "Init"), ("bounds", "Bounds"),
                             ("parameterization", "Param"), ("optimizer", "Optimizer")):
            f = h10["factors"].get(factor, {})
            if f.get("median_of_mean_tau") is not None:
                m[f"resTenTau{word}"] = {"value": f["median_of_mean_tau"], "fmt": ".2f"}
        si = h10.get("reference_S_I_first_interior", {})
        if si.get("n"):
            m["resTenSIFirst"] = {"value": 100 * si["fraction"], "fmt": ".0f"}
            m["resTenSIFirstN"] = {"value": si["n"], "fmt": ".0f"}
    # H22: second model class
    h22 = p9.get("h22", {})
    if h22.get("n_subjects"):
        m["resDmN"] = {"value": h22["n_subjects"], "fmt": ".0f"}
        for parameter, word in (("Vmx", "Vmx"), ("kabs", "Kabs"), ("kmax", "Kmax"), ("kmin", "Kmin")):
            d = h22["parameters"].get(parameter)
            if d:
                m[f"resDmBounded{word}"] = _ci(100 * d["fraction"], 100 * d["wilson"][0], 100 * d["wilson"][1])
                m[f"resDmPinned{word}"] = {"value": d["pinned"], "fmt": ".0f"}
        m["resDmRankMedian"] = {"value": h22["fisher"]["effective_rank_median"], "fmt": ".0f"}
        if h22["fisher"].get("condition_number_median"):
            m["resDmCondMedian"] = {"value": sci(h22["fisher"]["condition_number_median"])}
        m["resDmPolishGap"] = {"value": h22["polish_gap_median"], "fmt": ".2f"}
    # report-only
    d = p9.get("hall_trace", {}).get("1x", {})
    if d.get("n_subjects"):
        m["resHallTraceN"] = {"value": d["n_subjects"], "fmt": ".0f"}
        for pk, pw in (("insulin_sensitivity", "SI"), ("gastric_emptying", "Ke"), ("carb_absorption", "Ka")):
            x = d["parameters"][pk]
            m[f"resHallTraceBounded{pw}"] = _ci(100 * x["fraction"], 100 * x["wilson"][0], 100 * x["wilson"][1])
    level_word = {"90": "Ninety", "95": "NinetyFive", "99": "NinetyNine"}
    box_word = {"0.5x": "Half", "1x": "One", "2x": "Two"}
    for box, block in p9.get("delta_sensitivity", {}).items():
        for level, d in block.get("levels", {}).items():
            lw, bw = level_word[level], box_word[box]
            par = d["parameters"]
            if par["insulin_sensitivity"]["interior_fraction"] is not None:
                m[f"resLevel{lw}SIInterior{bw}"] = {"value": 100 * par["insulin_sensitivity"]["interior_fraction"],
                                                    "fmt": ".0f"}
            m[f"resLevel{lw}KeBounded{bw}"] = {"value": 100 * par["gastric_emptying"]["fraction"], "fmt": ".0f"}
            m[f"resLevel{lw}KaBounded{bw}"] = {"value": 100 * par["carb_absorption"]["fraction"], "fmt": ".0f"}
    sh = p9.get("shanghai_profile_subjects", {})
    if sh.get("2x"):
        m["resShanghaiTwoProfileN"] = {"value": sh["2x"], "fmt": ".0f"}
    for scale, word in (("0.75", "Low"), ("1.33", "High")):
        d = p9.get("carb_scale", {}).get(scale, {})
        if d.get("n"):
            m[f"resCarbScale{word}Expected"] = {
                "value": d["expected_log_ratio_if_S_I_absorbs_the_scale"], "fmt": ".2f"}
    return m


def _cohort_macros() -> dict:
    """Counts the paper reports for each cohort, computed from the cache rather than remembered."""
    from evaluation.cohort_data import cohort_summary, load_cohort
    m: dict = {}
    for cohort, minimum, word in (("cgmacros", 10, "Cgm"), ("shanghai", 10, "Shanghai"), ("hall", 5, "Hall")):
        try:
            s = cohort_summary(load_cohort(cohort, min_meals=minimum))
        except Exception:
            continue
        if not s.get("n_subjects"):
            continue
        m[f"resCohort{word}Subjects"] = {"value": s["n_subjects"], "fmt": ".0f"}
        m[f"resCohort{word}Meals"] = {"value": s["n_meals_total"], "fmt": ".0f"}
        m[f"resCohort{word}MealsMedian"] = {"value": s["meals_per_subject"]["median"], "fmt": ".0f"}
        m[f"resCohort{word}MealsMin"] = {"value": s["meals_per_subject"]["min"], "fmt": ".0f"}
        m[f"resCohort{word}MealsMax"] = {"value": s["meals_per_subject"]["max"], "fmt": ".0f"}
        m[f"resCohort{word}CarbMedian"] = {"value": s["carbs_g"]["median"], "fmt": ".0f"}
        m[f"resCohort{word}Window"] = {"value": s["window_min"], "fmt": ".0f"}
    return m


def _phase13_macros(p13: dict) -> dict:
    """Amendment 9: prediction against parameter error, clustered intervals, overlap, proxy, CV sweep, H26, R4."""
    m: dict = {}
    if not p13:
        return m

    def pct(name, block, key="fraction"):
        if block and block.get(key) is not None:
            m[name] = {"value": 100 * block[key], "fmt": ".0f"}

    for key, tag in (("fit_truth_seed0", "FitTruth"), ("random_truth_seed0", "RandomZero"),
                     ("random_truth_seed1", "RandomOne")):
        d = p13.get("r1", {}).get(key, {})
        if not d.get("n"):
            continue
        m[f"resPvp{tag}N"] = {"value": d["n"], "fmt": ".0f"}
        for field, name, fmt in (("median_grad3_mae", "Mae", ".0f"), ("median_floor", "Floor", ".0f"),
                                 ("median_ratio_to_floor", "Ratio", ".2f"), ("median_personal_mean_mae", "Mean", ".0f"),
                                 ("median_rms_log_error", "Rms", ".2f"), ("median_timing_rms_log_error", "Timing", ".2f"),
                                 ("median_si_log_error", "SI", ".2f"), ("median_ke_log_error", "Ke", ".2f"),
                                 ("median_ka_log_error", "Ka", ".2f"), ("floor_exact_median", "FloorExact", ".0f")):
            if d.get(field) is not None:
                m[f"resPvp{tag}{name}"] = {"value": d[field], "fmt": fmt}
        for x, xname in (("rms", "Rms"), ("timing", "Timing"), ("si", "SI")):
            for y, yname in (("mae", "Mae"), ("ratio", "Ratio")):
                c = d.get(f"spearman_{x}_{y}")
                if c:
                    m[f"resPvp{tag}Rho{xname}{yname}"] = _ci(c["estimate"], c["ci95"][0], c["ci95"][1], ".2f")
    for key, tag in (("S_I_pooled_iauc", "SI"), ("tau1_pooled_trace_seeds012", "Tau"), ("p_pooled_trace_seeds012", "P")):
        d = p13.get("r2", {}).get(key, {})
        if d.get("n_pairs"):
            m[f"resClust{tag}"] = _ci(100 * d["fraction"], 100 * d["clustered_ci95"][0], 100 * d["clustered_ci95"][1])
            m[f"resClust{tag}Count"] = {"value": d["bounded"], "fmt": ".0f"}
            m[f"resClust{tag}Pairs"] = {"value": d["n_pairs"], "fmt": ".0f"}
            m[f"resClust{tag}Subjects"] = {"value": d["n_subjects"], "fmt": ".0f"}
            if d.get("wilson_naive"):
                m[f"resClust{tag}NaiveLow"] = {"value": 100 * d["wilson_naive"][0], "fmt": ".0f"}
                m[f"resClust{tag}NaiveHigh"] = {"value": 100 * d["wilson_naive"][1], "fmt": ".0f"}
    r3 = p13.get("r3", {})
    if r3.get("n_gaps"):
        m["resOverlapGaps"] = {"value": r3["n_gaps"], "fmt": ".0f"}
        m["resOverlapMedianGap"] = {"value": r3["median_gap_hours"], "fmt": ".1f"}
        for hours, word in ((3, "Three"), (4, "Four"), (6, "Six")):
            b = r3.get(f"share_within_{hours}h")
            if b:
                m[f"resOverlapWithin{word}"] = _ci(100 * b["fraction"], 100 * b["wilson"][0], 100 * b["wilson"][1])
                m[f"resOverlapWithin{word}Count"] = {"value": b["count"], "fmt": ".0f"}
    r5 = p13.get("r5", {})
    if r5.get("n"):
        m["resProxyN"] = {"value": r5["n"], "fmt": ".0f"}
        for k, tag in (("raw", "Raw"), ("partial", "Partial")):
            c = r5["statistics"][k]
            m[f"resProxy{tag}"] = _ci(c["estimate"], c["low"], c["high"], ".2f")
    words = {"0": "Zero", "0.1": "Tenth", "0.25": "Quarter", "0.5": "Half", "0.75": "ThreeQuarter"}
    for cv, word in words.items():
        b = p13.get("r6", {}).get(cv, {})
        if not b.get("n"):
            continue
        for name, tag in (("tau1", "Tau"), ("p", "P")):
            pct(f"resCv{word}Bounded{tag}", b[name]["bounded_interior"])
            pct(f"resCv{word}Cover{tag}", b[name]["nominal_coverage"])
            pct(f"resCv{word}Covbd{tag}", b[name].get("bounded_coverage"))
    h26 = p13.get("h26", {})
    gap = h26.get("real", {}).get("gap", {})
    if gap.get("n"):
        m["resGlobalN"] = {"value": gap["n"], "fmt": ".0f"}
        m["resGlobalWithin"] = _ci(100 * gap["fraction"], 100 * gap["wilson"][0], 100 * gap["wilson"][1])
        m["resGlobalWithinCount"] = {"value": gap["within"], "fmt": ".0f"}
        m["resGlobalMedianGap"] = {"value": gap["median_gap"], "fmt": ".2f"}
        m["resGlobalMaxGap"] = {"value": gap["max_gap"], "fmt": ".2f"}
    for src, tag in (("real", "Real"), ("replica_seed0", "Replica")):
        at30 = h26.get(src, {}).get("at_30", {})
        if at30.get("n"):
            for name, word in (("tau1", "Tau"), ("p", "P")):
                pct(f"resGlobal{tag}{word}", at30[name]["bounded_interior"])
    g2 = h26.get("replica_seed0", {}).get("gap", {})
    if g2.get("n"):
        m["resGlobalReplicaMedianGap"] = {"value": g2["median_gap"], "fmt": ".2f"}
        m["resGlobalReplicaMaxGap"] = {"value": g2["max_gap"], "fmt": ".2f"}
    r4 = p13.get("r4", {})
    if r4.get("n"):
        m["resTolSubjects"] = {"value": r4["n"], "fmt": ".0f"}
        m["resTolMaxRel"] = {"value": 100 * r4["max_relative_eigenvalue_change_tight"], "fmt": ".1f"}
        m["resTolMedianRel"] = {"value": 100 * r4["median_relative_eigenvalue_change_tight"], "fmt": ".1f"}
        m["resTolMinCos"] = {"value": r4["min_abs_cosine_tight"], "fmt": ".3f"}
        steps = r4.get("profile_steps", {})
        if steps.get("n_profiles"):
            m["resStepsProfiles"] = {"value": steps["n_profiles"], "fmt": ".0f"}
            m["resStepsChanged"] = {"value": steps["changed"], "fmt": ".0f"}
    return m


def _phase14_macros(p14: dict) -> dict:
    """Amendment 10: inner-optimizer budget of the trace coordinate profiles, and the floors for peak time and trace."""
    m: dict = {}
    if not p14:
        return m
    settings = {"real": "Real", "replica": "Rep", "exact": "Exact", "quarter": "Quarter"}
    budgets = {"steps_100": "Hundred", "steps_500": "FiveHundred"}
    for key, tag in settings.items():
        block = p14.get("r7", {}).get(key, {})
        for steps, word in budgets.items():
            b = block.get(steps, {})
            if not b.get("n"):
                continue
            for name, coord in (("tau1", "Tau"), ("p", "P")):
                frac = (b.get(name, {}).get("bounded_interior") or {}).get("fraction")
                if frac is not None:
                    m[f"resInner{tag}{word}Bounded{coord}"] = {"value": 100 * frac, "fmt": ".0f"}
                cover = (b.get(name, {}).get("nominal_coverage") or {}).get("fraction")
                if cover is not None:
                    m[f"resInner{tag}{word}Cover{coord}"] = {"value": 100 * cover, "fmt": ".0f"}
                bounded_cover = ((b.get(name, {}).get("bounded_coverage") or {}).get("fraction"))
                if isinstance(bounded_cover, (int, float)):
                    m[f"resInner{tag}{word}Covbd{coord}"] = {"value": 100 * bounded_cover, "fmt": ".0f"}
            st = b.get("settling", {})
            if st.get("points"):
                m[f"resInner{tag}{word}Unsettled"] = {"value": 100 * st["unsettled_fraction"], "fmt": ".0f"}
                m[f"resInner{tag}{word}Better"] = {"value": st["better_than_estimate"], "fmt": ".0f"}
                m[f"resInner{tag}{word}Profiles"] = {"value": st["profiles"], "fmt": ".0f"}
        ch = block.get("changes", {})
        if ch.get("n"):
            m[f"resInner{tag}Changed"] = {"value": ch["changed"], "fmt": ".0f"}
            m[f"resInner{tag}ChangedOf"] = {"value": ch["n"], "fmt": ".0f"}
            m[f"resInner{tag}ToOpen"] = {"value": ch["bounded_to_open"], "fmt": ".0f"}
            m[f"resInner{tag}ToBounded"] = {"value": ch["open_to_bounded"], "fmt": ".0f"}
    diff = p14.get("h27", {}).get("difference_points")
    if diff:
        m["resInnerRealDiffTau"] = {"value": diff["tau1"], "fmt": ".0f"}
        m["resInnerRealDiffP"] = {"value": diff["p"], "fmt": ".0f"}
    for key, tag in (("fit_truth_seed0", "FitTruth"), ("random_truth_seed0", "RandomZero"),
                     ("random_truth_seed1", "RandomOne")):
        d = p14.get("r9", {}).get(key, {})
        if not d.get("n"):
            continue
        m[f"resFloor{tag}N"] = {"value": d["n"], "fmt": ".0f"}
        for cell, word in (("grad3", "Three"), ("grad1", "One")):
            for kind, name, fmt in (("peak", "Peak", ".1f"), ("trace", "Trace", ".1f")):
                c = d.get(f"{cell}_{kind}")
                if not c:
                    continue
                m[f"resFloor{name}{tag}{word}Fit"] = {"value": c["median_fit"], "fmt": fmt}
                m[f"resFloor{name}{tag}{word}Floor"] = {"value": c["median_floor"], "fmt": fmt}
                m[f"resFloor{name}{tag}{word}Ratio"] = {"value": c["median_ratio"], "fmt": ".2f"}
                rho = c["spearman_timing_ratio"]
                m[f"resFloor{name}{tag}{word}Rho"] = _ci(rho["estimate"], rho["ci95"][0], rho["ci95"][1], ".2f")
        for kind, name in (("peak", "Peak"), ("trace", "Trace")):
            c = d.get(f"free_minus_fixed_{kind}")
            if c:
                m[f"resFloor{name}{tag}FreeMinusFixed"] = {"value": c["median_difference_in_floors"], "fmt": ".2f"}
    return m


def _phase15_macros(p15: dict) -> dict:
    """Amendment 11: the symmetry broken by design, the fixed-parameter perturbation, the likelihood at the truth."""
    m: dict = {}
    if not p15:
        return m
    params = {"S_I": "SI", "k_e": "Ke", "k_a": "Ka"}

    def block_macros(prefix: str, block: dict, keys: dict) -> None:
        for name, tag in keys.items():
            b = block.get(name)
            if not b:
                continue
            frac = (b.get("bounded_interior") or {}).get("fraction")
            if frac is not None:
                m[f"{prefix}{tag}Bounded"] = {"value": 100 * frac, "fmt": ".0f"}
                m[f"{prefix}{tag}BoundedCount"] = {"value": b["bounded_interior"]["bounded"], "fmt": ".0f"}
                m[f"{prefix}{tag}BoundedN"] = {"value": b["bounded_interior"]["n"], "fmt": ".0f"}
            cover = (b.get("nominal_coverage") or {}).get("fraction")
            if cover is not None:
                m[f"{prefix}{tag}Cover"] = {"value": 100 * cover, "fmt": ".0f"}
            covbd = (b.get("bounded_coverage") or {}).get("fraction")
            if covbd is not None:
                m[f"{prefix}{tag}Covbd"] = {"value": 100 * covbd, "fmt": ".0f"}

    words = {"seed0_0": "Zero", "seed0_0.05": "Five", "seed0_0.1": "Ten", "seed0_0.2": "Twenty", "seed1_0.1": "SeedOneTen"}
    for key, word in words.items():
        block = p15.get("p1", {}).get(key, {})
        if block.get("n"):
            m[f"resRes{word}N"] = {"value": block["n"], "fmt": ".0f"}
            block_macros(f"resRes{word}", block, params)
    tags = {"baseline": "Base", "insulin_secretion_1.3": "GammaUp", "insulin_secretion_0.7": "GammaDown",
            "x_gain_1.3": "PThreeUp", "x_gain_0.7": "PThreeDown"}
    for key, tag in tags.items():
        block = p15.get("p2", {}).get(key, {})
        if not block.get("n"):
            continue
        block_macros(f"resPert{tag}", {"S_I": block["S_I"]}, {"S_I": "SI"})
        if block.get("median_abs_log_error") is not None:
            m[f"resPert{tag}Err"] = {"value": block["median_abs_log_error"], "fmt": ".2f"}
            m[f"resPert{tag}Bias"] = {"value": block["median_signed_log_error"], "fmt": ".2f"}
    for key, tag in (("up", "Up"), ("down", "Down")):
        eq = p15.get("p2", {}).get("equivalence", {}).get(key, {})
        if eq.get("n"):
            m[f"resPertSame{tag}"] = {"value": eq["same"], "fmt": ".0f"}
            m[f"resPertSame{tag}N"] = {"value": eq["n"], "fmt": ".0f"}
    for key, tag in (("exact", "Exact"), ("quarter", "Quarter"), ("replica", "Rep")):
        b = p15.get("p3", {}).get(key, {})
        if not b.get("n"):
            continue
        m[f"resCal{tag}N"] = {"value": b["n"], "fmt": ".0f"}
        m[f"resCal{tag}MedianLr"] = {"value": b["median_lr_half"], "fmt": ".1f"}
        m[f"resCal{tag}QNinetyLr"] = {"value": b["q90_lr_half"], "fmt": ".1f"}
        m[f"resCal{tag}Over"] = {"value": 100 * b["share_lr_over_delta"], "fmt": ".0f"}
        m[f"resCal{tag}GapWithin"] = {"value": b["optimizer_gap_within"], "fmt": ".0f"}
        m[f"resCal{tag}GapFraction"] = {"value": 100 * b["optimizer_gap_fraction"], "fmt": ".0f"}
        m[f"resCal{tag}GapMax"] = {"value": b["max_optimizer_gap"], "fmt": ".2f"}
        ref = b.get("reference_half_chi2_3")
        if ref:
            m["resCalRefMedian"] = {"value": ref[0], "fmt": ".1f"}
            m["resCalRefQNinety"] = {"value": ref[1], "fmt": ".1f"}
    return m


def _bounded_cells(m: dict, prefix: str, block: dict, keys: dict) -> None:
    """Bounded fraction (and count), nominal coverage and coverage of the bounded intervals for named parameters."""
    for name, tag in keys.items():
        b = (block or {}).get(name)
        if not b:
            continue
        bi = b.get("bounded_interior") or {}
        if bi.get("fraction") is not None:
            m[f"{prefix}{tag}Bounded"] = {"value": 100 * bi["fraction"], "fmt": ".0f"}
            m[f"{prefix}{tag}BoundedCount"] = {"value": bi["bounded"], "fmt": ".0f"}
            m[f"{prefix}{tag}BoundedN"] = {"value": bi["n"], "fmt": ".0f"}
        cover = (b.get("nominal_coverage") or {}).get("fraction")
        if cover is not None:
            m[f"{prefix}{tag}Cover"] = {"value": 100 * cover, "fmt": ".0f"}
        covbd = b.get("bounded_coverage")
        covbd = covbd.get("fraction") if isinstance(covbd, dict) else None
        if covbd is not None:
            m[f"{prefix}{tag}Covbd"] = {"value": 100 * covbd, "fmt": ".0f"}


def _phase16_macros(p16: dict) -> dict:
    """Amendment 12: the profile at the true value four ways, and the converged inner solver."""
    m: dict = {}
    if not p16:
        return m
    coords = {"log_tau1": "Tau", "log_p": "P"}
    for key, tag in (("quarter", "Quarter"), ("exact", "Exact"), ("replica", "Rep")):
        block = p16.get("r13", {}).get(key, {})
        if not block.get("n"):
            continue
        m[f"resTruth{tag}N"] = {"value": block["n"], "fmt": ".0f"}
        for name, word in coords.items():
            e = block.get(name) or {}
            if not e.get("n"):
                continue
            for field, label in (("stored", "Stored"), ("adam_point", "Adam"), ("lbfgsb_hat", "Direct"),
                                 ("lbfgsb_truth", "FromTruth")):
                c = e[field]
                m[f"resTruth{tag}{word}{label}Over"] = {"value": 100 * c["share_above_delta"], "fmt": ".0f"}
                m[f"resTruth{tag}{word}{label}Median"] = {"value": c["median"], "fmt": ".1f"}
            within = e["lbfgsb_hat"]["share_within_delta"]
            m[f"resTruth{tag}{word}DirectWithin"] = {"value": 100 * within, "fmt": ".0f"}
            m[f"resTruth{tag}{word}DirectWithinCount"] = {"value": round(within * e["lbfgsb_hat"]["n"]), "fmt": ".0f"}
            m[f"resTruth{tag}{word}DirectN"] = {"value": e["lbfgsb_hat"]["n"], "fmt": ".0f"}
            m[f"resTruth{tag}{word}Interp"] = {"value": e["median_interpolation"], "fmt": ".1f"}
            m[f"resTruth{tag}{word}Solver"] = {"value": abs(e["median_inner_solver"]), "fmt": ".3f"}
            m[f"resTruth{tag}{word}Start"] = {"value": abs(e["median_start"]), "fmt": ".3f"}
    settings = {"real": "Real", "quarter": "Quarter", "exact": "Exact", "replica": "Rep"}
    for key, tag in settings.items():
        block = p16.get("r14", {}).get(key, {})
        solved = block.get("lbfgsb", {})
        if not solved.get("n"):
            continue
        _bounded_cells(m, f"resConv{tag}", solved, {"tau1": "Tau", "p": "P"})
        ch = block.get("class_changes", {})
        if ch.get("n"):
            m[f"resConv{tag}Changed"] = {"value": ch["changed"], "fmt": ".0f"}
            m[f"resConv{tag}ChangedOf"] = {"value": ch["n"], "fmt": ".0f"}
        nc = block.get("lbfgsb_settling", {})
        if nc.get("fraction") is not None:
            m[f"resConv{tag}NotConverged"] = {"value": 100 * nc["fraction"], "fmt": ".1f"}
            m[f"resConv{tag}NotConvergedCount"] = {"value": nc["not_converged"], "fmt": ".0f"}
            m[f"resConv{tag}NotConvergedOf"] = {"value": nc["points"], "fmt": ".0f"}
    fine = p16.get("r14", {}).get("quarter_fine", {}).get("lbfgsb", {})
    if fine.get("n"):
        _bounded_cells(m, "resConvFine", fine, {"tau1": "Tau", "p": "P"})
    diff = p16.get("h32", {}).get("difference_points")
    if diff:
        m["resConvRealDiffTau"] = {"value": diff["tau1"], "fmt": ".0f"}
        m["resConvRealDiffP"] = {"value": diff["p"], "fmt": ".0f"}
    return m


def _phase17_macros(p17: dict) -> dict:
    """Amendment 13: the profiles on a 29-point grid in place of 15."""
    m: dict = {}
    if not p17:
        return m
    grids = {"grid_15": "Fifteen", "grid_29": "TwentyNine"}
    for key, tag in (("real", "Real"), ("exact", "Exact"), ("replica", "Rep")):
        block = p17.get("r15", {}).get(key, {})
        for grid, word in grids.items():
            b = block.get(grid, {})
            if b.get("n"):
                _bounded_cells(m, f"resGrid{tag}{word}", b, {"tau1": "Tau", "p": "P"})
        ch = block.get("class_changes", {})
        if ch.get("n"):
            m[f"resGrid{tag}Changed"] = {"value": ch["changed"], "fmt": ".0f"}
            m[f"resGrid{tag}ChangedOf"] = {"value": ch["n"], "fmt": ".0f"}
    for key, tag in (("residual_0", "Zero"), ("residual_0.1", "Ten")):
        block = p17.get("r16", {}).get(key, {})
        for grid, word in grids.items():
            b = block.get(grid, {})
            if b.get("n"):
                _bounded_cells(m, f"resGridRes{tag}{word}", b, {"S_I": "SI", "k_e": "Ke", "k_a": "Ka"})
    diff = p17.get("h34", {}).get("difference_points")
    if diff:
        m["resGridRealDiffTau"] = {"value": diff["tau1"], "fmt": ".0f"}
        m["resGridRealDiffP"] = {"value": diff["p"], "fmt": ".0f"}
    return m


def _extra_macros(summary: dict, verdicts: dict) -> dict:
    """Numbers the manuscript quotes that no other block declares: spectra, rank, bias, extensions, counts."""
    from pathlib import Path

    from evaluation.identifiability_tools import CHI2_DELTA
    from evaluation.results_io import ROOT
    from evaluation.stats_utils import N_RESAMPLES

    m: dict = {}
    m["resDeltaThreshold"] = {"value": CHI2_DELTA, "fmt": ".2f"}
    m["resBootstrapResamples"] = {"value": N_RESAMPLES, "fmt": ".0f"}
    m["resAmendmentCount"] = {"value": NUMBER_WORD.get(len(list(Path(ROOT).glob("PREREG_AMENDMENT_*.md"))), "several").lower()}
    statuses = [row["status"] for row in verdicts.values()]
    m["resCountMet"] = {"value": statuses.count("met"), "fmt": ".0f"}
    m["resCountNotMet"] = {"value": statuses.count("not met"), "fmt": ".0f"}
    m["resCountGraded"] = {"value": statuses.count("met") + statuses.count("not met"), "fmt": ".0f"}
    m["resCountTotal"] = {"value": len(statuses), "fmt": ".0f"}
    # planned (H1-H10) against declared later (H11-H25), and the primary endpoints
    def _tally(names, prefix):
        st = [verdicts[n]["status"] for n in names if n in verdicts]
        m[f"res{prefix}Met"] = {"value": st.count("met"), "fmt": ".0f"}
        m[f"res{prefix}NotMet"] = {"value": st.count("not met"), "fmt": ".0f"}
        m[f"res{prefix}Graded"] = {"value": st.count("met") + st.count("not met"), "fmt": ".0f"}
    _tally([f"H{i}" for i in range(1, 11)], "Planned")
    _tally([f"H{i}" for i in range(11, 100)], "Later")
    _tally([n for n, row in verdicts.items() if row.get("primary")], "Primary")
    # spectra at the primary box
    for key, word in (("lambda1", "One"), ("lambda2", "Two"), ("lambda3", "Three")):
        d = summary.get("eigenvalues", {}).get("1x", {}).get(key)
        if d:
            m[f"resEigMedian{word}"] = {"value": sci(d["median"])}
    # generic rank
    for obs, word in (("iauc", "Iauc"), ("iauc_centroid", "Centroid"), ("trace", "Trace")):
        d = summary.get("generic_rank", {}).get(obs)
        if d:
            counts = {int(k): v for k, v in d["rank_counts"].items()}
            m[f"resRankFull{word}"] = {"value": counts.get(3, 0), "fmt": ".0f"}
            m[f"resRankDraws{word}"] = {"value": sum(counts.values()), "fmt": ".0f"}
    # pinned subjects: how many on the upper bound, and the area deficit
    bias = summary.get("pinned_bias", {}).get("1x", {}).get("groups", {})
    for key, word in (("gastric_emptying", "Ke"), ("carb_absorption", "Ka")):
        g = bias.get(key)
        if not g:
            continue
        m[f"resPinnedUpper{word}One"] = {"value": g["n_pinned_upper"], "fmt": ".0f"}
        m[f"resBiasObsPinned{word}"] = {"value": g["obs"]["mean_pinned"], "fmt": ".0f"}
        m[f"resBiasObsOther{word}"] = {"value": g["obs"]["mean_other"], "fmt": ".0f"}
        ci = g["signed"].get("signed_mean_ci95_pinned")
        if ci:
            m[f"resBiasSigned{word}"] = _ci(ci["estimate"], ci["low"], ci["high"])
        m[f"resBiasSignedOther{word}"] = {"value": g["signed"]["mean_other"], "fmt": ".0f"}
    # the four-times extension of the timing profiles, and the profile improvement over the estimate
    for box, word in (("0.5x", "Half"), ("1x", "One"), ("2x", "Two")):
        q = summary.get("profile_quality", {}).get(box, {})
        for key, w in (("gastric_emptying", "Ke"), ("carb_absorption", "Ka")):
            ext = (q.get(key) or {}).get("four_times_extension")
            if ext:
                m[f"resExtFlat{w}{word}"] = {"value": ext.get("flat", 0), "fmt": ".0f"}
                m[f"resExtOneSided{w}{word}"] = {"value": ext.get("one-sided", 0), "fmt": ".0f"}
        if q.get("insulin_sensitivity"):
            m[f"resProfImproveMax{word}"] = {"value": max(v["improvement_over_theta_hat_max"] for v in q.values()),
                                              "fmt": ".2f"}
    # window sensitivity (H2)
    mc = summary.get("moment_checks", {})
    for window in (90, 180, 240, 360):
        word = NUMBER_WORD_BIG[window]
        for engine, ew in (("linear", "Lin"), ("nonlinear_default", "Nl")):
            block = mc.get(engine, {}).get(str(window))
            if block:
                for p, pw in (("ke", "Ke"), ("ka", "Ka")):
                    m[f"resWin{word}{ew}{pw}"] = {"value": block[p]["median"], "fmt": ".2f"}
    # peak-time and trace error of the cells (the trace objective improves what it targets)
    for key, tag in (("prediction_peak_time", "Peak"), ("prediction_trace_rmse", "TraceRmse")):
        for cell, d in summary.get(key, {}).get("cells", {}).items():
            if cell in ("grad3", "grad3_trace", "personal_mean", "grid3", "grad1", "snpe", "rf"):
                m[f"res{tag}{cell_word(cell)}"] = {"value": d["mean"], "fmt": ".1f"}
    # the gradient diagnostic with and without bound projection
    gd = summary.get("gradient_diagnostic", {}).get("1x", {}).get("all")
    if gd:
        m["resGradProjSIFirst"] = {"value": 100 * gd["projected_norm"]["S_I_largest_fraction"], "fmt": ".0f"}
        m["resGradRawSIFirst"] = {"value": 100 * gd["raw_norm"]["S_I_largest_fraction"], "fmt": ".0f"}
        m["resGradProjRatio"] = {"value": gd["projected_norm"]["median_timing_over_S_I"], "fmt": ".3f"}
        m["resGradRawRatio"] = {"value": gd["raw_norm"]["median_timing_over_S_I"], "fmt": ".2f"}
    # the 500-step check
    steps = summary.get("steps_sensitivity", {}).get("comparisons", {})
    for key, word in (("grad3_500_vs_grid3", "GradThreeFiveHundredVsGridThree"),
                      ("grad3_500_vs_grad1_500", "GradThreeFiveHundredVsGradOneFiveHundred")):
        d = steps.get(key)
        if d and d.get("ci95"):
            m[f"resSteps{word}"] = _ci(d["mean_difference"], d["ci95"]["low"], d["ci95"]["high"])
    return m


NUMBER_WORD_BIG = {90: "Ninety", 180: "OneEighty", 240: "TwoForty", 360: "ThreeSixty"}


def _stored_macros(p8: dict) -> dict:
    """The subject-level correlates (Spearman and Holm-adjusted p) and the tied-model interval fraction."""
    m: dict = {}
    corr = p8.get("stored", {}).get("correlates", {}).get("1x", {}).get("covariates", {})
    names = {"n_meals": "Meals", "carb_range": "CarbRange", "mean_iauc": "MeanIauc", "si_estimate": "Estimate"}
    outcomes = {"si_bounded": "SIBounded", "timing_pinned_upper": "Pinned", "si_max_rise": "SIRise"}
    for cov, cw in names.items():
        for out, ow in outcomes.items():
            d = corr.get(cov, {}).get(out)
            if d and "spearman" in d:
                m[f"resCorr{cw}{ow}"] = {"value": d["spearman"], "fmt": ".2f"}
                m[f"resCorrP{cw}{ow}"] = {"value": d.get("p_holm", d["p"]), "fmt": ".3f"}
    tied = p8.get("h12", {}).get("tau1_tied_interior")
    if tied:
        m["resTiedTauBounded"] = {"value": 100 * tied["fraction"], "fmt": ".0f"}
        m["resTiedTauCount"] = {"value": tied["bounded"], "fmt": ".0f"}
        m["resTiedTauInterior"] = {"value": tied["n"], "fmt": ".0f"}
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
