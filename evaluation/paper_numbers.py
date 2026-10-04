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
               8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve"}
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
