"""4c: statistics for the prediction comparison, from the stored A9 per-meal predictions.

Nothing is refitted here. Every number comes from the raw predictions in the A9 result files, so a
statistic can be changed or added without touching the expensive step.

Scoring. A unit is one (subject, repeat); within it every meal is held out exactly once, so a unit has
one mean absolute iAUC error per cell. A subject's score is the mean of that over the repeats, and the
spread across repeats is reported separately. The resampling unit for every interval is the SUBJECT.

The pre-specified pairs, and the question each one asks:

  H7   grad3 - grid3, grad1 - grid1        does the optimizer matter? (equivalence)
  H8   grad3 - grad1                        do the extra parameters matter? (equivalence)
  H9   grad3_trace - grad3                  does the objective matter? (superiority by a margin)
       grad1_trace - grad1
       grad3 - grid1, grad3 - rf, grad3 - snpe
       every cell - personal_mean

A difference is `first - second` in mg/dL*min, so a negative number means the first cell predicts
better. Equivalence is declared when the 90% interval lies inside the margin, at 90, 150 and 300.
"""
from __future__ import annotations

import numpy as np

from evaluation.stats_utils import (
    binomial_ci, holm, paired_bootstrap_ci, tost, wilcoxon,
)

ANALYSIS_ID = "A9_prediction_cv"

EQUIVALENCE_PAIRS = (("grad3", "grid3"), ("grad1", "grid1"), ("grad3", "grad1"))
SUPERIORITY_PAIRS = (("grad3_trace", "grad3"), ("grad1_trace", "grad1"))
OTHER_PAIRS = (("grad3", "grid1"), ("grad3", "rf"), ("grad3", "snpe"))
MARGINS = (90.0, 150.0, 300.0)
H9_MARGIN = 150.0
METRICS = {"iauc": "iauc_mae", "peak": "peak_mae", "trace": "trace_rmse_mean"}


def load_units(analysis_id: str = ANALYSIS_ID) -> list[dict]:
    """Every completed (subject, repeat) payload of the most complete A9 result set."""
    from evaluation.results_io import largest_result_set
    _, documents = largest_result_set(analysis_id)
    return [doc["payload"] for key, doc in documents.items() if key != "_unreadable"]


def subject_scores(units: list[dict], metric: str = "iauc") -> dict:
    """`{cell: {subject: mean over repeats}}` and `{cell: {subject: [per repeat]}}`."""
    field = METRICS[metric]
    per_repeat: dict = {}
    for unit in units:
        for cell, body in unit["cells"].items():
            value = body.get(field)
            if value is None:
                continue
            per_repeat.setdefault(cell, {}).setdefault(unit["subject_id"], []).append(
                (unit["repeat"], float(value)))
    means = {cell: {s: float(np.mean([v for _, v in vals])) for s, vals in subjects.items()}
             for cell, subjects in per_repeat.items()}
    return {"mean": means, "repeats": per_repeat}


def _aligned(scores: dict, first: str, second: str, required_repeats: int | None = None):
    a, b = scores["mean"].get(first, {}), scores["mean"].get(second, {})
    subjects = sorted(set(a) & set(b))
    if required_repeats is not None:
        subjects = [s for s in subjects
                    if len(scores["repeats"][first][s]) >= required_repeats
                    and len(scores["repeats"][second][s]) >= required_repeats]
    return subjects, np.array([a[s] for s in subjects]), np.array([b[s] for s in subjects])


def compare(scores: dict, first: str, second: str, margins=MARGINS) -> dict:
    subjects, x, y = _aligned(scores, first, second)
    if len(subjects) < 3:
        return {"first": first, "second": second, "n": len(subjects), "note": "too few subjects"}
    diff = x - y
    ci = paired_bootstrap_ci(x, y)
    wins = int(np.sum(x < y))
    win_ci = binomial_ci(wins, len(subjects))
    return {
        "first": first, "second": second, "n": len(subjects),
        "mean_first": float(x.mean()), "mean_second": float(y.mean()),
        "sd_first": float(x.std(ddof=1)), "sd_second": float(y.std(ddof=1)),
        "mean_difference": float(diff.mean()), "median_difference": float(np.median(diff)),
        "ci95": ci.as_dict(), "wilcoxon": wilcoxon(x, y),
        "wins_first": wins, "win_fraction": wins / len(subjects),
        "win_ci95": [win_ci.low, win_ci.high],
        "tost": {str(int(m)): tost(x, y, m) for m in margins},
    }


def spread_across_repeats(scores: dict, cell: str) -> dict:
    """How much a cell's per-subject score moves with the fold assignment alone."""
    per = scores["repeats"].get(cell, {})
    sds = [np.std([v for _, v in vals], ddof=1) for vals in per.values() if len(vals) > 1]
    if not sds:
        return {"n": 0}
    return {"n": len(sds), "median_within_subject_sd": float(np.median(sds)),
            "mean_within_subject_sd": float(np.mean(sds))}


def analyse(units: list[dict] | None = None, metric: str = "iauc") -> dict:
    units = load_units() if units is None else units
    if not units:
        return {"n_units": 0}
    scores = subject_scores(units, metric)
    cells = sorted(scores["mean"])
    out = {"metric": metric, "n_units": len(units),
           "n_subjects": len({u["subject_id"] for u in units}),
           "repeats": sorted({u["repeat"] for u in units}),
           "cells": {c: {"mean": float(np.mean(list(scores["mean"][c].values()))),
                         "sd": float(np.std(list(scores["mean"][c].values()), ddof=1)),
                         "n": len(scores["mean"][c]),
                         "spread_across_repeats": spread_across_repeats(scores, c)}
                     for c in cells}}

    pairs = []
    for group, pair_list in (("equivalence", EQUIVALENCE_PAIRS), ("superiority", SUPERIORITY_PAIRS),
                             ("other", OTHER_PAIRS)):
        for first, second in pair_list:
            if first in scores["mean"] and second in scores["mean"]:
                pairs.append((group, first, second))
    for cell in cells:
        if cell not in ("personal_mean",) and "personal_mean" in scores["mean"]:
            pairs.append(("versus_personal_mean", cell, "personal_mean"))

    comparisons = []
    for group, first, second in pairs:
        row = compare(scores, first, second)
        row["group"] = group
        comparisons.append(row)

    # Holm over the family of all pre-specified pairs that have a test.
    testable = [c for c in comparisons if "wilcoxon" in c]
    correction = holm([c["wilcoxon"]["p"] for c in testable])
    for c, adj, rej in zip(testable, correction["p_adjusted"], correction["reject"]):
        c["p_holm"], c["reject_holm"] = adj, bool(rej)
    out["comparisons"] = comparisons
    out["holm_family_size"] = len(testable)

    # H9: superiority by more than the margin. The CI of (trace - iauc) must lie entirely below -margin.
    verdicts = {}
    for c in comparisons:
        if c["group"] == "superiority" and "ci95" in c:
            ci = c["ci95"]
            verdicts[f"{c['first']}_vs_{c['second']}"] = {
                "mean_difference": c["mean_difference"], "ci95": [ci["low"], ci["high"]],
                "p_holm": c.get("p_holm"),
                "lowers_error_by_more_than_margin": bool(ci["high"] < -H9_MARGIN),
                "margin": H9_MARGIN}
    out["h9"] = verdicts
    out["h7_h8"] = {f"{c['first']}_vs_{c['second']}": {
        "mean_difference": c["mean_difference"],
        "equivalent_at": {m: c["tost"][m]["equivalent"] for m in c["tost"]}}
        for c in comparisons if c["group"] == "equivalence" and "tost" in c}
    return out


GAP_TOLERANCE = 0.01


def steps_sensitivity(long_run_match=lambda p: p.get("fit_settings", {}).get("steps") == 500) -> dict:
    """Amendment 3: does a 500-step budget change the H7 / H8 equivalence verdicts?

    The long run holds only `grad3` and `grad1` for the first repeats. They are compared with the
    exhaustive-grid cells of the primary run on the SAME repeats, and with each other, with the same
    paired bootstrap, 90% interval and margins as the primary analysis. The primary cells are rescored
    on those repeats too, so the comparison is like for like.
    """
    from evaluation.results_io import largest_matching
    long_units = largest_matching(ANALYSIS_ID, long_run_match)
    primary = [p for p in load_units() if p.get("fit_settings", {}).get("steps") == 150]
    if not long_units or not primary:
        return {"n": 0}
    repeats = sorted({u["repeat"] for u in long_units.values()})
    merged = {}
    for unit in primary:
        if unit["repeat"] in repeats:
            merged[(unit["subject_id"], unit["repeat"])] = {
                "subject_id": unit["subject_id"], "repeat": unit["repeat"],
                "cells": {k: v for k, v in unit["cells"].items() if k in ("grad3", "grad1", "grid3", "grid1")}}
    for unit in long_units.values():
        key = (unit["subject_id"], unit["repeat"])
        if key in merged:
            merged[key]["cells"]["grad3_500"] = unit["cells"]["grad3"]
            merged[key]["cells"]["grad1_500"] = unit["cells"]["grad1"]
    units = [u for u in merged.values() if "grad3_500" in u["cells"]]
    scores = subject_scores(units, "iauc")
    rows = {}
    for first, second in (("grad3", "grid3"), ("grad3_500", "grid3"), ("grad1", "grid1"),
                          ("grad1_500", "grid1"), ("grad3", "grad1"), ("grad3_500", "grad1_500")):
        c = compare(scores, first, second)
        rows[f"{first}_vs_{second}"] = {
            "mean_difference": c.get("mean_difference"), "ci95": c.get("ci95"),
            "equivalent_at": {m: v["equivalent"] for m, v in c.get("tost", {}).items()},
            "n": c.get("n")}
    changed = [k for k in ("grad3_vs_grid3", "grad1_vs_grid1", "grad3_vs_grad1")
               if rows[k]["equivalent_at"] != rows[k.replace("grad3", "grad3_500").replace(
                   "grad1", "grad1_500")]["equivalent_at"]]
    return {"n_units": len(units), "repeats": repeats, "comparisons": rows,
            "verdict_changed_for": changed, "robust_to_budget": not changed}


def inference_gap(units: list[dict] | None = None) -> dict:
    """How close does the gradient fit get to the exhaustive grid minimum of the SAME training loss?

    Per (subject, repeat, fold): `(loss_gradient - loss_grid) / loss_grid`, for grad3 against grid3 and
    grad1 against grid1. Both are the regularized training loss on the training meals of that fold, so
    the comparison is of optimizers on one objective. The grid has finite resolution, so it can sit
    ABOVE the true minimum and the gap can be negative; that means the gradient fit found a better point
    than any grid node, not an error. If the gradient fit is within `GAP_TOLERANCE` (one percent) of the
    grid minimum for nearly every fold, under-convergence at the submitted step count is ruled out as an
    explanation for the gradient and the grid predicting alike.
    """
    units = load_units() if units is None else units
    out = {"tolerance": GAP_TOLERANCE, "pairs": {}}
    for gradient, grid in (("grad3", "grid3"), ("grad1", "grid1")):
        gaps = []
        for unit in units:
            g, h = unit["cells"].get(gradient), unit["cells"].get(grid)
            if g is None or h is None:
                continue
            for fg, fh in zip(g["folds"], h["folds"]):
                if "final_loss" in fg and "final_loss" in fh and fh["final_loss"] > 0:
                    gaps.append((fg["final_loss"] - fh["final_loss"]) / fh["final_loss"])
        if not gaps:
            out["pairs"][f"{gradient}_vs_{grid}"] = {"n": 0}
            continue
        a = np.asarray(gaps)
        out["pairs"][f"{gradient}_vs_{grid}"] = {
            "n_folds": int(a.size), "median_relative_gap": float(np.median(a)),
            "q90_relative_gap": float(np.percentile(a, 90)), "max_relative_gap": float(a.max()),
            "fraction_within_tolerance": float(np.mean(a <= GAP_TOLERANCE)),
            "fraction_gradient_better": float(np.mean(a < 0)),
        }
    return out


def main() -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="iauc", choices=sorted(METRICS))
    args = ap.parse_args()
    print(json.dumps(analyse(metric=args.metric), indent=2))


if __name__ == "__main__":
    main()
