"""LaTeX tables generated from stored results. No number in these files is typed by hand.

Written to `paper/generated/` (not `paper/`), because `lint_numbers` guards the prose in `paper/*.tex`
and a generated table is by construction full of digits that came from an analysis.

  hypotheses.tex   plan rule, observed value and status for every hypothesis
  bounds.tex       fraction of subjects on a bound, three boxes
  profiles.tex     fraction with a bounded profile interval, three boxes
  ladder.tex       median timing-block condition number, own estimate and common reference
  prediction.tex   held-out error by cell, with the pre-specified comparisons

Run:  python -m evaluation.tables_aistats
"""
from __future__ import annotations

import json

from evaluation.hypotheses import evaluate_all, load_summary
from evaluation.results_io import ROOT

OUT = ROOT / "paper" / "generated"
BOXES = ("0.5x", "1x", "2x")
BOX_TEX = {"0.5x": r"$0.5\times$", "1x": r"$1\times$", "2x": r"$2\times$"}
PARAMS = (("insulin_sensitivity", r"$S_I$"), ("gastric_emptying", r"$k_e$"),
          ("carb_absorption", r"$k_a$"))


def _tex_escape(text: str) -> str:
    return (str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#"))


def _write(name: str, header: str, rows: list[str], caption: str, label: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    body = "\n".join(rows)
    tex = (f"\\begin{{table}}[t]\n\\centering\\small\n\\begin{{tabular}}{{{header}}}\n\\toprule\n"
           f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
           f"\\end{{table}}\n")
    (OUT / name).write_text(tex, encoding="utf-8")


def hypotheses_table(verdicts: dict) -> None:
    rows = [r"Hypothesis & Primary & Status \\", r"\midrule"]
    for name, row in verdicts.items():
        rows.append(f"{name} & {'yes' if row['primary'] else 'no'} & {_tex_escape(row['status'])} \\\\")
    _write("hypotheses.tex", "llp{5cm}", rows,
           "Status of every hypothesis, computed from the stored results by the rule in the analysis "
           "plan.", "tab:hypotheses")


def bounds_table(summary: dict) -> None:
    sweep = summary.get("bounds_sweep_fisher_fit", {})
    if not sweep:
        return
    rows = [r"Parameter & " + " & ".join(BOX_TEX[b] for b in BOXES) + r" \\", r"\midrule"]
    for key, label in PARAMS:
        cells = []
        for b in BOXES:
            d = sweep[b]["pinned"][key]
            cells.append(f"{d['n']} ({100 * d['fraction']:.0f}\\%)")
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    rows.append(r"interior for $S_I$ & " + " & ".join(
        str(sweep[b]["interior_for_parameter"]["insulin_sensitivity"]["n"]) for b in BOXES) + r" \\")
    rows.append(r"interior, original definition & " + " & ".join(
        str(sweep[b]["interior_converged_original"]) for b in BOXES) + r" \\")
    _write("bounds.tex", "lccc", rows,
           "Subjects whose maximum-likelihood estimate lies on a bound (iAUC objective, 45 subjects), "
           "for three box widths. The $1\\times$ box is the primary one.", "tab:bounds")


def profiles_table(summary: dict) -> None:
    rows = [r"Objective, parameter & " + " & ".join(BOX_TEX[b] for b in BOXES) + r" \\", r"\midrule"]
    names = (("profile_likelihood", "iAUC"), ("profile_likelihood_iauc_centroid", "iAUC+centroid"),
             ("profile_likelihood_trace", "trace"))
    any_row = False
    for key, objective in names:
        block = summary.get(key, {})
        if not any(block.get(b, {}).get("n_subjects") for b in BOXES):
            continue
        for pkey, label in PARAMS:
            cells = []
            for b in BOXES:
                d = block.get(b, {}).get("parameters", {}).get(pkey)
                cells.append("--" if not d else
                             f"{100 * d['fraction']:.0f} [{100 * d['wilson'][0]:.0f}, "
                             f"{100 * d['wilson'][1]:.0f}]")
            rows.append(f"{objective}, {label} & " + " & ".join(cells) + r" \\")
            any_row = True
    if any_row:
        _write("profiles.tex", "lccc", rows,
               "Percentage of subjects with a bounded 95\\% profile-likelihood interval (Wilson 95\\% "
               "interval in brackets). A dash means the analysis was not run.", "tab:profiles")


def ladder_table(summary: dict) -> None:
    ladder = summary.get("ladder", {})
    if not ladder:
        return
    rows = [r"Box, evaluated at & iAUC & iAUC+centroid & trace \\", r"\midrule"]
    for b in BOXES:
        for tag, key in (("own estimate", "h3_own_theta_hat"), ("common reference", "h3_common_reference")):
            med = ladder.get(b, {}).get(key, {}).get("median_condition_number")
            if med:
                rows.append(f"{BOX_TEX[b]}, {tag} & " + " & ".join(
                    f"{med[o]:.2g}" for o in ("iauc", "iauc_centroid", "trace")) + r" \\")
    _write("ladder.tex", "lccc", rows,
           "Median condition number of the Schur-complement timing block, at each objective's own "
           "maximum-likelihood estimate and at one common reference point per subject.", "tab:ladder")


def prediction_table(summary: dict) -> None:
    pred = summary.get("prediction", {})
    if not pred.get("cells"):
        return
    rows = [r"Cell & Mean & SD over subjects \\", r"\midrule"]
    for cell, c in sorted(pred["cells"].items(), key=lambda kv: kv[1]["mean"]):
        rows.append(f"{_tex_escape(cell)} & {c['mean']:.0f} & {c['sd']:.0f} \\\\")
    _write("prediction.tex", "lcc", rows,
           "Held-out iAUC mean absolute error (mg/dL$\\cdot$min), 45 subjects, five repeats of "
           "five-fold cross-validation.", "tab:prediction")
    rows = [r"Pair & Difference [95\% CI] & Wins & Holm $p$ \\", r"\midrule"]
    for c in pred["comparisons"]:
        if "ci95" not in c or c["group"] == "versus_personal_mean":
            continue
        ci = c["ci95"]
        rows.append(f"{_tex_escape(c['first'])} $-$ {_tex_escape(c['second'])} & "
                    f"{c['mean_difference']:.0f} [{ci['low']:.0f}, {ci['high']:.0f}] & "
                    f"{c['wins_first']}/{c['n']} & {c.get('p_holm', float('nan')):.3f} \\\\")
    _write("comparisons.tex", "lccc", rows,
           "Pre-specified paired comparisons (negative: the first cell predicts better).",
           "tab:comparisons")


def main() -> int:
    summary = load_summary()
    if not summary:
        print("no results/phase2_summary.json; run evaluation.phase2_summary first")
        return 1
    verdicts = evaluate_all(summary)
    hypotheses_table(verdicts)
    for fn in (bounds_table, profiles_table, ladder_table, prediction_table):
        fn(summary)
    print("tables:", ", ".join(sorted(p.name for p in OUT.glob("*.tex"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
