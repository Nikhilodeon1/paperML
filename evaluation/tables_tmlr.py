"""LaTeX tables for the TMLR manuscript, generated from stored results. No number here is typed by hand.

Written to `paper/generated/` (not `paper/`) because `lint_numbers` guards the prose in `paper/*.tex`, and a
generated table is by construction full of digits that came from an analysis.

  cohorts.tex           the cohorts: size, meals, sensor interval, window, carbohydrate source
  cohort_results.tex    replication: bound fractions, profile fractions, Fisher alignment, by cohort and box
  prediction_compact.tex held-out error by cell, CGMacros and Shanghai
  scorecard.tex         one line per hypothesis: question, origin, status
  register.tex          every hypothesis with its rule, origin and status (appendix)
  noise_budget.tex      the what-if noise budgets on the replica
  correlates.tex        subject-level correlates with Holm-adjusted p-values

The four tables of `tables_aistats` (bounds, profiles, ladder, comparisons) are regenerated as well.

Run:  python -m evaluation.tables_tmlr
"""
from __future__ import annotations

import json

import numpy as np

from evaluation import ladder, profile_lik, tables_aistats
from evaluation.hypotheses import evaluate_all, load_summary
from evaluation.results_io import ROOT

OUT = ROOT / "paper" / "generated"
STATUS_COLOR = {"met": "", "not met": ""}

TOPIC = {
    "H1": "Exchange structure of $\\ke$ and $\\ka$ under symbolic analysis",
    "H2": "The area is insensitive to timing at long windows",
    "H3": "Timing information rises along the ladder of observables",
    "H4": "The weakest Fisher direction is the exchange direction",
    "H5": "Bounded profile intervals: timing under iAUC and trace, $\\SI$ under iAUC",
    "H6": "The gradient diagnostic agrees with the profile likelihood",
    "H7": "Gradient fitting and grid search predict equally well",
    "H8": "The timing parameters add nothing to held-out iAUC error",
    "H9": "A full-trace objective lowers held-out iAUC error",
    "H10": "The diagnostic is robust to initialization, box, scale and optimizer",
    "H11": "$\\tau_1$ and $p$ are identifiable under richer observables",
    "H12": "A tied-rate model predicts as well and identifies $\\tau_1$",
    "H13": "Replica with the model true: bounded $\\SI$ interval",
    "H14": "Replica to real held-out error ratio (noise budgets)",
    "H15": "Replication on the Shanghai and standardized-meal cohorts",
    "H16": "Saturation, carbohydrate-scale and noise sensitivity (exploratory)",
    "H17": "The replica result over five seeds",
    "H18": "$\\tau_1$ and $p$ under the trace, true-model replica",
    "H19": "Synthetic recovery with random truths",
    "H20": "Prediction equivalence on the Shanghai cohort",
    "H21": "The coordinate estimates are likelihood optima",
    "H22": "A second model class (Dalla Man)",
}


def origin(name: str) -> str:
    n = int(name[1:])
    if n <= 10:
        return "plan"
    if n <= 12:
        return "amendment one"
    if n <= 16:
        return "amendment four (post hoc)"
    return "amendment five (post hoc)"


def _esc(text) -> str:
    return (str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
            .replace(">=", r"$\ge$").replace("<=", r"$\le$").replace("->", r"$\to$"))


def _write(name: str, tex: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(tex, encoding="utf-8")


def _pct(x, digits: int = 0) -> str:
    return "--" if x is None else f"{100 * x:.{digits}f}"


# --------------------------------------------------------------------------------------------------

def cohorts_table() -> None:
    from evaluation.cohort_data import cohort_summary, load_cohort
    label = {"cgmacros": "CGMacros", "shanghai": "Shanghai", "hall": "Standardized meals"}
    rows = []
    for cohort, minimum in (("cgmacros", 10), ("shanghai", 10), ("hall", 5)):
        try:
            s = cohort_summary(load_cohort(cohort, min_meals=minimum))
        except Exception:
            continue
        if not s.get("n_subjects"):
            continue
        mp = s["meals_per_subject"]
        rows.append(f"{label[cohort]} & {s['n_subjects']} & {s['n_meals_total']} & "
                    f"{mp['median']:.0f} ({mp['min']:.0f}--{mp['max']:.0f}) & {s['sampling_min']:.0f} & "
                    f"{s['window_min']:.0f} & {s['carbs_g']['median']:.0f} \\\\")
    if not rows:
        return
    tex = ("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lrrcccr}\n\\toprule\n"
           "cohort & subjects & meals & meals per subject & sensor (min) & window (min) & median carbs (g) "
           "\\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n\\caption{The cohorts after the selection rule (at least ten meals per "
           "subject, five for the standardized-meal cohort). Meals per subject: median (minimum--maximum). "
           "The Shanghai sensor samples every fifteen minutes; its stored five-minute grid is interpolated and "
           "trace objectives read the real samples only. Carbohydrate sources are described in the text.}\n"
           "\\label{tab:cohorts}\n\\end{table}\n")
    _write("cohorts.tex", tex)


def cohort_results_table() -> None:
    rows = []
    label = {"cgmacros": "CGMacros", "shanghai": "Shanghai", "hall": "Standardized meals"}
    for cohort in ("cgmacros", "shanghai", "hall"):
        for box in (1.0, 2.0):
            fisher = ladder.a4_results("iauc", box, cohort=cohort)
            profile = profile_lik.summarize({**profile_lik.default_config(), "objective": "iauc",
                                             "bounds_scale": box, "cohort": cohort})
            if not fisher or not profile.get("n_subjects"):
                continue
            n = len(fisher)
            pinned = {p: np.mean([r["fit"]["at_bound"].get(p) is not None for r in fisher.values()])
                      for p in ("insulin_sensitivity", "gastric_emptying", "carb_absorption")}
            cos = np.array([ladder.trade_off_cosine(r["timing_block"]["weak_direction"],
                                                    r["theta_ml"]["gastric_emptying"],
                                                    r["theta_ml"]["carb_absorption"]) for r in fisher.values()])
            par = profile["parameters"]
            interior = profile["interior_for_parameter"]
            rows.append(
                f"{label[cohort]} & ${box:g}\\times$ & {n} & {_pct(pinned['insulin_sensitivity'])} / "
                f"{_pct(pinned['gastric_emptying'])} / {_pct(pinned['carb_absorption'])} & "
                f"{_pct(par['insulin_sensitivity']['fraction'])} / {_pct(par['gastric_emptying']['fraction'])} / "
                f"{_pct(par['carb_absorption']['fraction'])} & {interior['gastric_emptying']} / "
                f"{interior['carb_absorption']} & {_pct(np.mean(cos >= 0.8))} \\\\")
    if not rows:
        return
    tex = ("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{llrcccc}\n\\toprule\n"
           "cohort & box & subjects & estimate on a bound (\\%) & bounded interval (\\%) & interior for $\\ke$ / $\\ka$ & "
           "$|\\cos|\\ge0.8$ (\\%) \\\\\n"
           " & & & $\\SI$ / $\\ke$ / $\\ka$ & $\\SI$ / $\\ke$ / $\\ka$ & & \\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n\\caption{Replication of the identifiability result on three cohorts, "
           "iAUC objective. The last column is the share of subjects whose weakest timing Fisher direction has "
           "an absolute cosine of at least $0.8$ with the exchange direction. Interior counts are subjects whose "
           "estimate of the parameter is more than one percent of the box width from both bounds.}\n"
           "\\label{tab:cohortresults}\n\\end{table}\n")
    _write("cohort_results.tex", tex)


def prediction_compact_table(summary: dict) -> None:
    pred = summary.get("prediction", {}).get("cells")
    if not pred:
        return
    sh = summary.get("phase9", {}).get("h20", {}).get("metrics", {}).get("iauc", {}).get("cell_means", {})
    peak = summary.get("prediction_peak_time", {}).get("cells", {})
    trace = summary.get("prediction_trace_rmse", {}).get("cells", {})
    names = [("grad3", "gradient, three parameters"), ("grid3", "grid search, three parameters"),
             ("grad1", "gradient, $\\SI$ only"), ("grid1", "grid search, $\\SI$ only"),
             ("grad3_trace", "gradient, trace objective"), ("rf", "random forest"), ("snpe", "SNPE"),
             ("personal_mean", "personal mean"), ("population", "population default"),
             ("persistence", "persistence")]
    rows = []
    for key, label in names:
        if key not in pred:
            continue
        rows.append(f"{label} & {pred[key]['mean']:.0f} & {sh[key]:.0f} & " if key in sh else
                    f"{label} & {pred[key]['mean']:.0f} & -- & ")
        rows[-1] += (f"{peak[key]['mean']:.1f} & " if key in peak else "-- & ")
        rows[-1] += (f"{trace[key]['mean']:.1f} \\\\" if key in trace else "-- \\\\")
    tex = ("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lcccc}\n\\toprule\n"
           "cell & iAUC, CGMacros & iAUC, Shanghai & peak time, CGMacros & trace, CGMacros \\\\\n"
           " & (mg\\,dL$^{-1}$\\,min) & (mg\\,dL$^{-1}$\\,min) & (min) & (mg\\,dL$^{-1}$) \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\caption{Held-out mean absolute error (iAUC, "
           "peak time) and root mean squared error (trace) by cell; mean over subjects of the mean over "
           "five repeats of five-fold cross-validation. The Shanghai column is empty where the cell was not run.}\n"
           "\\label{tab:prediction}\n\\end{table}\n")
    _write("prediction_compact.tex", tex)


def scorecard_table(verdicts: dict) -> None:
    rows = [f"{name} & {TOPIC[name]} & {origin(name)} & {_esc(row['status'])} \\\\"
            for name, row in verdicts.items()]
    tex = ("\\begin{table}[t]\n\\centering\\small\n\\begin{tabularx}{\\linewidth}{llXl}\n\\toprule\n"
           "& question & origin & status \\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabularx}\n\\caption{Every hypothesis and its status, computed from the stored "
           "results by the rule written in the plan or the amendment (Appendix~\\ref{app:register}). "
           "Primary endpoints of the plan: H1, H3, H5, H7, H9.}\n\\label{tab:scorecard}\n\\end{table}\n")
    _write("scorecard.tex", tex)


def register_table(verdicts: dict) -> None:
    rows = [f"{name} & {origin(name)} & {_esc(row['rule'])} & {_esc(row['status'])} \\\\"
            for name, row in verdicts.items()]
    tex = ("{\\small\n\\begin{longtable}{llp{0.55\\linewidth}l}\n\\toprule\n& origin & rule & status \\\\\n\\midrule\n"
           "\\endhead\n" + "\n".join(rows) + "\n\\bottomrule\n\\caption{The hypothesis register: rule as written "
           "in the plan or amendment, origin, and the status computed from the stored results.}\n"
           "\\label{tab:register}\n\\end{longtable}\n}\n")
    _write("register.tex", tex)


def noise_budget_table(summary: dict) -> None:
    settings = summary.get("phase8", {}).get("h14", {}).get("settings", {})
    names = [("cgm_off_carb_0", "none", "none"), ("cgm_off_carb_025", "none", "25\\%"),
             ("cgm_on_carb_0", "on", "none"), ("cgm_on_carb_025", "on", "25\\%"),
             ("cgm_on_carb_05", "on", "50\\%")]
    rows = []
    for key, cgm, carb in names:
        d = settings.get(key, {})
        if not d.get("n"):
            continue
        rows.append(f"{cgm} & {carb} & {d['replica_grad3_mae']:.0f} & {d['replica_grid3_mae']:.0f} & "
                    f"{d['replica_personal_mean_mae']:.0f} & {d['ratio']:.2f} "
                    f"({d['ratio_ci95'][0]:.2f}--{d['ratio_ci95'][1]:.2f}) \\\\")
    if not rows:
        return
    real = next(iter(settings.values()), {}).get("real_grad3_mae")
    tex = ("\\begin{table}[h]\n\\centering\\small\n\\begin{tabular}{llcccc}\n\\toprule\n"
           "CGM noise & carbohydrate error & gradient & grid & personal mean & ratio to real \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\caption{What-if noise budgets: held-out iAUC "
           f"error of the replica (mg\\,dL$^{{-1}}$\\,min) under each noise setting. The real-data error of the "
           f"gradient fit is {real:.0f}. The ratio carries a bootstrap interval over subjects. The settings are not "
           "additive shares of the real error.}\n\\label{tab:noise}\n\\end{table}\n")
    _write("noise_budget.tex", tex)


def levels_table(summary: dict) -> None:
    sens = summary.get("phase9", {}).get("delta_sensitivity", {})
    rows = []
    for box in ("0.5x", "1x", "2x"):
        for level, d in sens.get(box, {}).get("levels", {}).items():
            par = d["parameters"]
            si = par["insulin_sensitivity"]
            rows.append(f"${box[:-1]}\\times$ & {level} & {d['delta']:.2f} & {si['interior_bounded']} / "
                        f"{si['interior_n']} ({_pct(si['interior_fraction'])}) & "
                        f"{_pct(par['gastric_emptying']['fraction'])} & "
                        f"{_pct(par['carb_absorption']['fraction'])} \\\\")
    if not rows:
        return
    tex = ("\\begin{table}[h]\n\\centering\\small\n\\begin{tabular}{lcccrr}\n\\toprule\n"
           "box & level (\\%) & rise $\\delta$ & $\\SI$ bounded, interior (\\%) & $\\ke$ bounded (\\%) & "
           "$\\ka$ bounded (\\%) \\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n\\caption{Bounded profile intervals under iAUC at three interval "
           "levels, recomputed from the stored profiles. The rise $\\delta$ is half the corresponding quantile "
           "of $\\chi^2_1$.}\n\\label{tab:levels}\n\\end{table}\n")
    _write("levels.tex", tex)


def correlates_table(summary: dict) -> None:
    corr = summary.get("phase8", {}).get("stored", {}).get("correlates", {}).get("1x", {}).get("covariates", {})
    if not corr:
        return
    cov = {"n_meals": "number of meals", "carb_range": "carbohydrate range", "mean_iauc": "mean observed iAUC",
           "si_estimate": "$\\SI$ estimate"}
    out = {"si_bounded": "$\\SI$ interval bounded", "timing_pinned_upper": "timing rate on the upper bound",
           "si_max_rise": "rise of the $\\SI$ profile"}
    rows = []
    for c, cl in cov.items():
        cells = []
        for o in out:
            d = corr.get(c, {}).get(o, {})
            cells.append("--" if "spearman" not in d else
                         f"{d['spearman']:.2f} ({d['p']:.3f}; {d.get('p_holm', d['p']):.3f})")
        rows.append(f"{cl} & " + " & ".join(cells) + " \\\\")
    tex = ("\\begin{table}[h]\n\\centering\\small\n\\begin{tabular}{lccc}\n\\toprule\n covariate & "
           + " & ".join(out.values()) + " \\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n\\caption{Spearman correlation across subjects at the primary box, "
           "with the raw and the Holm-adjusted $p$-values in parentheses. Exploratory.}\n"
           "\\label{tab:correlates}\n\\end{table}\n")
    _write("correlates.tex", tex)


def main() -> int:
    summary = load_summary()
    verdicts = evaluate_all(summary)
    for fn in (lambda: cohorts_table(), lambda: cohort_results_table(),
               lambda: prediction_compact_table(summary), lambda: scorecard_table(verdicts),
               lambda: register_table(verdicts), lambda: noise_budget_table(summary), lambda: levels_table(summary),
               lambda: correlates_table(summary)):
        fn()
    # the tables of the earlier asset set that the appendix reuses (bounds, profiles, ladder, comparisons)
    for fn in (tables_aistats.bounds_table, tables_aistats.profiles_table, tables_aistats.ladder_table,
               tables_aistats.prediction_table):
        fn(summary)
    print("tables:", ", ".join(sorted(p.name for p in OUT.glob("*.tex"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
