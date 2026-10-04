"""The verdict on every hypothesis, computed from stored results by the rule written in the plan.

One function per hypothesis, each returning the rule, the observed numbers and a status in
{"met", "not met", "not evaluated", "formulation-dependent"}. Nothing here is typed in from memory: the
status is a comparison of the observed number with the threshold in `PREREG.md` or an amendment, so a
rerun with different results changes the verdicts without anyone editing a sentence.

The plan thresholds appear here once, as constants, with the document that states each.
"""
from __future__ import annotations

import json

from evaluation.results_io import RESULTS

# PREREG.md
H2_LINEAR, H2_NONLINEAR = 0.02, 0.25
H4_COSINE = 0.8
H5_TIMING_IAUC_MAX, H5_TIMING_TRACE_MIN, H5_SI_MIN = 0.25, 0.50, 0.80
H9_MARGIN = 150.0
H7_MARGIN = 150.0
# Amendment 1
H11_TAU_MIN, H11_P_IAUC_MAX, H11_P_TRACE_MIN = 0.70, 0.30, 0.50


def load_summary() -> dict:
    """The Phase 2 summary, with the Phase 8 summary (Amendment 4) under the key `phase8` if present."""
    path = RESULTS / "phase2_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    extra = RESULTS / "phase8_summary.json"
    if extra.exists():
        summary["phase8"] = json.loads(extra.read_text(encoding="utf-8"))
    return summary


def _row(rule: str, status: str, observed: dict | None = None, note: str = "") -> dict:
    return {"rule": rule, "status": status, "observed": observed or {}, "note": note}


def h1(summary):
    return _row("Report the tool output verbatim; if k_e and k_a are globally identifiable, delete all "
                "swap framing.", "formulation-dependent",
                note="Planned formulation: globally identifiable (rule triggered). Empty-gut model: "
                     "exact exchange symmetry, analytically and numerically. See Amendment 3.")


def h2(summary, moment=None):
    if not moment or "h2" not in moment:
        return _row("|d iAUC/d log k| / |d iAUC/d log S_I| <= 0.02 (linear, 6 h); median <= 0.25 "
                    "(nonlinear, 3 h).", "not evaluated")
    h = moment["h2"]
    return _row("Ratio <= 0.02 (linearized, 6 h window); median ratio <= 0.25 (nonlinear, 3 h window).",
                "met" if h["met"] else "not met", h)


def h3(summary):
    block = summary.get("ladder", {}).get("1x", {}).get("h3_own_theta_hat")
    if not block or "h3_met" not in block:
        return _row("Median timing-block condition number falls at each rung; paired Wilcoxon, Holm.",
                    "not evaluated")
    ref = summary["ladder"]["1x"].get("h3_common_reference", {})
    return _row("Median Schur-complement condition number decreases iAUC -> iAUC+centroid -> trace, "
                "paired Wilcoxon per step, Holm, alpha 0.05 (1x box, each objective's own estimate).",
                "met" if block["h3_met"] else "not met",
                {"own_estimate": block["median_condition_number"],
                 "common_reference": ref.get("median_condition_number"),
                 "common_reference_met": ref.get("h3_met")})


def h4(summary):
    cos = summary.get("ladder", {}).get("1x", {}).get("h4_cosine_timing_block", {}).get("iauc_centroid")
    if not cos or "median" not in cos:
        return _row("Median |cos| of the weak timing eigenvector with (1/k_a, -1/k_e) >= 0.8.",
                    "not evaluated")
    return _row("Median |cos| of the weak timing eigenvector with (1/k_a, -1/k_e), iAUC+centroid, >= 0.8.",
                "met" if cos["median"] >= H4_COSINE else "not met", cos)


def h5(summary):
    prof = summary.get("profile_likelihood", {}).get("1x")
    if not prof or not prof.get("n_subjects"):
        return _row("Bounded profile CI: k_e, k_a <= 25% (iAUC), >= 50% (trace); S_I >= 80% among "
                    "interior subjects.", "not evaluated")
    p = prof["parameters"]
    timing = max(p["gastric_emptying"]["fraction"], p["carb_absorption"]["fraction"])
    si = prof.get("S_I_bounded_among_interior_for_S_I", {})
    observed = {"timing_max_fraction_iauc": timing, "S_I_interior": si}
    status_timing = "met" if timing <= H5_TIMING_IAUC_MAX else "not met"
    status_si = ("met" if si and si["fraction"] >= H5_SI_MIN else "not met") if si else "not evaluated"
    trace = summary.get("profile_likelihood_trace", {}).get("1x")
    status_trace = "not evaluated"
    if trace and trace.get("n_subjects"):
        t = trace["parameters"]
        fr = min(t["gastric_emptying"]["fraction"], t["carb_absorption"]["fraction"])
        observed["timing_min_fraction_trace"] = fr
        status_trace = "met" if fr >= H5_TIMING_TRACE_MIN else "not met"
    overall = ("met" if {status_timing, status_si, status_trace} == {"met"}
               else "not met" if "not met" in (status_timing, status_si, status_trace)
               else "not evaluated")
    return _row("k_e and k_a bounded in <= 25% of subjects under iAUC and >= 50% under the trace; S_I "
                "bounded in >= 80% under iAUC among subjects interior for S_I (1x box).", overall,
                observed, note=f"timing/iAUC: {status_timing}; S_I: {status_si}; timing/trace: "
                               f"{status_trace}")


def h6(summary):
    rule = ("Median per-subject Spearman between the gradient diagnostic and the profile-likelihood "
            "ranking >= 0.7; AUC >= 0.85 for detecting profile-flat parameters in the synthetic study.")
    block = summary.get("phase8", {}).get("stored", {}).get("h6", {}).get("1x")
    if not block:
        return _row(rule, "not evaluated", note="Diagnostic-validation analysis not run.")
    median = block["spearman_vs_profile_strength"].get("median")
    observed = {"median_spearman_vs_profile": median, "median_spearman_vs_fisher":
                block["spearman_vs_fisher_information"].get("median"),
                "auc_real_data": block["auc_detecting_flat"]}
    real = median is not None and median >= 0.7
    return _row(rule, "partially evaluated" if real else "not met", observed,
                note="Real-data part only: the synthetic-study AUC (A11) was not run. With three "
                     "parameters per subject a Spearman coefficient takes few values.")


def h7(summary):
    block = summary.get("prediction", {}).get("h7_h8")
    if not block:
        return _row("Gradient vs grid: 90% bootstrap CI inside (-150, +150).", "not evaluated")
    pairs = ("grad3_vs_grid3", "grad1_vs_grid1")
    ok = all(block[k]["equivalent_at"][str(int(H7_MARGIN))] for k in pairs)
    robust = summary.get("steps_sensitivity", {}).get("robust_to_budget")
    return _row("Paired difference in held-out iAUC MAE, gradient vs exhaustive grid (three parameters "
                "and S_I only), 90% CI inside (-150, +150) mg/dL*min.", "met" if ok else "not met",
                {k: block[k] for k in pairs}, note=f"Robust to a 500-step budget: {robust}.")


def h8(summary):
    block = summary.get("prediction", {}).get("h7_h8", {}).get("grad3_vs_grad1")
    if not block:
        return _row("Gradient (3 parameters) vs gradient (S_I only), equivalence within 150.",
                    "not evaluated")
    return _row("Gradient three parameters vs S_I only: 90% CI inside (-150, +150).",
                "met" if block["equivalent_at"][str(int(H7_MARGIN))] else "not met", block)


def h9(summary):
    block = summary.get("prediction", {}).get("h9", {}).get("grad3_trace_vs_grad3")
    if not block:
        return _row("Full-trace fit lowers held-out iAUC MAE by more than 150.", "not evaluated")
    return _row("A full-trace fit lowers held-out iAUC MAE relative to an iAUC fit by more than 150 "
                "mg/dL*min (paired Wilcoxon, Holm).",
                "met" if block["lowers_error_by_more_than_margin"] and (block.get("p_holm") or 1) < 0.05
                else "not met", block, note="If not met: ceiling persists across observables.")


def h10(summary):
    return _row("Kendall tau >= 0.8 across 5 inits, 3 bound settings, parameterizations, optimizers; "
                "S_I ranked first in >= 95% of interior subjects.", "not evaluated",
                note="Robustness sweeps (Phase 5) not run.")


def h11(summary):
    rule = ("tau1 CI bounded in >= 70% (iAUC+centroid and trace); p bounded <= 30% (iAUC+centroid), "
            ">= 50% (trace); among subjects interior for the parameter.")
    block = summary.get("phase8", {}).get("h11")
    if not block or block.get("status") in (None, "not evaluated"):
        return _row(rule, "not evaluated", note="Profile in (log S_I, log tau1, log p) not available.")
    return _row(rule, block["status"], block.get("clauses", {}))


def h12(summary):
    rule = ("Tied-rate model within +/-150 of the three-parameter model (90% CI); tau1 bounded in >= 70% "
            "under iAUC+centroid.")
    block = summary.get("phase8", {}).get("h12")
    if not block or block.get("status") in (None, "not evaluated"):
        return _row(rule, "not evaluated", note="Tied-rate prediction cells or profile not available.")
    comp = block["comparisons"].get("iauc:grad2_tied-grad3", {})
    return _row(rule, block["status"], {"difference_vs_grad3": comp.get("mean_difference"),
                                        "ci95": comp.get("ci95"),
                                        "tau1_interior_bounded": block.get("tau1_tied_interior")},
                note=f"Coordinate fits on the tied boundary: {block.get('boundary')}")


def h13(summary):
    rule = ("Replica with the model true: S_I interval bounded in >= 50% of interior subjects, k_e and "
            "k_a in <= 25%.")
    block = summary.get("phase8", {}).get("h13")
    if not block or not block.get("n"):
        return _row(rule, "not evaluated", note="Replica profile not run.")
    met = block["S_I_clause_met"] and block["timing_clause_met"]
    return _row(rule, "met" if met else "not met",
                {"S_I_interior_bounded": block["S_I_interior_bounded"],
                 "timing_max_fraction": block["timing_max_fraction"]}, note=block["reading"])


def h14(summary):
    rule = ("Replica held-out iAUC MAE / real MAE: <= 0.7 substantial part unexplained by noise or "
            "optimization; >= 0.9 noise accounts for almost all; between, mixed.")
    block = summary.get("phase8", {}).get("h14")
    if not block or "main_ratio" not in block:
        return _row(rule, "not evaluated", note="Replica prediction cross-validation not run.")
    return _row(rule, "reported (no pass/fail)", {"ratio": block["main_ratio"]}, note=block["reading"])


def h15(summary):
    rule = ("Shanghai and Hall (iAUC, primary box): k_e, k_a bounded in <= 25%, weak Fisher direction "
            "cosine >= 0.8 with the exchange direction in >= 80% of subjects. S_I: no threshold.")
    block = summary.get("phase8", {}).get("h15")
    if not block or block.get("status") in (None, "not evaluated"):
        return _row(rule, "not evaluated", note="Replication on the other cohorts not run.")
    observed = {c: {"timing_max_bounded_fraction": block[c]["1x"].get("timing_max_bounded_fraction"),
                    "cosine_fraction": block[c]["1x"]["fisher"].get("cosine_fraction_at_least_0.8")}
                for c in ("shanghai", "hall")}
    return _row(rule, block["status"], observed)


def h16(summary):
    return _row("Exploratory: S_I saturation curve; carbohydrate-scale sensitivity; heteroskedastic noise "
                "(no thresholds).", "reported (exploratory)" if summary.get("phase8", {}).get("h16", {})
                .get("saturation", {}).get("n") else "not evaluated",
                note="Heteroskedastic-noise sensitivity was not run.")


ALL = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6, "H7": h7, "H8": h8, "H9": h9,
       "H10": h10, "H11": h11, "H12": h12, "H13": h13, "H14": h14, "H15": h15, "H16": h16}
PRIMARY = ("H1", "H3", "H5", "H7", "H9")


def evaluate_all(summary: dict | None = None, moment: dict | None = None) -> dict:
    summary = load_summary() if summary is None else summary
    if moment is None:
        try:
            from evaluation.moment_checks import summarize
            moment = summarize()
        except Exception:
            moment = None
    out = {}
    for name, fn in ALL.items():
        out[name] = (fn(summary, moment) if name == "H2" else fn(summary))
        out[name]["primary"] = name in PRIMARY
    return out
