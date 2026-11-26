"""Generate `paper/amendment16_macros.tex` from `results/phase20_amendment16/*.json`.

Keeps every Amendment 16 scalar macro-driven so `evaluation.lint_numbers` stays at 0 hand-typed numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.results_io import RESULTS, ROOT

IN_DIR = RESULTS / "phase20_amendment16"
OUT_PATH = ROOT / "paper" / "amendment16_macros.tex"


def _load(name: str) -> dict:
    return json.loads((IN_DIR / f"{name}.json").read_text(encoding="utf-8"))


def main() -> int:
    r25 = _load("r25_stored_mechanism")
    a21 = _load("a21_neg_control")
    a23 = _load("a23_hall_trace")
    a24 = _load("a24_sandwich_coverage")
    a25 = _load("a25_feasible_domain")
    a26 = _load("a26_exact_domain")
    a27 = _load("a27_misspecified_dalla_man")
    w60 = _load("a22_window_60")["summary"]
    w90 = _load("a22_window_90")["summary"]
    w120 = _load("a22_window_120")["summary"]

    m: dict[str, str] = {}

    # --- R25: Fixed-subject & mechanism check for 6/45 -> 28/45 ---
    for coord, tag in (("tau1", "Tau"), ("p", "P")):
        b = r25[coord]
        p0, p75 = b["per_cv"]["0.0"], b["per_cv"]["0.75"]
        m[f"resRtfOneSidedZero{tag}"] = f"{p0['one_sided']:.0f}"
        m[f"resRtfFlatZero{tag}"] = f"{p0['flat']:.0f}"
        m[f"resRtfBoundedZeroAll{tag}"] = f"{p0['bounded_all']:.0f}"
        m[f"resRtfBoundedHighAll{tag}"] = f"{p75['bounded_all']:.0f}"
        m[f"resRtfCommonN{tag}"] = f"{p0['fixed_common_n']:.0f}"
        m[f"resRtfCommonZero{tag}"] = f"{p0['fixed_common_bounded']:.0f}"
        m[f"resRtfCommonZeroPct{tag}"] = f"{p0['fixed_common_pct']:.0f}"
        m[f"resRtfCommonHigh{tag}"] = f"{p75['fixed_common_bounded']:.0f}"
        m[f"resRtfCommonHighPct{tag}"] = f"{p75['fixed_common_pct']:.0f}"
        m[f"resRtfBiasZero{tag}"] = f"{p0['median_signed_log_bias']:+.2f}"
        m[f"resRtfBiasQuarter{tag}"] = f"{b['per_cv']['0.25']['median_signed_log_bias']:+.2f}"
        m[f"resRtfBiasHigh{tag}"] = f"{p75['median_signed_log_bias']:+.2f}"
        m[f"resRtfDistZero{tag}"] = f"{p0['median_dist_to_bound']:.2f}"
        m[f"resRtfDistHigh{tag}"] = f"{p75['median_dist_to_bound']:.2f}"
        m[f"resRtfShift{tag}"] = f"{b['median_log_shift_0_to_075']:+.2f}"
        m[f"resRtfUpPct{tag}"] = f"{100.0 * b['frac_upward_shift_0_to_075']:.0f}"
        m[f"resRtfNewBd{tag}"] = f"{b['transitions_0_to_075']['one_sided_to_bounded']:.0f}"
        m[f"resRtfDistGainPct{tag}"] = f"{100.0 * b['frac_dist_gain_positive']:.0f}"

    # --- A21 (H39): Negative control (deliberately wrong corner gut rates) ---
    m["resVerdictHThirtyNine"] = a21["h39_status"]
    for s, stag in (("0", "Zero"), ("1", "One")):
        ss = a21["seeds"][s]["summary"]
        for cname, ctag in (("slow_corner", "Slow"), ("fast_corner", "Fast")):
            for mode, mtag in (("fixed_si", "Fixed"), ("refit_si", "Refit")):
                blk = ss[f"{cname}_{mode}"]
                m[f"resNeg{stag}{ctag}{mtag}Iauc"] = f"{blk['iauc_ratio']['median']:.2f}"
                m[f"resNeg{stag}{ctag}{mtag}Peak"] = f"{blk['peak_ratio']['median']:.2f}"
                m[f"resNeg{stag}{ctag}{mtag}Trace"] = f"{blk['trace_ratio']['median']:.2f}"

    # --- A22 (H40): CGMacros iAUC window sweep (60, 90, 120 min) ---
    h40_met = all(
        w[p]["bounded_all_pct"] <= 25.0
        for w in (w60, w90, w120)
        for p in ("k_e", "k_a")
    )
    m["resVerdictHForty"] = "met" if h40_met else "not met"
    for wdict, wtag in ((w60, "Sixty"), (w90, "Ninety"), (w120, "OneTwenty")):
        for pname, ptag in (("S_I", "SI"), ("k_e", "Ke"), ("k_a", "Ka")):
            blk = wdict[pname]
            m[f"resWinProf{wtag}{ptag}Bd"] = f"{blk['bounded']:.0f}"
            m[f"resWinProf{wtag}{ptag}Int"] = f"{blk['interior']:.0f}"
            m[f"resWinProf{wtag}{ptag}Pct"] = f"{blk['bounded_all_pct']:.0f}"

    # --- A23 (H41): Exploratory Hall (n=15) trace coordinate profiles ---
    hs = a23["summary"]
    m["resVerdictHFortyOne"] = a23["h41_status"]
    m["resHallTraceN"] = f"{hs['n']:.0f}"
    for pname, ptag in (("S_I", "SI"), ("tau1", "Tau"), ("p", "P")):
        blk = hs[pname]
        m[f"resHallTrace{ptag}Bd"] = f"{blk['bounded']:.0f}"
        m[f"resHallTrace{ptag}Int"] = f"{blk['interior']:.0f}"
        m[f"resHallTrace{ptag}IntPct"] = f"{blk['bounded_int_pct']:.0f}"
        m[f"resHallTrace{ptag}AllPct"] = f"{blk['bounded_all_pct']:.0f}"

    # --- A24 (H42): Meal-clustered sandwich variance remedy ---
    m["resVerdictHFortyTwo"] = a24["h42_status"]
    for cv, cvtag in (("0.0", "Zero"), ("0.25", "Quarter"), ("0.75", "ThreeQuarter")):
        cs = a24["cvs"][cv]["summary"]
        for pname, ptag in (("tau1", "Tau"), ("p", "P")):
            blk = cs[pname]
            m[f"resSand{cvtag}{ptag}Infl"] = f"{blk['median_inflation']:.2f}"
            m[f"resSand{cvtag}{ptag}NaiveCov"] = f"{blk['naive_cov_pct']:.0f}"
            m[f"resSand{cvtag}{ptag}ProfCov"] = f"{blk['sand_prof_cov_pct']:.0f}"
            m[f"resSand{cvtag}{ptag}WaldCov"] = f"{blk['sand_wald_cov_pct']:.0f}"
            m[f"resSand{cvtag}{ptag}NaiveBd"] = f"{blk['naive_bounded_all']:.0f}"
            m[f"resSand{cvtag}{ptag}ProfBd"] = f"{blk['sand_bounded_all']:.0f}"

    # --- A25: Real-pole log_p profiling & exact feasible-domain D endpoint audit ---
    for skey, stag in (
        ("cgmacros_real", "Real"),
        ("replica_cv0", "Zero"),
        ("replica_cv025", "Quarter"),
        ("replica_cv075", "ThreeQuarter"),
    ):
        t = a25[skey]["tau1"]
        p = a25[skey]["p_real_pole"]
        m[f"resFeas{stag}TauBd"] = f"{t['rect_bounded']:.0f}"
        m[f"resFeas{stag}TauHatInD"] = f"{t['hat_in_D']:.0f}"
        m[f"resFeas{stag}TauBothInD"] = f"{t['both_ep_in_D']:.0f}"
        m[f"resFeas{stag}TauBothInDPct"] = f"{t['both_ep_in_D_of_bd_pct']:.0f}"
        m[f"resFeas{stag}TauEpOut"] = f"{t['ep_outside_D']:.0f}"
        m[f"resFeas{stag}TauEpTot"] = f"{t['ep_total']:.0f}"
        m[f"resFeas{stag}TauEpOutPct"] = f"{t['ep_outside_D_pct']:.0f}"

        m[f"resFeas{stag}PInt"] = f"{p['n_interior']:.0f}"
        m[f"resFeas{stag}POldBd"] = f"{p['old_bounded']:.0f}"
        m[f"resFeas{stag}PNewBd"] = f"{p['real_pole_bounded']:.0f}"
        m[f"resFeas{stag}PNewIntPct"] = f"{p['real_pole_bounded_int_pct']:.0f}"
        m[f"resFeas{stag}PNewAllPct"] = f"{p['real_pole_bounded_all_pct']:.0f}"
        m[f"resFeas{stag}PBothInD"] = f"{p['both_ep_in_D']:.0f}"
        m[f"resFeas{stag}PBothInDPct"] = f"{p['both_ep_in_D_of_bd_pct']:.0f}"
        m[f"resFeas{stag}PEpOutPct"] = f"{p['ep_outside_D_pct']:.0f}"
        if p["cov_nom_pct"] is not None:
            m[f"resFeas{stag}PCovNom"] = f"{p['cov_nom_pct']:.0f}"
            m[f"resFeas{stag}PCovBd"] = f"{p['cov_bd_pct']:.0f}"

    # --- A26: Strictly feasible original-box D full-trace profiles ---
    for skey, stag in (
        ("cgmacros_real", "Real"),
        ("hall_real", "Hall"),
        ("replica_cv0", "Zero"),
        ("replica_cv025", "Quarter"),
        ("replica_cv075", "ThreeQuarter"),
    ):
        for coord, ctag in (("tau1", "Tau"), ("p", "P")):
            blk = a26[skey][coord]
            m[f"resExactD{stag}{ctag}N"] = f"{blk['n']:.0f}"
            m[f"resExactD{stag}{ctag}Int"] = f"{blk['n_interior']:.0f}"
            m[f"resExactD{stag}{ctag}BdAll"] = f"{blk['bounded_all']:.0f}"
            m[f"resExactD{stag}{ctag}BdInt"] = f"{blk['bounded_int']:.0f}"
            m[f"resExactD{stag}{ctag}AllPct"] = f"{blk['bounded_all_pct']:.0f}"
            m[f"resExactD{stag}{ctag}IntPct"] = f"{blk['bounded_int_pct']:.0f}"
            m[f"resExactD{stag}{ctag}Infeas"] = f"{blk['total_infeasible_grid_points']:.0f}"
            if blk["cov_nom_pct"] is not None:
                m[f"resExactD{stag}{ctag}CovNom"] = f"{blk['cov_nom_pct']:.0f}"
                m[f"resExactD{stag}{ctag}CovBd"] = f"{blk['cov_bd_pct']:.0f}"

    # --- A27: Misspecified-generator control (Dalla Man -> Bergman, strictly inside D) ---
    import numpy as np
    from scipy.stats import spearmanr
    s27 = a27["summary"]
    subs27 = a27["subjects"]
    rho_dm = float(spearmanr([x["true_tau1_dm"] for x in subs27], [x["hat_tau1"] for x in subs27]).statistic)
    rho_td = float(spearmanr([x["true_tau1_plus_td"] for x in subs27], [x["hat_tau1"] for x in subs27]).statistic)
    med_ci_w = float(np.median([
        x["tau1"]["ci_high"] - x["tau1"]["ci_low"]
        for x in subs27 if x["tau1"]["verdict"] == "identifiable"
    ]))
    m["resMisspecN"] = f"{s27['n']:.0f}"
    m["resMisspecTrueTauMed"] = f"{s27['true_tau1_dm_median']:.1f}"
    m["resMisspecHatTauMed"] = f"{s27['hat_tau1_median']:.1f}"
    m["resMisspecBiasDm"] = f"{s27['median_bias_dm_min']:+.1f}"
    m["resMisspecBiasTd"] = f"{s27['median_bias_td_min']:+.1f}"
    m["resMisspecCiWidthMin"] = f"{med_ci_w:.1f}"
    m["resMisspecTauBd"] = f"{s27['tau1_bounded_all']:.0f}"
    m["resMisspecTauPct"] = f"{s27['tau1_bounded_int_pct']:.0f}"
    m["resMisspecPBd"] = f"{s27['p_bounded_all']:.0f}"
    m["resMisspecPPct"] = f"{s27['p_bounded_int_pct']:.0f}"
    m["resMisspecCovDm"] = f"{s27['tau1_cov_nom_dm_pct']:.0f}"
    m["resMisspecCovTd"] = f"{s27['tau1_cov_nom_td_pct']:.0f}"
    m["resMisspecRhoDm"] = f"{rho_dm:.2f}"
    m["resMisspecRhoTd"] = f"{rho_td:.2f}"

    lines = [
        "% Generated by evaluation/make_amendment16_macros.py from results/phase20_amendment16/*.json -- DO NOT EDIT.",
        "",
    ]
    for k in sorted(m):
        lines.append(f"\\providecommand{{\\{k}}}{{{m[k]}}}")
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(m)} macros to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
