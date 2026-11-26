"""Phase 21 (Amendment 17): converged-solver sweeps, one coverage definition, pseudo-true Dalla Man, numerical checks.

Reads stored results only. Writes `results/phase21_summary.json` and `paper/amendment17_macros.tex` (a macro file of
`\\providecommand` lines, like the Amendment 16 macros; every number in the text that comes from this phase is one of them).

R26  coordinate profiles with the converged inner solver (L-BFGS-B at every grid point), a 29-point grid and polished
     estimates, hull coordinates (log S_I, log tau1, log p), real-pole projection respecting the fixed coordinate. Per setting
     (replica truth: fitted or random; carbohydrate error) and coordinate, the metrics of `coverage_metrics`;
     bounded counts among interior subjects and overall; and the same on the real CGMacros data.
R27  the pseudo-true tau1 of the Bergman-type fit to noise-free Dalla Man output (A28): its bias against the Dalla Man mean
     appearance time, coverage of the pseudo-true value by the A27 intervals, and rank agreement.
R28  the Fisher spectrum recomputed at a solver tolerance of 1e-8 for all 45 subjects (A17 with n_subjects = 45).
R29  the iAUC profiles at a box four times as wide for k_e and k_a (primary and 2x counts included).

Run:  python -m evaluation.phase21_summary
"""
from __future__ import annotations

import json
import math

import numpy as np

from evaluation import coverage_metrics as cm
from evaluation import profile_lik
from evaluation.identifiability_tools import CHI2_DELTA
from evaluation.recalibration import _rows
from evaluation.results_io import RESULTS, ROOT

OUT = RESULTS / "phase21_summary.json"
MACROS = ROOT / "paper" / "amendment17_macros.tex"
CV_WORD = {0.0: "Zero", 0.1: "Tenth", 0.25: "Quarter", 0.5: "Half", 0.75: "ThreeQuarter"}
INTERIOR = 0.01          # fraction of the box width, the paper's definition of an interior estimate


def _config(replica: dict | None, **extra) -> dict:
    return {**profile_lik.default_config(), "bounds_scale": 1.0, "objective": "trace", "parameterization": "coords",
            "polish": True, "grid_points": 29, "inner_solver": "lbfgsb", "replica": replica, **extra}


def _setting(replica: dict | None) -> dict:
    rows = _rows(_config(replica))
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for key, name in (("log_tau1", "tau1"), ("log_p", "p")):
        interior = [r for r in rows if r["distance_to_bound"][key] > INTERIOR]
        bounded_int = sum(r["profiles"][key]["classification"]["verdict"] == "identifiable" for r in interior)
        bounded_all = sum(r["profiles"][key]["classification"]["verdict"] == "identifiable" for r in rows)
        block = {"interior": len(interior), "bounded_interior": int(bounded_int), "bounded_all": int(bounded_all)}
        if replica is not None:
            block.update(cm.metrics(rows, key))
        out[name] = block
    return out


def settings() -> dict:
    out = {"real": _setting(None)}
    for truth, tag in (("fit", "fit"), ("random", "rand")):
        for cv in CV_WORD:
            replica = {"seed": 0, "cgm": True, "carb_cv": cv}
            if truth == "random":
                replica["truth"] = "random"
            out[f"{tag}_{cv}"] = _setting(replica)
    return out


def pseudo_true() -> dict:
    path = RESULTS / "phase20_amendment16" / "a28_dm_pseudo_true.json"
    if not path.exists():
        return {"n": 0}
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    pt = np.array([r["pseudo_tau1"] for r in rows], dtype=float)
    dm = np.array([r["true_tau1_dm"] for r in rows], dtype=float)
    dm_td = np.array([r["true_tau1_plus_td"] for r in rows], dtype=float)
    hat = np.array([r["hat_tau1"] for r in rows], dtype=float)
    bounded = [r for r in rows if r["tau1_verdict"] == "identifiable"]
    cover_pt = sum(r["tau1_ci"][0] <= r["pseudo_tau1"] <= r["tau1_ci"][1] for r in bounded)
    cover_dm = sum(r["tau1_ci"][0] <= r["true_tau1_dm"] <= r["tau1_ci"][1] for r in bounded)
    from scipy.stats import spearmanr
    tied = sum(abs(math.log(r["pseudo_ke"] / r["pseudo_ka"])) < 1e-3 for r in rows)
    return {"n": len(rows), "bounded": len(bounded), "bounded_covers_pseudo": int(cover_pt), "bounded_covers_dm": int(cover_dm),
            "median_bias_dm": float(np.median(pt - dm)), "median_bias_dm_td": float(np.median(pt - dm_td)),
            "median_gap_estimate_pseudo": float(np.median(hat - pt)), "median_abs_gap_estimate_pseudo": float(np.median(np.abs(hat - pt))),
            "median_log_width": float(np.median([math.log(r["tau1_ci"][1] / r["tau1_ci"][0]) for r in bounded])) if bounded else None,
            "rho_pseudo_dm": float(spearmanr(pt, dm)[0]), "rho_estimate_pseudo": float(spearmanr(hat, pt)[0]),
            "tied_pseudo": int(tied), "median_pseudo_tau1": float(np.median(pt)), "median_dm_tau1": float(np.median(dm))}


def box_four() -> dict:
    out = {}
    for scale, tag in ((1.0, "One"), (2.0, "Two"), (4.0, "Four")):
        s = profile_lik.summarize({**profile_lik.default_config(), "objective": "iauc", "bounds_scale": scale})
        if not s.get("n_subjects"):
            continue
        interior = s["interior_for_parameter"]
        bounded = {k: v.get("n_bounded") for k, v in s["parameters"].items()}
        out[tag] = {"n": s["n_subjects"], "interior": interior, "fraction": {k: v.get("fraction") for k, v in s["parameters"].items()},
                    "bounded": bounded}
    return out


def numerics() -> dict:
    from evaluation import numerics_check
    out = numerics_check.summarize()
    out.pop("profile_steps", None)
    return out


def summarize() -> dict:
    return {"r26": settings(), "r27": pseudo_true(), "r28": numerics(), "r29": box_four()}


def _put(m: dict, name: str, value, fmt: str) -> None:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return
    m[name] = format(value, fmt)


def _paper_macro(name: str) -> float | None:
    import re
    text = (ROOT / "paper" / "results_macros.tex").read_text(encoding="utf-8")
    hit = re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", text)
    try:
        return float(hit.group(1)) if hit else None
    except ValueError:
        return None


def macros(d: dict) -> dict:
    m: dict = {}
    mae, gain, mae_mean = (_paper_macro(n) for n in ("resMaeGradThree", "resDiffGradThreeVsPersonalMean", "resMaePersonalMean"))
    if mae and gain and mae_mean:
        exponent = int(math.floor(math.log10(mae ** 2)))
        m["resPenaltyLossScale"] = f"{mae ** 2 / 10 ** exponent:.1f}" + chr(92) + f"times10^{{{exponent}}}"
        m["resMarginShareGain"] = format(100 * 150 / abs(gain), ".0f")
        m["resGainPct"] = format(100 * abs(gain) / mae_mean, ".0f")
    real = d["r26"].get("real", {})
    for name, tag in (("tau1", "Tau"), ("p", "P")):
        b = real.get(name)
        if not b:
            continue
        _put(m, f"resCRealN{tag}Int", b["interior"], "d")
        _put(m, f"resCRealN{tag}Bd", b["bounded_interior"], "d")
        _put(m, f"resCRealN{tag}BdAll", b["bounded_all"], "d")
        if b["interior"]:
            _put(m, f"resCRealPct{tag}", 100 * b["bounded_interior"] / b["interior"], ".0f")
        _put(m, f"resCRealPctAll{tag}", 100 * b["bounded_all"] / real["n"], ".0f")
    if real.get("n"):
        _put(m, "resCRealN", real["n"], "d")
    for key, setting in d["r26"].items():
        if key == "real" or not setting.get("n"):
            continue
        truth, cv = key.split("_")
        word = ("Fit" if truth == "fit" else "Rand") + CV_WORD[float(cv)]
        for name, tag in (("tau1", "Tau"), ("p", "P")):
            b = setting[name]
            pre = f"resC{word}{tag}"
            _put(m, pre + "N", setting["n"], "d")
            _put(m, pre + "Int", b["interior"], "d")
            _put(m, pre + "BdInt", b["bounded_interior"], "d")
            if b["interior"]:
                _put(m, pre + "BdIntPct", 100 * b["bounded_interior"] / b["interior"], ".0f")
            _put(m, pre + "Bd", b["bounded_all"], "d")
            _put(m, pre + "BdPct", 100 * b["bounded_all"] / setting["n"], ".0f")
            if "covers" in b:
                _put(m, pre + "CoverPct", 100 * b["covers"] / b["n"], ".0f")
                _put(m, pre + "Narrow", b["narrow"], "d")
                _put(m, pre + "NarrowCovers", b["narrow_covers"], "d")
                if b["covers_given_narrow"] is not None:
                    _put(m, pre + "NarrowCoverPct", 100 * b["covers_given_narrow"], ".0f")
                _put(m, pre + "Score", b["median_score"], ".2f")
                _put(m, pre + "Width", b["median_log_width"], ".2f")
                _put(m, pre + "BoxRange", b["box_log_range"], ".2f")
    pt = d["r27"]
    if pt.get("n"):
        for k, name, fmt in (("n", "N", "d"), ("bounded", "Bounded", "d"), ("bounded_covers_pseudo", "CoversPseudo", "d"),
                             ("bounded_covers_dm", "CoversDm", "d"), ("median_bias_dm", "Bias", ".1f"),
                             ("median_bias_dm_td", "BiasTd", ".1f"), ("median_gap_estimate_pseudo", "Gap", ".1f"),
                             ("median_abs_gap_estimate_pseudo", "AbsGap", ".1f"), ("median_log_width", "Width", ".2f"),
                             ("rho_pseudo_dm", "RhoPseudoDm", ".2f"), ("rho_estimate_pseudo", "RhoEstPseudo", ".2f"),
                             ("tied_pseudo", "Tied", "d"), ("median_pseudo_tau1", "MedianPseudo", ".1f"),
                             ("median_dm_tau1", "MedianDm", ".1f")):
            _put(m, "resPt" + name, pt[k], fmt)
    tol = d["r28"]
    if tol.get("n"):
        _put(m, "resTolN", tol["n"], "d")
        _put(m, "resTolMedianRelChange", 100 * tol["median_relative_eigenvalue_change_tight"], ".1f")
        _put(m, "resTolMaxRelChange", 100 * tol["max_relative_eigenvalue_change_tight"], ".0f")
        _put(m, "resTolMinCosine", tol["min_abs_cosine_tight"], ".3f")
    for tag, block in d["r29"].items():
        for param, word in (("gastric_emptying", "Ke"), ("carb_absorption", "Ka"), ("insulin_sensitivity", "SI")):
            _put(m, f"resBox{tag}{word}Int", block["interior"][param], "d")
            frac = block["fraction"][param]
            if frac is not None:
                _put(m, f"resBox{tag}{word}BdAllCount", round(frac * block["n"]), "d")
    return m


def main() -> int:
    d = summarize()
    OUT.write_text(json.dumps(d, indent=2, default=float), encoding="utf-8")
    m = macros(d)
    lines = ["% Generated by evaluation/phase21_summary.py from results/phase21_summary.json -- DO NOT EDIT.", ""]
    lines += [f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    MACROS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} and {len(m)} macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
