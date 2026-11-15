"""Phase 15 (Amendment 11): a positive control with the symmetry broken by design, a misspecification test of the fixed
parameters, and the likelihood at the truth.

P1  trace profiles in rate coordinates (polished, 500 inner steps) on random-truth replicas with exact carbohydrate, the
    estimated noise and a designed residual intestinal content of 0, 5, 10 and 20% of each meal, seed 0, and 10% on seed 1
H29 at 10% (seed 0): k_e and k_a each bounded in >= 50% of interior subjects and covered by the nominal interval in >= 90%
P2  S_I profiles under iAUC on the standard replica with gamma (insulin secretion) or p3 (insulin action gain) of the truth
    multiplied by 1.3 or 0.7, the fit keeping the population values
P3  the likelihood at the truth (A18) for the trace coordinate profiles with exact carbohydrate at the estimated noise,
    at a quarter of it, and with carbohydrate error 0.25
H30 at exact carbohydrate and a quarter of the noise, the stored polished estimate exceeds the optimum found from the
    truth by less than delta in >= 90% of subjects

Nothing is fitted here. A missing result set gives `{"n": 0}` for its block.

Run:  python -m evaluation.phase15_summary          # writes results/phase15_summary.json
"""
from __future__ import annotations

import json

import numpy as np

from evaluation import profile_lik
from evaluation.identifiability_tools import CHI2_DELTA
from evaluation.phase13_summary import _iauc_config
from evaluation.recalibration import _rows, bounded_among_interior, coverage_at
from evaluation.results_io import RESULTS

RESIDUALS = (0.0, 0.05, 0.10, 0.20)
H29_RESIDUAL, H29_BOUNDED, H29_COVERAGE = 0.10, 0.50, 0.90
H30_FRACTION = 0.90
RATE_KEYS = (("S_I", "insulin_sensitivity"), ("k_e", "gastric_emptying"), ("k_a", "carb_absorption"))
PERTURBATIONS = ({"insulin_secretion": 1.3}, {"insulin_secretion": 0.7}, {"x_gain": 1.3}, {"x_gain": 0.7})
STANDARD_REPLICA = {"seed": 0, "cgm": True, "carb_cv": 0.25}
CALIBRATION = {"exact": {"seed": 0, "cgm": True, "carb_cv": 0.0},
               "quarter": {"seed": 0, "cgm": True, "carb_cv": 0.0, "cgm_scale": 0.25},
               "replica": {"seed": 0, "cgm": True, "carb_cv": 0.25}}


def p1_replica(residual: float, seed: int = 0) -> dict:
    replica = {"seed": seed, "cgm": True, "carb_cv": 0.0, "truth": "random"}
    if residual:
        replica["residual_gut"] = residual
    return replica


def _rates_config(replica: dict) -> dict:
    return {**profile_lik.default_config(), "bounds_scale": 1.0, "objective": "trace", "parameterization": "rates",
            "polish": True, "steps": 500, "replica": replica}


def _covered_of_bounded(rows: list[dict], key: str) -> dict:
    flags = [bool(r["profiles"][key]["truth_covered"]) for r in rows
             if key in r["profiles"] and "truth_covered" in r["profiles"][key]]
    if not flags:
        return {"n": 0, "covered": 0, "fraction": None}
    return {"n": len(flags), "covered": int(sum(flags)), "fraction": sum(flags) / len(flags)}


def rates_block(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for name, key in RATE_KEYS:
        out[name] = {"bounded_interior": bounded_among_interior(rows, key, CHI2_DELTA),
                     "nominal_coverage": coverage_at(rows, key, CHI2_DELTA),
                     "bounded_coverage": _covered_of_bounded(rows, key)}
    return out


def p1() -> dict:
    out = {}
    for residual in RESIDUALS:
        out[f"seed0_{residual:g}"] = rates_block(_rows(_rates_config(p1_replica(residual))))
    out["seed1_0.1"] = rates_block(_rows(_rates_config(p1_replica(0.10, seed=1))))
    return out


def h29(p1_out: dict) -> dict:
    """H29: with the symmetry broken by design (10% residual, exact carbohydrate), k_e and k_a bounded and calibrated."""
    rule = {"residual": H29_RESIDUAL, "bounded": H29_BOUNDED, "coverage": H29_COVERAGE}
    block = p1_out.get(f"seed0_{H29_RESIDUAL:g}", {})
    if not block.get("n"):
        return {"status": "not evaluated", "rule": rule}
    detail, ok = {}, True
    for name in ("k_e", "k_a"):
        frac = block[name]["bounded_interior"].get("fraction")
        cover = block[name]["nominal_coverage"].get("fraction")
        if frac is None or cover is None:
            return {"status": "not evaluated", "rule": rule}
        detail[name] = {"bounded": frac, "coverage": cover}
        ok = ok and frac >= H29_BOUNDED and cover >= H29_COVERAGE
    return {"status": "met" if ok else "not met", "rule": rule, "observed": detail}


def p2() -> dict:
    """S_I under iAUC when a fixed parameter of the truth differs from the population value the fit keeps."""
    out = {}
    sets = {"baseline": STANDARD_REPLICA}
    for perturb in PERTURBATIONS:
        (name, factor), = perturb.items()
        sets[f"{name}_{factor:g}"] = {**STANDARD_REPLICA, "perturb": perturb}
    for tag, replica in sets.items():
        rows = _rows(_iauc_config(replica))
        if not rows:
            out[tag] = {"n": 0}
            continue
        errors = []
        for r in rows:
            truth = (r.get("replica_truth") or {}).get("theta_true")
            if truth:
                errors.append(float(np.log(r["theta_ml"]["insulin_sensitivity"]) - np.log(truth["insulin_sensitivity"])))
        out[tag] = {"n": len(rows),
                    "S_I": {"bounded_interior": bounded_among_interior(rows, "insulin_sensitivity", CHI2_DELTA),
                            "nominal_coverage": coverage_at(rows, "insulin_sensitivity", CHI2_DELTA),
                            "bounded_coverage": _covered_of_bounded(rows, "insulin_sensitivity")},
                    "median_abs_log_error": float(np.median(np.abs(errors))) if errors else None,
                    "median_signed_log_error": float(np.median(errors)) if errors else None}
    return out


def p3() -> dict:
    from evaluation import truth_calibration as tc
    return {tag: tc.summarize({"replica": replica, "steps": 500}) for tag, replica in CALIBRATION.items()}


def h30(p3_out: dict) -> dict:
    """H30: at exact carbohydrate and a quarter of the noise, the stored polished estimate is within delta of the optimum."""
    rule = {"gap": CHI2_DELTA, "fraction_min": H30_FRACTION}
    block = p3_out.get("quarter", {})
    if not block.get("n"):
        return {"status": "not evaluated", "rule": rule}
    ok = block["optimizer_gap_fraction"] >= H30_FRACTION
    return {"status": "met" if ok else "not met", "rule": rule, "fraction": block["optimizer_gap_fraction"],
            "max_gap": block["max_optimizer_gap"]}


def summarize() -> dict:
    first, third = p1(), p3()
    return {"p1": first, "h29": h29(first), "p2": p2(), "p3": third, "h30": h30(third)}


def main() -> int:
    path = RESULTS / "phase15_summary.json"
    path.write_text(json.dumps(summarize(), indent=2, default=float), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
