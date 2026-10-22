"""H24 (Amendment 7): is the replica noise too large, and how well were the coordinate cutoffs determined?

The replica draws CGM noise from the subject's residuals around the real fit, which contain model misfit as
well as sensor noise, so it is an upper bound on sensor noise. If that is why the real data show more bounded
coordinate intervals than the replica does, a replica with a smaller noise scale should approach the real-data
fractions, and the nominal interval should then cover the truth. This module reads the stored profile results
of the replicas at scales 1, 0.5 and 0.25 and applies the rule of the amendment. Nothing is fitted here.

It also pools replica seeds 0 to 2 for the coordinate calibration of H23, report only.

Run:  python -m evaluation.noise_scale
"""
from __future__ import annotations

import json

from evaluation import profile_lik
from evaluation.identifiability_tools import CHI2_DELTA
from evaluation.recalibration import (bounded_among_interior, calibrated_cutoff, coverage_at, _rows)
from evaluation.results_io import RESULTS

SCALES = (1.0, 0.5, 0.25)
TAU1_MIN, P_MIN, COVERAGE_MIN = 0.73, 0.78, 0.90
POOLED_SEEDS = (0, 1, 2)


def replica(seed: int, scale: float = 1.0, carb_cv: float = 0.25) -> dict:
    out = {"seed": seed, "cgm": True, "carb_cv": carb_cv}
    if scale != 1.0:
        out["cgm_scale"] = scale
    return out


def _coords(extra: dict) -> dict:
    return {**profile_lik.default_config(), "bounds_scale": 1.0, "objective": "trace",
            "parameterization": "coords", "polish": True, **extra}


def _trace_block(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for name, key in (("tau1", "log_tau1"), ("p", "log_p"), ("S_I", "log_insulin_sensitivity")):
        interior = bounded_among_interior(rows, key, CHI2_DELTA)
        cover = coverage_at(rows, key, CHI2_DELTA)
        block = profile_lik.summarize_rows(rows, _coords({}))["parameters"][key].get("truth_coverage_of_bounded")
        out[name] = {"bounded_interior": interior, "nominal_coverage": cover, "bounded_coverage": block,
                     "cutoff": calibrated_cutoff(rows, key)}
    return out


def _iauc_block(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    config = {**profile_lik.default_config(), "objective": "iauc", "bounds_scale": 1.0}
    block = profile_lik.summarize_rows(rows, config)
    return {"n": len(rows),
            "S_I_interior": block.get("S_I_bounded_among_interior_for_S_I"),
            "timing_bounded": {k: block["parameters"][k]["fraction"]
                               for k in ("gastric_emptying", "carb_absorption")},
            "S_I_nominal_coverage": coverage_at(rows, "insulin_sensitivity", CHI2_DELTA)}


def h24() -> dict:
    base = {**profile_lik.default_config(), "bounds_scale": 1.0}
    real = _rows(_coords({}))
    out = {"scales": {}, "real_polished": {
        "tau1": bounded_among_interior(real, "log_tau1", CHI2_DELTA),
        "p": bounded_among_interior(real, "log_p", CHI2_DELTA)},
        "rule": {"tau1_min": TAU1_MIN, "p_min": P_MIN, "nominal_coverage_min": COVERAGE_MIN}}
    met_at = []
    for scale in SCALES:
        trace = _trace_block(_rows(_coords({"replica": replica(0, scale)})))
        iauc = _iauc_block(_rows({**base, "objective": "iauc", "replica": replica(0, scale)}))
        out["scales"][f"{scale:g}"] = {"trace": trace, "iauc": iauc}
        if scale != 1.0 and trace.get("n"):
            try:
                ok = (trace["tau1"]["bounded_interior"]["fraction"] >= TAU1_MIN
                      and trace["p"]["bounded_interior"]["fraction"] >= P_MIN
                      and trace["tau1"]["nominal_coverage"]["fraction"] >= COVERAGE_MIN
                      and trace["p"]["nominal_coverage"]["fraction"] >= COVERAGE_MIN)
            except (KeyError, TypeError):
                ok = False
            if ok:
                met_at.append(scale)
    graded = [s for s in out["scales"] if s != "1" and out["scales"][s]["trace"].get("n")]
    if len(graded) < 2:
        out["status"] = "not evaluated"
    else:
        out["met_at_scales"] = met_at
        out["status"] = "met" if met_at else "not met"
    return out


def h25() -> dict:
    """H25 (Amendment 8): is the undercoverage of the nominal interval caused by the carbohydrate logging error?

    Replica seed 0 with exact carbohydrate (CV 0) at CGM noise scales 1 and 0.25. The verdict is read at scale 1:
    under the trace the nominal interval must contain the true value for at least COVERAGE_MIN of the subjects,
    for both tau1 and p. The other cell and the iAUC results are reported.
    """
    base = {**profile_lik.default_config(), "bounds_scale": 1.0}
    out = {"scales": {}, "rule": {"nominal_coverage_min": COVERAGE_MIN, "read_at_scale": 1.0}}
    for scale in (1.0, 0.25):
        trace = _trace_block(_rows(_coords({"replica": replica(0, scale, 0.0)})))
        iauc = _iauc_block(_rows({**base, "objective": "iauc", "replica": replica(0, scale, 0.0)}))
        out["scales"][f"{scale:g}"] = {"trace": trace, "iauc": iauc}
    first = out["scales"]["1"]["trace"]
    if not first.get("n") or not out["scales"]["0.25"]["trace"].get("n"):
        out["status"] = "not evaluated"
        return out
    try:
        ok = (first["tau1"]["nominal_coverage"]["fraction"] >= COVERAGE_MIN
              and first["p"]["nominal_coverage"]["fraction"] >= COVERAGE_MIN)
    except (KeyError, TypeError):
        ok = False
    out["status"] = "met" if ok else "not met"
    return out


def pooled_calibration() -> dict:
    """The coordinate cutoffs of H23 recomputed with replica seeds 0 to 2 pooled; report only."""
    rows = []
    for seed in POOLED_SEEDS:
        rows += _rows(_coords({"replica": replica(seed)}))
    real = _rows(_coords({}))
    if not rows or not real:
        return {"n": 0}
    out = {"n": len(rows), "seeds": [s for s in POOLED_SEEDS
                                      if _rows(_coords({"replica": replica(s)}))]}
    for name, key in (("tau1", "log_tau1"), ("p", "log_p")):
        cut = calibrated_cutoff(rows, key)
        if not cut.get("n"):
            continue
        out[name] = {"cutoff": cut,
                     "real_nominal": bounded_among_interior(real, key, CHI2_DELTA),
                     "real_calibrated": bounded_among_interior(real, key, cut["cutoff"]),
                     "replica_nominal_coverage": coverage_at(rows, key, CHI2_DELTA)}
    return out


def main() -> None:
    summary = {"h24": h24(), "h25": h25(), "pooled_calibration": pooled_calibration()}
    path = RESULTS / "phase11_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
