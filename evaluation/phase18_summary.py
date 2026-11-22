"""Phase 18 (Amendment 14): robustness of the input-error result, and two checks that answer an outside review.

H35  carbohydrate error 0.25, trace coordinate profiles with the residual standard deviation and autocorrelation estimated again
     at the polished estimate: the nominal interval covers the truth for >= 90% of subjects, for tau1 and for p
H36  replica seeds 1 and 2, carbohydrate error 0 against 0.75: for tau1 and p the bounded fraction rises by >= 30 points and the
     nominal coverage falls by >= 30 points on each seed
H37  a systematic carbohydrate error (one lognormal multiplier per subject, coefficient of variation 0.25, no per-meal error):
     the nominal interval covers the truth for >= 90% of subjects, for tau1 and for p
H38  the pattern of H36 on a replica built from the Shanghai subjects (seed 0)
R17  the bounded fractions, coverage and S_I error under the systematic error
R18  the Shanghai real-data bounded fractions: iAUC profiles (S_I, k_e, k_a) and trace coordinate profiles (tau1, p)
R19  held-out iAUC error under random, blocked and forward (chronological) folds on CGMacros
R20  per carbohydrate error (seed 0): the pilot noise standard deviation, the factor on it that the 90% coverage cutoff implies,
     the bounded fraction at that cutoff and the all-subject bounded counts (stored results only, no new fits)

Nothing is fitted here. A missing result set gives `{"n": 0}` for its block.

Run:  python -m evaluation.phase18_summary          # writes results/phase18_summary.json
"""
from __future__ import annotations

import json

import numpy as np

from evaluation import profile_lik
from evaluation.identifiability_tools import CHI2_DELTA
from evaluation.phase13_summary import _block, _coords, _iauc_config
from evaluation.phase14_summary import COORDS
from evaluation.recalibration import _rows, bounded_among_interior
from evaluation.results_io import RESULTS, load_all

H35_COVERAGE, H36_POINTS, H37_COVERAGE, H38_POINTS = 0.90, 30.0, 0.90, 30.0
SEEDS = (0, 1, 2)
CVS = (0.0, 0.25, 0.75)
SCHEMES = ("random", "blocked", "forward")
CELLS = ("grad3", "grad1", "personal_mean", "population")


def _replica(seed: int, cv: float, **extra) -> dict:
    return {"seed": seed, "cgm": True, "carb_cv": cv, **extra}


def _cfg(replica: dict | None, **extra) -> dict:
    return _coords({"steps": 100, **({"replica": replica} if replica else {}), **extra})


def _fractions(block: dict, name: str):
    return (block.get(name) or {}).get("bounded_interior", {}).get("fraction")


def _coverage(block: dict, name: str):
    return (block.get(name) or {}).get("nominal_coverage", {}).get("fraction")


def pattern(cohort: str, seed: int) -> dict:
    """Blocks of the trace coordinate profiles at each carbohydrate error for one seed (and cohort)."""
    extra = {} if cohort == "cgmacros" else {"cohort": cohort, "min_meals": 10}
    return {f"{cv:g}": _block(_rows(_cfg(_replica(seed, cv), **extra))) for cv in CVS}


def pattern_rule(blocks: dict, points: float) -> dict:
    """The direction of the Phase 13 sweep: bounded fraction up and coverage down by at least `points` between 0 and 0.75."""
    low, high = blocks.get("0", {}), blocks.get("0.75", {})
    if not low.get("n") or not high.get("n"):
        return {"status": "not evaluated"}
    detail, ok = {}, True
    for name, _ in COORDS:
        b0, b1, c0, c1 = _fractions(low, name), _fractions(high, name), _coverage(low, name), _coverage(high, name)
        if None in (b0, b1, c0, c1):
            return {"status": "not evaluated"}
        detail[name] = {"bounded_rise_points": 100 * (b1 - b0), "coverage_drop_points": 100 * (c0 - c1)}
        ok = ok and 100 * (b1 - b0) >= points and 100 * (c0 - c1) >= points
    return {"status": "met" if ok else "not met", "detail": detail}


def h36(by_seed: dict) -> dict:
    rule = {"points": H36_POINTS, "seeds": [1, 2]}
    results = {f"seed{s}": pattern_rule(by_seed.get(f"seed{s}", {}), H36_POINTS) for s in (1, 2)}
    if any(r["status"] == "not evaluated" for r in results.values()):
        return {"status": "not evaluated", "rule": rule, "results": results}
    return {"status": "met" if all(r["status"] == "met" for r in results.values()) else "not met", "rule": rule, "results": results}


def h38(blocks: dict) -> dict:
    result = pattern_rule(blocks, H38_POINTS)
    return {**result, "rule": {"points": H38_POINTS, "cohort": "shanghai"}}


def _coverage_rule(block: dict, threshold: float, label: str) -> dict:
    rule = {"coverage": threshold}
    if not block.get("n"):
        return {"status": "not evaluated", "rule": rule}
    cover = {name: _coverage(block, name) for name, _ in COORDS}
    if any(v is None for v in cover.values()):
        return {"status": "not evaluated", "rule": rule}
    return {"status": "met" if all(v >= threshold for v in cover.values()) else "not met", "rule": rule, "coverage": cover}


def h35(block: dict) -> dict:
    return _coverage_rule(block, H35_COVERAGE, "noise re-estimated at the fit")


def h37(block: dict) -> dict:
    return _coverage_rule(block, H37_COVERAGE, "systematic carbohydrate error")


def r17() -> dict:
    replica = _replica(0, 0.0, carb_bias=0.25)
    rows = _rows(_cfg(replica))
    iauc_rows = _rows(_iauc_config(replica))
    errors = [float(np.log(r["theta_ml"]["insulin_sensitivity"]) - np.log(r["replica_truth"]["theta_true"]["insulin_sensitivity"]))
              for r in iauc_rows if (r.get("replica_truth") or {}).get("theta_true")]
    return {"trace": _block(rows), "iauc_S_I_median_log_error": float(np.median(errors)) if errors else None,
            "iauc_n": len(iauc_rows)}


def r18() -> dict:
    iauc_rows = _rows({**profile_lik.default_config(), "cohort": "shanghai", "min_meals": 10, "objective": "iauc",
                       "bounds_scale": 1.0})
    out = {"iauc": {"n": len(iauc_rows)}}
    for key, name in (("insulin_sensitivity", "S_I"), ("gastric_emptying", "k_e"), ("carb_absorption", "k_a")):
        if iauc_rows:
            out["iauc"][name] = bounded_among_interior(iauc_rows, key, CHI2_DELTA)
    out["trace"] = _block(_rows(_cfg(None, cohort="shanghai", min_meals=10)))
    return out


def _cv_by_subject(scheme: str) -> dict:
    """Per-subject mean held-out iAUC error of each cell, from the most complete CGMacros result set of this fold scheme."""
    best: dict = {}
    for units in load_all("A9_prediction_cv").values():
        rows = {k: d["payload"] for k, d in units.items() if k != "_unreadable"
                and d["payload"].get("fold_scheme", "random") == scheme and d["payload"].get("cohort") == "cgmacros"
                and d["payload"].get("replica") is None and not d["payload"].get("carb_scale")
                and "grad3" in d["payload"].get("cells", {})}
        if len(rows) > len(best):
            best = rows
    per: dict = {}
    for payload in best.values():
        for cell in CELLS:
            value = (payload["cells"].get(cell) or {}).get("iauc_mae")
            if value is not None:
                per.setdefault(payload["subject_id"], {}).setdefault(cell, []).append(float(value))
    return {sid: {c: float(np.mean(v)) for c, v in d.items()} for sid, d in per.items()}


def r19() -> dict:
    out = {}
    rng = np.random.default_rng(14)
    for scheme in SCHEMES:
        per = _cv_by_subject(scheme)
        ids = [s for s, d in per.items() if all(c in d for c in CELLS)]
        if len(ids) < 5:
            out[scheme] = {"n": 0}
            continue
        block = {"n": len(ids)}
        for cell in CELLS:
            block[cell] = float(np.mean([per[s][cell] for s in ids]))
        for other in ("personal_mean", "grad1"):
            diff = np.array([per[s]["grad3"] - per[s][other] for s in ids])
            draws = [float(np.mean(diff[rng.integers(0, len(diff), len(diff))])) for _ in range(2000)]
            block[f"grad3_minus_{other}"] = {"mean": float(np.mean(diff)), "ci95": [float(np.percentile(draws, 2.5)),
                                                                                  float(np.percentile(draws, 97.5))]}
        out[scheme] = block
    return out


def r20() -> dict:
    """Per carbohydrate error on seed 0 (trace coordinate profiles): the median pilot noise standard deviation, the factor on it that
    a cutoff covering the truth for 90% of subjects corresponds to, the bounded fraction at that cutoff, and the all-subject counts."""
    from evaluation.recalibration import calibrated_cutoff
    out = {}
    for cv in (0.0, 0.1, 0.25, 0.5, 0.75):
        rows = _rows(_cfg(_replica(0, cv)))
        if not rows:
            out[f"{cv:g}"] = {"n": 0}
            continue
        sigma = [float(list(r["noise"]["sigma"].values())[0]) for r in rows
                 if r.get("noise") and isinstance(r["noise"].get("sigma"), dict)]
        block = {"n": len(rows), "median_sigma": float(np.median(sigma)) if sigma else None}
        for name, key in (("tau1", "log_tau1"), ("p", "log_p")):
            cutoff = calibrated_cutoff(rows, key).get("cutoff")
            block[name] = {"all_bounded": int(sum(r["profiles"][key]["classification"]["verdict"] == "identifiable" for r in rows)),
                           "sigma_factor": float(np.sqrt(cutoff / CHI2_DELTA)) if cutoff else None,
                           "bounded_at_cutoff": (bounded_among_interior(rows, key, cutoff)["fraction"] if cutoff else None)}
        out[f"{cv:g}"] = block
    return out


def summarize() -> dict:
    by_seed = {f"seed{s}": pattern("cgmacros", s) for s in (1, 2)}
    shanghai = pattern("shanghai", 0)
    final = _block(_rows(_cfg(_replica(0, 0.25), noise_source="final")))
    final_high = _block(_rows(_cfg(_replica(0, 0.75), noise_source="final")))
    systematic = r17()
    return {"seeds": by_seed, "h36": h36(by_seed), "shanghai_replica": shanghai, "h38": h38(shanghai),
            "noise_final": {"0.25": final, "0.75": final_high}, "h35": h35(final),
            "r17": systematic, "h37": h37(systematic["trace"]), "r18": r18(), "r19": r19(), "r20": r20()}


def main() -> int:
    path = RESULTS / "phase18_summary.json"
    path.write_text(json.dumps(summarize(), indent=2, default=float), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
