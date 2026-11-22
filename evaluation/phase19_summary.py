"""Phase 19 (Amendment 15): report-only numbers added in the TMLR revision. No hypothesis, no rule.

R21  curvature of the trace likelihood against the carbohydrate logging error (A20): per error, the median Hessian eigenvalues
     of the negative log-likelihood at the polished estimate in (log S_I, log tau1, log p), the Schur complement of the
     (tau1, p) block, the pilot noise standard deviation and the ratio of the mean squared logged load to the mean squared true
     load; and the per-subject ratio of the Schur eigenvalues to those at exact carbohydrate
R24  per carbohydrate error (replica seed 0, trace coordinate profiles): the median absolute log error of the estimate of tau1 and p,
     the number of bounded intervals, their median width in log units, and the median ratio of error to half-width
R22  the median signed log error of S_I under the iAUC on the exact-carbohydrate replica, and the count of estimates on a bound
R23  the overall (not only interior) counts of bounded intervals on the real data under the iAUC

Run:  python -m evaluation.phase19_summary     # writes results/phase19_summary.json
"""
from __future__ import annotations

import json

import numpy as np

from evaluation import profile_lik
from evaluation.phase13_summary import _coords, _iauc_config
from evaluation.recalibration import _rows
from evaluation.results_io import RESULTS, largest_matching

CVS = (0.0, 0.25, 0.75)


def _curvature_rows(cv: float) -> dict:
    replica = {"seed": 0, "cgm": True, "carb_cv": cv}
    rows = largest_matching("A20_carb_curvature", lambda p, r=replica: p.get("replica") == r)
    return {u: r for u, r in rows.items() if r.get("timing_schur_eigenvalues")}


def curvature() -> dict:
    base = _curvature_rows(0.0)
    out = {}
    for cv in CVS:
        rows = _curvature_rows(cv)
        if not rows:
            out[str(cv)] = {"n": 0}
            continue
        schur = np.array([r["timing_schur_eigenvalues"] for r in rows.values()], dtype=float)
        eig = np.array([r["hessian_eigenvalues"] for r in rows.values()], dtype=float)
        entry = {"n": len(rows),
                 "load_ratio_squared": float(np.median([r["load_ratio_squared"] for r in rows.values()])),
                 "median_sigma": float(np.median([r["noise_innovation_sd"] for r in rows.values()
                                                  if r.get("noise_innovation_sd") is not None])),
                 "hessian_eig_median": np.median(eig, axis=0).tolist(),
                 "schur_eig_median": np.median(schur, axis=0).tolist()}
        shared = [u for u in rows if u in base]
        if shared:
            a = np.array([rows[u]["timing_schur_eigenvalues"] for u in shared], dtype=float)
            b = np.array([base[u]["timing_schur_eigenvalues"] for u in shared], dtype=float)
            ok = (b > 0).all(axis=1) & (a > 0).all(axis=1)
            entry["n_paired"] = int(ok.sum())
            entry["schur_ratio_median"] = np.median(a[ok] / b[ok], axis=0).tolist() if ok.any() else None
            entry["schur_trace_ratio_median"] = float(np.median(a[ok].sum(axis=1) / b[ok].sum(axis=1))) if ok.any() else None
        out[str(cv)] = entry
    return out


ERROR_CVS = (0.0, 0.1, 0.25, 0.5, 0.75)


def error_and_width() -> dict:
    """The coverage failure is an error that grows against a width that does not: measure both from the stored profiles."""
    out = {}
    for cv in ERROR_CVS:
        rows = _rows(_coords({"replica": {"seed": 0, "cgm": True, "carb_cv": cv}}))
        if not rows:
            out[str(cv)] = {"n": 0}
            continue
        block = {"n": len(rows)}
        for key, name in (("log_tau1", "tau1"), ("log_p", "p")):
            error, width, error_bounded, pinned = [], [], [], 0
            for r in rows:
                entry = r["profiles"][key]
                theta, truth = entry.get("theta_hat"), entry.get("truth")
                if theta is None or truth is None:
                    continue
                e = abs(float(np.log(theta / truth)))
                error.append(e)
                pinned += int(r["distance_to_bound"][key] <= 0.01)
                c = entry["classification"]
                if c["verdict"] == "identifiable" and c.get("ci_low") and c.get("ci_high"):
                    width.append(float(np.log(c["ci_high"] / c["ci_low"])))
                    error_bounded.append(e)
            rise = [float(r["profiles"][key]["classification"]["max_rise"]) for r in rows
                    if r["profiles"][key]["classification"].get("max_rise") is not None]
            block[name] = {"median_error": float(np.median(error)), "bounded": len(width), "pinned": pinned,
                           "median_max_rise": float(np.median(rise)) if rise else None,
                           "median_width": float(np.median(width)) if width else None,
                           "median_error_over_half_width": (float(np.median(np.array(error_bounded) / (np.array(width) / 2)))
                                                            if width else None)}
        out[str(cv)] = block
    return out


def exact_bias() -> dict:
    rows = _rows(_iauc_config({"seed": 0, "cgm": True, "carb_cv": 0.0}))
    if not rows:
        return {"n": 0}
    err = [np.log(r["theta_ml"]["insulin_sensitivity"]) - np.log(r["replica_truth"]["theta_true"]["insulin_sensitivity"])
           for r in rows]
    return {"n": len(rows), "median_log_error": float(np.median(err)),
            "pinned": int(sum(r["distance_to_bound"]["insulin_sensitivity"] <= 0.01 for r in rows))}


def real_counts() -> dict:
    rows = _rows({**profile_lik.default_config(), "objective": "iauc", "bounds_scale": 1.0})
    out = {"n": len(rows)}
    for key, name in (("insulin_sensitivity", "S_I"), ("gastric_emptying", "k_e"), ("carb_absorption", "k_a")):
        out[name] = int(sum(r["profiles"][key]["classification"]["verdict"] == "identifiable" for r in rows))
    return out


def summarize() -> dict:
    return {"r21": curvature(), "r22": exact_bias(), "r23": real_counts(), "r24": error_and_width()}


def main() -> int:
    path = RESULTS / "phase19_summary.json"
    path.write_text(json.dumps(summarize(), indent=2, default=float), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
