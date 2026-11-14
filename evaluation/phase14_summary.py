"""Phase 14 (Amendment 10): the inner-optimizer budget of the trace coordinate profiles, and the floor of the true model
for peak-time and trace error.

R7   trace coordinate profiles (polished, primary box) at 500 inner steps against the stored 100-step run, on the real
     data and on replica seed 0 with the carbohydrate error at 0.25, exact, and exact with a quarter of the CGM noise
R8   the read-out of R7 as a positive control: how often the combination is bounded and covered with exact inputs
     and low noise
R9   held-out peak-time error and trace error of the iAUC fits against the model at the true parameters, on the
     replicas (stored results only)
H27  the real-data bounded fractions of tau1 and p move by at most 10 percentage points between 100 and 500 steps
H28  with exact carbohydrate and a quarter of the noise, the nominal interval covers the truth for >= 90% of subjects
     at 500 steps, for tau1 and for p

Nothing is fitted here. A missing result set gives `{"n": 0}` for its block.

Run:  python -m evaluation.phase14_summary          # writes results/phase14_summary.json
"""
from __future__ import annotations

import json

import numpy as np

from evaluation import profile_lik
from evaluation.identifiability_tools import CHI2_DELTA, classify_profile
from evaluation.phase13_summary import (FIT_REPLICA, RANDOM_REPLICAS, _block, _coords, _floor_by_subject, _interval,
                                        _iauc_config, _rng, _spearman, N_BOOT, parameter_errors)
from evaluation.recalibration import _rows
from evaluation.results_io import RESULTS, load_all

H27_POINTS = 10.0                 # percentage points
H28_COVERAGE = 0.90
STEPS = (100, 500)
SETTINGS = {
    "real": None,
    "replica": {"seed": 0, "cgm": True, "carb_cv": 0.25},
    "exact": {"seed": 0, "cgm": True, "carb_cv": 0.0},
    "quarter": {"seed": 0, "cgm": True, "carb_cv": 0.0, "cgm_scale": 0.25},
}
COORDS = (("tau1", "log_tau1"), ("p", "log_p"))


# --------------------------------------------------------------------------------------------------
# R7 and R8
# --------------------------------------------------------------------------------------------------

def _config(replica: dict | None, steps: int) -> dict:
    return _coords({"steps": steps, **({"replica": replica} if replica else {})})


def settling(rows: list[dict]) -> dict:
    """Share of in-box grid points whose inner optimization had not stopped improving, and profiles beaten by a grid point."""
    points = unsettled = better = profiles = 0
    for row in rows:
        for entry in row["profiles"].values():
            points += int(entry["n_in_box"])
            unsettled += int(entry["unsettled"])
            better += bool(entry["better_than_theta_hat"])
            profiles += 1
    if not points:
        return {"n": 0}
    return {"profiles": profiles, "points": points, "unsettled": unsettled, "unsettled_fraction": unsettled / points,
            "better_than_estimate": better}


def class_changes(rows_a: list[dict], rows_b: list[dict], delta: float = CHI2_DELTA) -> dict:
    """Per (subject, coordinate) profile, whether the class (bounded, one-sided, flat) differs between two runs."""
    a = {r["subject_id"]: r for r in rows_a}
    b = {r["subject_id"]: r for r in rows_b}
    changed = total = to_unbounded = to_bounded = 0
    for sid in sorted(set(a) & set(b)):
        for _, key in COORDS:
            ea, eb = a[sid]["profiles"].get(key), b[sid]["profiles"].get(key)
            if ea is None or eb is None:
                continue
            ca = classify_profile(np.asarray(ea["grid"], float)[:int(ea["n_in_box"])],
                                  np.asarray(ea["profile"], float)[:int(ea["n_in_box"])],
                                  delta=delta, reference=float(ea["reference"]))["verdict"]
            cb = classify_profile(np.asarray(eb["grid"], float)[:int(eb["n_in_box"])],
                                  np.asarray(eb["profile"], float)[:int(eb["n_in_box"])],
                                  delta=delta, reference=float(eb["reference"]))["verdict"]
            total += 1
            if ca != cb:
                changed += 1
                to_unbounded += (ca == "identifiable") and (cb != "identifiable")
                to_bounded += (ca != "identifiable") and (cb == "identifiable")
    if not total:
        return {"n": 0}
    return {"n": total, "changed": changed, "bounded_to_open": int(to_unbounded), "open_to_bounded": int(to_bounded)}


def r7() -> dict:
    out = {}
    for tag, replica in SETTINGS.items():
        rows = {steps: _rows(_config(replica, steps)) for steps in STEPS}
        entry = {f"steps_{steps}": {**_block(rows[steps]), "settling": settling(rows[steps])} for steps in STEPS}
        entry["changes"] = class_changes(rows[100], rows[500])
        out[tag] = entry
    return out


def _fraction(block: dict, name: str):
    return (block.get(name) or {}).get("bounded_interior", {}).get("fraction")


def h27(r7_out: dict) -> dict:
    """H27: the real-data bounded fractions at 500 steps stay within 10 points of the 100-step ones."""
    real = r7_out.get("real", {})
    a, b = real.get("steps_100", {}), real.get("steps_500", {})
    if not a.get("n") or not b.get("n"):
        return {"status": "not evaluated", "rule": {"points": H27_POINTS}}
    diff = {}
    for name, _ in COORDS:
        fa, fb = _fraction(a, name), _fraction(b, name)
        if fa is None or fb is None:
            return {"status": "not evaluated", "rule": {"points": H27_POINTS}}
        diff[name] = 100.0 * (fb - fa)
    ok = all(abs(v) <= H27_POINTS for v in diff.values())
    return {"status": "met" if ok else "not met", "rule": {"points": H27_POINTS}, "difference_points": diff}


def h28(r7_out: dict) -> dict:
    """H28: exact carbohydrate and a quarter of the noise: nominal coverage >= 90% at 500 steps for tau1 and p."""
    block = r7_out.get("quarter", {}).get("steps_500", {})
    if not block.get("n"):
        return {"status": "not evaluated", "rule": {"coverage": H28_COVERAGE}}
    cover = {name: (block[name]["nominal_coverage"] or {}).get("fraction") for name, _ in COORDS}
    if any(v is None for v in cover.values()):
        return {"status": "not evaluated", "rule": {"coverage": H28_COVERAGE}}
    return {"status": "met" if all(v >= H28_COVERAGE for v in cover.values()) else "not met",
            "rule": {"coverage": H28_COVERAGE}, "coverage": cover}


# --------------------------------------------------------------------------------------------------
# R9: peak time and trace error against the floor of the true model
# --------------------------------------------------------------------------------------------------

def _cv_metrics_by_subject(replica: dict, cells=("grad3", "grad1")) -> dict:
    best: dict[str, dict] = {}
    for units in load_all("A9_prediction_cv").values():
        rows = {k: d["payload"] for k, d in units.items() if k != "_unreadable"
                and d["payload"].get("replica") == replica and d["payload"].get("cohort") == "cgmacros"
                and not d["payload"].get("carb_scale") and "grad3" in d["payload"].get("cells", {})}
        if len(rows) > len(best):
            best = rows
    per: dict[str, dict] = {}
    for payload in best.values():
        for cell in cells:
            block = payload["cells"].get(cell)
            if not block:
                continue
            for metric in ("peak_mae", "trace_rmse_mean"):
                if block.get(metric) is not None:
                    per.setdefault(payload["subject_id"], {}).setdefault((cell, metric), []).append(float(block[metric]))
    return {sid: {k: float(np.mean(v)) for k, v in d.items()} for sid, d in per.items()}


def peak_trace_vs_parameters(replica: dict) -> dict:
    cv = _cv_metrics_by_subject(replica)
    floor = _floor_by_subject(replica)
    params = {r["subject_id"]: r for r in _rows(_iauc_config(replica))}
    subjects = sorted(set(cv) & set(floor) & set(params))
    rows = []
    for sid in subjects:
        err = parameter_errors(params[sid])
        f = floor[sid]
        if err is None or not f.get("pred_peak"):
            continue
        floor_peak = float(np.mean(np.abs(np.asarray(f["pred_peak"], float) - np.asarray(f["obs_peak"], float))))
        floor_trace = float(f["trace_rmse"])
        row = {"subject_id": sid, "floor_peak": floor_peak, "floor_trace": floor_trace,
               "timing_rms": err["timing_rms"], "rms": err["rms"]}
        for cell in ("grad3", "grad1"):
            row[f"{cell}_peak"] = cv[sid].get((cell, "peak_mae"))
            row[f"{cell}_trace"] = cv[sid].get((cell, "trace_rmse_mean"))
        if all(row[k] is not None for k in ("grad3_peak", "grad3_trace", "grad1_peak", "grad1_trace")):
            rows.append(row)
    n = len(rows)
    if n < 5:
        return {"n": 0, "replica": replica}
    out = {"n": n, "replica": replica}
    timing = np.array([r["timing_rms"] for r in rows])
    rng = _rng("r9", json.dumps(replica, sort_keys=True))
    for cell in ("grad3", "grad1"):
        for kind, floor_key in (("peak", "floor_peak"), ("trace", "floor_trace")):
            fit = np.array([r[f"{cell}_{kind}"] for r in rows])
            fl = np.array([r[floor_key] for r in rows])
            ratio = fit / fl
            out[f"{cell}_{kind}"] = {"median_fit": float(np.median(fit)), "median_floor": float(np.median(fl)),
                                     "median_ratio": float(np.median(ratio))}
            point = _spearman(timing, ratio)
            draws = []
            for _ in range(N_BOOT):
                index = rng.integers(0, n, n)
                draws.append(_spearman(timing[index], ratio[index]))
            out[f"{cell}_{kind}"]["spearman_timing_ratio"] = {"estimate": point, "ci95": _interval(np.asarray(draws))}
    for kind in ("peak", "trace"):
        paired = np.array([(r[f"grad3_{kind}"] - r[f"grad1_{kind}"]) / r[f"floor_{kind}"] for r in rows])
        out[f"free_minus_fixed_{kind}"] = {"median_difference_in_floors": float(np.median(paired))}
    out["subjects"] = rows
    return out


def r9() -> dict:
    out = {"fit_truth_seed0": peak_trace_vs_parameters(FIT_REPLICA)}
    for replica in RANDOM_REPLICAS:
        out[f"random_truth_seed{replica['seed']}"] = peak_trace_vs_parameters(replica)
    return out


# --------------------------------------------------------------------------------------------------

def summarize() -> dict:
    seven = r7()
    return {"r7": seven, "h27": h27(seven), "h28": h28(seven), "r9": r9()}


def main() -> int:
    path = RESULTS / "phase14_summary.json"
    path.write_text(json.dumps(summarize(), indent=2, default=float), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
