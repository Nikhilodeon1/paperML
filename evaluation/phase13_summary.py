"""Phase 13 (Amendment 9): the reviewer-driven analyses, computed from stored results.

R1  prediction against parameter error, with the floor of the true model (replica seed 0 and random-truth seeds)
R2  clustered bootstrap intervals for the pooled replica fractions
R3  prior-meal overlap in the CGMacros logs
R5  a trivial insulin-sensitivity proxy against HbA1c
R6  the carbohydrate-error sweep of the trace coordinate profiles
H26 are the polished trace estimates global optima? (30 random starts against 5)

R4 (numerical checks) lives in `evaluation.numerics_check`. Nothing is fitted here. A missing result set gives
`{"n": 0}` for its block, so the summary can be built while a run is still in progress.

Run:  python -m evaluation.phase13_summary          # writes results/phase13_summary.json
"""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import numpy as np

from evaluation import profile_lik
from evaluation.identifiability_tools import CHI2_DELTA, classify_profile
from evaluation.recalibration import INTERIOR, bounded_among_interior, coverage_at, calibrated_cutoff, _rows
from evaluation.results_io import RESULTS, ROOT, load_all, largest_matching
from evaluation.stats_utils import wilson_ci

N_BOOT = 2000
H26_GAP, H26_FRACTION = CHI2_DELTA, 0.90        # polished 5-start estimate within delta of the 30-start one in 90%
PARAMETERS = ("insulin_sensitivity", "gastric_emptying", "carb_absorption")
FIT_REPLICA = {"seed": 0, "cgm": True, "carb_cv": 0.25}
RANDOM_REPLICAS = tuple({"seed": s, "cgm": True, "carb_cv": 0.25, "truth": "random"} for s in (0, 1))
CV_SWEEP = (0.0, 0.1, 0.25, 0.5, 0.75)
SEEDS_IAUC = (0, 1, 2, 3, 4)


def _rng(*parts) -> np.random.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in ("phase13",) + parts).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


# --------------------------------------------------------------------------------------------------
# R1: prediction against parameter error
# --------------------------------------------------------------------------------------------------

def _iauc_config(replica: dict, **extra) -> dict:
    return {**profile_lik.default_config(), "objective": "iauc", "bounds_scale": 1.0, "replica": replica, **extra}


def parameter_errors(row: dict) -> dict | None:
    """Absolute log errors of the iAUC maximum-likelihood estimate against the replica truth."""
    truth = (row.get("replica_truth") or {}).get("theta_true")
    if not truth:
        return None
    err = {p: abs(float(np.log(row["theta_ml"][p]) - np.log(truth[p]))) for p in PARAMETERS}
    err["rms"] = float(np.sqrt(np.mean([err[p] ** 2 for p in PARAMETERS])))
    err["timing_rms"] = float(np.sqrt(np.mean([err["gastric_emptying"] ** 2, err["carb_absorption"] ** 2])))
    return err


def _cv_by_subject(replica: dict, cells=("grad3", "personal_mean", "grad1", "population")) -> dict:
    """Held-out iAUC error of each cell, one number per subject (mean over repeats), from the A9 set that matches."""
    best: dict[str, dict] = {}
    for units in load_all("A9_prediction_cv").values():
        rows = {k: d["payload"] for k, d in units.items() if k != "_unreadable"
                and d["payload"].get("replica") == replica and d["payload"].get("cohort") == "cgmacros"
                and not d["payload"].get("carb_scale") and "grad3" in d["payload"].get("cells", {})}
        if len(rows) > len(best):
            best = rows
    per: dict[str, dict[str, list]] = {}
    for payload in best.values():
        for cell in cells:
            block = payload["cells"].get(cell)
            if block is not None and block.get("iauc_mae") is not None:
                per.setdefault(payload["subject_id"], {}).setdefault(cell, []).append(float(block["iauc_mae"]))
    return {sid: {cell: float(np.mean(v)) for cell, v in d.items()} for sid, d in per.items()}


def _floor_by_subject(replica: dict) -> dict:
    best: dict[str, dict] = {}
    for units in load_all("A16_truth_floor").values():
        rows = {k: d["payload"] for k, d in units.items() if k != "_unreadable"
                and d["payload"].get("replica") == replica}
        if len(rows) > len(best):
            best = rows
    return {p["subject_id"]: p for p in best.values()}


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def prediction_vs_parameters(replica: dict, exact_replica: dict | None = None) -> dict:
    """One replica: per-subject held-out error of the fit, parameter error of the iAUC estimate, and the floor.

    `exact_replica` is the same replica with exact carbohydrate, whose floor is the one that a model with the
    right parameters and the right inputs attains. When it is not stored the second floor is left out.
    """
    cv = _cv_by_subject(replica)
    params = {r["subject_id"]: r for r in _rows(_iauc_config(replica))}
    floor = _floor_by_subject(replica)
    floor_exact = _floor_by_subject(exact_replica) if exact_replica else {}
    subjects = sorted(set(cv) & set(params) & set(floor))
    if len(subjects) < 5:
        return {"n": 0, "replica": replica}
    rows = []
    for sid in subjects:
        err = parameter_errors(params[sid])
        if err is None:
            continue
        rows.append({"subject_id": sid, "grad3_mae": cv[sid]["grad3"], "floor": floor[sid]["iauc_mae"],
                     "floor_exact": floor_exact[sid]["iauc_mae"] if sid in floor_exact else None,
                     "personal_mean_mae": cv[sid].get("personal_mean"), **err})
    n = len(rows)
    mae = np.array([r["grad3_mae"] for r in rows])
    floor_v = np.array([r["floor"] for r in rows])
    ratio = mae / floor_v
    rms = np.array([r["rms"] for r in rows])
    timing = np.array([r["timing_rms"] for r in rows])
    si = np.array([r["insulin_sensitivity"] for r in rows])
    out = {"n": n, "replica": replica,
           "median_grad3_mae": float(np.median(mae)), "median_floor": float(np.median(floor_v)),
           "median_ratio_to_floor": float(np.median(ratio)),
           "median_personal_mean_mae": float(np.median([r["personal_mean_mae"] for r in rows
                                                        if r["personal_mean_mae"] is not None])),
           "median_rms_log_error": float(np.median(rms)), "median_timing_rms_log_error": float(np.median(timing)),
           "median_si_log_error": float(np.median(si)),
           "median_ke_log_error": float(np.median([r["gastric_emptying"] for r in rows])),
           "median_ka_log_error": float(np.median([r["carb_absorption"] for r in rows])),
           "floor_exact_median": (float(np.median([r["floor_exact"] for r in rows]))
                                  if all(r["floor_exact"] is not None for r in rows) else None)}
    rng = _rng("r1", json.dumps(replica, sort_keys=True))
    for name, x in (("rms", rms), ("timing", timing), ("si", si)):
        for target, y in (("mae", mae), ("ratio", ratio)):
            point = _spearman(x, y)
            draws = []
            for _ in range(N_BOOT):
                index = rng.integers(0, n, n)
                draws.append(_spearman(x[index], y[index]))
            out[f"spearman_{name}_{target}"] = {"estimate": point, "ci95": _interval(np.asarray(draws))}
    out["subjects"] = rows
    return out


def r1() -> dict:
    exact = {"seed": 0, "cgm": True, "carb_cv": 0.0}
    out = {"fit_truth_seed0": prediction_vs_parameters(FIT_REPLICA, exact)}
    for replica in RANDOM_REPLICAS:
        out[f"random_truth_seed{replica['seed']}"] = prediction_vs_parameters(replica)
    return out


# --------------------------------------------------------------------------------------------------
# R2: clustered intervals for the pooled replica fractions
# --------------------------------------------------------------------------------------------------

def _flags(rows: list[dict], key: str, delta: float) -> tuple[bool, bool]:
    """(interior for the parameter, bounded at `delta`) of one profile result."""
    if key not in rows["profiles"]:
        return False, False
    if rows["distance_to_bound"][key] <= INTERIOR:
        return False, False
    entry = rows["profiles"][key]
    n = int(entry["n_in_box"])
    verdict = classify_profile(np.asarray(entry["grid"], dtype=float)[:n], np.asarray(entry["profile"], dtype=float)[:n],
                               delta=delta, reference=float(entry["reference"]))["verdict"]
    return True, verdict == "identifiable"


def clustered_fraction(sets: list[list[dict]], key: str, delta: float = CHI2_DELTA, label: str = "") -> dict:
    """Bounded fraction among interior subject-seed pairs, with a percentile interval that resamples SUBJECTS.

    The seeds reuse the same subjects and meal schedules, so a Wilson interval on the pooled count treats the
    pairs as independent when they are not. Resampling subjects, each carrying all of its seeds, does not.
    """
    by_subject: dict[str, list[tuple[bool, bool]]] = {}
    for rows in sets:
        for row in rows:
            by_subject.setdefault(row["subject_id"], []).append(_flags(row, key, delta))
    ids = sorted(by_subject)
    if not ids:
        return {"n": 0}
    interior = np.array([sum(f[0] for f in by_subject[s]) for s in ids], dtype=float)
    bounded = np.array([sum(f[0] and f[1] for f in by_subject[s]) for s in ids], dtype=float)
    point = bounded.sum() / interior.sum() if interior.sum() else float("nan")
    rng = _rng("r2", key, label)
    draws = []
    for _ in range(N_BOOT):
        index = rng.integers(0, len(ids), len(ids))
        denominator = interior[index].sum()
        if denominator:
            draws.append(bounded[index].sum() / denominator)
    pairs = int(interior.sum())
    ci = wilson_ci(int(bounded.sum()), pairs) if pairs else None
    return {"n_pairs": pairs, "n_subjects": len(ids), "bounded": int(bounded.sum()), "fraction": float(point),
            "wilson_naive": [ci.low, ci.high] if ci else None, "clustered_ci95": _interval(np.asarray(draws))}


def r2() -> dict:
    iauc_sets = [_rows(_iauc_config({"seed": s, "cgm": True, "carb_cv": 0.25})) for s in SEEDS_IAUC]
    out = {"S_I_pooled_iauc": clustered_fraction([r for r in iauc_sets if r], "insulin_sensitivity", label="si")}
    coords = {**profile_lik.default_config(), "bounds_scale": 1.0, "objective": "trace", "parameterization": "coords",
              "polish": True}
    trace_sets = [_rows({**coords, "replica": {"seed": s, "cgm": True, "carb_cv": 0.25}}) for s in (0, 1, 2)]
    trace_sets = [r for r in trace_sets if r]
    for name, key in (("tau1", "log_tau1"), ("p", "log_p")):
        out[f"{name}_pooled_trace_seeds012"] = clustered_fraction(trace_sets, key, label=name)
    real = _rows(coords)
    out["real_polished"] = {name: bounded_among_interior(real, key, CHI2_DELTA)
                            for name, key in (("tau1", "log_tau1"), ("p", "log_p"))}
    return out


# --------------------------------------------------------------------------------------------------
# R3: prior-meal overlap
# --------------------------------------------------------------------------------------------------

def r3() -> dict:
    """Gap between a meal and the previous logged meal of the same subject, among the meals of the analysis cohort."""
    from evaluation import cgmacros
    from evaluation.cohort_data import load_cohort

    keep = {s.subject_id for s in load_cohort("cgmacros", min_meals=10)}
    gaps = []
    for sid, meals in cgmacros.subjects():
        if sid not in keep:
            continue
        times = sorted(m.t for m in meals)
        gaps += [(b - a) / timedelta(hours=1) for a, b in zip(times[:-1], times[1:])]
    if not gaps:
        return {"n": 0}
    g = np.asarray(gaps)
    out = {"n_gaps": int(len(g)), "median_gap_hours": float(np.median(g)),
           "q25_gap_hours": float(np.percentile(g, 25)), "q75_gap_hours": float(np.percentile(g, 75))}
    for hours in (3, 4, 6):
        k = int((g < hours).sum())
        ci = wilson_ci(k, len(g))
        out[f"share_within_{hours}h"] = {"count": k, "n": int(len(g)), "fraction": k / len(g),
                                         "wilson": [ci.low, ci.high]}
    return out


# --------------------------------------------------------------------------------------------------
# R5: a trivial insulin-sensitivity proxy against HbA1c
# --------------------------------------------------------------------------------------------------

def r5() -> dict:
    from evaluation import leakage_ci
    from evaluation.cohort_data import load_cohort

    stored = {r["subject_id"]: r for r in largest_matching(leakage_ci.ANALYSIS_ID, lambda p: "snpe_si" in p).values()
              if r["snpe_si"] is not None and r["hba1c"] is not None}
    proxy = {}
    for subject in load_cohort("shanghai", min_meals=3):
        values = [r["iauc"] / r["carbs_g"] for r in subject.records
                  if r.get("iauc") is not None and r.get("carbs_g") and r["carbs_g"] > 0]
        if values:
            proxy[subject.subject_id] = float(np.mean(values))
    ids = sorted(set(stored) & set(proxy))
    if len(ids) < 10:
        return {"n": len(ids)}
    arr = {"proxy": np.array([proxy[i] for i in ids]), "hba1c": np.array([stored[i]["hba1c"] for i in ids], dtype=float),
           "fasting": np.array([stored[i]["fasting"] for i in ids], dtype=float)}

    def stats(a):
        return {"raw": leakage_ci._corr(a["proxy"], a["hba1c"]),
                "partial": leakage_ci._partial(a["proxy"], a["hba1c"], a["fasting"])}
    point = stats(arr)
    rng = _rng("r5")
    draws = {k: [] for k in point}
    for _ in range(N_BOOT):
        index = rng.integers(0, len(ids), len(ids))
        s = stats({k: v[index] for k, v in arr.items()})
        for k, v in s.items():
            draws[k].append(v)
    out = {"n": len(ids), "statistics": {}}
    for k, v in point.items():
        d = np.asarray(draws[k], dtype=float)
        d = d[np.isfinite(d)]
        out["statistics"][k] = {"estimate": v, "low": float(np.percentile(d, 2.5)), "high": float(np.percentile(d, 97.5))}
    return out


# --------------------------------------------------------------------------------------------------
# R6 and H26: the trace coordinate profiles
# --------------------------------------------------------------------------------------------------

def _coords(extra: dict) -> dict:
    return {**profile_lik.default_config(), "bounds_scale": 1.0, "objective": "trace", "parameterization": "coords",
            "polish": True, **extra}


def _block(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for name, key in (("tau1", "log_tau1"), ("p", "log_p")):
        out[name] = {"bounded_interior": bounded_among_interior(rows, key, CHI2_DELTA),
                     "nominal_coverage": coverage_at(rows, key, CHI2_DELTA),
                     "bounded_coverage": profile_lik.summarize_rows(rows, _coords({}))["parameters"][key].get(
                         "truth_coverage_of_bounded"),
                     "cutoff": calibrated_cutoff(rows, key)}
    return out


def r6() -> dict:
    out = {}
    for cv in CV_SWEEP:
        out[f"{cv:g}"] = _block(_rows(_coords({"replica": {"seed": 0, "cgm": True, "carb_cv": cv}})))
    out["real_polished"] = {name: bounded_among_interior(_rows(_coords({})), key, CHI2_DELTA)
                            for name, key in (("tau1", "log_tau1"), ("p", "log_p"))}
    return out


def _gaps(rows5: list[dict], rows30: list[dict]) -> dict:
    """Per subject, the polished negative log-likelihood of the 5-start fit minus that of the 30-start fit."""
    a = {r["subject_id"]: r["polish"]["nll_polished"] for r in rows5 if r.get("polish")}
    b = {r["subject_id"]: r["polish"]["nll_polished"] for r in rows30 if r.get("polish")}
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0}
    gap = np.array([a[i] - b[i] for i in ids])
    within = int((gap < H26_GAP).sum())
    ci = wilson_ci(within, len(ids))
    return {"n": len(ids), "within": within, "fraction": within / len(ids), "wilson": [ci.low, ci.high],
            "median_gap": float(np.median(gap)), "max_gap": float(np.max(gap))}


def h26() -> dict:
    """H26 (Amendment 9): are the polished trace estimates already close to a global optimum?"""
    out = {"rule": {"gap": H26_GAP, "fraction_min": H26_FRACTION, "starts": 30}}
    real5, real30 = _rows(_coords({})), _rows(_coords({"polish_starts": 30}))
    rep = {"seed": 0, "cgm": True, "carb_cv": 0.25}
    rep5, rep30 = _rows(_coords({"replica": rep})), _rows(_coords({"replica": rep, "polish_starts": 30}))
    out["real"] = {"gap": _gaps(real5, real30), "at_5": _block(real5), "at_30": _block(real30)}
    out["replica_seed0"] = {"gap": _gaps(rep5, rep30), "at_5": _block(rep5), "at_30": _block(rep30)}
    gap = out["real"]["gap"]
    if not gap.get("n"):
        out["status"] = "not evaluated"
    else:
        out["status"] = "met" if gap["fraction"] >= H26_FRACTION else "not met"
    return out


def summarize() -> dict:
    out = {"r1": r1(), "r2": r2(), "r5": r5(), "r6": r6(), "h26": h26()}
    try:
        out["r3"] = r3()
    except Exception as exc:                                   # the raw logs are not on every machine
        out["r3"] = {"n": 0, "error": repr(exc)[:200]}
    try:
        from evaluation.numerics_check import summarize as numerics
        out["r4"] = numerics()
    except Exception as exc:
        out["r4"] = {"n": 0, "error": repr(exc)[:200]}
    return out


def main() -> int:
    path = RESULTS / "phase13_summary.json"
    path.write_text(json.dumps(summarize(), indent=2, default=float), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
