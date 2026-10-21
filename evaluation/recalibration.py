"""H23 (Amendment 6): an empirical cutoff for the profile intervals, calibrated on the true-model replica.

The nominal interval is the set of values whose profile rises less than `delta = 1.92` above the minimum. That
cutoff is a statement about a regular likelihood; a hard box, a nonlinear model and the tied boundary of the
coordinate parameterization can all make it wrong. On the replica the true value is known, so the cutoff can
be measured: it is the 95th percentile of the rise of the profile at the true value. Applying that cutoff to
the real-data profiles asks whether the real-data fractions survive a calibration that the nominal cutoff did
not have. Nothing is fitted here; every number comes from stored profile grids.

Run:  python -m evaluation.recalibration
"""
from __future__ import annotations

import json

import numpy as np

from evaluation import profile_lik
from evaluation.identifiability_tools import CHI2_DELTA, classify_profile
from evaluation.results_io import RESULTS, largest_matching
from evaluation.stats_utils import N_RESAMPLES, wilson_ci

QUANTILE = 0.95
S_I_BOUNDED_MAX = 0.50          # part (a): the S_I fraction stays below this
TAU1_MIN, P_MIN = 0.70, 0.50    # part (b): the trace clauses of H11 and H18
REPLICA_SEEDS = (0, 1, 2, 3, 4)
INTERIOR = 0.01


def _replica(seed: int) -> dict:
    return {"seed": seed, "cgm": True, "carb_cv": 0.25}


def excess_at_truth(entry: dict) -> float | None:
    """Rise of the profile at the true value above the reference, or None if the truth is off the grid.

    Interpolated linearly in the logarithm of the in-box grid. The in-box points only: the points beyond the
    physiological bound belong to a diagnostic and are not part of any classification.
    """
    truth = entry.get("truth")
    if truth is None or truth <= 0:
        return None
    n = int(entry["n_in_box"])
    grid = np.log(np.asarray(entry["grid"], dtype=float)[:n])
    profile = np.asarray(entry["profile"], dtype=float)[:n]
    order = np.argsort(grid)
    grid, profile = grid[order], profile[order]
    x = float(np.log(truth))
    if x < grid[0] - 1e-12 or x > grid[-1] + 1e-12:
        return None
    value = float(np.interp(x, grid, profile))
    return max(value - float(entry["reference"]), 0.0)


def calibrated_cutoff(rows: list[dict], parameter: str, quantile: float = QUANTILE,
                      resamples: int | None = None) -> dict:
    """The empirical cutoff for one parameter, with a bootstrap interval over real subjects."""
    excesses, subjects, excluded = [], [], 0
    for row in rows:
        entry = row["profiles"].get(parameter)
        if entry is None:
            continue
        value = excess_at_truth(entry)
        if value is None:
            excluded += 1
            continue
        excesses.append(value)
        subjects.append(row["subject_id"])
    if not excesses:
        return {"n": 0, "excluded": excluded}
    x = np.asarray(excesses, dtype=float)
    cutoff = float(np.quantile(x, quantile))
    ids = sorted(set(subjects))
    by_subject = {s: x[[i for i, t in enumerate(subjects) if t == s]] for s in ids}
    rng = np.random.default_rng(20261011)
    draws = []
    for _ in range(int(resamples or N_RESAMPLES)):
        pick = rng.integers(0, len(ids), size=len(ids))
        draws.append(float(np.quantile(np.concatenate([by_subject[ids[i]] for i in pick]), quantile)))
    return {"n": int(x.size), "n_subjects": len(ids), "excluded": excluded, "cutoff": cutoff,
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "nominal": CHI2_DELTA, "median_excess": float(np.median(x)),
            "fraction_within_nominal": float(np.mean(x <= CHI2_DELTA))}


def bounded_among_interior(rows: list[dict], parameter: str, delta: float) -> dict:
    """Bounded fraction among subjects interior for the parameter, recomputed at `delta` from stored grids."""
    inner = [r for r in rows if parameter in r["profiles"] and r["distance_to_bound"][parameter] > INTERIOR]
    if not inner:
        return {"n": 0, "bounded": 0, "fraction": None}
    bounded = 0
    for row in inner:
        entry = row["profiles"][parameter]
        n = int(entry["n_in_box"])
        verdict = classify_profile(np.asarray(entry["grid"], dtype=float)[:n],
                                   np.asarray(entry["profile"], dtype=float)[:n], delta=delta,
                                   reference=float(entry["reference"]))["verdict"]
        bounded += verdict == "identifiable"
    ci = wilson_ci(bounded, len(inner))
    return {"n": len(inner), "bounded": int(bounded), "fraction": bounded / len(inner),
            "wilson": [ci.low, ci.high]}


def coverage_at(rows: list[dict], parameter: str, delta: float) -> dict:
    """Share of replica subjects whose true value lies inside the interval at `delta` (open ends included)."""
    inside, n = 0, 0
    for row in rows:
        entry = row["profiles"].get(parameter)
        if entry is None:
            continue
        value = excess_at_truth(entry)
        if value is None:
            continue
        n += 1
        inside += value <= delta
    return {"n": n, "covered": int(inside), "fraction": (inside / n) if n else None}


def _rows(config: dict) -> list[dict]:
    return list(largest_matching(profile_lik.ANALYSIS_ID,
                                 lambda p, c=config: profile_lik._matches(p, c)).values())


def _part(replica_rows: list[dict], real_rows: list[dict], parameter: str) -> dict:
    if not replica_rows or not real_rows:
        return {"n": 0}
    cut = calibrated_cutoff(replica_rows, parameter)
    if not cut.get("n"):
        return {"n": 0}
    delta = cut["cutoff"]
    return {"n": cut["n"], "cutoff": cut,
            "real": {"nominal": bounded_among_interior(real_rows, parameter, CHI2_DELTA),
                     "calibrated": bounded_among_interior(real_rows, parameter, delta)},
            "replica": {"nominal": bounded_among_interior(replica_rows, parameter, CHI2_DELTA),
                        "calibrated": bounded_among_interior(replica_rows, parameter, delta),
                        "coverage_nominal": coverage_at(replica_rows, parameter, CHI2_DELTA),
                        "coverage_calibrated": coverage_at(replica_rows, parameter, delta)}}


def h23() -> dict:
    base = {**profile_lik.default_config(), "bounds_scale": 1.0}
    # (a) S_I under iAUC, rate parameterization
    replica_a = []
    for seed in REPLICA_SEEDS:
        replica_a += _rows({**base, "objective": "iauc", "replica": _replica(seed)})
    real_a = _rows({**base, "objective": "iauc"})
    part_a = _part(replica_a, real_a, "insulin_sensitivity")
    # (b) tau1 and p under the trace, coordinates, polished
    coords = {**base, "objective": "trace", "parameterization": "coords", "polish": True}
    replica_b = _rows({**coords, "replica": _replica(0)})
    real_b = _rows(coords)
    part_b = {"log_tau1": _part(replica_b, real_b, "log_tau1"), "log_p": _part(replica_b, real_b, "log_p")}
    out = {"S_I_iauc": part_a, "trace_coordinates": part_b,
           "rule": {"S_I_bounded_max": S_I_BOUNDED_MAX, "tau1_min": TAU1_MIN, "p_min": P_MIN}}
    try:
        a_ok = part_a["real"]["calibrated"]["fraction"] < S_I_BOUNDED_MAX
        tau = part_b["log_tau1"]["real"]["calibrated"]["fraction"]
        p = part_b["log_p"]["real"]["calibrated"]["fraction"]
        b_ok = tau >= TAU1_MIN and p >= P_MIN
    except (KeyError, TypeError):
        out["status"] = "not evaluated"
        return out
    out["part_a_met"], out["part_b_met"] = bool(a_ok), bool(b_ok)
    out["status"] = "met" if (a_ok and b_ok) else "not met"
    return out


# --- H16: heteroskedastic noise ------------------------------------------------------------------------

def heteroskedastic() -> dict:
    """S_I profile under the root and proportional noise structures against the homoskedastic baseline."""
    base = {**profile_lik.default_config(), "objective": "iauc", "bounds_scale": 1.0}
    baseline = {r["subject_id"]: r for r in _rows(base)}
    out = {"baseline": None, "structures": {}}
    if baseline:
        block = profile_lik.summarize_rows(list(baseline.values()), base)
        out["baseline"] = {"n": len(baseline), "S_I_interior": block.get("S_I_bounded_among_interior_for_S_I")}
    for structure in ("root", "proportional"):
        config = {**base, "noise_structure": structure, "profile_only": ["insulin_sensitivity"]}
        rows = _rows(config)
        if not rows:
            out["structures"][structure] = {"n": 0}
            continue
        block = profile_lik.summarize_rows(rows, config)
        changes = {"same": 0, "more_bounded": 0, "less_bounded": 0}
        for row in rows:
            before = baseline.get(row["subject_id"])
            if before is None:
                continue
            a = before["profiles"]["insulin_sensitivity"]["classification"]["verdict"] == "identifiable"
            b = row["profiles"]["insulin_sensitivity"]["classification"]["verdict"] == "identifiable"
            changes["same" if a == b else ("more_bounded" if b else "less_bounded")] += 1
        sigma = [r["noise"]["sigma"]["iauc"] for r in rows if r.get("noise")]
        out["structures"][structure] = {
            "n": len(rows), "S_I_interior": block.get("S_I_bounded_among_interior_for_S_I"),
            "all_bounded": block["parameters"]["insulin_sensitivity"]["bounded"],
            "verdicts": {k: block["parameters"]["insulin_sensitivity"][k]
                         for k in ("identifiable", "one_sided", "flat")},
            "changes_vs_baseline": changes, "median_sigma": float(np.median(sigma)) if sigma else None}
    done = [v for v in out["structures"].values() if v.get("n")]
    if len(done) < 2 or out["baseline"] is None:
        out["status"] = "not evaluated"
    else:
        fractions = [v["S_I_interior"]["fraction"] for v in done if v.get("S_I_interior")]
        out["h5_si_threshold"] = 0.80
        out["clause_unmet_under_all"] = bool(fractions and all(f < 0.80 for f in fractions))
        out["status"] = "reported (exploratory)"
    return out


def main() -> None:
    summary = {"h23": h23(), "heteroskedastic": heteroskedastic()}
    path = RESULTS / "phase10_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
