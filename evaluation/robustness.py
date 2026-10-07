"""H10 (diagnostic robustness) and the synthetic clause of H6, from stored results (Amendment 5).

**H10.** `PREREG.md`: Kendall tau at least 0.8 for the parameter ordering across 5 initializations, 3 bound
settings, log versus linear parameterization, and Adam versus L-BFGS; `S_I` ranked first in at least 95% of
subjects interior for `S_I`. Amendment 5 fixes the operational reading, before the sweeps were run:

* the ordering is the ordering of the three parameters by the bound-projected gradient diagnostic
  (`projected_norm`, the diagnostic of the submitted paper), reference = Adam, rate space, the population
  start, the primary box;
* for each subject and each of the four factors, Kendall's tau-b between the reference ordering and the
  ordering under each level of the factor, averaged over the levels of the factor (five initializations;
  the 0.5x and 2x boxes; log parameterization; L-BFGS);
* a factor passes when the median over subjects of that mean is at least 0.8. With three parameters, tau
  takes values in {-1, -1/3, 1/3, 1} and 0.8 or more therefore means an identical ordering;
* H10 is met when all four factors pass and `S_I` is ranked first in at least 95% of the subjects
  interior for `S_I` (estimate more than 1% of the box width from both bounds, at the reference fit).

**H6, synthetic clause.** The second half of the H6 rule needs a study where the truth is known. Here it is
the random-truth replica: each synthetic subject has the real meal schedule, true rates and insulin
sensitivity drawn uniformly inside the box, CGM noise from the subject's own residuals, and carbohydrate
error of CV 0.25. For each synthetic subject the gradient diagnostic and the iAUC profile likelihood are
computed on the synthetic data, and the original thresholds are applied: median per-subject Spearman between
the diagnostic and the profile strength at least 0.7, and AUC at least 0.85 for detecting a profile-flat
parameter. Seeds 0 and 1 are pooled (a seed redraws every truth).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from evaluation.gradient_diag import is_default
from evaluation.results_io import largest_matching

PARAMS = ("insulin_sensitivity", "gastric_emptying", "carb_absorption")
TAU_MIN = 0.80
SI_FIRST_MIN = 0.95
INIT_SEEDS = tuple(range(5))
SYNTHETIC_SEEDS = (0, 1)
H6_SPEARMAN_MIN, H6_AUC_MIN = 0.7, 0.85


def random_replica(seed: int) -> dict:
    return {"seed": seed, "cgm": True, "carb_cv": 0.25, "truth": "random"}


def _variant(payload: dict, bounds: float = 1.0, optimizer: str = "adam", init_seed=None,
             log_param: bool = False, replica=None) -> bool:
    return (payload.get("cohort", "cgmacros") == "cgmacros"
            and abs(payload["bounds_scale"] - bounds) < 1e-12
            and payload.get("optimizer", "adam") == optimizer
            and payload.get("init_seed") == init_seed
            and bool(payload.get("log_param", False)) == log_param
            and payload.get("replica") == replica
            and payload.get("carb_scale") in (None, 1, 1.0))


def _vector(row: dict) -> np.ndarray:
    return np.array([row["projected_norm"][p] for p in PARAMS], dtype=float)


def _tau(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.kendalltau(a, b).statistic)


def _interior_for_S_I(subjects) -> set[str]:
    from evaluation.ladder import a4_results
    rows = a4_results("iauc", 1.0)
    return {s for s in subjects if s in rows and rows[s]["fit"]["at_bound"].get("insulin_sensitivity") is None}


def _si_first(diag: dict, subjects) -> dict:
    subjects = [s for s in subjects if s in diag]
    if not subjects:
        return {"n": 0}
    first = [int(np.argmax(_vector(diag[s]))) == 0 for s in subjects]
    return {"n": len(subjects), "S_I_first": int(sum(first)), "fraction": float(np.mean(first))}


def h10() -> dict:
    reference = largest_matching("A8a_gradient_diag", lambda p: is_default(p, 1.0))
    if not reference:
        return {"n": 0, "status": "not evaluated"}
    levels = {
        "initialization": {f"seed_{k}": (lambda p, k=k: _variant(p, init_seed=k)) for k in INIT_SEEDS},
        "bounds": {"0.5x": lambda p: _variant(p, bounds=0.5), "2x": lambda p: _variant(p, bounds=2.0)},
        "parameterization": {"log": lambda p: _variant(p, log_param=True)},
        "optimizer": {"lbfgs": lambda p: _variant(p, optimizer="lbfgs")},
    }
    out = {"n_reference": len(reference), "factors": {}}
    interior = _interior_for_S_I(reference)
    out["reference_S_I_first_interior"] = _si_first(reference, sorted(interior))
    out["reference_S_I_first_all"] = _si_first(reference, sorted(reference))
    factor_pass = {}
    for factor, members in levels.items():
        per_level, by_subject = {}, {}
        for label, match in members.items():
            rows = largest_matching("A8a_gradient_diag", match)
            if not rows:
                per_level[label] = {"n": 0}
                continue
            taus = {s: _tau(_vector(reference[s]), _vector(rows[s])) for s in rows if s in reference}
            finite = np.array([v for v in taus.values() if np.isfinite(v)])
            per_level[label] = {
                "n": len(taus), "median_tau": float(np.median(finite)) if finite.size else None,
                "fraction_identical_ordering": float(np.mean(finite >= 0.999)) if finite.size else None,
                "S_I_first_interior": _si_first(rows, sorted(interior & set(rows)))}
            for s, v in taus.items():
                by_subject.setdefault(s, []).append(v)
        mean_tau = np.array([np.nanmean(v) for v in by_subject.values() if np.isfinite(v).any()])
        entry = {"levels": per_level, "n_subjects": int(mean_tau.size)}
        if mean_tau.size:
            entry["median_of_mean_tau"] = float(np.median(mean_tau))
            entry["passes"] = bool(np.median(mean_tau) >= TAU_MIN)
        else:
            entry["passes"] = None
        out["factors"][factor] = entry
        factor_pass[factor] = entry["passes"]
    si = out["reference_S_I_first_interior"]
    out["S_I_first_pass"] = bool(si.get("n") and si["fraction"] >= SI_FIRST_MIN)
    if any(v is None for v in factor_pass.values()):
        out["status"] = "not evaluated"
    else:
        out["status"] = "met" if (all(factor_pass.values()) and out["S_I_first_pass"]) else "not met"
    return out


# --- the synthetic study ------------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) > 0 and np.ptp(b) > 0:
        return float(stats.spearmanr(a, b).statistic)
    return float("nan")


def synthetic_rows(seeds=SYNTHETIC_SEEDS) -> list[dict]:
    """Per synthetic subject: the profile, the diagnostic and the truth, matched on (seed, subject)."""
    rows = []
    for seed in seeds:
        replica = random_replica(seed)
        profile = largest_matching(
            "A5_profile", lambda p: p["objective"] == "iauc" and abs(p["bounds_scale"] - 1.0) < 1e-12
            and p.get("cohort", "cgmacros") == "cgmacros" and p.get("replica") == replica
            and p.get("parameterization", "rates") == "rates" and p.get("polish") is None)
        diag = largest_matching("A8a_gradient_diag", lambda p: _variant(p, replica=replica))
        for subject in sorted(set(profile) & set(diag)):
            rows.append({"seed": seed, "subject": subject, "profile": profile[subject],
                         "diag": diag[subject]})
    return rows


def h6_synthetic(seeds=SYNTHETIC_SEEDS) -> dict:
    from evaluation.stored_analyses import _auc
    rows = synthetic_rows(seeds)
    if not rows:
        return {"n": 0, "status": "not evaluated"}
    rho, flat_labels, scores = [], [], []
    for row in rows:
        d = np.array([row["diag"]["projected_norm"][p] for p in PARAMS], dtype=float)
        strength = np.array([row["profile"]["profiles"][p]["classification"]["max_rise"] for p in PARAMS])
        rho.append(_spearman(d, strength))
        for j, p in enumerate(PARAMS):
            flat_labels.append(row["profile"]["profiles"][p]["classification"]["verdict"] == "flat")
            scores.append(d[j])
    rho = np.array([v for v in rho if np.isfinite(v)])
    flat_labels, scores = np.array(flat_labels), np.array(scores)
    auc = _auc(-scores[flat_labels], -scores[~flat_labels])
    out = {"n": len(rows), "seeds": list(seeds),
           "spearman_vs_profile_strength": {
               "n": int(rho.size), "median": float(np.median(rho)) if rho.size else None,
               "q1": float(np.percentile(rho, 25)) if rho.size else None,
               "q3": float(np.percentile(rho, 75)) if rho.size else None},
           "auc_detecting_flat": auc, "n_flat_pairs": int(flat_labels.sum()), "n_pairs": int(flat_labels.size),
           "thresholds": {"median_spearman_min": H6_SPEARMAN_MIN, "auc_min": H6_AUC_MIN}}
    median = out["spearman_vs_profile_strength"]["median"]
    met = (median is not None and median >= H6_SPEARMAN_MIN and np.isfinite(auc) and auc >= H6_AUC_MIN)
    out["status"] = "met" if met else "not met"
    out["recovery"] = recovery(rows)
    return out


def recovery(rows: list[dict]) -> dict:
    """Exploratory, no threshold: where in the box is each parameter recoverable?

    Bounded fraction among subjects interior for that parameter, by tercile of the TRUE value, and the
    coverage of the true value by the bounded intervals. Terciles are of the true values pooled over seeds.
    """
    from evaluation.stats_utils import wilson_ci
    out = {}
    for parameter in PARAMS:
        truth = np.array([r["profile"]["profiles"][parameter]["truth"] for r in rows])
        edges = np.percentile(truth, [100 / 3, 200 / 3])
        tercile = np.digitize(truth, edges)
        bounded = np.array([r["profile"]["profiles"][parameter]["classification"]["verdict"] == "identifiable"
                            for r in rows])
        interior = np.array([r["profile"]["distance_to_bound"][parameter] > 0.01 for r in rows])
        covered = [(r["profile"]["profiles"][parameter].get("truth_covered")) for r in rows]
        block = {"terciles": {}}
        for t, label in enumerate(("low", "middle", "high")):
            m = (tercile == t) & interior
            n = int(m.sum())
            b = int((bounded & m).sum())
            ci = wilson_ci(b, n) if n else None
            block["terciles"][label] = {
                "truth_range": [float(truth[tercile == t].min()), float(truth[tercile == t].max())],
                "n_interior": n, "bounded": b, "fraction": (b / n) if n else None,
                "wilson": [ci.low, ci.high] if ci else None}
        flags = [c for c in covered if c is not None]
        block["coverage_of_bounded"] = {"n": len(flags), "covered": int(sum(flags)),
                                        "fraction": float(np.mean(flags)) if flags else None}
        block["all_bounded"] = {"n": len(rows), "bounded": int(bounded.sum()),
                                "fraction": float(bounded.mean())}
        block["interior"] = {"n": int(interior.sum()), "bounded": int((bounded & interior).sum()),
                             "fraction": float((bounded & interior).sum() / max(interior.sum(), 1))}
        theta_ml = np.array([r["profile"]["theta_ml"][parameter] for r in rows])
        block["median_abs_log_error_of_estimate"] = float(np.median(np.abs(np.log(theta_ml / truth))))
        out[parameter] = block
    return out
