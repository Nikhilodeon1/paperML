"""T7: analyses that need no new simulation, only stored results.

(1) H6 on real data. Does the gradient-magnitude diagnostic of the submitted paper agree with the
    likelihood-based measures of what a parameter is worth?
      * per subject, the Spearman correlation of the three parameters' ranking by the (bound-projected)
        gradient diagnostic against their ranking by (a) profile-likelihood strength and (b) Fisher
        information;
      * across all subject-parameter pairs, the AUC of the diagnostic for detecting a profile-FLAT
        parameter (a flat parameter should have a SMALL diagnostic).
    Profile strength is the largest rise of the profile above its minimum inside the box (a bounded
    interval needs a rise of 1.92; a flat profile never gets there). Fisher information for parameter
    j is `1 / (F^+)_{jj}`: the information left after the other two are profiled out, so a compensating
    pair is not credited with information it does not have. With only three parameters per subject a
    Spearman coefficient can take few values, so the median and interquartile range are reported with
    the count of each value.
(2) Subject-level correlates (a reviewer suggestion). What distinguishes the subjects whose `S_I`
    profile interval is bounded, or whose timing parameters sit on a bound, from the rest: number of
    meals, spread of carbohydrate (max / min), mean observed iAUC, and the `S_I` estimate.

Both are reported whatever the number is.

Run:  python -m evaluation.stored_analyses
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from evaluation.ladder import a4_results
from evaluation.results_io import largest_matching
from evaluation.cohort_data import load_cohort

PARAMS = ("insulin_sensitivity", "gastric_emptying", "carb_absorption")


def _diagnostic(scale: float) -> dict:
    from evaluation.gradient_diag import is_default
    return largest_matching("A8a_gradient_diag", lambda p: is_default(p, scale))


def _profiles(scale: float) -> dict:
    return largest_matching("A5_profile", lambda p: p["objective"] == "iauc"
                            and abs(p["bounds_scale"] - scale) < 1e-12
                            and p.get("cohort", "cgmacros") == "cgmacros"
                            and p.get("replica") is None and p.get("carb_scale") in (None, 1, 1.0)
                            and p.get("parameterization", "rates") == "rates")


def _auc(scores_positive: np.ndarray, scores_negative: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties counted half (Mann-Whitney)."""
    if len(scores_positive) == 0 or len(scores_negative) == 0:
        return float("nan")
    a = np.asarray(scores_positive)[:, None]
    b = np.asarray(scores_negative)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


def h6(scale: float = 1.0, kind: str = "projected_norm") -> dict:
    diag, prof, fisher = _diagnostic(scale), _profiles(scale), a4_results("iauc", scale)
    subjects = sorted(set(diag) & set(prof) & set(fisher))
    rho_profile, rho_fisher = [], []
    flat_labels, scores = [], []
    for s in subjects:
        d = np.array([diag[s][kind][p] for p in PARAMS], dtype=float)
        strength = np.array([prof[s]["profiles"][p]["classification"]["max_rise"] for p in PARAMS])
        F = np.asarray(fisher[s]["fisher"]["fisher"], dtype=float)
        covariance = np.linalg.pinv(F, rcond=1e-12)
        information = 1.0 / np.maximum(np.diag(covariance), 1e-300)
        for target, bucket in ((strength, rho_profile), (information, rho_fisher)):
            if np.ptp(d) > 0 and np.ptp(target) > 0:
                bucket.append(float(stats.spearmanr(d, target).statistic))
            else:
                bucket.append(float("nan"))
        for j, p in enumerate(PARAMS):
            flat_labels.append(prof[s]["profiles"][p]["classification"]["verdict"] == "flat")
            scores.append(d[j])
    flat_labels = np.array(flat_labels)
    scores = np.array(scores)

    def summary(values):
        v = np.array([x for x in values if np.isfinite(x)])
        if v.size == 0:
            return {"n": 0}
        uniq, counts = np.unique(np.round(v, 3), return_counts=True)
        return {"n": int(v.size), "median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
                "q3": float(np.percentile(v, 75)),
                "values": {str(u): int(c) for u, c in zip(uniq, counts)}}

    auc = _auc(-scores[flat_labels], -scores[~flat_labels])        # flat = positive; small score = flat
    return {"bounds_scale": scale, "diagnostic": kind, "n_subjects": len(subjects),
            "spearman_vs_profile_strength": summary(rho_profile),
            "spearman_vs_fisher_information": summary(rho_fisher),
            "auc_detecting_flat": auc, "n_flat_pairs": int(flat_labels.sum()),
            "n_pairs": int(flat_labels.size),
            "h6_thresholds": {"median_spearman_min": 0.7, "auc_min": 0.85}}


def correlates(scale: float = 1.0) -> dict:
    prof, fisher = _profiles(scale), a4_results("iauc", scale)
    cohort = {s.subject_id: s for s in load_cohort("cgmacros", min_meals=10)}
    rows = []
    for sid in sorted(set(prof) & set(fisher) & set(cohort)):
        sub = cohort[sid]
        carbs = np.array([r["carbs_g"] for r in sub.records], dtype=float)
        carbs = carbs[carbs > 0]
        pr = prof[sid]["profiles"]["insulin_sensitivity"]["classification"]
        rows.append({
            "n_meals": sub.n_meals, "carb_range": float(carbs.max() / max(carbs.min(), 1e-9)),
            "mean_iauc": float(np.mean([r["iauc"] for r in sub.records])),
            "si_estimate": float(fisher[sid]["theta_ml"]["insulin_sensitivity"]),
            "si_bounded": pr["verdict"] == "identifiable", "si_max_rise": pr["max_rise"],
            "timing_pinned_upper": bool(fisher[sid]["fit"]["at_bound"].get("gastric_emptying") == "upper"
                                        or fisher[sid]["fit"]["at_bound"].get("carb_absorption") == "upper"),
        })
    out = {"bounds_scale": scale, "n": len(rows), "covariates": {}}
    for covariate in ("n_meals", "carb_range", "mean_iauc", "si_estimate"):
        x = np.array([r[covariate] for r in rows], dtype=float)
        entry = {}
        for outcome in ("si_max_rise", "si_bounded", "timing_pinned_upper"):
            y = np.array([r[outcome] for r in rows], dtype=float)
            if np.ptp(y) == 0:
                entry[outcome] = {"note": "outcome is constant"}
                continue
            rho = stats.spearmanr(x, y)
            entry[outcome] = {"spearman": float(rho.statistic), "p": float(rho.pvalue)}
        out["covariates"][covariate] = entry
    # Twelve exploratory tests per box: report the Holm-adjusted p-value next to the raw one.
    from evaluation.stats_utils import holm
    keys = [(c, o) for c, e in out["covariates"].items() for o, v in e.items() if "p" in v]
    adjusted = holm([out["covariates"][c][o]["p"] for c, o in keys])["p_adjusted"]
    for (c, o), value in zip(keys, adjusted):
        out["covariates"][c][o]["p_holm"] = value
    out["n_si_bounded"] = int(sum(r["si_bounded"] for r in rows))
    out["n_timing_pinned_upper"] = int(sum(r["timing_pinned_upper"] for r in rows))
    return out


if __name__ == "__main__":
    import json
    for scale in (1.0, 2.0):
        print(json.dumps(h6(scale), indent=1)[:1500])
        print(json.dumps(correlates(scale), indent=1)[:2500])
