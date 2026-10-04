"""A13: the feature-leakage finding on Shanghai, with bootstrap intervals.

The earlier draft reported that SNPE's insulin sensitivity correlates with HbA1c on the Shanghai cohort
(raw correlation about -0.4) and that this collapses once fasting glucose is controlled, because SNPE's
summary features include the absolute pre-meal baseline. The gradient estimate does not rely on that
feature. That statement had point estimates only. This recomputes it from the cohort cache with intervals:

per subject  SNPE posterior-mean S_I; gradient S_I (the regularized prediction fit, 150 steps, on all of
             the subject's meals); the mean pre-meal CGM baseline ("fasting"); SNPE's own baseline
             summary feature;
cohort       raw Pearson correlation of each S_I with HbA1c; partial correlation controlling fasting;
             their difference between the two estimators; and the correlation of SNPE's S_I with its own
             baseline feature. Bootstrap over SUBJECTS, 2000 resamples, percentile intervals.

Run:  python -m evaluation.runner evaluation.leakage_ci --workers 30
      python -m evaluation.leakage_ci            # summary from stored per-subject results
"""
from __future__ import annotations

import hashlib

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

from evaluation.cohort_data import load_cohort                          # noqa: E402
from evaluation.subject_source import window_for                        # noqa: E402

ANALYSIS_ID = "A13_leakage"
N_BOOT = 2000


def default_config() -> dict:
    return {"cohort": "shanghai", "min_meals": 3, "limit": None, "steps": 150}


def units(config: dict) -> list[str]:
    return [s.subject_id for s in load_cohort(config["cohort"], min_meals=config["min_meals"],
                                              limit=config["limit"])]


_POSTERIOR = {}


def _posterior():
    if "p" not in _POSTERIOR:
        from evaluation.clinical_recovery import load_snpe
        _POSTERIOR["p"] = load_snpe()
    return _POSTERIOR["p"]


def run_unit(unit: str, config: dict) -> dict:
    from personalization import npe, snpe_infer
    from personalization.fit_general import fit_subject
    from personalization.objectives import ObjectiveSpec

    subject = next(s for s in load_cohort(config["cohort"], min_meals=config["min_meals"],
                                          limit=config["limit"]) if s.subject_id == unit)
    records = list(subject.records)
    meals = [{"carbs_g": r["carbs_g"], "glucose": r["glucose"]} for r in records]
    estimate = snpe_infer.point_estimate(meals, _posterior(), profile=subject.profile)
    snpe_si = estimate.get("insulin_sensitivity") if estimate else None

    fit = fit_subject(subject, ObjectiveSpec(window=window_for(subject)), steps=config["steps"])
    gradient_si = fit.theta["insulin_sensitivity"]

    baseline = [float(np.mean(r["glucose"]["values"][:6])) for r in records]
    feature = [npe.summary_stats(r["glucose"]["values"], -30.0, 5.0, 0.0)[3] for r in records]
    feature = [f for f in feature if np.isfinite(f)]
    return {"subject_id": unit, "n_meals": len(records), "snpe_si": snpe_si, "gradient_si": gradient_si,
            "hba1c": subject.clinical.get("HbA1c"), "fasting": float(np.mean(baseline)),
            "snpe_baseline_feature": float(np.mean(feature)) if feature else float(np.mean(baseline)),
            "macros": {}}


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _partial(x, y, z):
    z1 = np.column_stack([np.ones(len(z)), z])
    rx = x - z1 @ np.linalg.lstsq(z1, x, rcond=None)[0]
    ry = y - z1 @ np.linalg.lstsq(z1, y, rcond=None)[0]
    return _corr(rx, ry)


def statistics_of(arr: dict) -> dict:
    s, g, h = arr["snpe_si"], arr["gradient_si"], arr["hba1c"]
    f, b = arr["fasting"], arr["snpe_baseline_feature"]
    out = {
        "raw_snpe": _corr(s, h), "raw_gradient": _corr(g, h),
        "partial_snpe": _partial(s, h, f), "partial_gradient": _partial(g, h, f),
        "snpe_vs_own_baseline_feature": _corr(b, s), "fasting_vs_hba1c": _corr(f, h),
    }
    out["raw_difference"] = out["raw_snpe"] - out["raw_gradient"]
    out["partial_difference"] = out["partial_snpe"] - out["partial_gradient"]
    out["drop_snpe"] = out["raw_snpe"] - out["partial_snpe"]
    out["drop_gradient"] = out["raw_gradient"] - out["partial_gradient"]
    return out


def summarize(seed: int = 0) -> dict:
    from evaluation.results_io import largest_matching
    rows = [r for r in largest_matching(ANALYSIS_ID, lambda p: "snpe_si" in p).values()
            if r["snpe_si"] is not None and r["hba1c"] is not None
            and np.isfinite(r["snpe_si"]) and np.isfinite(r["gradient_si"])]
    if len(rows) < 10:
        return {"n": len(rows), "note": "too few subjects"}
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in
           ("snpe_si", "gradient_si", "hba1c", "fasting", "snpe_baseline_feature")}
    point = statistics_of(arr)
    digest = hashlib.sha256(f"leakage|{seed}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    draws = {k: [] for k in point}
    n = len(rows)
    for _ in range(N_BOOT):
        index = rng.integers(0, n, n)
        stats = statistics_of({k: v[index] for k, v in arr.items()})
        for k, v in stats.items():
            draws[k].append(v)
    out = {"n": n, "n_resamples": N_BOOT, "statistics": {}}
    for k, value in point.items():
        d = np.asarray(draws[k], dtype=float)
        d = d[np.isfinite(d)]
        out["statistics"][k] = {"estimate": value, "low": float(np.percentile(d, 2.5)),
                                "high": float(np.percentile(d, 97.5))}
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(summarize(), indent=2))
