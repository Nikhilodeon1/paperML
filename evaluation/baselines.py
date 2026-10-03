"""Baselines (paper §Results) — the comparisons a reviewer requires before believing the claim.

Two families:

1. Predictive baselines for held-out iAUC — population constant (Si=1.0), personal-mean iAUC,
   and persistence. These are computed inside ``snpe_kfold.py`` alongside the estimators; they are
   re-exported here as thin helpers for completeness.

2. The DEMOGRAPHIC baseline for the clinical-recovery claim — the one prior scripts omitted. The
   worry: is r(Si, HbA1c) just demographics in disguise (older/heavier people have both lower Si
   and higher HbA1c)? We test it directly:
     * how well age+BMI+sex ALONE predict HbA1c (multiple-R),
     * r(Si, HbA1c) on its own,
     * the PARTIAL correlation r(Si, HbA1c | age, BMI, sex) — Si's unique contribution after
       regressing demographics out of both,
     * the full model (demographics + Si) multiple-R.
   If the partial correlation stays clearly negative and the full model beats demographics-only,
   the recovered Si carries physiological signal beyond body size — the SBI claim survives.

Run:  python -m evaluation.baselines --estimator snpe
"""
from __future__ import annotations

import numpy as np

import paper_config as cfg
from evaluation.cgmacros import load_bio
from evaluation import clinical_recovery


def _ols_predict(Z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit y ~ [1, Z], return (prediction, multiple-R = corr(pred, y))."""
    X = np.column_stack([np.ones(len(Z)), Z]) if Z.ndim == 2 else np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    R = float(np.corrcoef(pred, y)[0, 1]) if pred.std() > 0 else float("nan")
    return pred, R


def partial_corr(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """Partial correlation r(x, y | Z): correlate the residuals of x~Z and y~Z."""
    Z1 = np.column_stack([np.ones(len(Z)), Z])

    def resid(v):
        beta, *_ = np.linalg.lstsq(Z1, v, rcond=None)
        return v - Z1 @ beta

    rx, ry = resid(x), resid(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else float("nan")


def demographic_baseline(estimator: str = "snpe", limit: int | None = None) -> dict:
    """Compare the recovered Si to an age+BMI+sex demographic model for predicting HbA1c."""
    rows, _ = clinical_recovery.collect(estimator, limit=limit)
    bio = load_bio()
    recs = []
    for r in rows:
        b = bio.get(r["subject"], {})
        if b.get("age") and b.get("bmi") and r["hba1c"] is not None and np.isfinite(r["si"]):
            recs.append((r["si"], r["hba1c"], b["age"], b["bmi"],
                         1.0 if b.get("sex") == "male" else 0.0))
    if len(recs) < 5:
        return {"n": len(recs), "error": "too few subjects with full demographics"}

    si = np.array([x[0] for x in recs])
    a1c = np.array([x[1] for x in recs])
    Z = np.array([[x[2], x[3], x[4]] for x in recs])   # age, bmi, sex

    _, r_demo = _ols_predict(Z, a1c)                    # demographics-only multiple-R
    r_si = float(np.corrcoef(si, a1c)[0, 1])            # Si alone
    _, r_full = _ols_predict(np.column_stack([Z, si]), a1c)  # demographics + Si
    pr = partial_corr(si, a1c, Z)                       # Si unique, controlling demographics

    return {
        "n": len(recs), "estimator": estimator,
        "r_demographics_only": r_demo,          # |R| of age+BMI+sex predicting HbA1c
        "r_si_alone": r_si,
        "r_full_model": r_full,                 # demographics + Si
        "partial_r_si_given_demographics": pr,  # the decisive number
        "si_adds_over_demographics": bool(abs(r_full) > abs(r_demo) + 1e-6 and abs(pr) > 0.2),
    }


def main() -> None:
    import argparse
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", choices=["grid", "rf", "snpe"], default="snpe")
    ap.add_argument("--subjects", type=int, default=None)
    args = ap.parse_args()

    d = demographic_baseline(args.estimator, limit=args.subjects)
    print("=" * 76)
    print(f"DEMOGRAPHIC BASELINE — does {args.estimator} Si beat age+BMI+sex for HbA1c?")
    print("=" * 76)
    if "error" in d:
        print("  " + d["error"])
        return
    print(f"  n subjects: {d['n']}")
    print(f"  |R| demographics only (age+BMI+sex -> HbA1c) : {abs(d['r_demographics_only']):.3f}")
    print(f"  r(Si, HbA1c) alone                           : {d['r_si_alone']:+.3f}")
    print(f"  |R| full model (demographics + Si)           : {abs(d['r_full_model']):.3f}")
    print(f"  PARTIAL r(Si, HbA1c | age, BMI, sex)         : {d['partial_r_si_given_demographics']:+.3f}")
    print(f"\n  VERDICT: Si carries signal beyond demographics: "
          f"{'YES' if d['si_adds_over_demographics'] else 'WEAK — demographics explain most of it'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
