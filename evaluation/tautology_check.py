"""Is SNPE's Shanghai Si-vs-HbA1c real, or tautological via fasting glucose? (integrity check)

On the all-T2D Shanghai cohort, HbA1c is driven by chronic fasting hyperglycemia. SNPE's summary
features include the absolute pre-meal baseline, so its "Si" may be partly reading fasting glucose —
which trivially predicts HbA1c. This module quantifies that:

  1. partial r(SNPE_Si,  HbA1c | fasting_glucose)   — near 0 => SNPE result is tautological
  2. partial r(gradient_Si, HbA1c | fasting_glucose) — expected near 0 (gradient fits iAUC, which
     already subtracts the baseline) — confirms gradient isn't fasting-contaminated
  3. r(SNPE_baseline_feature, SNPE_Si)               — high (>0.6) => direct feature leakage

`fasting_glucose` = per-subject mean pre-meal CGM baseline (what the feature actually sees).

Run:  python -m evaluation.tautology_check
"""
from __future__ import annotations

import numpy as np

import paper_config as cfg
from evaluation.baselines import partial_corr


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def run(limit: int | None = None, n_steps: int = 150) -> dict:
    from evaluation.shanghai_loader import load_shanghai_t2dm
    from evaluation.cross_dataset_validation import _profile, _gradient_fit, _snpe_si
    from evaluation.clinical_recovery import build_rf, load_snpe
    from personalization import npe

    cfg.set_all_seeds()
    subs = load_shanghai_t2dm()
    if limit:
        subs = subs[:limit]
    posterior = load_snpe()

    snpe_si, grad_si, a1c, fasting, base_feat = [], [], [], [], []
    for sub in subs:
        prof = _profile(sub)
        meals = sub["meals"]
        ssi = _snpe_si(meals, prof, posterior)
        gsi = _gradient_fit(meals, prof, n_steps)["Si"]
        if ssi is None or gsi is None or not np.isfinite(ssi) or not np.isfinite(gsi):
            continue
        # per-subject fasting = mean pre-meal CGM baseline; SNPE baseline feature = summary stat #3
        bl = [np.mean(m["cgm_curve"][:6]) for m in meals]
        feats = [npe.summary_stats(m["cgm_curve"], -30.0, 5.0, 0.0)[3] for m in meals]
        feats = [f for f in feats if np.isfinite(f)]
        snpe_si.append(ssi); grad_si.append(gsi); a1c.append(sub["HbA1c"])
        fasting.append(float(np.mean(bl)))
        base_feat.append(float(np.mean(feats)) if feats else float(np.mean(bl)))

    Z = np.array(fasting).reshape(-1, 1)
    return {
        "n": len(a1c),
        "raw_snpe_hba1c": _corr(snpe_si, a1c),
        "raw_grad_hba1c": _corr(grad_si, a1c),
        "partial_snpe_hba1c_given_fasting": partial_corr(np.array(snpe_si), np.array(a1c), Z),
        "partial_grad_hba1c_given_fasting": partial_corr(np.array(grad_si), np.array(a1c), Z),
        "r_baseline_feature_vs_snpe_si": _corr(base_feat, snpe_si),
        "r_fasting_vs_hba1c": _corr(fasting, a1c),
    }


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    r = run()
    print("=" * 74)
    print(f"TAUTOLOGY CHECK — SNPE vs gradient on Shanghai (n={r['n']})")
    print("=" * 74)
    print(f"  context: r(fasting glucose, HbA1c) = {r['r_fasting_vs_hba1c']:+.3f}")
    print(f"  raw     r(SNPE Si, HbA1c)                    = {r['raw_snpe_hba1c']:+.3f}")
    print(f"  raw     r(gradient Si, HbA1c)                = {r['raw_grad_hba1c']:+.3f}")
    print(f"  PARTIAL r(SNPE Si, HbA1c | fasting glucose)  = {r['partial_snpe_hba1c_given_fasting']:+.3f}")
    print(f"  PARTIAL r(gradient Si, HbA1c | fasting)      = {r['partial_grad_hba1c_given_fasting']:+.3f}")
    print(f"  LEAKAGE r(SNPE baseline feature, SNPE Si)    = {r['r_baseline_feature_vs_snpe_si']:+.3f}")
    print("-" * 74)
    ps = r["partial_snpe_hba1c_given_fasting"]
    print("  VERDICT: SNPE Shanghai result is "
          + ("TAUTOLOGICAL (partial ~0) — do not cite as generalization"
             if abs(ps) < 0.25 else "REAL beyond fasting (partial <=-0.25)"))
    print("=" * 74)


if __name__ == "__main__":
    main()
