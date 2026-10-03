"""Carb-estimation sensitivity — attributes the Shanghai gradient miss to estimated carbs.

Gradient inference recovers Si-vs-HbA1c on CGMacros (Spearman -0.73) with MEASURED carbs, but not on
ShanghaiT2DM (-0.14) where carbs are keyword-ESTIMATED. To test whether estimated carbs are the
cause (rather than a fundamental method failure), we inject multiplicative carb-estimate noise into
CGMacros — keeping the real observed iAUC, but feeding the gradient fit a NOISY carb value (exactly
the Shanghai situation) — and sweep the noise level. If the CGMacros correlation degrades toward the
Shanghai value as carb noise grows, the external miss is a data limitation (estimated carbs), not a
method failure.

Run:  python -m evaluation.carb_sensitivity
"""
from __future__ import annotations

import numpy as np

import paper_config as cfg


def run(cv_levels=(0.0, 0.15, 0.30, 0.45), n_steps: int = 120, seed: int = cfg.SEED) -> dict:
    from evaluation.cgmacros import subjects, load_bio, _profile
    from evaluation.gradient_inference_results import _subject_meals, _base_for
    from personalization.gradient_fit import fit_parameters
    from evaluation.clinical_recovery import bootstrap_corr

    cfg.set_all_seeds(seed)
    bio = load_bio()
    data = []
    for sid, meals in subjects():
        b = bio.get(sid)
        prof = _profile(b)
        ms = _subject_meals(meals)
        if prof and b and b["hba1c"] is not None and len(ms) >= 4:
            data.append((ms, prof, b["hba1c"]))

    out = {}
    for cv in cv_levels:
        rng = np.random.default_rng(seed)
        si, a1c = [], []
        for ms, prof, hb in data:
            pert = [{**m, "carbs_g": max(1.0, m["carbs_g"] * (1.0 + rng.normal(0, cv)))} for m in ms]
            res = fit_parameters(pert, base=_base_for(prof), n_steps=n_steps)
            si.append(res["Si"]); a1c.append(hb)
        r = bootstrap_corr(si, a1c, "spearman")
        out[round(cv, 2)] = {"spearman": r["r"], "ci": (r["lo"], r["hi"]), "n": r["n"]}
    return out


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    res = run()
    print("=" * 68)
    print("CARB-ESTIMATION SENSITIVITY (CGMacros, gradient Si vs HbA1c)")
    print("=" * 68)
    print(f"  {'carb noise CV':>15}{'Spearman(Si,HbA1c)':>22}{'95% CI':>18}")
    for cv, d in res.items():
        print(f"  {cv:>15.2f}{d['spearman']:>+22.3f}   [{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]")
    print("-" * 68)
    print("  Shanghai (estimated carbs) reference: gradient Spearman -0.14")
    print("  => if the correlation decays toward -0.14 as carb noise grows, the")
    print("     external miss is attributable to estimated carbs, not the method.")
    print("=" * 68)


if __name__ == "__main__":
    main()
