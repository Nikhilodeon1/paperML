"""Hall 2018 external validation — gradient Si vs HbA1c / SSPG on standardized-meal CGM.

Second external cohort (after ShanghaiT2DM). Frozen transfer: gradient fit from population defaults,
no Hall-specific tuning. Primary target HbA1c (n>=20, citable); SSPG reported with an n<20 caveat.
Also the identifiability replication (Si vs gastric/carb gradient-norm ratio) on a THIRD cohort.

Run:  python -m evaluation.hall_validation
"""
from __future__ import annotations

import statistics
import time

import numpy as np

import paper_config as cfg
from simulation import PhysioParams
from evaluation.clinical_recovery import bootstrap_corr


def _profile(sub) -> dict:
    d = sub["demographics"]
    return {"weight_kg": d["weight_kg"], "height_cm": d["height_cm"], "age": d["age"],
            "sex": "male"}   # Hall clinical table has no sex; default (affects glucose negligibly)


def run(n_steps: int = 150) -> dict:
    from simulation.jax_engine import JaxPhysioParams
    from personalization.gradient_fit import fit_parameters
    from evaluation.hall_loader import load_hall, summary

    cfg.set_all_seeds()
    subs = load_hall()
    si, a1c, sspg = [], [], []
    sspg_si, gnorms = [], {"insulin_sensitivity": [], "gastric_emptying": [], "carb_absorption": []}
    t0 = time.time()
    for sub in subs:
        prof = _profile(sub)
        base = JaxPhysioParams.from_numpy(PhysioParams.from_profile(prof))
        res = fit_parameters(sub["meals"], n_steps=n_steps, base=base)
        s = res["Si"]
        if not np.isfinite(s):
            continue
        for k in gnorms:
            gnorms[k].append(res["gradient_norms"][k])
        if sub["HbA1c"] is not None:
            si.append(s); a1c.append(sub["HbA1c"])
        if sub["SSPG"] is not None:
            sspg_si.append(s); sspg.append(sub["SSPG"])

    gm = {k: statistics.fmean(v) for k, v in gnorms.items()}
    return {
        "summary": summary(subs),
        "ms_per_subject": round(1000 * (time.time() - t0) / max(len(subs), 1)),
        "n_hba1c": len(a1c),
        "hba1c": bootstrap_corr(si, a1c, "spearman"),
        "hba1c_pearson": bootstrap_corr(si, a1c, "pearson"),
        "n_sspg": len(sspg),
        "sspg": bootstrap_corr(sspg_si, sspg, "spearman") if len(sspg) >= 3 else None,
        "grad_ratio_Si_over_gastric": round(gm["insulin_sensitivity"] / max(gm["gastric_emptying"], 1e-9), 1),
        "grad_ratio_Si_over_carb": round(gm["insulin_sensitivity"] / max(gm["carb_absorption"], 1e-9), 1),
    }


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    r = run()
    s = r["summary"]
    print("=" * 74)
    print(f"HALL 2018 EXTERNAL VALIDATION — {s['n_subjects']} subjects, {s['n_meals_total']} meals "
          f"({r['ms_per_subject']} ms/subj)")
    print("=" * 74)
    print(f"  cohort: {s['dx']}  HbA1c {s['hba1c_range']}  SSPG {s['sspg_range']}")
    h, hp = r["hba1c"], r["hba1c_pearson"]
    print(f"  PRIMARY  Si vs HbA1c (n={r['n_hba1c']}): Spearman {h['r']:+.3f} "
          f"[{h['lo']:+.2f},{h['hi']:+.2f}]  Pearson {hp['r']:+.3f}")
    if r["sspg"]:
        sp = r["sspg"]
        flag = "  [n<20: NOT citable per protocol]" if r["n_sspg"] < 20 else ""
        print(f"  SSPG     Si vs SSPG  (n={r['n_sspg']}): Spearman {sp['r']:+.3f} "
              f"[{sp['lo']:+.2f},{sp['hi']:+.2f}]{flag}")
    print(f"  IDENTIFIABILITY: Si grad-norm / gastric = {r['grad_ratio_Si_over_gastric']}x, "
          f"/ carb = {r['grad_ratio_Si_over_carb']}x  (CGMacros ~20-60x, Shanghai 10.7x)")
    print("=" * 74)


if __name__ == "__main__":
    main()
