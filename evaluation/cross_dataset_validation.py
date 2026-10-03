"""External validation on ShanghaiT2DM (weakness-1 fix) — pure transfer, frozen CGMacros artifacts.

Runs all four estimators on the Shanghai cohort with NO Shanghai-specific tuning: gradient fit from
population defaults, the RF trained on CGMacros synthetic data, the SNPE posterior artifact, and a
1-D Si grid with the same bounds. Correlates each method's recovered Si against HbA1c. Also
replicates the gradient identifiability structure (Si vs gastric/carb) on this independent cohort.

Run:  python -m evaluation.cross_dataset_validation           # full (~20 min)
      python -m evaluation.cross_dataset_validation --subjects 6
"""
from __future__ import annotations

import statistics
import time

import numpy as np

import paper_config as cfg
from simulation import PhysioParams
from evaluation.cgmacros import predict_meal_iauc
from evaluation.clinical_recovery import bootstrap_corr

# Shanghai meal cgm_curve is the -30..+180 min 5-min grid (meal at t=0, index 6).
_T0, _STEP, _MEAL_T = -30.0, 5.0, 0.0


def _profile(sub) -> dict:
    d = sub["demographics"]
    return {"weight_kg": d["weight_kg"], "height_cm": d["height_cm"], "age": d["age"],
            "sex": d["sex"]}   # demographics only — no diagnosis leaked into the prior


# --- per-method Si on the Shanghai per-subject schema -----------------------------------------
def _grid_si(meals, prof):
    grid = np.linspace(0.2, 1.6, 15)
    best, berr = float(grid[0]), float("inf")
    for si in grid:
        err = sum(abs(predict_meal_iauc(m["carbs_g"], si, prof, m["fat_g"], m["fiber_g"])[0]
                      - m["observed_iAUC"]) for m in meals)
        if err < berr:
            best, berr = float(si), err
    return best


def _rf_si(meals, prof, rf):
    from personalization import npe
    feats = []
    for m in meals:
        s = npe.summary_stats(m["cgm_curve"], _T0, _STEP, _MEAL_T)
        if not np.isnan(s).any():
            feats.append(npe.features_for(s, m["carbs_g"], prof))
    if not feats:
        return None
    return float(np.median(rf.predict(np.array(feats))[:, npe.PARAM_NAMES.index("insulin_sensitivity")]))


def _snpe_si(meals, prof, posterior):
    from personalization import snpe_infer
    sm = [{"carbs_g": m["carbs_g"],
           "glucose": {"values": m["cgm_curve"], "t0_min": _T0, "step_min": _STEP, "meal_t_min": _MEAL_T}}
          for m in meals]
    pe = snpe_infer.point_estimate(sm, posterior, profile=prof)
    return pe.get("insulin_sensitivity") if pe else None


def _gradient_fit(meals, prof, n_steps=150):
    from simulation.jax_engine import JaxPhysioParams
    from personalization.gradient_fit import fit_parameters
    base = JaxPhysioParams.from_numpy(PhysioParams.from_profile(prof))
    return fit_parameters(meals, n_steps=n_steps, base=base)


def validate_on_shanghai(limit: int | None = None, n_steps: int = 150,
                         methods=("gradient", "grid", "rf", "snpe")) -> dict:
    from evaluation.shanghai_loader import load_shanghai_t2dm, validate_shanghai_format
    from evaluation.clinical_recovery import build_rf, load_snpe

    cfg.set_all_seeds()
    subs = load_shanghai_t2dm()
    if limit:
        subs = subs[:limit]
    rf = build_rf() if "rf" in methods else None
    posterior = load_snpe() if "snpe" in methods else None

    si = {m: [] for m in methods}
    a1c, homa, grad_norms = [], [], {"insulin_sensitivity": [], "gastric_emptying": [],
                                     "carb_absorption": []}
    times = {m: 0.0 for m in methods}
    for sub in subs:
        prof = _profile(sub)
        meals = sub["meals"]
        vals = {}
        for m in methods:
            t = time.time()
            if m == "grid":
                vals[m] = _grid_si(meals, prof)
            elif m == "rf":
                vals[m] = _rf_si(meals, prof, rf)
            elif m == "snpe":
                vals[m] = _snpe_si(meals, prof, posterior)
            elif m == "gradient":
                res = _gradient_fit(meals, prof, n_steps)
                vals[m] = res["Si"]
                for k in grad_norms:
                    grad_norms[k].append(res["gradient_norms"][k])
            times[m] += time.time() - t
        if any(vals[m] is None or not np.isfinite(vals[m]) for m in methods):
            continue
        for m in methods:
            si[m].append(vals[m])
        a1c.append(sub["HbA1c"])
        homa.append(sub["HOMA_IR"])

    out = {"n_subjects": len(a1c), "n_meals_total": sum(len(s["meals"]) for s in subs),
           "ms_per_subject": {m: round(1000 * times[m] / max(len(a1c), 1), 1) for m in methods}}
    for m in methods:
        sp = bootstrap_corr(si[m], a1c, "spearman")
        pe = bootstrap_corr(si[m], a1c, "pearson")
        out[m] = {"r_spearman": sp["r"], "spearman_ci": (sp["lo"], sp["hi"]),
                  "r_pearson": pe["r"], "pearson_ci": (pe["lo"], pe["hi"]),
                  "mean_si": round(statistics.fmean(si[m]), 3)}
    # identifiability replication (normalized gradient norms already scale-normalized in gradient_fit)
    if "gradient" in methods and grad_norms["insulin_sensitivity"]:
        gm = {k: statistics.fmean(v) for k, v in grad_norms.items()}
        out["identifiability"] = {
            "mean_grad_norm": {k: round(gm[k], 1) for k in gm},
            "ratio_Si_over_gastric": round(gm["insulin_sensitivity"] / max(gm["gastric_emptying"], 1e-9), 1),
            "ratio_Si_over_carb": round(gm["insulin_sensitivity"] / max(gm["carb_absorption"], 1e-9), 1),
        }
    out["ood_summary"] = validate_shanghai_format(subs)
    return out


def cross_dataset_identifiability(limit: int | None = None, n_steps: int = 150) -> dict:
    """Addition A — does the identifiability STRUCTURE replicate on an independent cohort?

    Runs the gradient sensitivity diagnostic on ShanghaiT2DM and on CGMacros and compares the
    scale-normalized gradient-norm ratios (Si vs gastric, Si vs carb). If both cohorts show Si
    dominating by a large factor, the finding is a property of what postprandial CGM can reveal
    about Bergman parameters, not a dataset artifact — the paper's key generalization claim.
    """
    from evaluation.shanghai_loader import load_shanghai_t2dm
    from evaluation.gradient_inference_results import collect as cgmacros_collect

    cfg.set_all_seeds()

    def _ratios(norm_lists):
        gm = {k: statistics.fmean(v) for k, v in norm_lists.items()}
        return {"mean_grad_norm": {k: round(gm[k], 1) for k in gm},
                "Si_over_gastric": round(gm["insulin_sensitivity"] / max(gm["gastric_emptying"], 1e-9), 1),
                "Si_over_carb": round(gm["insulin_sensitivity"] / max(gm["carb_absorption"], 1e-9), 1)}

    # Shanghai
    subs = load_shanghai_t2dm()
    if limit:
        subs = subs[:limit]
    sh = {"insulin_sensitivity": [], "gastric_emptying": [], "carb_absorption": []}
    for sub in subs:
        res = _gradient_fit(sub["meals"], _profile(sub), n_steps)
        for k in sh:
            sh[k].append(res["gradient_norms"][k])

    # CGMacros
    cg_rows = cgmacros_collect(limit=limit, n_steps=n_steps)["rows"]
    cg = {k: [r["grad_norms"][k] for r in cg_rows] for k in sh}

    return {"shanghai": _ratios(sh), "cgmacros": _ratios(cg),
            "n_shanghai": len(subs), "n_cgmacros": len(cg_rows)}


def main() -> None:
    import argparse
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=None)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--methods", type=str, default="gradient,grid,rf,snpe")
    args = ap.parse_args()
    methods = tuple(m.strip() for m in args.methods.split(","))

    r = validate_on_shanghai(limit=args.subjects, n_steps=args.steps, methods=methods)
    print("=" * 80)
    print(f"SHANGHAI EXTERNAL VALIDATION — n={r['n_subjects']} T2D subjects, {r['n_meals_total']} meals")
    print("=" * 80)
    print("  Si vs HbA1c (frozen CGMacros artifacts, pure transfer):")
    for m in methods:
        d = r[m]
        print(f"    {m:9} Spearman {d['r_spearman']:+.3f} [{d['spearman_ci'][0]:+.2f},{d['spearman_ci'][1]:+.2f}]"
              f"  Pearson {d['r_pearson']:+.3f}  meanSi {d['mean_si']}  {r['ms_per_subject'][m]:.0f}ms/subj")
    if "identifiability" in r:
        idn = r["identifiability"]
        print(f"  IDENTIFIABILITY (Shanghai): Si grad-norm / gastric = {idn['ratio_Si_over_gastric']}x, "
              f"/ carb = {idn['ratio_Si_over_carb']}x  (CGMacros ~20-60x)")
    print(f"  OOD: mean glucose {r['ood_summary']['mean_glucose']:.0f}, iAUC mean "
          f"{r['ood_summary']['iauc_mean']:.0f}, carb-estimated {r['ood_summary']['carb_estimated_fraction']*100:.0f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
