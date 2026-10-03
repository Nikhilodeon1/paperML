"""Protocol-aligned held-out k-fold for ALL FOUR estimators (Table 2, submission-grade).

Fixes the protocol mismatch a reviewer would catch: gradient inference previously used a
chronological 50/50 split while snpe_kfold used 5-fold random. Here every method — gradient, grid,
RF, SNPE — is scored on the SAME 5-fold random assignments (same seed, same per-subject meal
splits), and we report BOTH the marginal MAE and the PAIRED per-subject comparison (does gradient
beat grid on each subject, not just in the mean — robust to a few outliers driving the mean).

Prediction is through each method's native engine (numpy for the Si-scalar methods, jax for the
gradient method's full fitted params); the two engines agree to 1.5%, so this is fair.

Run:  python -m evaluation.unified_kfold                 # full (slow, ~1-1.5h)
      python -m evaluation.unified_kfold --subjects 6    # smoke
"""
from __future__ import annotations

import statistics
import time

import numpy as np

import paper_config as cfg
from simulation import PhysioParams
from evaluation.cgmacros import _profile, load_bio, subjects
from evaluation.snpe_kfold import _meal_records, _fit_si, _pred_iauc

_METHODS = ("adaptive", "gradient", "grid", "rf", "snpe")


def _train_meals(train_recs):
    return [{"carbs_g": r["carbs_g"], "fat_g": r["fat_g"], "fiber_g": r["fiber_g"],
             "observed_iAUC": r["iauc"]} for r in train_recs]


def _jax_test_mae(p, test_recs):
    from simulation.jax_engine import run_meal
    from simulation.jax_observation import iauc
    errs = [abs(float(iauc(*run_meal(p, r["carbs_g"], r["fat_g"], r["fiber_g"]))) - r["iauc"])
            for r in test_recs]
    return statistics.fmean(errs)


def _grad_predict(train_recs, test_recs, prof, n_steps):
    from simulation.jax_engine import JaxPhysioParams
    from personalization.gradient_fit import fit_parameters
    base = JaxPhysioParams.from_numpy(PhysioParams.from_profile(prof))
    res = fit_parameters(_train_meals(train_recs), n_steps=n_steps, base=base)
    return _jax_test_mae(res["params"], test_recs), None


def _adaptive_predict(train_recs, test_recs, prof, n_steps):
    from simulation.jax_engine import JaxPhysioParams
    from personalization.gradient_fit import adaptive_fit
    base = JaxPhysioParams.from_numpy(PhysioParams.from_profile(prof))
    res = adaptive_fit(_train_meals(train_recs), n_steps=n_steps, base=base)
    return _jax_test_mae(res["params"], test_recs), res["free_params"]


def run(n_folds: int = 5, min_meals: int = 10, limit: int | None = None, seed: int = cfg.SEED,
        gradient_steps: int = 150, methods=_METHODS) -> dict:
    from evaluation.clinical_recovery import build_rf, load_snpe
    cfg.set_all_seeds(seed)
    need_si = any(m in ("grid", "rf", "snpe") for m in methods)
    rf = build_rf() if "rf" in methods else None
    posterior = load_snpe() if "snpe" in methods else None
    bio = load_bio()
    sel_count = {"insulin_sensitivity": 0, "gastric_emptying": 0, "carb_absorption": 0}
    sel_total = 0

    data = []
    for sid, meals in subjects(limit=limit):
        recs = _meal_records(meals)
        prof = _profile(bio.get(sid))
        if prof and len(recs) >= min_meals:
            data.append((sid, recs, prof))

    subj_mae = {m: {sid: [] for sid, _, _ in data} for m in methods}   # per-subject, over folds
    fold_mae = {m: [] for m in methods}
    fold_win = {m: [] for m in methods}                                # vs personal-mean
    base_mae = {"personal_mean": [], "persistence": [], "population": []}

    for fold in range(n_folds):
        fm = {m: [] for m in methods}
        fw = {m: [] for m in methods}
        fpm, fpers, fpop = [], [], []
        for sid, recs, prof in data:
            rng = np.random.default_rng(seed * 1000 + abs(hash(sid)) % 100000)  # same as snpe_kfold
            assign = rng.integers(0, n_folds, size=len(recs))
            train = [recs[i] for i in range(len(recs)) if assign[i] != fold]
            test = [recs[i] for i in range(len(recs)) if assign[i] == fold]
            if not train or not test:
                continue
            pm = statistics.fmean(r["iauc"] for r in train)
            pm_mae = statistics.fmean(abs(pm - r["iauc"]) for r in test)
            fpm.append(pm_mae)
            prev = train[-1]["iauc"]
            pers = []
            for r in test:
                pers.append(abs(prev - r["iauc"])); prev = r["iauc"]
            fpers.append(statistics.fmean(pers))
            fpop.append(statistics.fmean(abs(_pred_iauc(r, 1.0, prof) - r["iauc"]) for r in test))
            for m in methods:
                if m == "gradient":
                    emae, _ = _grad_predict(train, test, prof, gradient_steps)
                elif m == "adaptive":
                    emae, free = _adaptive_predict(train, test, prof, gradient_steps)
                    sel_total += 1
                    for k in free:
                        sel_count[k] += 1
                else:
                    si = _fit_si(m, train, prof, rf, posterior)
                    if si is None or not np.isfinite(si):
                        continue
                    emae = statistics.fmean(abs(_pred_iauc(r, si, prof) - r["iauc"]) for r in test)
                fm[m].append(emae)
                fw[m].append(emae < pm_mae)
                subj_mae[m][sid].append(emae)
        for m in methods:
            fold_mae[m].append(statistics.fmean(fm[m]))
            fold_win[m].append(100 * statistics.fmean(fw[m]))
        base_mae["personal_mean"].append(statistics.fmean(fpm))
        base_mae["persistence"].append(statistics.fmean(fpers))
        base_mae["population"].append(statistics.fmean(fpop))

    def ms(v):
        return {"mean": statistics.fmean(v), "std": statistics.pstdev(v)}

    # paired per-subject comparison (average each subject's MAE over folds, then compare)
    subj_avg = {m: {sid: statistics.fmean(v) for sid, v in subj_mae[m].items() if v}
                for m in methods}

    def paired_winrate(a, b):
        if a not in subj_avg or b not in subj_avg:
            return None
        common = [s for s in subj_avg[a] if s in subj_avg[b]]
        return (round(100 * statistics.fmean(subj_avg[a][s] < subj_avg[b][s] for s in common), 0),
                len(common))

    pairs = {}
    for a in methods:
        if a != "grid":
            pairs[f"{a}_vs_grid"] = paired_winrate(a, "grid")
    if "adaptive" in methods and "gradient" in methods:
        pairs["adaptive_vs_gradient"] = paired_winrate("adaptive", "gradient")

    selection = ({k: round(100 * sel_count[k] / sel_total) for k in sel_count}
                 if sel_total else {})

    return {
        "n_subjects": len(data), "n_folds": n_folds, "methods": list(methods),
        "estimator_mae": {m: ms(fold_mae[m]) for m in methods},
        "estimator_winrate_vs_personal": {m: ms(fold_win[m]) for m in methods},
        "baseline_mae": {b: ms(v) for b, v in base_mae.items()},
        "paired": pairs,
        "adaptive_selection_pct": selection,   # how often each param was kept as identifiable
    }


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
    ap.add_argument("--methods", type=str, default=",".join(_METHODS),
                    help="comma list, e.g. adaptive,gradient,grid")
    args = ap.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(","))
    r = run(limit=args.subjects, gradient_steps=args.steps, methods=methods)
    print("=" * 80)
    print(f"UNIFIED 5-FOLD (aligned protocol) — {r['n_subjects']} subjects")
    print("  held-out iAUC MAE (lower better) | % beats personal-mean")
    for m in r["methods"]:
        mae, wr = r["estimator_mae"][m], r["estimator_winrate_vs_personal"][m]
        print(f"    {m:9} MAE {mae['mean']:.0f}±{mae['std']:.0f}   beats {wr['mean']:.0f}±{wr['std']:.0f}%")
    for b, d in r["baseline_mae"].items():
        print(f"    {b:9} MAE {d['mean']:.0f}")
    print("  PAIRED per-subject (row method wins on X% of subjects):")
    for k, v in r["paired"].items():
        if v:
            print(f"    {k}: {v[0]:.0f}%  (n={v[1]})")
    if r["adaptive_selection_pct"]:
        print("  adaptive kept each param (% of fits): " +
              "  ".join(f"{k.split('_')[0]} {v}%" for k, v in r["adaptive_selection_pct"].items()))
    print("=" * 80)


if __name__ == "__main__":
    main()
