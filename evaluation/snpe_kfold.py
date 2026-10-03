"""Predictive utility (paper §4.2 / Table 2) — held-out k-fold, all estimators at once.

Parallels ``evaluation/kfold_calibration.py`` (which calibrates the fat/fibre blunting) but here
the blunting is held fixed at its calibrated live value and the ONE thing that varies is the
source of each subject's insulin sensitivity Si. For every subject we split their meals into 5
folds; on the train meals we estimate Si with each estimator, predict the held-out meals' iAUC
through the engine, and compare to three baselines.

Estimators: ``snpe`` (posterior mean), ``rf`` (RandomForest median), ``grid`` (1-D iAUC grid).
Baselines: personal-mean iAUC, persistence (previous meal), population (Si=1.0 through engine).

The bar that justifies personalization: the engine's per-subject Si must beat the subject's own
average iAUC on held-out meals. Reported per fold as mean +/- std of the MAE gap and win-rate.

Run:  python -m evaluation.snpe_kfold
"""
from __future__ import annotations

import statistics

import numpy as np

import paper_config as cfg
from evaluation.cgmacros import _profile, load_bio, predict_meal_iauc, subjects
from personalization import npe, snpe_infer

_SI_GRID = np.linspace(0.2, 1.6, 15)
_SI_IDX = npe.PARAM_NAMES.index("insulin_sensitivity")


def _meal_records(meals) -> list[dict]:
    recs = []
    for m in meals:
        r = m.real_iauc()
        if r is None:
            continue
        vals, t0 = m._grid()
        recs.append({"carbs_g": m.carbs_g, "protein_g": m.protein_g, "fat_g": m.fat_g,
                     "fiber_g": m.fiber_g, "iauc": r,
                     "glucose": {"values": vals, "t0_min": t0, "step_min": 5.0, "meal_t_min": 0.0}})
    return recs


def _pred_iauc(rec, si, prof):
    return predict_meal_iauc(rec["carbs_g"], si, prof, rec["protein_g"], rec["fat_g"],
                             rec["fiber_g"])[0]


def _fit_si(estimator, train, prof, rf, posterior):
    if estimator == "grid":
        best, best_err = float(_SI_GRID[0]), float("inf")
        for si in _SI_GRID:
            err = sum(abs(_pred_iauc(r, si, prof) - r["iauc"]) for r in train)
            if err < best_err:
                best, best_err = float(si), err
        return best
    if estimator == "rf":
        feats = []
        for r in train:
            g = r["glucose"]
            s = npe.summary_stats(g["values"], g["t0_min"], g["step_min"], g["meal_t_min"])
            if not np.isnan(s).any():
                feats.append(npe.features_for(s, r["carbs_g"], prof))
        if not feats:
            return None
        return float(np.median(rf.predict(np.array(feats))[:, _SI_IDX]))
    if estimator == "snpe":
        meals = [{"carbs_g": r["carbs_g"], "glucose": r["glucose"]} for r in train]
        pe = snpe_infer.point_estimate(meals, posterior, profile=prof)
        return pe.get("insulin_sensitivity") if pe else None
    raise ValueError(estimator)


def run(estimators=("snpe", "rf", "grid"), n_folds: int = 5, min_meals: int = 10,
        limit: int | None = None, seed: int = cfg.SEED) -> dict:
    from evaluation.clinical_recovery import build_rf, load_snpe

    cfg.set_all_seeds(seed)
    rf = build_rf()
    posterior = load_snpe()
    bio = load_bio()

    data = []
    for sid, meals in subjects(limit=limit):
        recs = _meal_records(meals)
        prof = _profile(bio.get(sid))
        if prof and len(recs) >= min_meals:
            data.append((sid, recs, prof))

    est_mae = {e: [] for e in estimators}
    est_gap = {e: [] for e in estimators}
    est_win = {e: [] for e in estimators}
    base_mae = {"personal_mean": [], "persistence": [], "population": []}

    for fold in range(n_folds):
        fmae = {e: [] for e in estimators}
        fwin = {e: [] for e in estimators}
        fpm, fpers, fpop = [], [], []
        for sid, recs, prof in data:
            rng = np.random.default_rng(seed * 1000 + abs(hash(sid)) % 100000)
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
                pers.append(abs(prev - r["iauc"]))
                prev = r["iauc"]
            fpers.append(statistics.fmean(pers))
            fpop.append(statistics.fmean(abs(_pred_iauc(r, 1.0, prof) - r["iauc"]) for r in test))
            for e in estimators:
                si = _fit_si(e, train, prof, rf, posterior)
                if si is None or not np.isfinite(si):
                    continue
                emae = statistics.fmean(abs(_pred_iauc(r, si, prof) - r["iauc"]) for r in test)
                fmae[e].append(emae)
                fwin[e].append(emae < pm_mae)
        for e in estimators:
            est_mae[e].append(statistics.fmean(fmae[e]))
            est_gap[e].append(statistics.fmean(fpm) - statistics.fmean(fmae[e]))
            est_win[e].append(100 * statistics.fmean(fwin[e]))
        base_mae["personal_mean"].append(statistics.fmean(fpm))
        base_mae["persistence"].append(statistics.fmean(fpers))
        base_mae["population"].append(statistics.fmean(fpop))

    def ms(v):
        return {"mean": statistics.fmean(v), "std": statistics.pstdev(v), "folds": [round(x, 1) for x in v]}

    return {
        "n_subjects": len(data), "n_folds": n_folds,
        "estimators": {e: {"mae": ms(est_mae[e]), "gap": ms(est_gap[e]), "winrate": ms(est_win[e])}
                       for e in estimators},
        "baselines": {b: ms(base_mae[b]) for b in base_mae},
    }


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    r = run()
    print("=" * 80)
    print("HELD-OUT k-FOLD UTILITY — iAUC MAE by Si estimator vs baselines (lower = better)")
    print("=" * 80)
    print(f"  subjects: {r['n_subjects']}   folds: {r['n_folds']}")
    print(f"  {'method':16}{'iAUC MAE':>12}{'gap vs pers.mean':>20}{'% beats pers.mean':>20}")
    for e, d in r["estimators"].items():
        print(f"  {e:16}{d['mae']['mean']:>9.0f}±{d['mae']['std']:>2.0f}"
              f"{d['gap']['mean']:>+15.0f}±{d['gap']['std']:>2.0f}"
              f"{d['winrate']['mean']:>16.0f}±{d['winrate']['std']:>2.0f}")
    for b, d in r["baselines"].items():
        print(f"  {b:16}{d['mean']:>9.0f}±{d['std']:>2.0f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
