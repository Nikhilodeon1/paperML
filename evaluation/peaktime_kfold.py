"""Aligned 5-fold k-fold with PEAK-TIME as the target (parallel to unified_kfold's iAUC version).

Same fold assignments (same seed), same four methods, same paired per-subject comparison — the only
change is the observable: minutes-to-glucose-peak instead of iAUC. Peak time is a TIMING observable
governed by gastric emptying / carb absorption, the parameters non-identifiable from iAUC. If
gradient (which fits those) now BEATS grid (Si-only), the iAUC prediction tie was observable-
specific and the timing parameters are identifiable/predictive from the right target.

Method setup (parallel to the iAUC experiment): gradient fits Si+gastric+carb to peak time; grid
fits Si to peak time by 1-D search; RF/SNPE keep their native (amortized, iAUC-trained) Si estimate
and predict peak time through the engine with default timing (they cannot personalize timing). All
peak-time predictions go through the same jax engine (hard argmax) for a fair comparison.

Run:  python -m evaluation.peaktime_kfold                # full (~40 min)
      python -m evaluation.peaktime_kfold --subjects 6   # smoke
"""
from __future__ import annotations

import statistics
import time

import numpy as np

import paper_config as cfg
from simulation import PhysioParams
from evaluation.cgmacros import _profile, load_bio, subjects
from evaluation.snpe_kfold import _meal_records, _fit_si

_METHODS = ("gradient", "grid", "rf", "snpe")
_SI_GRID = np.linspace(0.2, 1.6, 15)


def _observed_peaktime(rec) -> float:
    """Minutes to peak of the real CGM curve, post-meal (meal at index 6, 5-min grid)."""
    g = rec["glucose"]
    vals = np.asarray(g["values"], dtype=float)
    step, mt, t0 = g["step_min"], g["meal_t_min"], g["t0_min"]
    i0 = int(round((mt - t0) / step))            # index of meal (t=0)
    post = vals[i0:]
    return float(np.argmax(post) * step)


def _si_params(prof, si):
    from simulation.jax_engine import JaxPhysioParams
    import equinox as eqx
    import jax.numpy as jnp
    base = JaxPhysioParams.from_numpy(PhysioParams.from_profile(prof))
    return eqx.tree_at(lambda m: m.insulin_sensitivity, base, jnp.asarray(float(si), jnp.float32))


def _pred_pt(params_or_si, carbs, prof):
    from personalization.gradient_fit_peaktime import predict_peaktime
    p = params_or_si if not isinstance(params_or_si, float) else _si_params(prof, params_or_si)
    return predict_peaktime(p, carbs)


def _method_mae(m, train, test, prof, rf, posterior, n_steps):
    if m == "gradient":
        from personalization.gradient_fit_peaktime import fit_parameters
        from simulation.jax_engine import JaxPhysioParams
        meals = [{"carbs_g": r["carbs_g"], "observed_peaktime": _observed_peaktime(r)} for r in train]
        res = fit_parameters(meals, n_steps=n_steps,
                             base=JaxPhysioParams.from_numpy(PhysioParams.from_profile(prof)))
        p = res["params"]
        return statistics.fmean(abs(_pred_pt(p, r["carbs_g"], prof) - _observed_peaktime(r)) for r in test)
    if m == "grid":
        best, berr = float(_SI_GRID[0]), float("inf")
        for si in _SI_GRID:
            e = sum(abs(_pred_pt(float(si), r["carbs_g"], prof) - _observed_peaktime(r)) for r in train)
            if e < berr:
                best, berr = float(si), e
        return statistics.fmean(abs(_pred_pt(best, r["carbs_g"], prof) - _observed_peaktime(r)) for r in test)
    # rf / snpe: native Si estimate -> predict peak time with default timing
    si = _fit_si(m, train, prof, rf, posterior)
    if si is None or not np.isfinite(si):
        return None
    return statistics.fmean(abs(_pred_pt(float(si), r["carbs_g"], prof) - _observed_peaktime(r)) for r in test)


def run(n_folds: int = 5, min_meals: int = 10, limit: int | None = None, seed: int = cfg.SEED,
        n_steps: int = 150) -> dict:
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

    subj_mae = {m: {sid: [] for sid, _, _ in data} for m in _METHODS}
    fold_mae = {m: [] for m in _METHODS}
    persist = []
    t0 = time.time()
    for fold in range(n_folds):
        fm = {m: [] for m in _METHODS}
        fpers = []
        for sid, recs, prof in data:
            rng = np.random.default_rng(seed * 1000 + abs(hash(sid)) % 100000)  # SAME as unified_kfold
            assign = rng.integers(0, n_folds, size=len(recs))
            train = [recs[i] for i in range(len(recs)) if assign[i] != fold]
            test = [recs[i] for i in range(len(recs)) if assign[i] == fold]
            if not train or not test:
                continue
            # persistence baseline: previous meal's peak time
            prev = _observed_peaktime(train[-1])
            pe = []
            for r in test:
                pe.append(abs(prev - _observed_peaktime(r))); prev = _observed_peaktime(r)
            fpers.append(statistics.fmean(pe))
            for m in _METHODS:
                e = _method_mae(m, train, test, prof, rf, posterior, n_steps)
                if e is not None:
                    fm[m].append(e); subj_mae[m][sid].append(e)
        for m in _METHODS:
            fold_mae[m].append(statistics.fmean(fm[m]))
        persist.append(statistics.fmean(fpers))

    def ms(v):
        return {"mean": statistics.fmean(v), "std": statistics.pstdev(v)}

    subj_avg = {m: {s: statistics.fmean(v) for s, v in subj_mae[m].items() if v} for m in _METHODS}

    def paired(a, b):
        common = [s for s in subj_avg[a] if s in subj_avg[b]]
        return (round(100 * statistics.fmean(subj_avg[a][s] < subj_avg[b][s] for s in common)), len(common))

    return {
        "n_subjects": len(data), "target": "peak_time_min",
        "mae": {m: ms(fold_mae[m]) for m in _METHODS},
        "persistence_mae": ms(persist),
        "paired_gradient_vs_grid": paired("gradient", "grid"),
        "paired_gradient_vs_rf": paired("gradient", "rf"),
        "paired_gradient_vs_snpe": paired("gradient", "snpe"),
        "wall_seconds": round(time.time() - t0),
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
    args = ap.parse_args()
    r = run(limit=args.subjects, n_steps=args.steps)
    print("=" * 76)
    print(f"PEAK-TIME 5-FOLD (aligned) — n={r['n_subjects']} ({r['wall_seconds']}s)")
    print("  held-out peak-time MAE (minutes, lower better)")
    for m in _METHODS:
        d = r["mae"][m]
        print(f"    {m:9} {d['mean']:6.1f} ± {d['std']:.1f} min")
    print(f"    {'persist':9} {r['persistence_mae']['mean']:6.1f} min")
    print("  PAIRED (gradient wins on X% of subjects):")
    for k in ("paired_gradient_vs_grid", "paired_gradient_vs_rf", "paired_gradient_vs_snpe"):
        w, n = r[k]
        print(f"    {k.replace('paired_', '')}: {w}%  (n={n})")
    print("=" * 76)


if __name__ == "__main__":
    main()
