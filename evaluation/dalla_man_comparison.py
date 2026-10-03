"""Does gradient inference through Dalla Man beat grid on held-out iAUC? (the deciding experiment)

Same aligned 5-fold protocol and fold assignments as `evaluation/unified_kfold.py`, so the numbers
drop straight into Table 2. Methods compared:
  * dalla_man — gradient fit through the Dalla Man model (observation = subcutaneous glucose)
  * bergman   — gradient fit through the Bergman engine (the committed method)
  * grid      — 1-D Si grid on the Bergman engine (the baseline that currently ties)

If Dalla Man gradient < grid on held-out iAUC MAE, the prediction tie breaks and the paper's
Table 2 becomes a win. If it does not, the CGM observation (baseline-subtracted iAUC) is the
bottleneck rather than Bergman's expressiveness — which vindicates the current framing.

Run:  python -m evaluation.dalla_man_comparison --subjects 5   # smoke
      python -m evaluation.dalla_man_comparison                # full
"""
from __future__ import annotations

import statistics
import time

import numpy as np

import paper_config as cfg
from simulation import PhysioParams
from evaluation.cgmacros import _profile, load_bio, subjects
from evaluation.snpe_kfold import _meal_records, _fit_si, _pred_iauc

_METHODS = ("dalla_man", "bergman", "grid")


def _subject_Gb(recs) -> float:
    """Subject basal glucose = mean pre-meal CGM baseline across their meals."""
    bl = []
    for r in recs:
        g = r["glucose"]
        vals, t0, step, mt = g["values"], g["t0_min"], g["step_min"], g["meal_t_min"]
        pre = [v for i, v in enumerate(vals) if mt - 30.0 <= t0 + i * step < mt]
        if pre:
            bl.append(float(np.mean(pre)))
    return float(np.mean(bl)) if bl else 100.0


def _blunt(r):
    """Same fat/fibre blunting Bergman applies (metabolic.on_impulse) -> effective carbs."""
    return 1.0 / (1.0 + 0.08 * r["fiber_g"] + 0.005 * r["fat_g"])


def _dalla_man_mae(train, test, prof, Gb, n_steps, parity=False):
    from simulation.dalla_man import DallaManParams
    from personalization.gradient_fit_dalla_man import fit_parameters, predict_iauc
    base = DallaManParams.defaults(prof["weight_kg"], Gb)
    ceff = (lambda r: r["carbs_g"] * _blunt(r)) if parity else (lambda r: r["carbs_g"])
    meals = [{"carbs_g": ceff(r), "observed_iAUC": r["iauc"]} for r in train]
    res = fit_parameters(meals, n_steps=n_steps, base=base,
                         weight_kg=prof["weight_kg"], Gb=Gb)
    p = res["params"]
    errs = [abs(predict_iauc(p, ceff(r)) - r["iauc"]) for r in test]
    return statistics.fmean(errs), res["gradient_norms"]


def _bergman_mae(train, test, prof, n_steps):
    from simulation.jax_engine import JaxPhysioParams, run_meal
    from simulation.jax_observation import iauc
    from personalization.gradient_fit import fit_parameters
    base = JaxPhysioParams.from_numpy(PhysioParams.from_profile(prof))
    meals = [{"carbs_g": r["carbs_g"], "fat_g": r["fat_g"], "fiber_g": r["fiber_g"],
              "observed_iAUC": r["iauc"]} for r in train]
    res = fit_parameters(meals, n_steps=n_steps, base=base)
    p = res["params"]
    errs = [abs(float(iauc(*run_meal(p, r["carbs_g"], r["fat_g"], r["fiber_g"]))) - r["iauc"])
            for r in test]
    return statistics.fmean(errs)


def run(n_folds: int = 5, min_meals: int = 10, limit: int | None = None, seed: int = cfg.SEED,
        n_steps: int = 150, parity: bool = False) -> dict:
    cfg.set_all_seeds(seed)
    bio = load_bio()
    data = []
    for sid, meals in subjects(limit=limit):
        recs = _meal_records(meals)
        prof = _profile(bio.get(sid))
        if prof and len(recs) >= min_meals:
            data.append((sid, recs, prof, _subject_Gb(recs)))

    subj_mae = {m: {sid: [] for sid, _, _, _ in data} for m in _METHODS}
    fold_mae = {m: [] for m in _METHODS}
    fold_win = {m: [] for m in _METHODS}
    pm_fold = []
    gnorms = []
    t0 = time.time()

    for fold in range(n_folds):
        fm = {m: [] for m in _METHODS}
        fw = {m: [] for m in _METHODS}
        fpm = []
        for sid, recs, prof, Gb in data:
            rng = np.random.default_rng(seed * 1000 + abs(hash(sid)) % 100000)  # same as unified_kfold
            assign = rng.integers(0, n_folds, size=len(recs))
            train = [recs[i] for i in range(len(recs)) if assign[i] != fold]
            test = [recs[i] for i in range(len(recs)) if assign[i] == fold]
            if not train or not test:
                continue
            pm = statistics.fmean(r["iauc"] for r in train)
            pm_mae = statistics.fmean(abs(pm - r["iauc"]) for r in test)
            fpm.append(pm_mae)
            for m in _METHODS:
                if m == "dalla_man":
                    e, gn = _dalla_man_mae(train, test, prof, Gb, n_steps, parity=parity)
                    gnorms.append(gn)
                elif m == "bergman":
                    e = _bergman_mae(train, test, prof, n_steps)
                else:
                    si = _fit_si("grid", train, prof, None, None)
                    e = statistics.fmean(abs(_pred_iauc(r, si, prof) - r["iauc"]) for r in test)
                fm[m].append(e)
                fw[m].append(e < pm_mae)
                subj_mae[m][sid].append(e)
        for m in _METHODS:
            fold_mae[m].append(statistics.fmean(fm[m]))
            fold_win[m].append(100 * statistics.fmean(fw[m]))
        pm_fold.append(statistics.fmean(fpm))

    def ms(v):
        return {"mean": statistics.fmean(v), "std": statistics.pstdev(v)}

    subj_avg = {m: {s: statistics.fmean(v) for s, v in subj_mae[m].items() if v} for m in _METHODS}

    def paired(a, b):
        common = [s for s in subj_avg[a] if s in subj_avg[b]]
        return (round(100 * statistics.fmean(subj_avg[a][s] < subj_avg[b][s] for s in common)),
                len(common))

    mean_gn = ({k: statistics.fmean(g[k] for g in gnorms) for k in gnorms[0]} if gnorms else {})
    return {
        "n_subjects": len(data), "n_folds": n_folds,
        "mae": {m: ms(fold_mae[m]) for m in _METHODS},
        "winrate_vs_personal": {m: ms(fold_win[m]) for m in _METHODS},
        "personal_mean_mae": ms(pm_fold),
        "paired_dalla_vs_grid": paired("dalla_man", "grid"),
        "paired_dalla_vs_bergman": paired("dalla_man", "bergman"),
        "dalla_man_grad_norms": mean_gn,
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
    ap.add_argument("--parity", action="store_true",
                    help="give Dalla Man the same fat/fibre effective-carb adjustment as Bergman")
    args = ap.parse_args()

    r = run(limit=args.subjects, n_steps=args.steps, parity=args.parity)
    if args.parity:
        print("  [parity mode: Dalla Man carbs blunted by fat/fibre, same as Bergman]")
    print("=" * 78)
    print(f"DALLA MAN vs BERGMAN vs GRID — aligned 5-fold, n={r['n_subjects']} ({r['wall_seconds']}s)")
    print("=" * 78)
    for m in _METHODS:
        mae, wr = r["mae"][m], r["winrate_vs_personal"][m]
        print(f"  {m:10} held-out iAUC MAE {mae['mean']:7.0f}±{mae['std']:.0f}   "
              f"beats personal-mean {wr['mean']:.0f}%")
    print(f"  {'personal':10} held-out iAUC MAE {r['personal_mean_mae']['mean']:7.0f}")
    print("-" * 78)
    print(f"  PAIRED dalla_man vs grid    : {r['paired_dalla_vs_grid'][0]}% of subjects "
          f"(n={r['paired_dalla_vs_grid'][1]})")
    print(f"  PAIRED dalla_man vs bergman : {r['paired_dalla_vs_bergman'][0]}%")
    if r["dalla_man_grad_norms"]:
        gn = r["dalla_man_grad_norms"]
        order = sorted(gn, key=gn.get, reverse=True)
        print("  Dalla Man identifiability (normalized |grad|, high->low): " +
              "  ".join(f"{k} {gn[k]:.3g}" for k in order))
    print("=" * 78)


if __name__ == "__main__":
    main()
