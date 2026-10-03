"""K-fold calibration of the meal fat/fibre blunting — the citable held-out result.

The held-out run (`cgmacros --mode heldout`) showed the engine only beats a personal constant
AFTER fat/fibre are modelled, but with two weaknesses a reviewer catches instantly: (1) the
blunting coefficients were hand-picked, (2) it was a single train/test split. This fixes both:

  - FIT the coefficients (a = per-gram fibre effect, b = per-gram fat effect in
    blunt = 1/(1 + a*fibre + b*fat)) on TRAIN meals, never the held-out ones.
  - 5-FOLD cross-validation, so the reported gap is mean +/- std across folds, not one lucky
    split.

Nested structure, done right: a,b are POPULATION physiology (same for everyone); insulin
sensitivity Si is PER SUBJECT. So for each candidate (a,b) we refit every subject's Si on
their train meals, then the held-out score uses that subject's Si with the shared (a,b).

Tractability: blunting only SCALES the carb load, so engine iAUC is a function of
(effective_carbs, Si). We precompute a per-subject iAUC table over that grid ONCE (~5 min of
simulation); every fold's (a,b) search is then 1-D interpolation, no simulation in the loop.

Run:  python -m evaluation.kfold_calibration
"""

from __future__ import annotations

import statistics

import numpy as np

from evaluation.cgmacros import _profile, load_bio, predict_meal_iauc, subjects

_CARB_GRID = np.arange(5.0, 175.0, 12.0)        # effective-carb axis of the precomputed table
_SI_GRID = np.linspace(0.2, 1.6, 15)            # per-subject Si is chosen on this grid
# (a, b) candidates. a ~ 1/fibre_k, b ~ 1/fat_k; includes (0,0) = carbs-only baseline.
# hand-picked guess was a=1/30=0.033, b=1/100=0.010.
_A_GRID = [0.0, 0.033, 0.08, 0.13, 0.20]         # fibre
_B_GRID = [0.0, 0.005, 0.010, 0.020, 0.035]      # fat
_N_FOLDS = 5

# PROTEIN was tested as a third blunting coefficient (2026-07-17). It is collinear with fat in
# real meals (fatty meals are protein-rich): the fit swapped fat->0 for protein~0.008 with no
# mean improvement and TRIPLED the fold variance (+/-72 -> +/-215). A redundant, destabilising
# parameter -> rejected. The robust model is fibre-dominant + a small fat term.


def _blunt(fiber, fat, a, b):
    return 1.0 / (1.0 + a * fiber + b * fat)


def _iauc_table(profile) -> np.ndarray:
    """[carb, si] -> engine iAUC for a pure-carb meal (blunting handled by scaling carbs)."""
    tab = np.empty((len(_CARB_GRID), len(_SI_GRID)))
    for i, c in enumerate(_CARB_GRID):
        for j, si in enumerate(_SI_GRID):
            tab[i, j] = predict_meal_iauc(float(c), float(si), profile)[0]
    return tab


def _pred(table, eff_carbs, si_idx):
    return float(np.interp(eff_carbs, _CARB_GRID, table[:, si_idx]))


def _fit_si_idx(table, meals_iauc, a, b) -> int:
    """Index of the Si grid point minimising train error for this subject given (a,b)."""
    best_j, best_err = 0, float("inf")
    for j in range(len(_SI_GRID)):
        err = sum(abs(_pred(table, cg * _blunt(fi, fa, a, b), j) - r)
                  for cg, fi, fa, r in meals_iauc)
        if err < best_err:
            best_j, best_err = j, err
    return best_j


def _meal_tuple(m):
    r = m.real_iauc()
    return None if r is None else (m.carbs_g, m.fiber_g, m.fat_g, r)


def run(limit: int | None = None, seed: int = 0) -> dict:
    bio = load_bio()
    # gather subjects with enough scorable meals + precompute their tables
    data = []
    for sid, meals in subjects(limit=limit):
        tuples = [t for t in (_meal_tuple(m) for m in meals) if t is not None]
        if len(tuples) >= 10:
            data.append((sid, tuples, _iauc_table(_profile(bio.get(sid)))))

    rng = np.random.default_rng(seed)
    fold_gaps, fold_winrates, fold_ab = [], [], []
    for fold in range(_N_FOLDS):
        # per-subject fold assignment of meals
        train_test = []
        for sid, tuples, table in data:
            idx = np.arange(len(tuples))
            rng2 = np.random.default_rng(seed * 100 + hash(sid) % 1000)
            assign = rng2.integers(0, _N_FOLDS, size=len(tuples))
            train = [tuples[i] for i in idx if assign[i] != fold]
            test = [tuples[i] for i in idx if assign[i] == fold]
            if train and test:
                train_test.append((table, train, test))

        # choose (a,b) that minimises TRAIN error across all subjects (Si refit per subject)
        best_ab, best_err = (0.0, 0.0), float("inf")
        for a in _A_GRID:
            for b in _B_GRID:
                err = 0.0
                for table, train, _ in train_test:
                    j = _fit_si_idx(table, train, a, b)
                    err += sum(abs(_pred(table, cg * _blunt(fi, fa, a, b), j) - r)
                               for cg, fi, fa, r in train)
                if err < best_err:
                    best_ab, best_err = (a, b), err
        a, b = best_ab
        fold_ab.append(best_ab)

        # score HELD-OUT test meals: engine (calibrated a,b + per-subject Si) vs personal mean
        eng_maes, pm_maes, wins, n = [], [], 0, 0
        for table, train, test in train_test:
            j = _fit_si_idx(table, train, a, b)
            pm = statistics.fmean(r for *_, r in train)
            e = statistics.fmean(abs(_pred(table, cg * _blunt(fi, fa, a, b), j) - r)
                                 for cg, fi, fa, r in test)
            p = statistics.fmean(abs(pm - r) for *_, r in test)
            eng_maes.append(e); pm_maes.append(p); n += 1
            wins += (e < p)
        fold_gaps.append(statistics.fmean(pm_maes) - statistics.fmean(eng_maes))
        fold_winrates.append(100 * wins / n)

    return {"n_subjects": len(data), "fold_ab": fold_ab,
            "gap_mean": statistics.fmean(fold_gaps), "gap_std": statistics.pstdev(fold_gaps),
            "gaps": [round(g, 0) for g in fold_gaps],
            "winrate_mean": statistics.fmean(fold_winrates),
            "winrate_std": statistics.pstdev(fold_winrates),
            "winrates": [round(w) for w in fold_winrates]}


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    r = run()
    print("=" * 76)
    print("CGMacros — 5-FOLD calibrated held-out validation (fit fat/fibre blunting on train)")
    print("=" * 76)
    print(f"  subjects: {r['n_subjects']}   folds: {_N_FOLDS}")
    print(f"  calibrated (a=fibre, b=fat) per fold: {r['fold_ab']}  (protein tested + rejected: collinear w/ fat)")
    print(f"\n  iAUC MAE gap (personal-mean - engine), per fold: {r['gaps']}")
    print(f"    mean {r['gap_mean']:+.0f} +/- {r['gap_std']:.0f} mg/dL*min   "
          f"(positive = engine better than a personal constant)")
    print(f"  % subjects engine beats personal-mean, per fold: {r['winrates']}")
    print(f"    mean {r['winrate_mean']:.0f}% +/- {r['winrate_std']:.0f}%")
    folds_won = sum(g > 0 for g in r["gaps"])
    print(f"\n  VERDICT: engine beats personal constant on {folds_won}/{_N_FOLDS} folds. "
          f"{'CITABLE — consistent held-out win.' if folds_won >= 4 and r['gap_mean'] > 0 else 'MARGINAL — Dalla Man moves up.'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
