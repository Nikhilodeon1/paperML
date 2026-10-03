"""Does modelling CIRCADIAN glucose tolerance cut held-out error on real data?

The engine now scales insulin sensitivity by time of day (worse in the evening), controlled by
`circadian_amp`. This calibrates that amplitude on CGMacros meal timestamps and asks the only
question that matters: does it reduce HELD-OUT per-meal iAUC error vs amp=0?

Trick that keeps it fast + exact: circadian only multiplies insulin sensitivity, so
iAUC(base_Si, amp, clock_hour) == iAUC(base_Si * (1 - amp*resistance(hour)), amp=0). So the
amp=0 iAUC table (over eff-carbs x Si) is reused — a meal at a given hour is just looked up at
its EFFECTIVE Si. No re-simulation per amp.

5-fold: fit each subject's base Si on train meals (given amp), predict held-out meals, score.
Compare the held-out iAUC MAE across amp values. If a nonzero amp wins, circadian earns its
place and the live default is set to it.

Run:  python -m evaluation.circadian_validation
"""

from __future__ import annotations

import statistics

import numpy as np

from evaluation.cgmacros import _profile, load_bio, predict_meal_iauc, subjects
from simulation.modules.metabolic import _circadian_resistance

_CARB_GRID = np.arange(5.0, 175.0, 12.0)
_SI_GRID = np.linspace(0.2, 1.6, 15)
_BLUNT_A, _BLUNT_B = 0.08, 0.005              # calibrated fibre/fat (held fixed here)
_AMP_GRID = [0.0, 0.10, 0.20, 0.30, 0.40]
_N_FOLDS = 5


def _blunt(fi, fa):
    return 1.0 / (1.0 + _BLUNT_A * fi + _BLUNT_B * fa)


def _table(profile):
    tab = np.empty((len(_CARB_GRID), len(_SI_GRID)))
    for i, c in enumerate(_CARB_GRID):
        for j, si in enumerate(_SI_GRID):
            tab[i, j] = predict_meal_iauc(float(c), float(si), profile)[0]
    return tab


def _lookup(table, eff_carbs, si):
    """Bilinear iAUC(eff_carbs, si) — si continuous (circadian gives non-grid effective Si)."""
    si = float(np.clip(si, _SI_GRID[0], _SI_GRID[-1]))
    j = int(np.clip(np.searchsorted(_SI_GRID, si) - 1, 0, len(_SI_GRID) - 2))
    s0, s1 = _SI_GRID[j], _SI_GRID[j + 1]
    v0 = np.interp(eff_carbs, _CARB_GRID, table[:, j])
    v1 = np.interp(eff_carbs, _CARB_GRID, table[:, j + 1])
    w = (si - s0) / (s1 - s0)
    return float(v0 * (1 - w) + v1 * w)


def _si_eff(base_si, amp, hour):
    return base_si * (1.0 - amp * _circadian_resistance(hour))


def _pred(table, meal, base_si, amp):
    cg, fi, fa, r, hour = meal
    return _lookup(table, cg * _blunt(fi, fa), _si_eff(base_si, amp, hour))


def _fit_base_si(table, meals, amp):
    best, best_err = float(_SI_GRID[0]), float("inf")
    for si in _SI_GRID:
        err = sum(abs(_pred(table, m, si, amp) - m[3]) for m in meals)
        if err < best_err:
            best, best_err = float(si), err
    return best


def _meal(m):
    r = m.real_iauc()
    return None if r is None else (m.carbs_g, m.fiber_g, m.fat_g, r, m.clock_hour)


def run(limit: int | None = None, seed: int = 0) -> dict:
    bio = load_bio()
    data = []
    for sid, meals in subjects(limit=limit):
        tuples = [t for t in (_meal(x) for x in meals) if t is not None]
        if len(tuples) >= 12:
            data.append((tuples, _table(_profile(bio.get(sid)))))

    # per-amp held-out MAE, averaged over folds
    amp_mae = {a: [] for a in _AMP_GRID}
    for fold in range(_N_FOLDS):
        for tuples, table in data:
            rng = np.random.default_rng(seed * 100 + fold + len(tuples))
            assign = rng.integers(0, _N_FOLDS, size=len(tuples))
            train = [tuples[i] for i in range(len(tuples)) if assign[i] != fold]
            test = [tuples[i] for i in range(len(tuples)) if assign[i] == fold]
            if not train or not test:
                continue
            for a in _AMP_GRID:
                si = _fit_base_si(table, train, a)
                amp_mae[a].append(statistics.fmean(abs(_pred(table, m, si, a) - m[3]) for m in test))
    return {"n_subjects": len(data),
            "mae_by_amp": {a: round(statistics.fmean(v), 0) for a, v in amp_mae.items() if v}}


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    r = run()
    print("=" * 72)
    print("CIRCADIAN validation — held-out per-meal iAUC MAE vs circadian_amp (CGMacros)")
    print("=" * 72)
    print(f"  subjects: {r['n_subjects']}")
    base = r["mae_by_amp"].get(0.0)
    for a, mae in r["mae_by_amp"].items():
        delta = f"  ({100*(mae-base)/base:+.1f}% vs amp=0)" if base and a != 0.0 else "  (baseline)"
        print(f"    circadian_amp {a:.2f}:  held-out iAUC MAE {mae:.0f}{delta}")
    best_amp = min(r["mae_by_amp"], key=r["mae_by_amp"].get)
    improved = r["mae_by_amp"][best_amp] < base
    print(f"\n  best amp: {best_amp:.2f}  "
          f"({'circadian HELPS -> set live default' if improved and best_amp > 0 else 'no help -> keep amp=0'})")
    print("=" * 72)


if __name__ == "__main__":
    main()
