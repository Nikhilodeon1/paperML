"""Calibration harness for the metabolic module (brief 4).

Same coverage check as calibrate_hepatic, applied to the projected final weight:
draw a synthetic individual whose true RMR multiplier / PAL / energy-density are
sampled from the Layer 1 priors, compute their TRUE final weight under a random
intake scenario, then check how often the model's prior-only predictive interval
contains it. Honest intervals => empirical coverage ~= nominal.

Horizon is 120 days here (long enough for trajectories to spread, short enough to
keep the day-by-day simulation fast). Run:

    python -m evaluation.calibrate_metabolic
"""

from __future__ import annotations

import numpy as np

from knowledge_base import load_system
from modules.metabolic import _rmr_mifflin, _simulate_weight

MAX_CALIBRATION_ERROR = 0.05
NOMINAL_LEVELS = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])


def _tn(rng, mu, sd, lo, hi, size):
    return np.clip(rng.normal(mu, sd, size=size), lo, hi)


def run(n_individuals: int = 400, n_pred_samples: int = 300,
        horizon_days: int = 120, seed: int = 0) -> int:
    kb = load_system("metabolic")
    rmr_sd = kb["rmr_individual_sd"].population_sd
    kcal = kb["kcal_per_kg_fat"]
    pal_params = {a: kb[f"pal_{a}"] for a in ("sedentary", "moderate", "active")}

    rng = np.random.default_rng(seed)
    days = np.arange(1, horizon_days + 1)
    hits = np.zeros(len(NOMINAL_LEVELS))

    for _ in range(n_individuals):
        sex = rng.choice(["male", "female"])
        weight = float(rng.uniform(55, 105))
        height = float(rng.uniform(155, 190))
        age = float(rng.uniform(20, 65))
        activity = rng.choice(["sedentary", "moderate", "active"])
        pal_p = pal_params[activity]

        # Maintenance for the prior-mean person, then a random surplus/deficit.
        maint = pal_p.value * _rmr_mifflin(weight, height, age, sex, kb)
        intake = maint + float(rng.uniform(-700, 700))

        # TRUE individual from the priors.
        t_rmr = float(_tn(rng, 1.0, rmr_sd, 0.7, 1.3, 1)[0])
        t_pal = float(_tn(rng, pal_p.value, pal_p.population_sd, *pal_p.plausible_range, 1)[0])
        t_kcal = float(_tn(rng, kcal.value, kcal.value * 0.03, *kcal.plausible_range, 1)[0])
        true_final = _simulate_weight(weight, height, age, sex, intake,
                                      t_rmr, t_pal, t_kcal, days, kb)[-1]

        # Model predictive distribution from the priors (no personal data).
        r_s = _tn(rng, 1.0, rmr_sd, 0.7, 1.3, n_pred_samples)
        p_s = _tn(rng, pal_p.value, pal_p.population_sd, *pal_p.plausible_range, n_pred_samples)
        k_s = _tn(rng, kcal.value, kcal.value * 0.03, *kcal.plausible_range, n_pred_samples)
        pred = np.array([
            _simulate_weight(weight, height, age, sex, intake,
                             r_s[i], p_s[i], k_s[i], days, kb)[-1]
            for i in range(n_pred_samples)
        ])

        for j, level in enumerate(NOMINAL_LEVELS):
            lo = np.percentile(pred, (1 - level) / 2 * 100)
            hi = np.percentile(pred, (1 + level) / 2 * 100)
            if lo <= true_final <= hi:
                hits[j] += 1

    empirical = hits / n_individuals
    gaps = np.abs(empirical - NOMINAL_LEVELS)

    print("=" * 60)
    print("METABOLIC MODULE CALIBRATION  (reliability diagram)")
    print(f"individuals={n_individuals}  pred_samples={n_pred_samples}  "
          f"horizon={horizon_days}d")
    print("=" * 60)
    print(f"{'nominal':>9} {'empirical':>11} {'gap':>8}")
    print("-" * 60)
    for level, emp, gap in zip(NOMINAL_LEVELS, empirical, gaps):
        bar = "#" * int(round(emp * 30))
        print(f"{level*100:>7.0f}% {emp*100:>10.1f}% {gap*100:>6.1f}%  {bar}")
    print("-" * 60)
    mce = float(np.mean(gaps))
    print(f"Mean calibration error: {mce*100:.2f}%  (tol {MAX_CALIBRATION_ERROR*100:.0f}%)")
    ok = mce <= MAX_CALIBRATION_ERROR
    print("RESULT:", "PASS" if ok else "FAIL")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
