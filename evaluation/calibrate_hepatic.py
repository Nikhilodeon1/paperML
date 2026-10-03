"""Calibration harness for the hepatic module (brief 4).

Accuracy (backtest_hepatic.py) asks "is the point estimate right?". Calibration
asks a different, often-ignored question: "when the model says 90% confident, is it
actually right 90% of the time?". For a probabilistic model with intervals, that
means INTERVAL COVERAGE — a nominal X% predictive interval should contain the true
value X% of the time.

Method (a proper frequentist coverage check of a Bayesian-style predictive interval):
  1. Draw a synthetic individual whose TRUE parameters (Widmark r, beta, ka) are
     sampled from the population priors in Layer 1.
  2. Compute that individual's TRUE peak BAC for a random scenario.
  3. The model, knowing only the population prior (no personal data), forms a
     predictive distribution of peak BAC by Monte-Carlo over the same priors.
  4. For each nominal level, check whether the true peak falls inside the predictive
     interval. Average over many individuals => empirical coverage.

If the interval construction is honest, empirical coverage ~= nominal across levels.
This is the reliability diagram that backs the product's confidence labels.

Run:  python -m evaluation.calibrate_hepatic
"""

from __future__ import annotations

import numpy as np

from knowledge_base import load_system
from modules.hepatic import Drink, _simulate_bac

# Mean abs gap between nominal and empirical coverage we tolerate.
MAX_CALIBRATION_ERROR = 0.05

NOMINAL_LEVELS = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])


def _truncated_normal(rng, mu, sd, lo, hi, size):
    s = rng.normal(mu, sd, size=size)
    return np.clip(s, lo, hi)


def run(n_individuals: int = 400, n_pred_samples: int = 300, seed: int = 0) -> int:
    kb = load_system("hepatic")
    g = kb["standard_drink_grams"].value
    r_male, r_female = kb["widmark_r_male"], kb["widmark_r_female"]
    beta_p, ka_p = kb["elimination_rate_beta"], kb["absorption_rate_ka"]

    rng = np.random.default_rng(seed)
    hits = np.zeros(len(NOMINAL_LEVELS))

    for _ in range(n_individuals):
        # Random scenario.
        sex = rng.choice(["male", "female"])
        r_param = r_male if sex == "male" else r_female
        weight = float(rng.uniform(55, 100))
        n_drinks = float(rng.integers(1, 5))
        times = np.arange(0.0, 16.0, 0.1)
        drinks = [Drink(grams_ethanol=n_drinks * g, hour=0.0)]

        # 1-2) TRUE individual drawn from the priors.
        true_r = float(_truncated_normal(rng, r_param.value, r_param.population_sd, *r_param.plausible_range, 1)[0])
        true_beta = float(_truncated_normal(rng, beta_p.value, beta_p.population_sd, *beta_p.plausible_range, 1)[0])
        true_ka = float(_truncated_normal(rng, ka_p.value, ka_p.population_sd, *ka_p.plausible_range, 1)[0])
        true_peak = _simulate_bac(drinks, weight, true_r, true_beta, true_ka, times).max()

        # 3) Model's predictive distribution from the prior (no personal data).
        r_s = _truncated_normal(rng, r_param.value, r_param.population_sd, *r_param.plausible_range, n_pred_samples)
        b_s = _truncated_normal(rng, beta_p.value, beta_p.population_sd, *beta_p.plausible_range, n_pred_samples)
        k_s = _truncated_normal(rng, ka_p.value, ka_p.population_sd, *ka_p.plausible_range, n_pred_samples)
        pred_peaks = np.array([
            _simulate_bac(drinks, weight, r_s[i], b_s[i], k_s[i], times).max()
            for i in range(n_pred_samples)
        ])

        # 4) Coverage at each nominal level.
        for j, level in enumerate(NOMINAL_LEVELS):
            lo = np.percentile(pred_peaks, (1 - level) / 2 * 100)
            hi = np.percentile(pred_peaks, (1 + level) / 2 * 100)
            if lo <= true_peak <= hi:
                hits[j] += 1

    empirical = hits / n_individuals
    gaps = np.abs(empirical - NOMINAL_LEVELS)

    print("=" * 60)
    print("HEPATIC MODULE CALIBRATION  (reliability diagram)")
    print(f"individuals={n_individuals}  pred_samples={n_pred_samples}")
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
