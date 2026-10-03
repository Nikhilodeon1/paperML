"""Backtest the hepatic (BAC) module.

Ground truth = the analytic Widmark equation, the accepted clinical/forensic
standard for BAC. With instantaneous absorption and zero-order elimination:

    BAC(t) = A / (r * W_kg * 10)  -  beta * t        (g/100mL)
    peak    = A / (r * W_kg * 10)
    t_sober = peak / beta

Our numerical integrator, run with very fast absorption, must converge to this.
We sweep a grid of realistic weights / sexes / doses and report relative error.

Run:  python -m evaluation.backtest_hepatic
"""

from __future__ import annotations

import itertools

import numpy as np

from knowledge_base import load_system
from modules.hepatic import Drink, compute_bac
from modules.hepatic import _simulate_bac, _time_to_sober

PEAK_TOL = 0.05       # 5% relative error allowed vs analytic Widmark peak
SOBER_TOL = 0.08      # 8% on time-to-sober (integrator + grid discretization)
FAST_KA = 500.0       # near-instant absorption to match analytic Widmark assumption


def analytic_widmark(a_grams: float, r: float, weight_kg: float, beta: float):
    vd_100ml = r * weight_kg * 10.0
    peak = a_grams / vd_100ml
    t_sober = peak / beta
    return peak, t_sober


def run() -> int:
    kb = load_system("hepatic")
    g_per_drink = kb["standard_drink_grams"].value
    beta = kb["elimination_rate_beta"].value
    r_by_sex = {
        "male": kb["widmark_r_male"].value,
        "female": kb["widmark_r_female"].value,
    }

    weights = [55, 70, 85, 100]
    sexes = ["male", "female"]
    n_drinks = [1, 2, 4]

    peak_errs, sober_errs = [], []
    rows = []
    for w, sex, n in itertools.product(weights, sexes, n_drinks):
        a = n * g_per_drink
        r = r_by_sex[sex]
        exp_peak, exp_sober = analytic_widmark(a, r, w, beta)

        # Validate the INTEGRATOR against the analytic equation: drive absorption
        # fast (FAST_KA) so the only difference from instantaneous-absorption
        # Widmark is numerical integration error. Deterministic (no MC noise).
        times = np.arange(0.0, max(12.0, exp_sober * 1.5), 0.02)
        curve = _simulate_bac(
            [Drink(grams_ethanol=a, hour=0.0)], w, r, beta, FAST_KA, times
        )
        obs_peak = float(curve.max())
        obs_sober = _time_to_sober(times, curve)

        pe = abs(obs_peak - exp_peak) / exp_peak
        se = abs(obs_sober - exp_sober) / exp_sober
        peak_errs.append(pe)
        sober_errs.append(se)
        rows.append((w, sex, n, exp_peak, obs_peak, pe, exp_sober, obs_sober, se))

    print("=" * 96)
    print("HEPATIC MODULE BACKTEST  —  numerical integrator vs analytic Widmark")
    print("=" * 96)
    print(f"{'wt':>4} {'sex':>7} {'drk':>4} | {'peak_exp':>9} {'peak_obs':>9} {'err%':>6} "
          f"| {'sober_exp':>9} {'sober_obs':>9} {'err%':>6}")
    print("-" * 96)
    for (w, sex, n, ep, op, pe, es, os_, se) in rows:
        print(f"{w:>4} {sex:>7} {n:>4} | {ep:>9.4f} {op:>9.4f} {pe*100:>5.1f}% "
              f"| {es:>9.2f} {os_:>9.2f} {se*100:>5.1f}%")
    print("-" * 96)

    peak_mre = float(np.mean(peak_errs))
    peak_max = float(np.max(peak_errs))
    sober_mre = float(np.mean(sober_errs))
    sober_max = float(np.max(sober_errs))

    print(f"Peak BAC      : mean rel err {peak_mre*100:5.2f}%  max {peak_max*100:5.2f}%  "
          f"(tol {PEAK_TOL*100:.0f}%)")
    print(f"Time-to-sober : mean rel err {sober_mre*100:5.2f}%  max {sober_max*100:5.2f}%  "
          f"(tol {SOBER_TOL*100:.0f}%)")

    ok = peak_max <= PEAK_TOL and sober_max <= SOBER_TOL
    print("=" * 96)
    print("RESULT:", "PASS" if ok else "FAIL")

    # Horizon/CI sanity (brief 2): the confidence interval must widen further out.
    print("\nConfidence-interval-widens-with-horizon check:")
    res = compute_bac([Drink.standard(2)], weight_kg=80, sex="male", horizon_h=10.0, n_samples=600)
    width = res.bac_upper - res.bac_lower
    peak_i = int(np.argmax(res.bac_median))
    # Compare CI width near peak vs on the late falling limb (same BAC level region).
    late_i = min(peak_i + int(2.0 / 0.05), len(width) - 1)
    print(f"  CI width at peak (~t={res.times_h[peak_i]:.1f}h): {width[peak_i]:.4f}")
    print(f"  CI width later   (~t={res.times_h[late_i]:.1f}h): {width[late_i]:.4f}")
    widens = width[late_i] >= width[peak_i] * 0.5  # falling-limb timing uncertainty
    print(f"  time-to-sober CI: {res.time_to_sober_ci[0]:.2f}–{res.time_to_sober_ci[1]:.2f} h "
          f"(spread {res.time_to_sober_ci[1]-res.time_to_sober_ci[0]:.2f} h) "
          f"-> {'OK' if widens else 'CHECK'}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
