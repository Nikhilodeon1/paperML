"""Backtest the metabolic module.

Ground truth sources:
  - RMR: the published Mifflin-St Jeor worked example. For a 30 y, 80 kg, 180 cm
    male: RMR = 10*80 + 6.25*180 - 5*30 + 5 = 1780 kcal/day. We check the module's
    RMR equation reproduces canonical values exactly.
  - Short-horizon weight change: the energy-balance identity. A sustained daily
    deficit D held for N days removes ~ D*N / 7700 kg, slightly LESS once RMR falls
    with weight. We check sign, rough magnitude, and that adaptation reduces loss
    below the naive linear estimate.
  - Maintenance: intake == TDEE => flat weight.

Run:  python -m evaluation.backtest_metabolic
"""

from __future__ import annotations

import numpy as np

from knowledge_base import load_system
from modules.metabolic import _rmr_mifflin, project_weight

RMR_TOL = 0.5          # kcal — exact equation, allow rounding only
MAINTENANCE_TOL = 0.2  # kg drift over a year at true maintenance


def _maintenance_intake(weight, height, age, sex, activity, kb):
    pal = kb[{"sedentary": "pal_sedentary", "moderate": "pal_moderate",
              "active": "pal_active"}[activity]].value
    return pal * _rmr_mifflin(weight, height, age, sex, kb)


def run() -> int:
    kb = load_system("metabolic")
    ok = True

    # --- 1. RMR canonical values --------------------------------------------
    print("=" * 70)
    print("METABOLIC BACKTEST")
    print("=" * 70)
    cases = [
        ("male", 80, 180, 30, 1780.0),
        ("female", 65, 165, 30, 10 * 65 + 6.25 * 165 - 5 * 30 - 161),
    ]
    print("RMR (Mifflin-St Jeor) vs hand-computed:")
    for sex, w, h, a, expected in cases:
        got = _rmr_mifflin(w, h, a, sex, kb)
        err = abs(got - expected)
        ok &= err <= RMR_TOL
        print(f"  {sex:>6} {w}kg {h}cm {a}y : expected {expected:.1f}  got {got:.1f}  (|err| {err:.2f})")

    # --- 2. Maintenance => flat weight --------------------------------------
    intake = _maintenance_intake(80, 180, 30, "male", "sedentary", kb)
    # Check the MEDIAN trajectory (priors centered at the maintenance assumption);
    # a single noisy draw would not sit exactly at maintenance.
    res = project_weight(80, 180, 30, "male", daily_intake_kcal=intake,
                         activity="sedentary", horizon_days=365, n_samples=600)
    drift = abs(res.final_weight - 80)
    ok &= drift <= MAINTENANCE_TOL
    print(f"\nMaintenance (intake={intake:.0f} kcal): 1-yr drift {drift:.3f} kg "
          f"(tol {MAINTENANCE_TOL}) -> {'OK' if drift <= MAINTENANCE_TOL else 'FAIL'}")

    # --- 3. Deficit: sign, magnitude, adaptation ----------------------------
    deficit = 500.0
    res = project_weight(80, 180, 30, "male", daily_intake_kcal=intake - deficit,
                         activity="sedentary", horizon_days=30, n_samples=1)
    naive = deficit * 30 / kb["kcal_per_kg_fat"].value  # linear, no adaptation
    actual_loss = 80 - res.final_weight
    print(f"\n500 kcal/day deficit, 30 d: naive linear loss {naive:.2f} kg, "
          f"model loss {actual_loss:.2f} kg")
    sign_ok = actual_loss > 0
    adapt_ok = actual_loss < naive          # adaptation must reduce loss
    mag_ok = actual_loss > naive * 0.85     # but not wildly less over just 30 d
    ok &= sign_ok and adapt_ok and mag_ok
    print(f"  loses weight: {sign_ok} | below naive (adaptation): {adapt_ok} | "
          f"within 15% of naive: {mag_ok}")

    # --- 4. CI widens with horizon ------------------------------------------
    short = project_weight(80, 180, 30, "male", daily_intake_kcal=intake - 300,
                           activity="sedentary", horizon_days=30)
    long = project_weight(80, 180, 30, "male", daily_intake_kcal=intake - 300,
                          activity="sedentary", horizon_days=365)
    sw = short.final_weight_ci[1] - short.final_weight_ci[0]
    lw = long.final_weight_ci[1] - long.final_weight_ci[0]
    widens = lw > sw
    ok &= widens
    print(f"\nFinal-weight CI width: 30 d = {sw:.2f} kg, 365 d = {lw:.2f} kg "
          f"-> {'widens OK' if widens else 'FAIL'}")
    print(f"  evidence label 30 d: {short.evidence.value} | 365 d: {long.evidence.value} "
          "(long horizon downgraded as designed)")

    print("=" * 70)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
