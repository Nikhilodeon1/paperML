"""Horizon ML playground — test every capability in one place.

Two modes:

  Capabilities tour (default, offline, no API):
      python -m evaluation.playground

  Interactive Q&A (you type questions):
      python -m evaluation.playground --chat            # deterministic router (free)
      python -m evaluation.playground --chat --gemini   # Gemini routing (uses API calls)

The tour walks through all 5 modules, both personalization layers, both cross-system
edges, and the orchestration router (including the honest "no model" refusal).
"""

from __future__ import annotations

import sys

import numpy as np

from knowledge_base import load_system
from modules.cardiovascular import compute_cvd_risk
from modules.hepatic import Drink, compute_bac
from modules.metabolic import _rmr_mifflin, _simulate_weight, project_weight
from modules.sleep import SleepModel
from modules.stress import StressModel
from personalization.hepatic import BacReading, personal_params_from_readings
from personalization.metabolic import Weighin, personal_params_from_logs
from pipeline.graph import run_alcohol_then_sleep, run_stress_then_sleep
from orchestration.router import UserProfile, route


def _hdr(title):
    print("\n" + "=" * 68 + f"\n  {title}\n" + "=" * 68)


def tour():
    print("HORIZON ML — CAPABILITIES TOUR")
    sleep_model = SleepModel().fit()
    stress_model = StressModel().fit()

    # 1. Hepatic ------------------------------------------------------------
    _hdr("1. HEPATIC — blood alcohol (mechanistic, Widmark)")
    r = compute_bac([Drink.standard(4)], weight_kg=80, sex="male")
    print(f"4 drinks, 80kg male: peak BAC {r.peak_bac:.3f}, sober in "
          f"{r.time_to_sober_h:.1f}h ({r.time_to_sober_ci[0]:.1f}-{r.time_to_sober_ci[1]:.1f}) "
          f"[{r.evidence.value}]")

    # 2. Metabolic ----------------------------------------------------------
    _hdr("2. METABOLIC — weight trajectory (Mifflin-St Jeor + energy balance)")
    w = project_weight(80, 180, 35, "male", daily_intake_kcal=2000, horizon_days=180)
    print(f"2000 kcal/day, 180d: {w.delta_kg:+.1f} kg "
          f"({w.delta_kg_ci[0]:+.1f} to {w.delta_kg_ci[1]:+.1f}), "
          f"maintenance {w.tdee_estimate:.0f} kcal [{w.evidence.value}]")

    # 3. Metabolic personalization -----------------------------------------
    _hdr("3. PERSONALIZATION — metabolic RMR from weigh-ins (Bayesian/Kalman)")
    kb = load_system("metabolic")
    pal, kcal = kb["pal_sedentary"].value, kb["kcal_per_kg_fat"].value
    days = np.arange(1, 15 * 14 + 1)
    traj = _simulate_weight(82, 178, 35, "male", 2200, 0.87, pal, kcal, days, kb)
    logs = [Weighin(0, 82, 2200)] + [Weighin(i * 14, float(traj[i * 14 - 1]), 2200)
                                     for i in range(1, 15)]
    pp = personal_params_from_logs(logs, 178, 35, "male")
    print(f"True hidden RMR multiplier 0.87 -> estimated {pp.rmr_multiplier:.3f} "
          f"(posterior sd {pp.rmr_multiplier_sd:.3f}) from {len(logs)} weigh-ins")

    # 4. Hepatic personalization -------------------------------------------
    _hdr("4. PERSONALIZATION — personal alcohol elimination rate")
    readings = [BacReading(1.0 + i * 0.5, max(0, 0.10 - 0.020 * (i * 0.5)))
                for i in range(8)]
    hp = personal_params_from_readings(readings)
    print(f"True elimination 0.020 -> estimated {hp.elimination_beta:.4f} "
          f"(sd {hp.elimination_beta_sd:.4f}) from falling-limb readings")

    # 5. Cardiovascular -----------------------------------------------------
    _hdr("5. CARDIOVASCULAR — 10-yr CVD risk (Framingham)")
    c = compute_cvd_risk(age=55, sex="male", total_chol=240, hdl=40, sbp=145, smoker=True)
    print(f"55yM, chol240 HDL40 SBP145 smoker: risk {c.risk_10yr*100:.1f}% "
          f"({c.risk_ci[0]*100:.1f}-{c.risk_ci[1]*100:.1f}%), {c.heart_age_note}")
    print(f"   biggest lever: {c.drivers[0][0]} (-{c.drivers[0][1]*100:.1f}%)")

    # 6. Sleep --------------------------------------------------------------
    _hdr("6. SLEEP — architecture prediction (ML, gradient-boosted)")
    s = sleep_model.predict(age=35, caffeine_mg_afternoon=150, bedtime_regularity=0.9)
    print(f"35y, 150mg afternoon caffeine: REM {s.metrics['rem_pct']:.0f}%, "
          f"deep {s.metrics['deep_pct']:.0f}%, efficiency {s.metrics['sleep_efficiency']:.0f}%")

    # 7. Stress -------------------------------------------------------------
    _hdr("7. STRESS — arousal index from wearables (ML)")
    st = stress_model.predict(heart_rate=95, rmssd=22, eda=6.0)
    print(f"HR95 RMSSD22 EDA6: stress index {st.stress_index:.0f}/100 "
          f"({st.interval[0]:.0f}-{st.interval[1]:.0f})")

    # 8. Cross-system: alcohol -> sleep ------------------------------------
    _hdr("8. CROSS-SYSTEM — alcohol -> sleep (strong edge)")
    for n in (0, 3, 6):
        cs = run_alcohol_then_sleep(drinks=[Drink.standard(n)] if n else [],
                                    weight_kg=80, sex="male", bedtime_hour=2.0, age=35,
                                    sleep_model=sleep_model)
        print(f"  {n} drinks: bedtime BAC {cs.bedtime_bac:.3f} -> "
              f"REM {cs.sleep.metrics['rem_pct']:.0f}%, "
              f"awakenings {cs.sleep.metrics['awakenings']:.1f}")

    # 9. Cross-system: stress -> sleep -------------------------------------
    _hdr("9. CROSS-SYSTEM — stress -> sleep (weak edge, low confidence)")
    calm = run_stress_then_sleep(heart_rate=60, rmssd=60, eda=1.5, age=35,
                                 sleep_model=sleep_model, stress_model=stress_model)
    tense = run_stress_then_sleep(heart_rate=105, rmssd=18, eda=8.0, age=35,
                                  sleep_model=sleep_model, stress_model=stress_model)
    print(f"  calm  (stress {calm.stress_index:.0f}): efficiency "
          f"{calm.baseline_efficiency:.0f}% -> {calm.adjusted_efficiency:.0f}%")
    print(f"  tense (stress {tense.stress_index:.0f}): efficiency "
          f"{tense.baseline_efficiency:.0f}% -> {tense.adjusted_efficiency:.0f}%")

    # 10. Orchestration -----------------------------------------------------
    _hdr("10. ORCHESTRATION — natural-language routing (deterministic)")
    profile = UserProfile(80, 180, 45, "male", total_chol=230, hdl=42, sbp=138, smoker=True)
    for q in ["4 beers tonight, when am I sober?",
              "eat 1800 calories a day for 3 months, what happens to my weight?",
              "what's my heart disease risk?",
              "how will 3 drinks before bed affect my sleep?",
              "will standing on my head improve my eyesight?"]:
        a = route(q, profile)
        print(f"  Q: {q}\n     [{a.evidence}] {a.text[:110]}")

    print("\n" + "=" * 68)
    print("Tour complete. For live typing:  python -m evaluation.playground --chat")
    print("=" * 68)


def chat(use_gemini: bool):
    profile = UserProfile(80, 180, 45, "male", total_chol=230, hdl=42, sbp=138, smoker=True)
    router = None
    if use_gemini:
        from orchestration.llm_gemini import route_with_gemini
        router = lambda q: route_with_gemini(q, profile)
    else:
        router = lambda q: route(q, profile)

    print(f"Interactive mode ({'Gemini' if use_gemini else 'deterministic'}). "
          f"Profile: 45y male, 80kg. Type a question, or 'quit'.")
    print("Examples: '5 pints tonight?', 'lose weight on 1800 kcal?', 'my heart risk?'")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q", ""):
            break
        a = router(q)
        print(f"[{a.evidence}] {a.text}")


def main():
    args = sys.argv[1:]
    if "--chat" in args:
        chat(use_gemini="--gemini" in args)
    else:
        tour()


if __name__ == "__main__":
    main()
