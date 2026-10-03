"""Validate the body-simulation engine — plausibility + coupling + metamorphic checks.

The engine has no single ground-truth trajectory to compare against, so (like the other
modules) we validate the PROPERTIES a correct coupled body simulator must have: numbers
stay physiological, and perturbing one input moves the right variables the right way
THROUGH the shared state (the whole point of the framework).

Run:  python -m evaluation.backtest_simulation
"""

from __future__ import annotations

from simulation import (Simulator, PhysioParams, Schedule, Meal, Drink, Caffeine,
                        Exercise, Stressor)


def _sim(**kw):
    return Simulator(PhysioParams(weight_kg=80, sex="male", age=35, **kw))


def run() -> int:
    ok = True
    print("=" * 70)
    print("BODY-SIMULATION ENGINE — plausibility, coupling & metamorphic checks")
    print("=" * 70)
    p = PhysioParams(weight_kg=80, sex="male", age=35)
    sim = Simulator(p)

    # 1. Postprandial glucose is physiological and returns toward baseline.
    meal = sim.run(Schedule().add(Meal(0, carbs_g=60)), duration_min=210)
    g = meal.series["glucose_mg_dl"]
    gpk, gend, ins = max(g), g[-1], meal.peak("insulin_uU_ml")
    c1 = 120 <= gpk <= 190 and 80 <= gend <= 105 and 30 <= ins <= 130
    ok &= c1
    print(f"1. Meal 60g: glucose peak {gpk:.0f} (120-190), returns to {gend:.0f} (80-105), "
          f"insulin peak {ins:.0f} (30-130) -> {'OK' if c1 else 'FAIL'}")

    # 2. Bigger meal -> higher glucose peak (monotonic dose-response).
    big = sim.run(Schedule().add(Meal(0, carbs_g=100)), duration_min=210)
    c2 = big.peak("glucose_mg_dl") > gpk
    ok &= c2
    print(f"2. 100g meal peak {big.peak('glucose_mg_dl'):.0f} > 60g peak {gpk:.0f}: "
          f"{'OK' if c2 else 'FAIL'}")

    # 3. Exercise after a meal LOWERS glucose (cross-system: activity -> metabolism).
    ex = sim.run(Schedule().add(Meal(0, carbs_g=60)).add(Exercise(45, 75, 8)), duration_min=210)
    c3 = ex.at("glucose_mg_dl", 90) < meal.at("glucose_mg_dl", 90) and ex.peak("heart_rate_bpm") > 120
    ok &= c3
    print(f"3. Exercise lowers post-meal glucose ({ex.at('glucose_mg_dl',90):.0f} < "
          f"{meal.at('glucose_mg_dl',90):.0f}) and raises HR to {ex.peak('heart_rate_bpm'):.0f}: "
          f"{'OK' if c3 else 'FAIL'}")

    # 4. Insulin resistance (diabetic) -> higher, more sustained glucose than healthy.
    dia = Simulator(PhysioParams(weight_kg=80, sex="male", age=35,
                                 insulin_sensitivity=0.4, insulin_secretion=0.1))
    dmeal = dia.run(Schedule().add(Meal(0, carbs_g=60)), duration_min=210)
    c4 = dmeal.peak("glucose_mg_dl") > gpk and dmeal.at("glucose_mg_dl", 120) > meal.at("glucose_mg_dl", 120)
    ok &= c4
    print(f"4. Insulin-resistant peak {dmeal.peak('glucose_mg_dl'):.0f} > healthy {gpk:.0f} and "
          f"higher at 2h: {'OK' if c4 else 'FAIL'}")

    # 5. Acute stressor raises cortisol AND heart rate (stress -> autonomic coupling).
    st = sim.run(Schedule().add(Stressor(20, 80, 0.8)), duration_min=180)
    c5 = 13 <= st.peak("cortisol_ug_dl") <= 25 and st.peak("heart_rate_bpm") > p.hr_rest + 3
    ok &= c5
    print(f"5. Stressor: cortisol {st.peak('cortisol_ug_dl'):.1f} (13-25) and HR up to "
          f"{st.peak('heart_rate_bpm'):.0f}: {'OK' if c5 else 'FAIL'}")

    # 6. Caffeine raises HR and clears with a ~5h half-life.
    caf = sim.run(Schedule().add(Caffeine(0, 200)), duration_min=360)
    c6 = caf.peak("heart_rate_bpm") > p.hr_rest + 5 and caf.at("caffeine_mg", 300) < 130
    ok &= c6
    print(f"6. Caffeine 200mg: HR peak {caf.peak('heart_rate_bpm'):.0f}, "
          f"{caf.at('caffeine_mg',300):.0f}mg left at 5h (<130): {'OK' if c6 else 'FAIL'}")

    # 7. Alcohol BAC peaks then eliminates to zero (Widmark, matches hepatic module).
    alc = sim.run(Schedule().add(Drink(0, 3)), duration_min=420)
    c7 = 0.04 <= alc.peak("bac_g_dl") <= 0.09 and alc.final()["bac_g_dl"] < 0.005
    ok &= c7
    print(f"7. 3 drinks: BAC peak {alc.peak('bac_g_dl'):.3f} (0.04-0.09), sober by end: "
          f"{'OK' if c7 else 'FAIL'}")

    # 8. Sleep pressure builds while awake and dissipates during sleep.
    from simulation import Sleep, Water
    awake = sim.run(Schedule(), duration_min=480)
    slept = sim.run(Schedule().add(Sleep(0, 480)), duration_min=480)
    c8 = awake.final()["sleep_pressure"] > awake.series["sleep_pressure"][0] \
        and slept.final()["sleep_pressure"] < slept.series["sleep_pressure"][0]
    ok &= c8
    print(f"8. Sleep pressure rises awake ({awake.final()['sleep_pressure']:.2f}) and falls "
          f"asleep ({slept.final()['sleep_pressure']:.2f}): {'OK' if c8 else 'FAIL'}")

    # 9. ACCURACY: postprandial glucose peaks at a realistic ~45-60 min (2-compartment gut).
    gt = meal.times_min[meal.series["glucose_mg_dl"].index(max(meal.series["glucose_mg_dl"]))]
    c9 = 40 <= gt <= 70
    ok &= c9
    print(f"9. Glucose peak timing {gt:.0f} min (physiological 40-70): {'OK' if c9 else 'FAIL'}")

    # 10. Hydration coupling: long hard exercise dehydrates -> HR drifts up; drinking helps.
    dry = sim.run(Schedule().add(Exercise(0, 90, 9)), duration_min=120)
    wet = sim.run(Schedule().add(Exercise(0, 90, 9)).add(Water(30, 500)).add(Water(60, 500)),
                  duration_min=120)
    c10 = dry.final()["water_deficit_ml"] > 1000 and wet.final()["water_deficit_ml"] < dry.final()["water_deficit_ml"] \
        and dry.at("heart_rate_bpm", 95) >= wet.at("heart_rate_bpm", 95)
    ok &= c10
    print(f"10. Dehydration drift: deficit {dry.final()['water_deficit_ml']:.0f}ml -> HR "
          f"{dry.at('heart_rate_bpm',95):.0f}; hydrated -> {wet.at('heart_rate_bpm',95):.0f}: "
          f"{'OK' if c10 else 'FAIL'}")

    # 11. STRATEGIC: selecting outputs runs fewer modules but gives identical results.
    full = sim.run(Schedule().add(Meal(0, carbs_g=60)), duration_min=180)
    sel = sim.run(Schedule().add(Meal(0, carbs_g=60)), duration_min=180, outputs=["glucose_mg_dl"])
    n_full, n_sel = len(sim.modules), len(sim.modules_for(["glucose_mg_dl"]))
    c11 = n_sel < n_full and abs(sel.peak("glucose_mg_dl") - full.peak("glucose_mg_dl")) < 1e-6
    ok &= c11
    print(f"11. Strategic selection: glucose needs {n_sel}/{n_full} modules, identical result: "
          f"{'OK' if c11 else 'FAIL'}")

    # 12. DATA-DRIVEN PHARMACOLOGY: a stimulant raises HR, a sedative lowers alertness,
    #     and an unknown substance is safely ignored (extensible by JSON, no crash).
    from simulation import Dose
    base = sim.run(Schedule(), duration_min=180)
    nic = sim.run(Schedule().add(Dose(0, "nicotine", 1)), duration_min=180)
    mel = sim.run(Schedule().add(Dose(0, "melatonin", 3)), duration_min=240)
    unk = sim.run(Schedule().add(Dose(0, "unobtainium", 999)), duration_min=180)
    c12 = (nic.peak("heart_rate_bpm") - base.peak("heart_rate_bpm") > 4
           and min(mel.series["alertness"]) < min(base.series["alertness"]) - 0.05
           and unk.series == base.series)     # unknown substance -> identical trajectory
    ok &= c12
    print(f"12. Pharmacology (data-driven): nicotine HR +{nic.peak('heart_rate_bpm')-base.peak('heart_rate_bpm'):.0f}, "
          f"melatonin alertness {min(mel.series['alertness']):.2f}, unknown-substance safe: "
          f"{'OK' if c12 else 'FAIL'}")

    # 13. FASTING / KETOSIS: no ketosis fed; multi-day fast builds ketones + defends
    #     glucose lower; refeeding clears ketones (classic fasting physiology).
    fed = sim.run(Schedule().add(Meal(0, 70)).add(Meal(300, 80)).add(Meal(600, 75)), 900,
                  outputs=["ketones_mmol_l"])
    fast = sim.run(Schedule(), 72 * 60, outputs=["ketones_mmol_l", "glucose_mg_dl"])
    k48, k72 = fast.at("ketones_mmol_l", 48 * 60), fast.at("ketones_mmol_l", 72 * 60)
    gfast = fast.at("glucose_mg_dl", 72 * 60)
    refeed = sim.run(Schedule().add(Meal(48 * 60, 80)), 56 * 60, outputs=["ketones_mmol_l"])
    cleared = refeed.at("ketones_mmol_l", 54 * 60) < refeed.at("ketones_mmol_l", 48 * 60)
    c13 = (fed.final()["ketones_mmol_l"] < 0.3 and 1.0 <= k48 <= 4.0 and k72 > k48
           and 65 <= gfast <= 88 and cleared)
    ok &= c13
    print(f"13. Fasting/ketosis: fed ketones {fed.final()['ketones_mmol_l']:.1f}, 48h {k48:.1f}, "
          f"72h {k72:.1f}, fasting glucose {gfast:.0f}, refeed clears: {'OK' if c13 else 'FAIL'}")

    # 14. THERMOREGULATION: exercise + heat raise core temp (and HR via drift), cold lowers
    #     it, and a hot environment drives sweating.
    from simulation import Ambient
    rest = sim.run(Schedule(), 120, outputs=["core_temp_c"])
    exo = sim.run(Schedule().add(Exercise(0, 60, 10)), 90, outputs=["core_temp_c", "heart_rate_bpm"])
    hot = sim.run(Schedule().add(Ambient(0, 180, 38)), 180, outputs=["core_temp_c", "water_deficit_ml"])
    cold = sim.run(Schedule().add(Ambient(0, 120, 2)), 120, outputs=["core_temp_c"])
    c14 = (abs(rest.final()["core_temp_c"] - 37.0) < 0.1
           and 38.3 <= max(exo.series["core_temp_c"]) <= 40.0 and max(exo.series["heart_rate_bpm"]) > 150
           and 37.3 <= hot.final()["core_temp_c"] <= 38.2 and hot.final()["water_deficit_ml"] > 200
           and 36.2 <= cold.final()["core_temp_c"] <= 36.9)
    ok &= c14
    print(f"14. Thermoregulation: rest {rest.final()['core_temp_c']:.1f}, exercise "
          f"{max(exo.series['core_temp_c']):.1f}, hot {hot.final()['core_temp_c']:.1f} "
          f"(+sweat {hot.final()['water_deficit_ml']:.0f}ml), cold {cold.final()['core_temp_c']:.1f}: "
          f"{'OK' if c14 else 'FAIL'}")

    # 15. FITNESS / HRV / RECOVERY: a fitter body (higher VO2max) runs a lower HR at the
    #     same effort; HRV drops during exercise and rebounds afterward; alcohol lowers HRV.
    fit_p = PhysioParams(weight_kg=75, sex="male", age=30, vo2max=55, hrv_rest=70)
    unfit_p = PhysioParams(weight_kg=90, sex="male", age=50, vo2max=28, hrv_rest=28)
    ef = Simulator(fit_p).run(Schedule().add(Exercise(0, 30, 8)), 90,
                              outputs=["heart_rate_bpm", "hrv_rmssd_ms"])
    eu = Simulator(unfit_p).run(Schedule().add(Exercise(0, 30, 8)), 90, outputs=["heart_rate_bpm"])
    alc = Simulator(fit_p).run(Schedule().add(Drink(0, 4)), 240, outputs=["hrv_rmssd_ms"])
    hrv_exercise = min(ef.series["hrv_rmssd_ms"])
    hrv_recovered = ef.at("hrv_rmssd_ms", 60)
    c15 = (max(ef.series["heart_rate_bpm"]) < max(eu.series["heart_rate_bpm"]) - 15
           and hrv_exercise < fit_p.hrv_rest * 0.7 and hrv_recovered > hrv_exercise + 10
           and min(alc.series["hrv_rmssd_ms"]) < fit_p.hrv_rest * 0.7)
    ok &= c15
    print(f"15. Fitness/HRV: fit HR {max(ef.series['heart_rate_bpm']):.0f} < unfit "
          f"{max(eu.series['heart_rate_bpm']):.0f}; HRV exercise {hrv_exercise:.0f}->recover "
          f"{hrv_recovered:.0f}; alcohol lowers HRV: {'OK' if c15 else 'FAIL'}")

    # 16. CIRCADIAN + HORMONAL: cortisol follows a diurnal rhythm (morning > evening); a
    #     luteal-phase cycle raises basal core temp and lowers HRV vs the follicular phase.
    am = sim.run(Schedule(), 120, outputs=["cortisol_ug_dl"], start_hour=7).final()["cortisol_ug_dl"]
    pm = sim.run(Schedule(), 120, outputs=["cortisol_ug_dl"], start_hour=20).final()["cortisol_ug_dl"]
    foll = Simulator(PhysioParams(weight_kg=62, sex="female", age=30, cycle_day=7, hrv_rest=60))
    lut = Simulator(PhysioParams(weight_kg=62, sex="female", age=30, cycle_day=22, hrv_rest=60))
    cf = foll.run(Schedule(), 60, outputs=["core_temp_c", "hrv_rmssd_ms"]).final()
    cl = lut.run(Schedule(), 60, outputs=["core_temp_c", "hrv_rmssd_ms"]).final()
    c16 = (am > pm + 4 and cl["core_temp_c"] > cf["core_temp_c"] + 0.2
           and cl["hrv_rmssd_ms"] < cf["hrv_rmssd_ms"])
    ok &= c16
    print(f"16. Circadian/hormonal: cortisol AM {am:.1f} > PM {pm:.1f}; luteal core "
          f"{cl['core_temp_c']:.2f} > follicular {cf['core_temp_c']:.2f}, HRV lower: "
          f"{'OK' if c16 else 'FAIL'}")

    # 17. DATA-DRIVEN FOODS: the SAME grams of a high-GI food (white rice) and a low-GI,
    #     high-fibre food (lentils) produce very different glucose peaks; protein-only food
    #     barely moves glucose; an unknown food is safely ignored.
    from simulation import Food
    rice = sim.run(Schedule().add(Food(0, "white_rice_cooked", 200)), 200, outputs=["glucose_mg_dl"])
    lentil = sim.run(Schedule().add(Food(0, "lentils_cooked", 200)), 200, outputs=["glucose_mg_dl"])
    chicken = sim.run(Schedule().add(Food(0, "chicken_breast", 200)), 200, outputs=["glucose_mg_dl"])
    unknown = sim.run(Schedule().add(Food(0, "unicorn_meat", 100)), 120, outputs=["glucose_mg_dl"])
    c17 = (max(rice.series["glucose_mg_dl"]) > max(lentil.series["glucose_mg_dl"]) + 25
           and max(chicken.series["glucose_mg_dl"]) < 100
           and max(unknown.series["glucose_mg_dl"]) < 92)
    ok &= c17
    print(f"17. Data-driven foods: white rice peak {max(rice.series['glucose_mg_dl']):.0f} > "
          f"lentils {max(lentil.series['glucose_mg_dl']):.0f}, chicken flat, unknown safe: "
          f"{'OK' if c17 else 'FAIL'}")

    print("=" * 70)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
