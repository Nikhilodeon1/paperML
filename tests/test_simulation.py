"""Body-simulation framework tests — shared-state coupling + personalization.

Fast guards that the engine evolves a physiological trajectory and that perturbing one
input moves the right variables through the shared state (the coupling that makes it a
body simulator). Exhaustive plausibility lives in evaluation/backtest_simulation.py.
"""

from __future__ import annotations

from simulation import (Simulator, PhysioParams, Schedule, Meal, Drink, Caffeine,
                        Exercise, Stressor, Sleep)


def _sim():
    return Simulator(PhysioParams(weight_kg=80, sex="male", age=35))


def test_meal_glucose_is_physiological_and_recovers():
    tr = _sim().run(Schedule().add(Meal(0, carbs_g=60)), duration_min=210)
    g = tr.series["glucose_mg_dl"]
    assert g[0] == 90.0
    assert 120 <= max(g) <= 190           # postprandial peak
    assert 80 <= g[-1] <= 105             # returns toward baseline
    assert 30 <= tr.peak("insulin_uU_ml") <= 130


def test_bigger_meal_higher_peak():
    a = _sim().run(Schedule().add(Meal(0, carbs_g=40)), duration_min=180).peak("glucose_mg_dl")
    b = _sim().run(Schedule().add(Meal(0, carbs_g=90)), duration_min=180).peak("glucose_mg_dl")
    assert b > a


def test_exercise_lowers_glucose_and_raises_hr_via_shared_state():
    meal = _sim().run(Schedule().add(Meal(0, carbs_g=60)), duration_min=210)
    ex = _sim().run(Schedule().add(Meal(0, carbs_g=60)).add(Exercise(45, 75, 8)), duration_min=210)
    assert ex.at("glucose_mg_dl", 90) < meal.at("glucose_mg_dl", 90)
    assert ex.peak("heart_rate_bpm") > 120


def test_stress_couples_to_cortisol_and_heart_rate():
    tr = _sim().run(Schedule().add(Stressor(20, 80, 0.8)), duration_min=180)
    assert tr.peak("cortisol_ug_dl") > 12.5
    assert tr.peak("heart_rate_bpm") > 63


def test_alcohol_matches_widmark_shape():
    tr = _sim().run(Schedule().add(Drink(0, 3)), duration_min=420)
    assert 0.04 <= tr.peak("bac_g_dl") <= 0.09
    assert tr.final()["bac_g_dl"] < 0.005          # zero-order elimination to sober


def test_personalization_diabetic_runs_higher():
    healthy = _sim().run(Schedule().add(Meal(0, carbs_g=75)), duration_min=210)
    dia = Simulator(PhysioParams(weight_kg=80, sex="male", age=35,
                                 insulin_sensitivity=0.4, insulin_secretion=0.1))
    d = dia.run(Schedule().add(Meal(0, carbs_g=75)), duration_min=210)
    assert d.peak("glucose_mg_dl") > healthy.peak("glucose_mg_dl")


def test_params_from_profile():
    p = PhysioParams.from_profile({"sex": "female", "weight_kg": 62, "height_cm": 168,
                                   "age": 40, "diabetic": True})
    assert p.sex == "female" and p.weight_kg == 62
    assert p.insulin_sensitivity < 1.0            # diabetic lowers sensitivity
    assert p.rmr_kcal_min > 0.5


def test_sleep_pressure_two_process():
    awake = _sim().run(Schedule(), duration_min=300)
    slept = _sim().run(Schedule().add(Sleep(0, 300)), duration_min=300)
    assert awake.final()["sleep_pressure"] > awake.series["sleep_pressure"][0]
    assert slept.final()["sleep_pressure"] < slept.series["sleep_pressure"][0]


def test_glucose_peak_timing_is_physiological():
    tr = _sim().run(Schedule().add(Meal(0, carbs_g=60)), duration_min=240)
    g = tr.series["glucose_mg_dl"]
    t_peak = tr.times_min[g.index(max(g))]
    assert 40 <= t_peak <= 70            # two-compartment gut -> realistic delayed peak


def test_hydration_dehydration_raises_hr_and_water_repays():
    from simulation import Water
    dry = _sim().run(Schedule().add(Exercise(0, 90, 9)), duration_min=120)
    wet = _sim().run(Schedule().add(Exercise(0, 90, 9)).add(Water(30, 500)).add(Water(60, 500)),
                     duration_min=120)
    assert dry.final()["water_deficit_ml"] > 1000
    assert wet.final()["water_deficit_ml"] < dry.final()["water_deficit_ml"]
    assert dry.at("heart_rate_bpm", 95) >= wet.at("heart_rate_bpm", 95)


def test_fasting_produces_ketosis_and_defends_glucose():
    p = PhysioParams(weight_kg=80, sex="male", age=35)
    # fed day -> no ketosis
    fed = Simulator(p).run(Schedule().add(Meal(0, 70)).add(Meal(300, 80)).add(Meal(600, 75)),
                           900, outputs=["ketones_mmol_l"])
    assert fed.final()["ketones_mmol_l"] < 0.3
    # multi-day fast -> nutritional ketosis + glucose defended lower
    fast = Simulator(p).run(Schedule(), 72 * 60, outputs=["ketones_mmol_l", "glucose_mg_dl"])
    assert fast.at("ketones_mmol_l", 12 * 60) < 0.5        # not ketotic overnight
    assert 1.0 <= fast.at("ketones_mmol_l", 48 * 60) <= 4.0
    assert fast.at("ketones_mmol_l", 72 * 60) > fast.at("ketones_mmol_l", 48 * 60)
    assert 65 <= fast.at("glucose_mg_dl", 72 * 60) <= 88   # defended, not crashed


def test_refeeding_clears_ketones():
    p = PhysioParams(weight_kg=80, sex="male", age=35)
    rf = Simulator(p).run(Schedule().add(Meal(48 * 60, 80)), 56 * 60, outputs=["ketones_mmol_l"])
    assert rf.at("ketones_mmol_l", 54 * 60) < rf.at("ketones_mmol_l", 48 * 60)


def test_foods_glycemic_response_is_data_driven():
    from simulation import Food
    from simulation.foods import known_foods
    assert {"white_rice_cooked", "lentils_cooked", "banana"} <= set(known_foods())
    p = PhysioParams(weight_kg=75, sex="male", age=35)
    rice = Simulator(p).run(Schedule().add(Food(0, "white_rice_cooked", 200)), 200, outputs=["glucose_mg_dl"])
    lentil = Simulator(p).run(Schedule().add(Food(0, "lentils_cooked", 200)), 200, outputs=["glucose_mg_dl"])
    chicken = Simulator(p).run(Schedule().add(Food(0, "chicken_breast", 200)), 200, outputs=["glucose_mg_dl"])
    assert max(rice.series["glucose_mg_dl"]) > max(lentil.series["glucose_mg_dl"]) + 25  # GI + fibre
    assert max(chicken.series["glucose_mg_dl"]) < 100                                    # no carbs
    # unknown food is ignored, not a crash
    unk = Simulator(p).run(Schedule().add(Food(0, "unicorn_meat", 100)), 120, outputs=["glucose_mg_dl"])
    assert max(unk.series["glucose_mg_dl"]) < 92


def test_diurnal_cortisol_and_menstrual_cycle():
    p = PhysioParams(weight_kg=80, sex="male", age=35)
    am = Simulator(p).run(Schedule(), 120, outputs=["cortisol_ug_dl"], start_hour=7).final()["cortisol_ug_dl"]
    pm = Simulator(p).run(Schedule(), 120, outputs=["cortisol_ug_dl"], start_hour=20).final()["cortisol_ug_dl"]
    assert am > pm + 4                              # morning cortisol higher than evening
    foll = Simulator(PhysioParams(weight_kg=62, sex="female", age=30, cycle_day=7, hrv_rest=60))
    lut = Simulator(PhysioParams(weight_kg=62, sex="female", age=30, cycle_day=22, hrv_rest=60))
    cf = foll.run(Schedule(), 60, outputs=["core_temp_c", "hrv_rmssd_ms"]).final()
    cl = lut.run(Schedule(), 60, outputs=["core_temp_c", "hrv_rmssd_ms"]).final()
    assert cl["core_temp_c"] > cf["core_temp_c"] + 0.2     # luteal basal temp rise
    assert cl["hrv_rmssd_ms"] < cf["hrv_rmssd_ms"]         # luteal HRV dip
    assert not PhysioParams(weight_kg=80, sex="male", cycle_day=22).luteal


def test_fitness_and_hrv_dynamics():
    from simulation import Drink
    fit = PhysioParams(weight_kg=75, sex="male", age=30, vo2max=55, hrv_rest=70)
    unfit = PhysioParams(weight_kg=90, sex="male", age=50, vo2max=28, hrv_rest=28)
    ef = Simulator(fit).run(Schedule().add(Exercise(0, 30, 8)), 90,
                            outputs=["heart_rate_bpm", "hrv_rmssd_ms"])
    eu = Simulator(unfit).run(Schedule().add(Exercise(0, 30, 8)), 90, outputs=["heart_rate_bpm"])
    assert max(ef.series["heart_rate_bpm"]) < max(eu.series["heart_rate_bpm"]) - 15  # fitter = lower HR
    assert min(ef.series["hrv_rmssd_ms"]) < fit.hrv_rest * 0.7                        # HRV drops
    assert ef.at("hrv_rmssd_ms", 60) > min(ef.series["hrv_rmssd_ms"]) + 10            # rebounds
    alc = Simulator(fit).run(Schedule().add(Drink(0, 4)), 240, outputs=["hrv_rmssd_ms"])
    assert min(alc.series["hrv_rmssd_ms"]) < fit.hrv_rest * 0.7                       # alcohol lowers HRV


def test_thermoregulation_responds_to_exercise_heat_cold():
    from simulation import Ambient
    p = PhysioParams(weight_kg=80, sex="male", age=35)
    assert abs(Simulator(p).run(Schedule(), 90, outputs=["core_temp_c"]).final()["core_temp_c"] - 37.0) < 0.1
    exo = Simulator(p).run(Schedule().add(Exercise(0, 60, 10)), 90,
                           outputs=["core_temp_c", "heart_rate_bpm"])
    assert 38.3 <= max(exo.series["core_temp_c"]) <= 40.0     # exercise hyperthermia
    hot = Simulator(p).run(Schedule().add(Ambient(0, 180, 38)), 180,
                           outputs=["core_temp_c", "water_deficit_ml"])
    assert hot.final()["core_temp_c"] > 37.3 and hot.final()["water_deficit_ml"] > 200
    cold = Simulator(p).run(Schedule().add(Ambient(0, 120, 2)), 120, outputs=["core_temp_c"])
    assert cold.final()["core_temp_c"] < 36.9


def test_pharmacology_is_data_driven_and_directional():
    from simulation import Dose
    from simulation.modules.pharmacology import known_substances
    assert {"nicotine", "melatonin", "diphenhydramine"} <= set(known_substances())
    base = _sim().run(Schedule(), 180)
    # stimulant raises heart rate
    nic = _sim().run(Schedule().add(Dose(0, "nicotine", 1)), 180)
    assert nic.peak("heart_rate_bpm") - base.peak("heart_rate_bpm") > 4
    # sedative lowers alertness
    dip = _sim().run(Schedule().add(Dose(0, "diphenhydramine", 25)), 240)
    assert min(dip.series["alertness"]) < min(base.series["alertness"]) - 0.1
    # decongestant raises blood pressure
    pse = _sim().run(Schedule().add(Dose(0, "pseudoephedrine", 60)), 240)
    assert pse.peak("sbp_mmhg") - base.peak("sbp_mmhg") > 3


def test_unknown_substance_is_ignored_safely():
    from simulation import Dose
    base = _sim().run(Schedule(), 120)
    unk = _sim().run(Schedule().add(Dose(0, "unobtainium", 1e6)), 120)
    assert abs(unk.peak("heart_rate_bpm") - base.peak("heart_rate_bpm")) < 1e-6


def test_strategic_selection_is_minimal_and_exact():
    sim = _sim()
    needed = sim.modules_for(["glucose_mg_dl"])
    assert len(needed) < len(sim.modules)          # skips alcohol/autonomic/sleep/hydration
    # glucose is written by metabolic + substrate; they read cortisol/caffeine/insulin, so
    # stress, caffeine and pharmacology join the closure too.
    assert {m.name for m in needed} == {"metabolic", "substrate", "stress", "caffeine", "pharmacology"}
    assert sim.modules_for(["bac_g_dl"])[0].name == "alcohol"
    assert len(sim.modules_for(["bac_g_dl"])) == 1
    # selecting a subset gives identical numbers for the selected variable
    full = sim.run(Schedule().add(Meal(0, 60)), 180)
    sel = sim.run(Schedule().add(Meal(0, 60)), 180, outputs=["glucose_mg_dl"])
    assert abs(sel.peak("glucose_mg_dl") - full.peak("glucose_mg_dl")) < 1e-6
