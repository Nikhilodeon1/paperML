"""Calibrate more of the engine against published reference physiology.

Beyond glucose and BAC, this checks the engine's resting and response values land in the
ranges the literature reports — insulin, caffeine pharmacokinetics, the heart-rate/effort
relationship, and the diurnal cortisol rhythm — so 'accurate' is quantified across the
board, not just for the two headline curves.

Run:  python -m evaluation.calibrate_engine_physiology
"""

from __future__ import annotations

from simulation import Simulator, PhysioParams, Schedule, Meal, Caffeine, Exercise


def run() -> int:
    print("=" * 70)
    print("ENGINE PHYSIOLOGY CALIBRATION — insulin, caffeine PK, HR/effort, cortisol")
    print("=" * 70)
    p = PhysioParams(weight_kg=75, sex="male", age=35, vo2max=42)
    checks = []

    # 1. resting fasting glucose + insulin
    f = Simulator(p).run(Schedule(), 60, outputs=["glucose_mg_dl", "insulin_uU_ml"]).final()
    checks.append(("Fasting glucose 70-99 mg/dL", 70 <= f["glucose_mg_dl"] <= 99, f"{f['glucose_mg_dl']:.0f}"))
    checks.append(("Fasting insulin 2-25 uU/mL", 2 <= f["insulin_uU_ml"] <= 25, f"{f['insulin_uU_ml']:.1f}"))

    # 2. postprandial insulin (75 g)
    m = Simulator(p).run(Schedule().add(Meal(0, 75)), 240, outputs=["insulin_uU_ml"])
    ipk, i3h = max(m.series["insulin_uU_ml"]), m.at("insulin_uU_ml", 180)
    checks.append(("75g insulin peak 40-110 uU/mL", 40 <= ipk <= 110, f"{ipk:.0f}"))
    checks.append(("Insulin returns toward baseline by 3h (<30)", i3h < 30, f"{i3h:.0f}"))

    # 3. caffeine PK: ~5 h half-life; HR bump 4-15 bpm at 200 mg
    cf = Simulator(p).run(Schedule().add(Caffeine(0, 200)), 600, outputs=["caffeine_mg"])
    frac5h = cf.at("caffeine_mg", 300) / 200.0
    hb = max(Simulator(p).run(Schedule().add(Caffeine(0, 200)), 300,
                              outputs=["heart_rate_bpm"]).series["heart_rate_bpm"]) - p.hr_rest
    checks.append(("Caffeine ~5h half-life (0.4-0.6 left at 5h)", 0.4 <= frac5h <= 0.6, f"{frac5h:.2f}"))
    checks.append(("Caffeine 200mg HR bump 4-15 bpm", 4 <= hb <= 15, f"{hb:.0f}"))

    # 4. heart-rate / effort relationship (% of HRmax)
    half = Simulator(p).run(Schedule().add(Exercise(0, 20, p.vo2max * 0.5 / 3.5)), 40,
                            outputs=["heart_rate_bpm"])
    mx = Simulator(p).run(Schedule().add(Exercise(0, 20, p.vo2max / 3.5)), 40, outputs=["heart_rate_bpm"])
    h50 = 100 * max(half.series["heart_rate_bpm"]) / p.hr_max
    hmax = 100 * max(mx.series["heart_rate_bpm"]) / p.hr_max
    checks.append(("HR at 50% VO2max = 55-78% HRmax", 55 <= h50 <= 78, f"{h50:.0f}%"))
    checks.append(("HR at max effort = 92-112% HRmax", 92 <= hmax <= 112, f"{hmax:.0f}%"))

    # 5. diurnal cortisol
    am = Simulator(p).run(Schedule(), 90, outputs=["cortisol_ug_dl"], start_hour=8).final()["cortisol_ug_dl"]
    pm = Simulator(p).run(Schedule(), 90, outputs=["cortisol_ug_dl"], start_hour=21).final()["cortisol_ug_dl"]
    checks.append(("Morning cortisol 10-20 ug/dL", 10 <= am <= 20, f"{am:.1f}"))
    checks.append(("Evening cortisol 3-9 ug/dL", 3 <= pm <= 9, f"{pm:.1f}"))

    ok = True
    for label, passed, val in checks:
        ok &= passed
        print(f"  [{'OK' if passed else 'FAIL'}] {label:<44} = {val}")
    print("=" * 70)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
