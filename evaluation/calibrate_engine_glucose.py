"""Calibrate the engine's postprandial glucose against published reference physiology.

No local continuous-glucose dataset, so we calibrate against the well-documented mean
glucose response to a ~75 g carbohydrate load in healthy adults, and against the ADA
diagnostic thresholds for glucose tolerance. This quantifies the engine's error in real
mg/dL rather than trusting a plausible-looking curve.

Reference (healthy adult, ~75 g carbohydrate, venous plasma glucose, mg/dL) — a composite
of standardized meal / OGTT mean curves (American Diabetes Association 2021 diagnostic
criteria; Ceriello 2008 postprandial reviews):

    t(min):   0    30    45    60    90   120   180
    G(mg/dL): 90   140   150   145   125   108    93

Run:  python -m evaluation.calibrate_engine_glucose
"""

from __future__ import annotations

import statistics

from simulation import Simulator, PhysioParams, Schedule, Meal

_REF = {0: 90, 30: 140, 45: 150, 60: 145, 90: 125, 120: 108, 180: 93}


def run() -> int:
    print("=" * 70)
    print("ENGINE GLUCOSE CALIBRATION vs healthy 75 g reference + ADA thresholds")
    print("=" * 70)
    p = PhysioParams(weight_kg=75, sex="male", age=35)
    tr = Simulator(p).run(Schedule().add(Meal(0, 75)), 200,
                          outputs=["glucose_mg_dl", "insulin_uU_ml"])

    print("  t(min)  engine  reference  error")
    errs = []
    for t, ref in _REF.items():
        g = tr.at("glucose_mg_dl", t)
        errs.append(abs(g - ref))
        print(f"  {t:>4}    {g:>5.0f}    {ref:>5}     {g-ref:+.0f}")
    mae = statistics.mean(errs)
    g = tr.series["glucose_mg_dl"]
    peak, peak_t = max(g), tr.times_min[g.index(max(g))]
    g2h = tr.at("glucose_mg_dl", 120)

    # ADA glucose-tolerance behaviour: healthy 2 h < 140; a diabetic parameter set must
    # cross into impaired/diabetic territory (>= 140, trending to >= 200 for T2D).
    pd = PhysioParams(weight_kg=75, sex="male", age=55,
                      insulin_sensitivity=0.35, insulin_secretion=0.08)
    d2h = Simulator(pd).run(Schedule().add(Meal(0, 75)), 200,
                            outputs=["glucose_mg_dl"]).at("glucose_mg_dl", 120)

    mae_ok = mae < 15.0
    peak_ok = 130 <= peak <= 185 and 40 <= peak_t <= 70
    ada_ok = g2h < 140 and d2h >= 140
    ok = mae_ok and peak_ok and ada_ok
    print("-" * 70)
    print(f"Mean absolute error vs reference: {mae:.1f} mg/dL (tol 15): {'OK' if mae_ok else 'FAIL'}")
    print(f"Peak {peak:.0f} mg/dL @ {peak_t:.0f} min (physiological 130-185 / 40-70 min): "
          f"{'OK' if peak_ok else 'FAIL'}")
    print(f"ADA tolerance: healthy 2h {g2h:.0f} (<140), diabetic 2h {d2h:.0f} (>=140): "
          f"{'OK' if ada_ok else 'FAIL'}")
    print("=" * 70)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
