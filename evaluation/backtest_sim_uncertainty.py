"""Validate uncertainty quantification on the body-simulation engine.

Confidence bands are only useful if they're honest. These checks assert the properties a
correct Monte-Carlo UQ must have:

  1. the ensemble MEDIAN tracks the deterministic (best-estimate) run,
  2. bands are ordered lo <= median <= hi everywhere,
  3. the deterministic run is COVERED by the 5-95% band (calibration sanity),
  4. MORE parameter uncertainty -> WIDER bands (monotone in the inputs),
  5. PERSONALIZATION tightens the band for a parameter we actually learn (RMR -> energy),
  6. determinism under a fixed seed.

Run:  python -m evaluation.backtest_sim_uncertainty
"""

from __future__ import annotations

import numpy as np

from simulation import Simulator, PhysioParams, Schedule, Meal, Exercise
from simulation.uncertainty import run_ensemble, ParamUncertainty, Spec


def run() -> int:
    ok = True
    print("=" * 72)
    print("BODY-SIMULATION ENGINE — uncertainty quantification (Monte-Carlo bands)")
    print("=" * 72)
    p = PhysioParams(weight_kg=80, sex="male", age=35)
    sched = Schedule().add(Meal(0, carbs_g=70))

    det = Simulator(p).run(sched, 210, outputs=["glucose_mg_dl"])
    det_peak = max(det.series["glucose_mg_dl"])
    ens = run_ensemble(p, sched, 210, n=150, outputs=["glucose_mg_dl"], seed=1)
    med = ens.bands["glucose_mg_dl"]["median"]
    lo, hi = ens.bands["glucose_mg_dl"]["lo"], ens.bands["glucose_mg_dl"]["hi"]
    med_peak = max(med)

    # 1. median tracks deterministic
    c1 = abs(med_peak - det_peak) / det_peak < 0.06
    ok &= c1
    print(f"1. Ensemble median peak {med_peak:.0f} vs deterministic {det_peak:.0f} "
          f"(<6% off): {'OK' if c1 else 'FAIL'}")

    # 2. band ordering lo <= median <= hi
    c2 = all(lo[i] - 1e-6 <= med[i] <= hi[i] + 1e-6 for i in range(len(med)))
    ok &= c2
    print(f"2. Bands ordered lo <= median <= hi at every time: {'OK' if c2 else 'FAIL'}")

    # 3. deterministic run covered by the 5-95% band (allow small Monte-Carlo slack)
    g = det.series["glucose_mg_dl"]
    covered = sum(1 for i in range(len(g)) if lo[i] - 2 <= g[i] <= hi[i] + 2) / len(g)
    c3 = covered >= 0.9
    ok &= c3
    print(f"3. Deterministic run inside 5-95% band {covered*100:.0f}% of the time "
          f"(>=90%): {'OK' if c3 else 'FAIL'}")

    # 4. more parameter uncertainty -> wider band
    wide = ParamUncertainty({"carb_absorption": Spec(0.5), "insulin_secretion": Spec(0.5),
                             "glucose_effectiveness": Spec(0.5)})
    narrow = ParamUncertainty({"carb_absorption": Spec(0.05), "insulin_secretion": Spec(0.05),
                              "glucose_effectiveness": Spec(0.05)})
    ew = run_ensemble(p, sched, 210, n=150, unc=wide, outputs=["glucose_mg_dl"], seed=2)
    en = run_ensemble(p, sched, 210, n=150, unc=narrow, outputs=["glucose_mg_dl"], seed=2)
    wsum = ew.summary(["glucose_mg_dl"])["glucose_mg_dl"]["peak"]
    nsum = en.summary(["glucose_mg_dl"])["glucose_mg_dl"]["peak"]
    c4 = (wsum["hi"] - wsum["lo"]) > (nsum["hi"] - nsum["lo"])
    ok &= c4
    print(f"4. Wider priors -> wider band (peak width {wsum['hi']-wsum['lo']:.0f} > "
          f"{nsum['hi']-nsum['lo']:.0f}): {'OK' if c4 else 'FAIL'}")

    # 5. personalization tightens a LEARNED parameter's band (RMR posterior -> energy)
    exsched = Schedule().add(Exercise(0, 60, 6))
    pop = run_ensemble(p, exsched, 120, n=150, outputs=["energy_expended_kcal"], seed=3)
    learned = {"derived": {"metabolic": {"rmr_multiplier": 0.9, "rmr_multiplier_sd": 0.03,
                                         "n_observations": 12}}}
    per = run_ensemble(p, exsched, 120, n=150, unc=ParamUncertainty.from_user(learned),
                       outputs=["energy_expended_kcal"], seed=3)
    pw = pop.summary(["energy_expended_kcal"])["energy_expended_kcal"]["end"]
    rw = per.summary(["energy_expended_kcal"])["energy_expended_kcal"]["end"]
    c5 = (rw["hi"] - rw["lo"]) < (pw["hi"] - pw["lo"])
    ok &= c5
    print(f"5. Personalization sharpens forecast (energy band {rw['hi']-rw['lo']:.0f} < "
          f"population {pw['hi']-pw['lo']:.0f} kcal): {'OK' if c5 else 'FAIL'}")

    # 6. determinism under fixed seed
    a = run_ensemble(p, sched, 120, n=40, seed=7, outputs=["glucose_mg_dl"]).bands
    b = run_ensemble(p, sched, 120, n=40, seed=7, outputs=["glucose_mg_dl"]).bands
    c6 = a == b
    ok &= c6
    print(f"6. Determinism under fixed seed: {'OK' if c6 else 'FAIL'}")

    print("=" * 72)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
