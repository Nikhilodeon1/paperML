"""Robustness fuzzing for the body-simulation engine.

Plausibility (backtest_simulation) proves the engine is RIGHT on hand-picked cases; this
proves it is ROBUST on everything else: thousands of randomized bodies x randomized
scenarios, including deliberately abusive inputs (500 g meals, 40-MET efforts, 30 drinks,
5 g caffeine, 3 kg / NaN weights, multi-day runs). Hard invariants that must hold for
EVERY run:

  1. never raises,
  2. every value stays finite (no NaN/inf ever propagates),
  3. every variable stays within its declared physiological bounds (outputs saturate),
  4. numerical stability: halving the step barely moves the trajectory (Euler is adequate),
  5. determinism: same inputs -> identical output.

Run:  python -m evaluation.backtest_sim_robustness
"""

from __future__ import annotations

import math

import numpy as np

from simulation import (Simulator, PhysioParams, Schedule, Meal, Food, Drink, Caffeine, Water,
                        Dose, Exercise, Stressor, Sleep, Ambient)
from simulation.modules.pharmacology import known_substances
from simulation.foods import known_foods
from simulation.state import BodyState


def _random_params(rng) -> PhysioParams:
    # mostly realistic, sometimes abusive (the clamps must absorb it)
    extreme = rng.random() < 0.15
    return PhysioParams(
        weight_kg=float(rng.choice([rng.normal(78, 18), rng.choice([0, 3, 1e4, float("nan")])])
                        if extreme else rng.normal(78, 18)),
        height_cm=float(rng.normal(172, 12)),
        age=float(rng.choice([rng.normal(45, 18), rng.choice([-5, 0, 200])]) if extreme else rng.normal(45, 18)),
        sex=str(rng.choice(["male", "female", "x"] if extreme else ["male", "female"])),
        insulin_sensitivity=float(rng.choice([rng.uniform(0.3, 1.6), rng.choice([0, -1, 100])])
                                  if extreme else rng.uniform(0.3, 1.6)),
        insulin_secretion=float(rng.uniform(0.05, 0.3)))


def _random_schedule(rng) -> Schedule:
    sch = Schedule()
    big = rng.random() < 0.2
    for _ in range(int(rng.integers(0, 4))):
        sch.add(Meal(float(rng.uniform(0, 300)),
                     carbs_g=float(rng.choice([rng.uniform(10, 120), 500]) if big else rng.uniform(10, 120))))
    foods = known_foods() + ["unicorn_meat"]
    for _ in range(int(rng.integers(0, 3))):
        sch.add(Food(float(rng.uniform(0, 300)), str(rng.choice(foods)),
                     grams=float(rng.choice([rng.uniform(30, 400), 1e5]) if big else rng.uniform(30, 400))))
    for _ in range(int(rng.integers(0, 3))):
        sch.add(Drink(float(rng.uniform(0, 300)),
                      standard_drinks=float(rng.choice([rng.uniform(1, 5), 30]) if big else rng.uniform(1, 5))))
    for _ in range(int(rng.integers(0, 3))):
        sch.add(Caffeine(float(rng.uniform(0, 300)),
                         mg=float(rng.choice([rng.uniform(40, 250), 5000]) if big else rng.uniform(40, 250))))
    for _ in range(int(rng.integers(0, 2))):
        sch.add(Water(float(rng.uniform(0, 300)), ml=float(rng.uniform(100, 1000))))
    subs = known_substances() + ["unobtainium"]      # incl. an unknown for safety
    for _ in range(int(rng.integers(0, 3))):
        sch.add(Dose(float(rng.uniform(0, 300)), str(rng.choice(subs)),
                     mg=float(rng.choice([rng.uniform(1, 100), 1e5]) if big else rng.uniform(1, 100))))
    if rng.random() < 0.5:
        a = float(rng.uniform(0, 200)); sch.add(Exercise(a, a + rng.uniform(10, 120),
                  intensity_mets=float(rng.choice([rng.uniform(2, 12), 40]) if big else rng.uniform(2, 12))))
    if rng.random() < 0.4:
        a = float(rng.uniform(0, 200)); sch.add(Stressor(a, a + rng.uniform(20, 150), level=float(rng.uniform(0, 1))))
    if rng.random() < 0.3:
        a = float(rng.uniform(0, 200)); sch.add(Sleep(a, a + rng.uniform(60, 480)))
    if rng.random() < 0.4:
        a = float(rng.uniform(0, 200))
        sch.add(Ambient(a, a + rng.uniform(30, 300),
                        temp_c=float(rng.choice([rng.uniform(-10, 45), 60]) if big else rng.uniform(-10, 45))))
    return sch


def run() -> int:
    print("=" * 72)
    print("BODY-SIMULATION ENGINE — robustness fuzzing (randomized bodies x scenarios)")
    print("=" * 72)
    rng = np.random.default_rng(20260711)
    N = 600
    bounds = BodyState._BOUNDS
    crashes = 0
    nonfinite = 0
    oob = 0
    examples: list[str] = []

    for i in range(N):
        p = _random_params(rng)
        sch = _random_schedule(rng)
        dur = float(rng.choice([rng.uniform(30, 600), 100000]))   # sometimes absurd duration
        try:
            tr = Simulator(p).run(sch, dur, outputs=None if rng.random() < 0.7 else ["glucose_mg_dl"])
        except Exception as e:
            crashes += 1
            if len(examples) < 3:
                examples.append(f"crash {type(e).__name__}: {e}")
            continue
        for var, xs in tr.series.items():
            for v in xs:
                if not math.isfinite(v):
                    nonfinite += 1
                    break
            if var in bounds:
                lo, hi = bounds[var]
                if any(v < lo - 1e-6 or v > hi + 1e-6 for v in xs):
                    oob += 1
                    if len(examples) < 3:
                        examples.append(f"{var} out of [{lo},{hi}]: {min(xs):.1f}..{max(xs):.1f}")

    print(f"1. {N} randomized runs (incl. abusive inputs) — crashes: {crashes}, "
          f"non-finite series: {nonfinite}, out-of-bounds series: {oob}")

    # 4. numerical stability: dt=1.0 vs dt=0.5 agree on peaks for normal scenarios
    max_rel = 0.0
    for _ in range(40):
        p = PhysioParams(weight_kg=float(rng.normal(78, 12)), age=float(rng.uniform(25, 65)))
        sch = Schedule().add(Meal(0, float(rng.uniform(30, 90))))
        if rng.random() < 0.5:
            sch.add(Exercise(60, 90, float(rng.uniform(4, 10))))
        a = Simulator(p).run(sch, 210, dt=1.0)
        b = Simulator(p).run(sch, 210, dt=0.5)
        for var in ("glucose_mg_dl", "heart_rate_bpm"):
            pa, pb = max(a.series[var]), max(b.series[var])
            if pa > 1:
                max_rel = max(max_rel, abs(pa - pb) / pa)
    stable = max_rel < 0.10
    print(f"2. Numerical stability (dt 1.0 vs 0.5): max peak deviation {max_rel*100:.1f}% "
          f"(<10%): {'OK' if stable else 'FAIL'}")

    # 5. determinism
    p = PhysioParams(weight_kg=80)
    s1 = Simulator(p).run(Schedule().add(Meal(0, 60)).add(Caffeine(0, 95)), 180).series
    s2 = Simulator(p).run(Schedule().add(Meal(0, 60)).add(Caffeine(0, 95)), 180).series
    deterministic = s1 == s2
    print(f"3. Determinism (identical inputs -> identical output): {'OK' if deterministic else 'FAIL'}")

    if examples:
        print("   examples:", "; ".join(examples))

    ok = crashes == 0 and nonfinite == 0 and oob == 0 and stable and deterministic
    print("=" * 72)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
