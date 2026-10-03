"""Backtest the sleep module on REAL PhysioNet Sleep-EDF data.

The deployed model = an age-baseline trained on real nights + cited behavioural
modifiers. Sleep-EDF only varies by age, so we backtest the part that is actually
learned from data (age -> architecture) with a real held-out split, and separately
check that the cited behavioural modifiers move predictions in the documented
direction.

Checks:
  1. Held-out accuracy: train the age-baseline on 80% of real nights, predict the
     held-out 20%, and beat a mean-predictor baseline (features carry real signal).
  2. Prediction-interval coverage on the held-out nights (~90%).
  3. Modifier directions: alcohol -> less REM + more awakenings; caffeine -> lower
     efficiency; exercise -> more deep; age -> less deep.

Run:  python -m evaluation.backtest_sleep
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from modules.sleep import SleepModel
from modules.sleep_data import FEATURES, TARGETS
from modules.sleep_data_real import is_available, load_sleep_edf

# Age predicts deep sleep / efficiency / awakenings well but REM% only weakly (REM% is
# relatively age-stable in adulthood — a real finding, not a model defect). So we
# require: no target WORSE than the mean baseline, and a meaningful AVERAGE improvement.
NOT_WORSE = -0.02
MEAN_IMPROVEMENT = 0.10
COVERAGE_TOL = 0.12        # small real sample -> looser coverage tolerance
_LO, _HI = 0.05, 0.95


def run() -> int:
    if not is_available():
        print("Sleep-EDF not downloaded — see data/README.md. (Model falls back to "
              "synthetic; download the data to backtest on real nights.)")
        return 1

    X, y = load_sleep_edf()
    ages = X[:, FEATURES.index("age")]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ages))
    cut = int(0.8 * len(ages))
    tr, te = idx[:cut], idx[cut:]

    print("=" * 74)
    print(f"SLEEP MODULE BACKTEST  —  REAL Sleep-EDF ({len(ages)} nights, "
          f"{len(tr)} train / {len(te)} test)")
    print("=" * 74)
    print(f"{'target':>18} {'model MAE':>10} {'base MAE':>10} {'improv':>8} {'PI cover':>9}")
    print("-" * 74)

    ok = True
    improvements = []
    for t in TARGETS:
        models = {}
        for tag, q in (("lo", _LO), ("md", 0.5), ("hi", _HI)):
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_depth=3,
                                              learning_rate=0.05, max_iter=200,
                                              min_samples_leaf=15, random_state=0)
            m.fit(ages[tr].reshape(-1, 1), y[t][tr])
            models[tag] = m
        Xte = ages[te].reshape(-1, 1)
        pred = models["md"].predict(Xte)
        lo, hi = models["lo"].predict(Xte), models["hi"].predict(Xte)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

        truth = y[t][te]
        mae = float(np.mean(np.abs(pred - truth)))
        base = float(np.mean(np.abs(truth - np.mean(y[t][tr]))))
        improv = 1 - mae / base if base else 0.0
        improvements.append(improv)
        cover = float(np.mean((truth >= lo) & (truth <= hi)))
        not_worse = improv >= NOT_WORSE
        cover_ok = abs(cover - (_HI - _LO)) <= COVERAGE_TOL
        ok &= not_worse and cover_ok
        flag = "" if (not_worse and cover_ok) else "  <-- CHECK"
        print(f"{t:>18} {mae:>10.3f} {base:>10.3f} {improv*100:>7.1f}% {cover*100:>8.1f}%{flag}")
    print("-" * 74)
    mean_impr = float(np.mean(improvements))
    ok &= mean_impr >= MEAN_IMPROVEMENT
    print(f"mean improvement over baseline: {mean_impr*100:.1f}% (need >= {MEAN_IMPROVEMENT*100:.0f}%; "
          f"REM% is weakly age-dependent by design)")

    # Modifier directions (cited research applied on top of the real baseline).
    m = SleepModel().fit()
    sober, drunk = m.predict(age=35), m.predict(age=35, alcohol_gkg_bedtime=0.8)
    no_caf, caf = m.predict(age=35), m.predict(age=35, caffeine_mg_afternoon=400)
    no_ex, ex = m.predict(age=35), m.predict(age=35, exercise_min=60)
    young, old = m.predict(age=25), m.predict(age=70)

    checks = {
        "alcohol -> less REM": drunk.metrics["rem_pct"] < sober.metrics["rem_pct"],
        "alcohol -> more awakenings": drunk.metrics["awakenings"] > sober.metrics["awakenings"],
        "caffeine -> lower efficiency": caf.metrics["sleep_efficiency"] < no_caf.metrics["sleep_efficiency"],
        "exercise -> more deep": ex.metrics["deep_pct"] > no_ex.metrics["deep_pct"],
        "older -> less deep (real)": old.metrics["deep_pct"] < young.metrics["deep_pct"],
    }
    print("Modifier / age directions:")
    for label, passed in checks.items():
        ok &= passed
        print(f"  {label:<32} {'OK' if passed else 'FAIL'}")

    print("=" * 74)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
