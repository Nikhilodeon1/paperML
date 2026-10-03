"""Validate the sleep knowledge base against REAL Sleep-EDF data.

Sleep-EDF has no behavioural inputs, so it cannot validate the behavioural edges. What
it CAN do (and this is genuine Layer 1 cohort validation per the brief): check that our
architecture norms and the age->deep-sleep relationship match a real cohort.

Reports, for 153 nights:
  1. Population means for REM%, deep%, efficiency, sustained awakenings vs the KB norms.
  2. The fitted age -> deep-sleep slope vs the KB coefficient `deep_pct_age_slope`
     (Ohayon 2004: SWS declines ~2%/decade -> ~ -0.18 pp/yr). This is the key
     scientific check and, if it disagrees, the KB value we should update.

Run:  python -m evaluation.calibrate_sleep_edf
"""

from __future__ import annotations

import numpy as np

from knowledge_base import load_system
from modules.sleep_data_real import _iter_nights, is_available

# Sleep-EDF skews older & includes fragmented nights; allow generous norm bands.
NORM_TOL = {"rem_pct": 6.0, "deep_pct": 10.0, "sleep_efficiency": 15.0}
SLOPE_SIGN_MUST_MATCH = True


def run() -> int:
    if not is_available():
        print("Sleep-EDF not found — see data/README.md.")
        return 1

    ages, rem, deep, eff, awk = [], [], [], [], []
    for age, m in _iter_nights():
        ages.append(age); rem.append(m["rem_pct"]); deep.append(m["deep_pct"])
        eff.append(m["sleep_efficiency"]); awk.append(m["awakenings"])
    ages = np.array(ages)

    kb = load_system("sleep")
    print("=" * 68)
    print(f"SLEEP-EDF CALIBRATION  ({len(ages)} nights, age {ages.min():.0f}-{ages.max():.0f})")
    print("=" * 68)

    ok = True
    print(f"{'metric':>18} {'real mean':>10} {'KB norm':>9} {'within?':>8}")
    print("-" * 68)
    norm_key = {"rem_pct": "rem_pct_mean", "deep_pct": "deep_pct_mean",
                "sleep_efficiency": "sleep_efficiency_mean"}
    for name, data in (("rem_pct", rem), ("deep_pct", deep), ("sleep_efficiency", eff)):
        real = float(np.mean(data))
        norm = kb[norm_key[name]].value
        within = abs(real - norm) <= NORM_TOL[name]
        ok &= within
        print(f"{name:>18} {real:>10.1f} {norm:>9.1f} {'OK' if within else 'CHECK':>8}")
    print(f"{'awakenings(>=5m)':>18} {np.mean(awk):>10.1f} {kb['awakenings_mean'].value:>9.1f} "
          f"{'(context)':>8}")

    # Age -> deep-sleep slope (the key scientific relationship).
    slope, intercept = np.polyfit(ages, deep, 1)
    kb_slope = kb["deep_pct_age_slope"].value
    r = np.corrcoef(ages, deep)[0, 1]
    print("-" * 68)
    print(f"age -> deep-sleep slope: real {slope:+.3f} pp/yr (r={r:+.2f}), "
          f"KB {kb_slope:+.3f} pp/yr")
    sign_ok = (slope < 0) == (kb_slope < 0)
    ok &= (sign_ok or not SLOPE_SIGN_MUST_MATCH)
    print(f"  direction matches KB (deep sleep declines with age): "
          f"{'OK' if sign_ok else 'FAIL'}")
    if sign_ok:
        print(f"  -> Layer 1 refinement: consider updating deep_pct_age_slope "
              f"{kb_slope:+.3f} -> {slope:+.3f} (cohort-fitted).")

    print("=" * 68)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
