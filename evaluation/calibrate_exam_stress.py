"""Validate the stress module physiology against REAL wearable data.

PhysioNet Wearable Exam Stress (Empatica E4) has no stress label / rest baseline, so
(like Sleep-EDF) it validates physiology, not a supervised target. Two honest checks:

  1. Stress elevates heart rate: mean exam-condition HR should exceed the resting HR
     norm in Layer 1.
  2. Autonomic arousal signature: across windows, higher HR should coincide with lower
     HRV (RMSSD) — a NEGATIVE correlation. This is the core physiological relationship
     the synthetic stress generator encodes, now checked on real data.

(We do NOT require RMSSD below the resting norm: E4 wrist-PPG HRV runs high and the
cohort is young; the robust, device-independent check is the within-data HR-HRV
relationship.)

Run:  python -m evaluation.calibrate_exam_stress
"""

from __future__ import annotations

import numpy as np

from knowledge_base import load_system
from modules.stress_data_real import is_available, load_exam_features


def run() -> int:
    if not is_available():
        print("Wearable Exam Stress not found — see data/README.md.")
        return 1

    X = load_exam_features()
    hr, rmssd, eda = X[:, 0], X[:, 1], X[:, 2]
    m = ~np.isnan(rmssd)
    kb = load_system("stress")
    rest_hr = kb["rest_hr_mean"].value

    print("=" * 62)
    print(f"WEARABLE EXAM STRESS CALIBRATION  ({X.shape[0]} 5-min windows)")
    print("=" * 62)
    print(f"  mean HR    {np.nanmean(hr):6.1f} bpm   (resting norm {rest_hr:.0f})")
    print(f"  mean RMSSD {np.nanmean(rmssd):6.1f} ms")
    print(f"  mean EDA   {np.nanmean(eda):6.2f} uS")

    hr_elevated = float(np.nanmean(hr)) > rest_hr
    corr = float(np.corrcoef(hr[m], rmssd[m])[0, 1])
    arousal_ok = corr < 0

    print("-" * 62)
    print(f"  1. exam HR > resting norm (stress elevates HR): "
          f"{'OK' if hr_elevated else 'FAIL'}")
    print(f"  2. HR-RMSSD correlation {corr:+.3f} < 0 (arousal signature): "
          f"{'OK' if arousal_ok else 'FAIL'}")
    ok = hr_elevated and arousal_ok
    print("=" * 62)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
