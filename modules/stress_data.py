"""Synthetic stress-sample generator (bootstrap before the real dataset).

Grounds a synthetic dataset in the norms in knowledge_base/stress.json. As with
sleep, training AND testing on this only proves the model learned the generator — NOT
physiological validation (brief 4). Re-run on the PhysioNet Wearable Exam Stress
Dataset (modules/stress_data_real.py) to validate.

Features = wearable-derived signals; target = a 0-100 stress index. The synthetic
ground truth makes the stress index rise with HR and EDA and fall with HRV (RMSSD),
matching the established physiology of sympathetic arousal.
"""

from __future__ import annotations

import numpy as np

from knowledge_base import load_system

FEATURES = ["heart_rate", "rmssd", "eda", "resp_rate"]
TARGET = "stress_index"


def generate(n: int = 4000, seed: int = 0):
    """Return (X, y) where X is (n, len(FEATURES)) and y is (n,) stress index 0-100."""
    rng = np.random.default_rng(seed)
    kb = load_system("stress")

    hr = np.clip(rng.normal(kb["rest_hr_mean"].value, kb["rest_hr_mean"].population_sd, n)
                 + rng.exponential(8, n), 40, 160)          # arousal pushes HR up
    rmssd = np.clip(rng.normal(kb["rmssd_mean"].value, kb["rmssd_mean"].population_sd, n),
                    *kb["rmssd_mean"].plausible_range)
    eda = np.clip(rng.normal(kb["eda_mean"].value, kb["eda_mean"].population_sd, n)
                  + rng.exponential(1.0, n), *kb["eda_mean"].plausible_range)
    resp = np.clip(rng.normal(15, 3, n), 8, 30)

    # Operational ground-truth stress index (z-scored drivers -> 0..100).
    z_hr = (hr - kb["rest_hr_mean"].value) / kb["rest_hr_mean"].population_sd
    z_rmssd = (rmssd - kb["rmssd_mean"].value) / kb["rmssd_mean"].population_sd
    z_eda = (eda - kb["eda_mean"].value) / kb["eda_mean"].population_sd
    raw = 50 + 12 * z_hr - 12 * z_rmssd + 10 * z_eda + 4 * (resp - 15) / 3
    stress = np.clip(raw + rng.normal(0, 6, n), 0, 100)

    X = np.column_stack([hr, rmssd, eda, resp])
    return X, {TARGET: stress}
