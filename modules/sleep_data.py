"""Synthetic sleep-night generator (bootstrap before real wearable data).

Grounds a realistic-but-synthetic dataset in the published norms and effect sizes in
knowledge_base/sleep.json (brief 3: "synthetic data bootstrapped from published
sleep architecture studies if no real data is accessible early on").

IMPORTANT (brief 4): a model trained AND tested on this only proves it learned this
generator — it is NOT physiological validation. The real validation is re-running the
backtest on PhysioNet Sleep-EDF once that data clears licensing. Treat every accuracy
number derived from synthetic data as provisional.

Feature columns (FEATURES) are shared with modules/sleep.py. `alcohol_gkg_bedtime`
is the cross-system input the hepatic module will supply (alcohol metabolized / kg at
bedtime), wiring the alcohol -> sleep edge (brief 2.5).
"""

from __future__ import annotations

import numpy as np

from knowledge_base import load_raw, load_system, parameter_from_dict

FEATURES = [
    "age",
    "caffeine_mg_afternoon",
    "exercise_min",
    "screen_min_before_bed",
    "bedtime_regularity",      # 0..1
    "alcohol_gkg_bedtime",     # g ethanol / kg body weight at bedtime (from hepatic)
]
TARGETS = ["rem_pct", "deep_pct", "sleep_efficiency", "awakenings"]

_REF_AGE = 30.0


def _sample_features(n, rng):
    return {
        "age": rng.uniform(20, 70, n),
        "caffeine_mg_afternoon": np.clip(rng.exponential(60, n), 0, 400),
        "exercise_min": np.clip(rng.exponential(30, n), 0, 180),
        "screen_min_before_bed": np.clip(rng.normal(45, 30, n), 0, 180),
        "bedtime_regularity": np.clip(rng.beta(5, 2, n), 0, 1),
        "alcohol_gkg_bedtime": np.where(rng.random(n) < 0.4,
                                        np.clip(rng.exponential(0.4, n), 0, 1.5), 0.0),
    }


def generate(n: int = 4000, seed: int = 0):
    """Return (X, y) where X is (n, len(FEATURES)) and y is dict[target] -> (n,)."""
    rng = np.random.default_rng(seed)
    kb = load_system("sleep")
    mods = load_raw("sleep")["cross_system_modifiers"]["alcohol_bedtime"]
    rem_alc = parameter_from_dict("rem_alc", mods["rem_pct_per_gkg"]).value
    awk_alc = parameter_from_dict("awk_alc", mods["awakenings_per_gkg"]).value

    f = _sample_features(n, rng)
    age_c = f["age"] - _REF_AGE

    # Ground-truth physiology = KB norms + documented effects + idiosyncratic noise.
    rem = (kb["rem_pct_mean"].value
           + rem_alc * f["alcohol_gkg_bedtime"]
           - 0.03 * age_c
           + rng.normal(0, 2.5, n))

    deep = (kb["deep_pct_mean"].value
            + kb["deep_pct_age_slope"].value * age_c
            + 0.03 * f["exercise_min"]
            + 1.5 * f["alcohol_gkg_bedtime"]              # alcohol boosts early SWS
            - 0.01 * f["caffeine_mg_afternoon"]
            + rng.normal(0, 3.0, n))

    eff = (kb["sleep_efficiency_mean"].value
           + 6.0 * (f["bedtime_regularity"] - 0.75)
           - 0.02 * f["caffeine_mg_afternoon"]
           - 0.015 * f["screen_min_before_bed"]
           - 4.0 * f["alcohol_gkg_bedtime"]
           + rng.normal(0, 3.0, n))

    awk = (kb["awakenings_mean"].value
           + awk_alc * f["alcohol_gkg_bedtime"]
           + 0.01 * age_c
           + rng.normal(0, 0.6, n))

    clamp = lambda v, p: np.clip(v, *kb[p].plausible_range)
    y = {
        "rem_pct": clamp(rem, "rem_pct_mean"),
        "deep_pct": clamp(deep, "deep_pct_mean"),
        "sleep_efficiency": clamp(eff, "sleep_efficiency_mean"),
        "awakenings": np.clip(awk, 0, 10),
    }
    X = np.column_stack([f[c] for c in FEATURES])
    return X, y
