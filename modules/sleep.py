"""Sleep module — real age-baseline + cited behavioral modifiers.

Design (updated to use REAL data): Sleep-EDF only varies by AGE (it has no behavioural
inputs), so the honest split is:

  - BASELINE architecture (REM%, deep%, efficiency, awakenings vs age) is an ML model
    trained on REAL PhysioNet Sleep-EDF (153 nights) with quantile regression for the
    prediction interval. This replaces the old synthetic generator.
  - BEHAVIOURAL / cross-system effects (alcohol, caffeine, exercise, screen, bedtime
    regularity) are applied as additive shifts using CITED effect sizes from Layer 1
    (knowledge_base/sleep.json). Their direction is well established; magnitudes are
    approximate and graded — never invented by a synthetic generator.

So every number is now grounded in either real data (baseline) or cited research
(modifiers). `alcohol_gkg_bedtime` remains the hepatic->sleep cross-system input.

If the real dataset is unavailable, fit() falls back to the synthetic generator so the
system still runs (clearly labelled provisional in that case).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from knowledge_base import load_raw, load_system
from modules.hepatic import EvidenceLevel
from modules.sleep_data import FEATURES, TARGETS, generate

_LOWER_Q, _UPPER_Q = 0.05, 0.95  # 90% prediction interval
_REF_REGULARITY = 0.75


@dataclass
class SleepPrediction:
    metrics: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    drivers: list[tuple[str, float]]
    evidence: EvidenceLevel
    confidence_label: str
    citations: list[str] = field(default_factory=list)


def _load_real_or_synthetic(seed: int):
    """Prefer real Sleep-EDF; fall back to synthetic if the data isn't downloaded."""
    from modules.sleep_data_real import is_available, load_sleep_edf
    if is_available():
        X, y = load_sleep_edf()
        return X, y, True
    X, y = generate(4000, seed)
    return X, y, False


class SleepModel:
    """Age-baseline quantile GBMs (real data) + cited behavioural modifiers."""

    def __init__(self, n_train: int | None = None, seed: int = 0, data_fn=None):
        # n_train kept for signature compatibility; ignored for the real loader.
        self._seed = seed
        self._data_fn = data_fn
        self._models: dict[str, dict[str, HistGradientBoostingRegressor]] = {}
        self._real = False
        self._trained = False

    def fit(self) -> "SleepModel":
        if self._data_fn is not None:
            X, y = self._data_fn(4000, self._seed)
            self._real = getattr(self._data_fn, "_is_real", False)
        else:
            X, y, self._real = _load_real_or_synthetic(self._seed)

        ages = X[:, FEATURES.index("age")].reshape(-1, 1)  # baseline depends on age only
        for t in TARGETS:
            self._models[t] = {}
            for tag, q in (("lower", _LOWER_Q), ("median", 0.5), ("upper", _UPPER_Q)):
                m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=q, max_depth=3,
                    learning_rate=0.05, max_iter=200, min_samples_leaf=15,
                    random_state=self._seed)
                m.fit(ages, y[t])
                self._models[t][tag] = m
        self._trained = True
        return self

    def _ensure(self):
        if not self._trained:
            self.fit()

    # --- cited behavioural modifiers (additive shifts) ---------------------
    def _modifier_shifts(self, caffeine, exercise, screen, regularity, alcohol_gkg):
        b = load_raw("sleep")["behavioral_modifiers"]
        alc = load_raw("sleep")["cross_system_modifiers"]["alcohol_bedtime"]

        def clamp(v, p):
            lo, hi = b[p]["plausible_range"]
            return max(lo, min(hi, v))

        eff = (clamp(b["caffeine_efficiency_per_mg"]["value"], "caffeine_efficiency_per_mg") * caffeine
               + clamp(b["screen_efficiency_per_min"]["value"], "screen_efficiency_per_min") * screen
               + b["regularity_efficiency_per_unit"]["value"] * (regularity - _REF_REGULARITY))
        deep = b["exercise_deep_per_min"]["value"] * min(exercise, 90.0)
        rem = alc["rem_pct_per_gkg"]["value"] * alcohol_gkg
        awk = alc["awakenings_per_gkg"]["value"] * alcohol_gkg
        return {"sleep_efficiency": eff, "deep_pct": deep, "rem_pct": rem, "awakenings": awk}

    def predict(self, *, age: float, caffeine_mg_afternoon: float = 0.0,
                exercise_min: float = 0.0, screen_min_before_bed: float = 0.0,
                bedtime_regularity: float = 0.8,
                alcohol_gkg_bedtime: float = 0.0) -> SleepPrediction:
        self._ensure()
        x = np.array([[age]])
        kb = load_system("sleep")
        ranges = {"rem_pct": kb["rem_pct_mean"].plausible_range,
                  "deep_pct": kb["deep_pct_mean"].plausible_range,
                  "sleep_efficiency": kb["sleep_efficiency_mean"].plausible_range,
                  "awakenings": (0.0, 10.0)}
        shifts = self._modifier_shifts(caffeine_mg_afternoon, exercise_min,
                                       screen_min_before_bed, bedtime_regularity,
                                       alcohol_gkg_bedtime)

        metrics, intervals = {}, {}
        for t in TARGETS:
            lo = float(self._models[t]["lower"].predict(x)[0]) + shifts[t]
            md = float(self._models[t]["median"].predict(x)[0]) + shifts[t]
            hi = float(self._models[t]["upper"].predict(x)[0]) + shifts[t]
            lo, hi = min(lo, hi), max(lo, hi)
            clo, chi = ranges[t]
            metrics[t] = float(np.clip(md, clo, chi))
            intervals[t] = (float(np.clip(lo, clo, chi)), float(np.clip(hi, clo, chi)))

        any_behaviour = any([caffeine_mg_afternoon, exercise_min, screen_min_before_bed,
                             alcohol_gkg_bedtime, bedtime_regularity != _REF_REGULARITY])
        if self._real:
            evidence = EvidenceLevel.STRONG
            label = ("age baseline trained on PhysioNet Sleep-EDF (153 nights)"
                     + ("; behavioural effects from cited research (approximate)"
                        if any_behaviour else ""))
        else:
            evidence = EvidenceLevel.WEAK
            label = "SYNTHETIC baseline (Sleep-EDF not found) — provisional"

        return SleepPrediction(
            metrics=metrics, intervals=intervals,
            drivers=self._drivers(age, shifts),
            evidence=evidence, confidence_label=label,
            citations=["PhysioNet Sleep-EDF (Kemp et al. 2000)",
                       "Ohayon et al. 2004, Sleep", "Ebrahim et al. 2013, ACER"],
        )

    def _drivers(self, age: float, shifts: dict) -> list[tuple[str, float]]:
        """Rank what moved this prediction: the age baseline + each applied modifier."""
        out = [("age (baseline sleep architecture)",
                abs(float(self._models["deep_pct"]["median"].predict([[age]])[0])
                    - float(self._models["deep_pct"]["median"].predict([[30.0]])[0])))]
        for name, key in (("alcohol at bedtime", "rem_pct"),
                          ("caffeine / screen / regularity", "sleep_efficiency"),
                          ("exercise", "deep_pct")):
            out.append((name, abs(shifts.get(key, 0.0))))
        out.sort(key=lambda z: z[1], reverse=True)
        return out
