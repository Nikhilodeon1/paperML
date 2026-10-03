"""Stress module — supervised stress classifier trained on WESAD (labelled real data).

Upgrade from the earlier HR-only composite: WESAD provides ground-truth stress labels
alongside wrist-E4 signals, so we now train a real classifier on [HR, RMSSD, EDA]
(derived from the wrist wearable, matching deployment) and define

    stress_index = 100 * P(stress | your wearable signals)

This is supervised on real labels and validated leave-one-subject-out (see
evaluation/backtest_stress.py). If WESAD isn't present, the module falls back to the
HR-elevation composite (calibrated on real exam-stress HR) so the system still runs,
clearly labelled as the weaker operational mode.

Feature reliability note: HR and EDA separate stress cleanly in WESAD; wrist-PPG RMSSD
is noisy, so the classifier naturally leans on HR/EDA. Numbers stay grounded — the LLM
never produces a stress number, it comes from this model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from knowledge_base import load_system
from modules.hepatic import EvidenceLevel

_HR_NOISE_BPM = 5.0
_INDEX_AT_EXAM = 65.0
_FEATURES = ["heart_rate", "rmssd", "eda"]


@dataclass
class StressPrediction:
    stress_index: float
    interval: tuple[float, float]
    category: str
    context: dict
    drivers: list[tuple[str, float]]
    evidence: EvidenceLevel
    confidence_label: str
    citations: list[str] = field(default_factory=list)


def _category(idx: float) -> str:
    return "low" if idx < 33 else "moderate" if idx < 66 else "high"


class StressModel:
    """Supervised WESAD classifier (P(stress) -> 0-100 index); HR-composite fallback."""

    def __init__(self, *args, **kwargs):
        self._clf = None
        self._supervised = False
        self._trained = False

    def fit(self) -> "StressModel":
        from modules.stress_data_wesad import is_available, load_wesad_windows
        if is_available():
            from sklearn.ensemble import HistGradientBoostingClassifier
            X, y, _ = load_wesad_windows()
            self._clf = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.06, max_iter=250,
                min_samples_leaf=15, class_weight="balanced", random_state=0)
            self._clf.fit(X, y)
            self._supervised = True
        self._trained = True
        return self

    def _ensure(self):
        if not self._trained:
            self.fit()

    # --- fallback composite (used only if WESAD is absent) -----------------
    def _composite_index(self, hr: float) -> float:
        kb = load_system("stress")
        rest, exam = kb["rest_hr_mean"].value, kb["exam_hr_reference"].value
        return float(np.clip(_INDEX_AT_EXAM / (exam - rest) * (hr - rest), 0.0, 100.0))

    def predict(self, *, heart_rate: float, rmssd: float, eda: float,
                resp_rate: float = 15.0) -> StressPrediction:
        self._ensure()
        context = {"heart_rate": heart_rate, "rmssd": rmssd, "eda": eda}

        if self._supervised:
            x = np.array([[heart_rate, rmssd, eda]])
            idx = float(100.0 * self._clf.predict_proba(x)[0, 1])
            drivers = self._supervised_drivers(x, idx)
            # Probability-based band: tighter when the model is confident.
            p = idx / 100.0
            half = 8.0 + 22.0 * (1.0 - abs(2 * p - 1))  # widest near p=0.5
            interval = (round(max(0.0, idx - half), 1), round(min(100.0, idx + half), 1))
            evidence = EvidenceLevel.STRONG
            label = ("supervised stress classifier trained on WESAD (labelled wrist-E4 "
                     "data); index = 100 x P(stress); validated leave-one-subject-out")
            citations = ["Schmidt et al. 2018, WESAD (ICMI)"]
        else:
            idx = self._composite_index(heart_rate)
            lo = self._composite_index(heart_rate - _HR_NOISE_BPM)
            hi = self._composite_index(heart_rate + _HR_NOISE_BPM)
            interval = (round(min(lo, hi), 1), round(max(lo, hi), 1))
            drivers = [("heart-rate elevation above resting", abs(idx))]
            evidence = EvidenceLevel.WEAK
            label = ("WESAD not found — operational HR-elevation composite "
                     "(resting ~68 -> exam-stress ~95 bpm); not supervised")
            kb = load_system("stress")
            citations = [kb["rest_hr_mean"].citation, kb["exam_hr_reference"].citation]

        return StressPrediction(
            stress_index=round(idx, 1), interval=interval, category=_category(idx),
            context=context, drivers=drivers, evidence=evidence,
            confidence_label=label, citations=citations)

    def _supervised_drivers(self, x: np.ndarray, base_idx: float):
        """How much each wearable signal moved the predicted stress probability."""
        steps = {"heart_rate": 10.0, "rmssd": 15.0, "eda": 1.0}
        out = []
        for i, feat in enumerate(_FEATURES):
            xp = x.copy(); xp[0, i] += steps[feat]
            bumped = float(100.0 * self._clf.predict_proba(xp)[0, 1])
            out.append((feat, abs(bumped - base_idx)))
        out.sort(key=lambda z: z[1], reverse=True)
        return out
