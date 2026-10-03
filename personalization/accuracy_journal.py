"""The Baseline Journal — how well does a running mean predict this user's daily metrics?

READ THIS BEFORE QUOTING ANY NUMBER FROM HERE.

This module does NOT test the simulation engine. Its predictor is `statistics.mean(history)`
— a personal running mean. It never imports the simulator. What it measures is whether a
person's own average beats a population constant, which is true by construction for anyone
whose average differs from the population's. An earlier framing of this as "the personalized
model beats the population average by 50-68%" was therefore measuring the baseline learner,
not the twin, and has been retired.

What it IS good for: these daily aggregates (resting HR, HRV, sleep efficiency, steps) are
mostly behaviour and slow-moving baseline, which the engine does not model and should not be
judged on. Resting HR is a PARAMETER the twin learns, not a prediction it makes. So this is
a legitimate check on the baseline learner — and nothing more.

It reports PERSISTENCE ("tomorrow = today") alongside, because that is the honest bar for
anything slow-moving; a running mean often loses to it. Beating a population constant is
not an achievement.

For actual engine validation — predicting a response to logged events on held-out days —
see `evaluation/forward_validation.py`.
"""

from __future__ import annotations

import statistics

# metric -> (label, population-average baseline used by the 'naive' predictor)
_METRICS = {
    "resting_hr": ("Resting heart rate", 66.0),
    "hrv_rmssd": ("HRV (RMSSD)", 45.0),
    "sleep_efficiency": ("Sleep efficiency", 85.0),
    "steps": ("Daily steps", 7500.0),
}
_Z90 = 1.645


def build_journal(user: dict, metric: str, min_history: int = 7) -> list[dict]:
    """Walk the wearable history: predict each day from the prior days (personalized =
    running mean + 90% band from day-to-day spread), record actual vs predicted."""
    daily = user.get("wearable", {}).get("daily", [])
    _, pop = _METRICS.get(metric, (metric, 0.0))
    entries: list[dict] = []
    hist: list[float] = []
    for d in daily:
        actual = d.get(metric)
        if not isinstance(actual, (int, float)):
            continue
        if len(hist) >= min_history:
            pred = statistics.mean(hist)
            sd = statistics.pstdev(hist) or 1.0
            lo, hi = pred - _Z90 * sd, pred + _Z90 * sd
            entries.append({
                "date": d.get("date"), "predicted": round(pred, 1),
                "actual": round(float(actual), 1), "error": round(actual - pred, 1),
                "abs_error": round(abs(actual - pred), 1),
                "in_90_band": lo <= actual <= hi,
                # the two baselines: a textbook constant (trivial to beat) and yesterday's
                # value (genuinely hard to beat for anything slow-moving)
                "population_abs_error": round(abs(actual - pop), 1),
                "persistence_abs_error": round(abs(actual - hist[-1]), 1),
            })
        hist.append(float(actual))
    return entries


def accuracy_summary(user: dict) -> dict:
    """Per-metric rolling accuracy of the RUNNING-MEAN baseline (not the engine): MAE, bias,
    90% coverage, and how it compares to a population constant AND to persistence.

    `predictor` is returned explicitly so no surface can quietly present this as the twin's
    accuracy. `vs_persistence_pct` is the number that actually means something — and it is
    frequently negative, which is honest.
    """
    out: dict = {}
    for metric, (label, _) in _METRICS.items():
        j = build_journal(user, metric)
        if len(j) < 3:
            continue
        mae = statistics.mean(e["abs_error"] for e in j)
        pop_mae = statistics.mean(e["population_abs_error"] for e in j)
        pers_mae = statistics.mean(e["persistence_abs_error"] for e in j)
        bias = statistics.mean(e["error"] for e in j)
        coverage = sum(e["in_90_band"] for e in j) / len(j)
        out[metric] = {
            "label": label, "n": len(j),
            "predictor": "personal running mean (NOT the simulation engine)",
            "mae": round(mae, 1), "bias": round(bias, 1),
            "coverage_90": round(coverage, 2),
            "population_mae": round(pop_mae, 1),
            "persistence_mae": round(pers_mae, 1),
            "vs_population_pct": round(100 * (1 - mae / pop_mae)) if pop_mae > 0 else 0,
            "vs_persistence_pct": round(100 * (1 - mae / pers_mae)) if pers_mae > 0 else 0,
        }
    return out
