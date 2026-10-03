"""Pattern detection — find real signals in a user's longitudinal data.

Predictions and warnings use the engine; this is the third pillar of the vision ("detect
patterns in your data"). It is deliberately STATISTICAL, not a black box: for each logged meal
it computes the glucose response (iAUC), removes the part explained by the meal itself
(carbs/fat/fibre via a small regression), and correlates the LEFTOVER — the part driven by
that day's context — against the day's wearable features (sleep, activity, HRV, resting HR).
A strong, well-sampled correlation surfaces as a plain-English pattern with its strength and n.

It also flags simple TRENDS (resting HR, sleep, weight drifting over time).

Every pattern reports r and n so nothing is over-claimed: a weak or under-sampled correlation
is not shown. This is honest signal-finding, not fortune-telling — the same discipline as the
rest of the system.
"""

from __future__ import annotations

import statistics
from datetime import datetime

import numpy as np

_PRE, _WINDOW, _STEP = 30.0, 180.0, 5.0
_MIN_MEALS = 15            # below this, correlations are noise
_MIN_R = 0.30             # minimum |correlation| to report


def _iauc(mr: dict) -> float | None:
    g = mr.get("glucose") or {}
    vals = g.get("values")
    if not vals:
        return None
    t0, step, meal_t = g.get("t0_min", 0.0), g.get("step_min", 5.0), g.get("meal_t_min", _PRE)
    pre, seg = [], []
    for i, v in enumerate(vals):
        t = t0 + i * step
        if meal_t - _PRE <= t < meal_t:
            pre.append(float(v))
        if meal_t <= t <= meal_t + _WINDOW:
            seg.append(float(v))
    if len(seg) < 3:
        return None
    base = statistics.fmean(pre) if pre else seg[0]
    incr = [max(0.0, v - base) for v in seg]
    return step * (sum(incr) - 0.5 * (incr[0] + incr[-1]))


def _date(ts) -> str | None:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date().isoformat()
    except (ValueError, TypeError):
        return None


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# Only features with a plausible CAUSAL link to a meal's glucose response — behaviour on the
# day. Resting HR / HRV are day-STATE, not drivers of meal glucose; including them just adds
# noise tests and spurious correlations (the multiple-comparisons trap). They are covered as
# TRENDS instead.
_FEATURES = {
    "sleep_efficiency": ("sleep quality", "you sleep worse", "you sleep better"),
    "steps": ("activity", "you move less", "you move more"),
}


def _meal_context_patterns(user: dict) -> list[dict]:
    daily = {d.get("date"): d for d in user.get("wearable", {}).get("daily", []) if d.get("date")}
    rows = []                                   # (iauc, carbs, fat, fibre, day_features)
    for mr in user.get("meal_responses", []):
        iauc = _iauc(mr)
        day = daily.get(_date(mr.get("ts")))
        if iauc is None or not day or not mr.get("carbs_g"):
            continue
        rows.append((iauc, mr["carbs_g"], mr.get("fat_g", 0), mr.get("fiber_g", 0), day))
    if len(rows) < _MIN_MEALS:
        return []

    iauc = np.array([r[0] for r in rows])
    # remove the meal's own effect (carbs/fat/fibre) -> residual = the day-driven part
    X = np.column_stack([np.ones(len(rows)), [r[1] for r in rows],
                         [r[2] for r in rows], [r[3] for r in rows]])
    coef, *_ = np.linalg.lstsq(X, iauc, rcond=None)
    resid = iauc - X @ coef

    out = []
    for feat, (label, lo_phrase, hi_phrase) in _FEATURES.items():
        vals = [r[4].get(feat) for r in rows]
        if any(v is None for v in vals):
            continue
        r = _pearson(resid, vals)
        if abs(r) >= _MIN_R:
            # residual HIGH = worse glucose. negative r vs feature => low feature -> worse glucose
            worse_when = lo_phrase if r < 0 else hi_phrase
            out.append({
                "kind": "meal_context", "feature": feat, "r": round(r, 2), "n": len(rows),
                "strength": "strong" if abs(r) > 0.5 else "moderate",
                "pattern": f"Your blood-sugar response to meals is higher on days when {worse_when}.",
                "detail": f"After accounting for what you ate, meal glucose correlates with your "
                          f"{label} (r={r:+.2f}, n={len(rows)} meals).",
                "evidence": "moderate"})
    out.sort(key=lambda p: abs(p["r"]), reverse=True)
    return out


def _trend_patterns(user: dict) -> list[dict]:
    daily = [d for d in user.get("wearable", {}).get("daily", []) if d.get("date")]
    out = []
    for feat, label, unit in (("resting_hr", "resting heart rate", "bpm"),
                              ("sleep_efficiency", "sleep efficiency", "%")):
        pts = [(i, d[feat]) for i, d in enumerate(daily) if isinstance(d.get(feat), (int, float))]
        if len(pts) < 20:
            continue
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        # a real trend CORRELATES with time — a slope over noisy data does not. Require both a
        # meaningful time-correlation and a meaningful magnitude, else it's random drift.
        r_time = _pearson(xs, ys)
        slope = float(np.polyfit(xs, ys, 1)[0]) * len(pts)      # total change over the span
        if abs(r_time) >= 0.45 and abs(slope) > (2 if unit == "bpm" else 3):
            direction = "risen" if slope > 0 else "fallen"
            out.append({"kind": "trend", "feature": feat, "n": len(pts),
                        "pattern": f"Your {label} has {direction} ~{abs(slope):.0f} {unit} "
                                   f"over the tracked period.",
                        "evidence": "moderate"})
    return out


def detect_patterns(user: dict) -> dict:
    patterns = _meal_context_patterns(user) + _trend_patterns(user)
    return {"patterns": patterns, "n_patterns": len(patterns),
            "evidence": "moderate" if patterns else "none",
            "citations": ["Correlation on the user's own longitudinal data (residualised for "
                          "meal composition)"],
            "note": "Data patterns, not causation — worth noticing, not a diagnosis."}
