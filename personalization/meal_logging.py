"""Real-user meal ingestion — turn logged meals + a CGM stream into `meal_responses`.

This is the missing link that makes personalization real for an actual user instead of only
CGMacros subjects. The Si learner (`personalization.insulin_sensitivity`) consumes
`user["meal_responses"]`; this builds them from what a real user + device produce:

    user["logs"]["meals"]     = [{ts, carbs_g, protein_g, fat_g, fiber_g, source}]
    user["wearable"]["cgm"]   = [{ts, glucose}]          # 5-min interstitial stream
        -> build_meal_responses -> user["meal_responses"] -> Si refit on refresh_derived

A meal_response is built only when the CGM actually covers that meal's window (pre-meal
baseline + a 3h postprandial), so a meal logged without CGM nearby simply doesn't produce a
fittable response — honest, no fabricated curve. Timestamps are ISO-8601; the window is
aligned to the same 5-min grid (meal at +30 min) the learner and CGMacros loader use.

The meal source can be a photo (via the vision pipeline) or manual entry — this layer doesn't
care how the carbs were obtained, only that they're logged with a timestamp.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

_PRE, _WINDOW, _STEP = 30.0, 180.0, 5.0
_MIN_SAMPLES = 20            # CGM points needed in the window to trust the response


def _parse_ts(ts) -> datetime | None:
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def add_cgm(user: dict, readings: list[dict]) -> int:
    """Append CGM readings [{ts, glucose}] to the user's stream (dedup by ts). Returns count."""
    stream = user.setdefault("wearable", {}).setdefault("cgm", [])
    seen = {r.get("ts") for r in stream}
    added = 0
    for r in readings:
        ts, g = r.get("ts"), r.get("glucose")
        if ts is None or ts in seen or _parse_ts(ts) is None:
            continue
        try:
            stream.append({"ts": str(ts), "glucose": float(g)})
            seen.add(ts); added += 1
        except (TypeError, ValueError):
            continue
    return added


def add_meal(user: dict, meal: dict) -> dict:
    """Log one meal [{ts, carbs_g, protein_g?, fat_g?, fiber_g?, source?}]."""
    entry = {"ts": str(meal.get("ts")), "carbs_g": float(meal.get("carbs_g", 0) or 0),
             "protein_g": float(meal.get("protein_g", 0) or 0),
             "fat_g": float(meal.get("fat_g", 0) or 0),
             "fiber_g": float(meal.get("fiber_g", 0) or 0),
             "source": meal.get("source", "manual")}
    user.setdefault("logs", {}).setdefault("meals", []).append(entry)
    return entry


def _window_values(meal_ts: datetime, cgm: list[dict]):
    """5-min grid of glucose from meal-30 to meal+180 (meal at +30), or None if too sparse."""
    samples = []
    for r in cgm:
        t = _parse_ts(r.get("ts"))
        if t is None:
            continue
        dt = (t - meal_ts).total_seconds() / 60.0
        if -_PRE - 5 <= dt <= _WINDOW + 5:
            samples.append((dt + _PRE, float(r["glucose"])))     # shift so grid starts at 0
    if len(samples) < _MIN_SAMPLES:
        return None
    samples.sort()
    grid = np.arange(0.0, _PRE + _WINDOW + 1e-6, _STEP)
    xs = np.array([s[0] for s in samples]); ys = np.array([s[1] for s in samples])
    return [round(float(v), 1) for v in np.interp(grid, xs, ys)]


def build_meal_responses(user: dict) -> int:
    """(Re)build `user["meal_responses"]` from logged meals + the CGM stream. Returns count."""
    meals = user.get("logs", {}).get("meals", [])
    cgm = user.get("wearable", {}).get("cgm", [])
    out = []
    for m in meals:
        mt = _parse_ts(m.get("ts"))
        if mt is None or not m.get("carbs_g"):
            continue
        vals = _window_values(mt, cgm)
        if vals is None:
            continue
        out.append({"carbs_g": m["carbs_g"], "protein_g": m.get("protein_g", 0.0),
                    "fat_g": m.get("fat_g", 0.0), "fiber_g": m.get("fiber_g", 0.0),
                    "glucose": {"values": vals, "t0_min": 0.0, "step_min": _STEP,
                                "meal_t_min": _PRE}, "source": m.get("source", "manual"),
                    "ts": m["ts"]})
    user["meal_responses"] = out
    return len(out)
