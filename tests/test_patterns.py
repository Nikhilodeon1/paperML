"""Pattern detection — must find a REAL embedded pattern and NOT invent fake ones."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from personalization.patterns import _iauc, detect_patterns
from orchestration.planner import deterministic_plan


def _day(i, sleep_eff, steps=8000):
    d = (datetime(2026, 6, 1) + timedelta(days=i)).date().isoformat()
    return {"date": d, "sleep_efficiency": sleep_eff, "steps": steps,
            "resting_hr": 60, "hrv_rmssd": 45}


def _meal(day_i, iauc_level, carbs=60):
    """A meal on day `day_i` whose window integrates to roughly `iauc_level`."""
    d = datetime(2026, 6, 1) + timedelta(days=day_i, hours=12)
    base = 90.0
    vals = [base] * 6 + [base + iauc_level / 90.0] * 36 + [base]   # rough rectangle
    return {"carbs_g": carbs, "fat_g": 10, "fiber_g": 4, "ts": d.isoformat(),
            "glucose": {"values": vals, "t0_min": 0.0, "step_min": 5.0, "meal_t_min": 30.0}}


def _user_with_sleep_pattern(n=25, rng=None):
    """Higher glucose on low-sleep days; sleep alternates high/low."""
    rng = rng or np.random.default_rng(0)
    daily, meals = [], []
    for i in range(n):
        low = i % 2 == 0
        sleep = 70 if low else 92
        daily.append(_day(i, sleep + rng.normal(0, 1)))
        iauc = (5500 if low else 3000) + rng.normal(0, 300)      # clear sleep effect
        meals.append(_meal(i, iauc))
    return {"wearable": {"daily": daily}, "meal_responses": meals}


def _user_no_pattern(n=25, rng=None):
    rng = rng or np.random.default_rng(1)
    daily, meals = [], []
    for i in range(n):
        daily.append(_day(i, rng.uniform(75, 95)))
        meals.append(_meal(i, 4000 + rng.normal(0, 300)))        # iauc independent of sleep
    return {"wearable": {"daily": daily}, "meal_responses": meals}


def test_finds_a_real_sleep_glucose_pattern():
    r = detect_patterns(_user_with_sleep_pattern())
    sleep = [p for p in r["patterns"] if p.get("feature") == "sleep_efficiency"]
    assert sleep, "missed the embedded sleep->glucose pattern"
    assert sleep[0]["r"] < -0.3 and "sleep worse" in sleep[0]["pattern"]


def test_no_false_pattern_when_none_exists():
    assert detect_patterns(_user_no_pattern())["patterns"] == []


def test_too_little_data_finds_nothing():
    assert detect_patterns(_user_with_sleep_pattern(n=6))["patterns"] == []


def test_resting_hr_is_not_a_meal_context_feature():
    """RHR/HRV are excluded from meal-context correlation (they were the false-positive source);
    only behaviour (sleep, steps) is tested."""
    feats = {p.get("feature") for p in detect_patterns(_user_with_sleep_pattern())["patterns"]}
    assert "resting_hr" not in feats and "hrv_rmssd" not in feats


def test_planner_routes_pattern_questions():
    for q in ("what patterns do you see in my data", "have you noticed any trends"):
        assert "patterns" in {s.capability for s in deterministic_plan(q, {"profile": {}})}
