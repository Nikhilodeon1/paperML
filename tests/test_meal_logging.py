"""Real-user meal ingestion: logged meals + CGM stream -> meal_responses -> Si refit."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from personalization.meal_logging import add_cgm, add_meal, build_meal_responses

_BASE = datetime(2026, 7, 17, 8, 0, 0)


def _flat_cgm_around(meal_ts, glucose=100.0, n=50):
    """A CGM reading every 5 min from meal-40 to meal+200."""
    start = meal_ts - timedelta(minutes=40)
    return [{"ts": (start + timedelta(minutes=5 * i)).isoformat(), "glucose": glucose}
            for i in range(n)]


def test_add_cgm_dedups_and_counts():
    u = {}
    r = _flat_cgm_around(_BASE)
    assert add_cgm(u, r) == len(r)
    assert add_cgm(u, r) == 0                    # same timestamps -> nothing added
    assert len(u["wearable"]["cgm"]) == len(r)


def test_add_cgm_skips_malformed():
    u = {}
    assert add_cgm(u, [{"ts": None, "glucose": 5}, {"ts": "bad", "glucose": 5},
                       {"ts": _BASE.isoformat(), "glucose": "x"}]) == 0


def test_meal_with_cgm_coverage_builds_a_response():
    u = {}
    add_cgm(u, _flat_cgm_around(_BASE))
    add_meal(u, {"ts": _BASE.isoformat(), "carbs_g": 60, "fat_g": 20, "source": "photo"})
    assert build_meal_responses(u) == 1
    mr = u["meal_responses"][0]
    assert mr["carbs_g"] == 60 and mr["fat_g"] == 20 and mr["source"] == "photo"
    g = mr["glucose"]
    assert g["meal_t_min"] == 30 and g["step_min"] == 5 and len(g["values"]) == 43


def test_meal_without_cgm_builds_nothing():
    """No fabricated curve when the device didn't cover the meal — honest."""
    u = {}
    add_meal(u, {"ts": _BASE.isoformat(), "carbs_g": 60})
    assert build_meal_responses(u) == 0
    assert u["meal_responses"] == []


def test_meal_with_sparse_cgm_is_skipped():
    u = {}
    # only 3 readings in the window -> below the coverage threshold
    add_cgm(u, [{"ts": (_BASE + timedelta(minutes=m)).isoformat(), "glucose": 100}
                for m in (-20, 40, 120)])
    add_meal(u, {"ts": _BASE.isoformat(), "carbs_g": 60})
    assert build_meal_responses(u) == 0


def test_api_cgm_and_meal_endpoints_refit(tmp_path, monkeypatch):
    import personalization.user_store as us
    monkeypatch.setattr(us, "USER_DIR", tmp_path)
    us.create("u_api", weight_kg=82, height_cm=178, age=45, sex="male")

    from fastapi.testclient import TestClient
    import api.app as app
    c = TestClient(app.app)

    r = c.post("/users/u_api/cgm", json={"readings": _flat_cgm_around(_BASE)})
    assert r.status_code == 200 and r.json()["cgm_added"] > 0
    r = c.post("/users/u_api/meals",
               json={"ts": _BASE.isoformat(), "carbs_g": 55, "source": "manual"})
    body = r.json()
    assert r.status_code == 200 and body["logged"]["carbs_g"] == 55
    assert body["n_meal_responses"] == 1
    # one meal -> below the >=4 gate -> Si stays population prior (source not 'fitted')
    assert body["source"] in ("prior", "weak")


def test_api_rejects_unknown_user(tmp_path, monkeypatch):
    import personalization.user_store as us
    monkeypatch.setattr(us, "USER_DIR", tmp_path)
    from fastapi.testclient import TestClient
    import api.app as app
    c = TestClient(app.app)
    assert c.post("/users/nope/meals", json={"ts": _BASE.isoformat(), "carbs_g": 40}).status_code == 404
