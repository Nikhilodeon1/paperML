"""Tests for the per-user data store + average-vs-you comparison."""

from __future__ import annotations

import pytest

from personalization import user_store as us


@pytest.fixture()
def demo(tmp_path, monkeypatch):
    # Redirect the store to a temp dir so tests don't touch real user data.
    monkeypatch.setattr(us, "USER_DIR", tmp_path)
    return us.seed_demo("u_test")


def test_seed_creates_full_record(demo):
    assert demo["user_id"] == "u_test"
    assert len(demo["logs"]["weighins"]) >= 10
    assert len(demo["wearable"]["daily"]) >= 30       # mock wearable present
    assert demo["surveys"] and demo["account"]        # all data channels present
    assert "metabolic" in demo["derived"]             # personalization computed


def test_list_users_excludes_reserved_files(tmp_path, monkeypatch):
    # The auth store lives in the same dir as user records; it must not be listed as a
    # user (regression: `_auth` surfaced as a user and 500'd GET /users/_auth).
    monkeypatch.setattr(us, "USER_DIR", tmp_path)
    us.create("u_real", sex="male", age=30, height_cm=175, weight_kg=75)
    (tmp_path / "_auth.json").write_text('{"accounts": {}, "tokens": {}}', encoding="utf-8")
    assert us.list_users() == ["u_real"]


def test_personalization_recovers_signal(demo):
    mp = us.metabolic_params(demo)
    assert mp is not None
    assert 0.80 < mp.rmr_multiplier < 1.0             # hidden truth was 0.90
    assert mp.n_observations >= 10


def test_open_schema_accepts_arbitrary_data(demo):
    demo["uploads"].append({"kind": "lab_pdf", "hba1c": 5.4})
    demo["profile"]["blood_type"] = "O+"              # arbitrary extra field allowed
    us.save(demo)
    reloaded = us.load("u_test")
    assert reloaded["profile"]["blood_type"] == "O+"
    assert reloaded["uploads"][-1]["hba1c"] == 5.4


def test_record_result_tracks_over_time(demo):
    from orchestration.router import Answer
    us.record_result(demo, "test q", Answer("t", tool="compute_bmi", evidence="strong",
                                             headline="BMI 24"))
    assert us.load("u_test")["metrics_history"][-1]["tool"] == "compute_bmi"


def test_average_vs_you_differs_with_personalization(demo):
    from personalization.compare import average_vs_you
    from orchestration.router import Answer
    ans = Answer("t", tool="project_weight", evidence="weak",
                 indicators={"daily_intake_kcal": 2000, "horizon_days": 365})
    cmp = average_vs_you(ans, demo)
    assert cmp and cmp["average_person"] != cmp["you"]
    assert "%" in cmp["why"]
