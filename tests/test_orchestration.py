"""Tests for Layer 4 orchestration (tools + deterministic router)."""

from __future__ import annotations

import pytest

from orchestration.router import UserProfile, route
from orchestration.tools import TOOL_SCHEMAS, dispatch

PROFILE = UserProfile(weight_kg=80, height_cm=180, age=45, sex="male",
                      total_chol=230, hdl=42, sbp=138, smoker=True)


def test_all_tools_have_valid_schemas():
    for s in TOOL_SCHEMAS:
        assert s["name"] and s["description"]
        assert s["parameters"]["type"] == "object"
        # `required` is optional in JSON schema — some tools (e.g. simulate_scenario,
        # remember_about_user) take only optional args or read the current user. When
        # present it must list real properties.
        req = s["parameters"].get("required", [])
        assert isinstance(req, list)
        assert all(r in s["parameters"].get("properties", {}) for r in req)


def test_dispatch_unknown_tool_raises():
    with pytest.raises(ValueError):
        dispatch("not_a_tool", {})


def test_dispatch_bac():
    out = dispatch("estimate_bac", dict(standard_drinks=3, weight_kg=80, sex="male"))
    assert out["peak_bac"] > 0
    assert out["evidence"] == "strong"
    assert out["citations"]


def test_route_bac_question():
    a = route("If I have 4 beers tonight when am I sober?", PROFILE)
    assert a.tool == "estimate_bac"
    assert a.evidence == "strong"
    assert "BAC" in a.text


def test_route_alcohol_sleep_is_cross_system():
    a = route("how will 3 drinks before bed affect my sleep?", PROFILE)
    assert a.tool == "alcohol_effect_on_sleep"
    assert "REM" in a.text


def test_route_weight_question():
    a = route("if I eat 2000 calories a day for 6 months what happens to my weight?", PROFILE)
    assert a.tool == "project_weight"
    assert "kg" in a.text


def test_route_cvd_question():
    a = route("what is my heart disease risk?", PROFILE)
    assert a.tool == "estimate_cvd_risk"
    assert "%" in a.text


def test_cvd_question_without_labs_asks_for_them():
    bare = UserProfile(weight_kg=80, height_cm=180, age=45, sex="male")
    a = route("what is my cardiovascular risk?", bare)
    assert a.tool is None
    assert a.evidence == "none"


def test_unsupported_question_returns_third_outcome():
    """No model => honest refusal, never a fabricated answer (brief 8)."""
    a = route("will standing on my head improve my eyesight?", PROFILE)
    assert a.tool is None
    assert a.evidence == "none"
    assert "don't have" in a.text.lower() or "won't guess" in a.text.lower()
