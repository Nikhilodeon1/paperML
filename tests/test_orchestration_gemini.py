"""Network-free tests for the Gemini router (no API calls)."""

from __future__ import annotations

from orchestration.llm_gemini import _merge_profile, route_with_gemini
from orchestration.router import UserProfile
from orchestration.tools import TOOL_SCHEMAS

PROFILE = UserProfile(weight_kg=80, height_cm=180, age=45, sex="male",
                      total_chol=230, hdl=42, sbp=138, smoker=True)


def test_merge_profile_fills_missing_args():
    merged = _merge_profile({"standard_drinks": 3}, PROFILE)
    assert merged["standard_drinks"] == 3        # preserved
    assert merged["weight_kg"] == 80             # filled from profile
    assert merged["sex"] == "male"


def test_merge_profile_does_not_override_explicit_args():
    merged = _merge_profile({"weight_kg": 99}, PROFILE)
    assert merged["weight_kg"] == 99


def test_falls_back_to_rules_without_api_key(monkeypatch):
    """No key => fall back to the deterministic router so the system still answers.
    Stub the key resolver directly so this never depends on a real .env or makes a
    live call, regardless of test order."""
    import orchestration.config as cfg
    monkeypatch.setattr(cfg, "gemini_api_key", lambda: None)
    a = route_with_gemini("If I have 4 beers tonight when am I sober?", PROFILE)
    assert a.tool == "estimate_bac"
    assert a.evidence == "strong"


def test_build_gemini_tools_when_sdk_available():
    """If google-genai is installed, every tool schema converts to a declaration."""
    try:
        from orchestration.llm_gemini import build_gemini_tools
        tools = build_gemini_tools()
    except ImportError:
        import pytest
        pytest.skip("google-genai not installed")
    names = {d.name for t in tools for d in t.function_declarations}
    assert names == {s["name"] for s in TOOL_SCHEMAS}
