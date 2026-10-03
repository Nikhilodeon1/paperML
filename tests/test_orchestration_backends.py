"""Network-free tests for the Groq backend + the unified backend selector."""

from __future__ import annotations

from orchestration.llm_groq import _openai_tools, route_with_groq
from orchestration.router import Answer, UserProfile
from orchestration.tools import TOOL_SCHEMAS

PROFILE = UserProfile(80, 180, 45, "male", total_chol=230, hdl=42, sbp=138, smoker=True)


def test_openai_tools_conversion():
    tools = _openai_tools()
    assert len(tools) == len(TOOL_SCHEMAS)
    assert all(t["type"] == "function" and t["function"]["name"] for t in tools)


def test_groq_falls_back_without_key(monkeypatch):
    import orchestration.config as cfg
    monkeypatch.setattr(cfg, "groq_api_key", lambda: None)
    a = route_with_groq("4 beers tonight when am I sober?", PROFILE)
    assert a.tool == "estimate_bac"


def test_active_backend_prefers_groq(monkeypatch):
    import orchestration.orchestrate as orch
    monkeypatch.setattr(orch, "groq_api_key", lambda: "g")
    monkeypatch.setattr(orch, "gemini_api_key", lambda: "gem")
    assert orch.active_backend() == "groq"


def test_active_backend_gemini_when_no_groq(monkeypatch):
    import orchestration.orchestrate as orch
    monkeypatch.setattr(orch, "groq_api_key", lambda: None)
    monkeypatch.setattr(orch, "gemini_api_key", lambda: "gem")
    assert orch.active_backend() == "gemini"


def test_active_backend_deterministic_when_no_keys(monkeypatch):
    import orchestration.orchestrate as orch
    monkeypatch.setattr(orch, "groq_api_key", lambda: None)
    monkeypatch.setattr(orch, "gemini_api_key", lambda: None)
    assert orch.active_backend() == "deterministic"


def test_route_llm_dispatches_to_active_backend(monkeypatch):
    import orchestration.orchestrate as orch
    monkeypatch.setattr(orch, "active_backend", lambda: "groq")
    import orchestration.llm_groq as g
    monkeypatch.setattr(g, "route_with_groq",
                        lambda q, p, **kw: Answer("groq answer", tool="compute_bmi", evidence="strong"))
    a = orch.route_llm("what's my bmi", PROFILE)
    assert a.text == "groq answer"
