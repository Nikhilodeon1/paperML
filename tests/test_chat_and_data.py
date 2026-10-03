"""Tests for the follow-up chat, modifiable user data (medical/notes), and web search."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def _demo():
    client.post("/users/seed_demo")


def test_web_search_tool_registered_and_flagged():
    from orchestration.tools import TOOL_SCHEMAS
    assert any(s["name"] == "web_search" for s in TOOL_SCHEMAS)


def test_web_search_result_shape(monkeypatch):
    import orchestration.websearch as ws
    monkeypatch.setattr(ws, "search", lambda q, n=3: (
        {"title": "T", "snippet": "S", "url": "u"},))
    out = ws.search_facts("anything")
    assert out["source"] == "web" and out["evidence"] == "weak"
    assert out["results"][0]["title"] == "T"


def test_patch_medical_and_notes_appends():
    _demo()
    r = client.patch("/users/u_demo", json={
        "medical": {"conditions": ["asthma"], "allergies": ["pollen"]},
        "notes": ["low iron intake"]})
    body = r.json()
    assert "asthma" in body["medical"]["conditions"]
    assert "low iron intake" in body["notes"]


def test_context_summary_reaches_medical():
    _demo()
    from personalization import user_store
    ctx = user_store.context_summary(user_store.load("u_demo"))
    assert "penicillin" in ctx.lower() and "calcium" in ctx.lower()


def test_arbitrary_user_data_editable():
    _demo()
    r = client.patch("/users/u_demo", json={"profile": {"resting_hr_pref": 55}})
    assert r.status_code == 200
    got = client.get("/users/u_demo").json()
    assert got["profile"]["resting_hr_pref"] == 55


def test_chat_uses_ml_tool_and_keeps_history(monkeypatch):
    """Chat routes via the orchestrator (mocked) and threads history back to the client."""
    import orchestration.orchestrate as orch
    from orchestration.router import Answer

    seen = {}

    def fake_chat(messages, profile, user_context=""):
        seen["n_messages"] = len(messages)
        seen["ctx"] = user_context
        return Answer("Wait ~6 hours.", tool="estimate_bac", evidence="strong",
                      headline="Wait ~6 hours")

    monkeypatch.setattr(orch, "chat_llm", fake_chat)
    _demo()
    msgs = [{"role": "user", "content": "4 beers, safe to drive?"}]
    r = client.post("/chat", json={"user_id": "u_demo", "messages": msgs}).json()
    assert r["tool"] == "estimate_bac"
    assert len(r["messages"]) == 2 and r["messages"][-1]["role"] == "assistant"
    assert seen["ctx"]  # medical/notes context was passed through


def test_chat_deterministic_fallback_no_backend(monkeypatch):
    """With no LLM backend, chat still answers the latest turn via the rules engine."""
    import orchestration.orchestrate as orch
    monkeypatch.setattr(orch, "active_backend", lambda: "deterministic")
    r = client.post("/chat", json={"messages": [
        {"role": "user", "content": "what's my bmi?"}],
        "weight_kg": 80, "height_cm": 180, "age": 30, "sex": "male"}).json()
    assert "bmi" in r["reply"].lower()
