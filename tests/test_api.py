"""Tests for the FastAPI backend (api/app.py) — the one interface a real app
would call. Uses TestClient, no running server needed."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

# Force the deterministic router for plumbing tests so they're network-free and
# reproducible regardless of whether a GEMINI_API_KEY happens to be configured.
BASE = dict(weight_kg=80, height_cm=180, age=45, sex="male",
           total_chol=230, hdl=42, sbp=138, smoker=True, use_llm=False)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "Horizon" in r.text


def test_ask_cross_system_vodka_to_sleep():
    r = client.post("/ask", json={
        **BASE, "question": "if I drink 2 shots of vodka now, how will my sleep be affected?"})
    assert r.status_code == 200
    body = r.json()
    assert body["tool"] == "alcohol_effect_on_sleep"
    assert body["evidence"] == "strong"
    assert body["explanation"]                 # plain-language bullets present
    assert body["indicators"]["standard_drinks"] == 2
    assert body["citations"]


def test_ask_bac():
    r = client.post("/ask", json={**BASE, "question": "4 beers tonight when am I sober?"})
    body = r.json()
    assert body["tool"] == "estimate_bac"
    assert "peak_bac_g_per_100ml" in body["indicators"]
    assert len(body["explanation"]) >= 1


def test_ask_cvd():
    r = client.post("/ask", json={**BASE, "question": "what's my heart disease risk?"})
    body = r.json()
    assert body["tool"] == "estimate_cvd_risk"
    assert "risk_10yr_pct" in body["indicators"]


def test_ask_unsupported_question_is_honest():
    r = client.post("/ask", json={**BASE, "question": "will standing on my head improve my eyesight?"})
    body = r.json()
    assert body["tool"] is None
    assert body["evidence"] == "none"
    assert body["explanation"] == []


def test_ask_invalid_sex_rejected():
    r = client.post("/ask", json={**{**BASE, "sex": "other"}, "question": "4 beers, when sober?"})
    assert r.status_code == 400


def test_use_llm_false_never_calls_gemini(monkeypatch):
    """use_llm=False must never call the Gemini path, even if a key is configured."""
    import orchestration.llm_gemini as g

    def _boom(*a, **k):
        raise AssertionError("Gemini router must not be called when use_llm=False")

    monkeypatch.setattr(g, "route_with_gemini", _boom)
    r = client.post("/ask", json={**BASE, "question": "4 beers tonight when am I sober?"})
    assert r.status_code == 200
    assert r.json()["tool"] == "estimate_bac"


def test_auto_mode_routes_to_llm_when_key_present(monkeypatch):
    """When use_llm is unset, the API routes through the unified LLM orchestrator iff a
    backend is available. Stub the backend + call so no network happens."""
    from orchestration.router import Answer
    import orchestration.orchestrate as orch

    monkeypatch.setattr(orch, "active_backend", lambda: "groq")
    monkeypatch.setattr(orch, "route_llm",
                        lambda q, p, **kw: Answer("llm answer", tool="estimate_bac", evidence="strong"))

    body = {k: v for k, v in BASE.items() if k != "use_llm"}  # unset => auto
    r = client.post("/ask", json={**body, "question": "4 beers, when sober?"})
    assert r.status_code == 200
    assert r.json()["text"] == "llm answer"


def test_bmi_via_api_deterministic():
    r = client.post("/ask", json={**BASE, "question": "what's my bmi?"})
    body = r.json()
    assert body["tool"] == "compute_bmi"
    assert "bmi" in body["indicators"]
    assert body["headline"]  # compact takeaway present


def test_user_store_flow_and_average_vs_you():
    """Seed the demo user, ask a weight question by user_id, and get the average-vs-you
    comparison grounded in their personalization."""
    client.post("/users/seed_demo")
    assert "u_demo" in client.get("/users").json()["users"]
    r = client.post("/ask", json={
        "user_id": "u_demo", "use_llm": False,
        "question": "if I eat 2000 cal a day for a year, what happens to my weight?"})
    body = r.json()
    assert body["user_id"] == "u_demo"
    p = body["personalization"]
    assert p and p["average_person"] != p["you"]   # personalization actually changes it
    assert "%" in p["why"] or "weigh-ins" in p["why"]


def test_unknown_user_id_404():
    r = client.post("/ask", json={"user_id": "nope", "question": "what's my bmi?", "use_llm": False})
    assert r.status_code == 404
