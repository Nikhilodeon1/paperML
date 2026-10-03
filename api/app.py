"""Horizon backend API — the ONE interface for testing the model end to end.

This is what a real app's backend would expose: a single /ask endpoint that takes a
user profile + a natural-language question and returns a structured, explainable
answer. Everything upstream (Flutter app, this test page, curl, etc.) talks to this.

Default routing is the DETERMINISTIC/mechanistic router (ML + equations, no LLM) —
per your preference for a reliable, reproducible, data-grounded backend. Pass
`"use_llm": true` in the request to route via Gemini instead (uses your API quota);
useful for messy phrasing, but not the default computation path.

Run:
    uvicorn api.app:app --reload --port 8000
Then open http://127.0.0.1:8000/ in a browser for the test page, or POST to /ask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from orchestration.router import Answer, UserProfile, route

app = FastAPI(title="Horizon ML API", version="0.1.0")

# Allow the Flutter app (web/desktop/mobile) to call this dev backend from any origin.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _seed_demo_account():
    from personalization import auth
    auth.ensure_demo_account()  # dummy@test.com / dummy -> u_demo


@app.on_event("startup")
def _prewarm_models():
    """Fit the sleep + stress ML singletons in a background thread so the first
    /systems or simulate_body call isn't a ~75s cold start."""
    import threading

    def _warm():
        try:
            from personalization.systems import _get_sleep, _get_stress
            _get_sleep(); _get_stress()
        except Exception:
            pass
    threading.Thread(target=_warm, daemon=True).start()


# --- Auth -------------------------------------------------------------------
class SignupRequest(BaseModel):
    email: str
    password: str
    phone: Optional[str] = None
    enable_2fa: bool = False


class LoginRequest(BaseModel):
    email: str
    password: str


class Verify2FARequest(BaseModel):
    email: str
    code: str


@app.post("/auth/signup")
def auth_signup(req: SignupRequest):
    from personalization import auth
    try:
        auth.signup(req.email, req.password, req.phone, req.enable_2fa)
        return auth.login(req.email, req.password)  # sign in (or trigger 2FA)
    except auth.AuthError as e:
        raise HTTPException(400, str(e))


@app.post("/auth/login")
def auth_login(req: LoginRequest):
    from personalization import auth
    try:
        return auth.login(req.email, req.password)
    except auth.AuthError as e:
        raise HTTPException(401, str(e))


@app.post("/auth/verify_2fa")
def auth_verify(req: Verify2FARequest):
    from personalization import auth
    try:
        return auth.verify_2fa(req.email, req.code)
    except auth.AuthError as e:
        raise HTTPException(401, str(e))


@app.get("/auth/me")
def auth_me(authorization: str = Header(default="")):
    from personalization import auth
    token = authorization.replace("Bearer ", "").strip()
    who = auth.user_for_token(token)
    if not who:
        raise HTTPException(401, "invalid token")
    return who


@app.post("/auth/logout")
def auth_logout(authorization: str = Header(default="")):
    from personalization import auth
    auth.logout(authorization.replace("Bearer ", "").strip())
    return {"ok": True}

_STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural-language question, e.g. "
                          "'if I drink 2 shots of vodka now, how will my sleep be affected?'")
    # Either pass a user_id (profile + personalization pulled from the store) OR the
    # individual profile fields below.
    user_id: Optional[str] = None
    weight_kg: float = 80
    height_cm: float = 180
    age: float = 35
    sex: str = "male"
    total_chol: Optional[float] = None
    hdl: Optional[float] = None
    sbp: Optional[float] = None
    smoker: bool = False
    diabetic: bool = False
    # None = auto (LLM orchestrator when an API key is configured, else deterministic).
    # Numbers are grounded either way; the LLM only adds understanding + presentation.
    use_llm: Optional[bool] = None


class AskResponse(BaseModel):
    tool: Optional[str]
    evidence: str
    headline: str
    text: str
    explanation: list[str]
    indicators: dict
    citations: list[str]
    personalization: dict          # average-human vs you (empty if none)
    user_id: Optional[str] = None


def _to_profile(req: AskRequest) -> UserProfile:
    return UserProfile(
        weight_kg=req.weight_kg, height_cm=req.height_cm, age=req.age, sex=req.sex,
        total_chol=req.total_chol, hdl=req.hdl, sbp=req.sbp,
        smoker=req.smoker, diabetic=req.diabetic,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if req.sex not in ("male", "female"):
        raise HTTPException(400, "sex must be 'male' or 'female'")

    # Load profile (and later, personalization) from the user store when a user_id is
    # given; otherwise use the profile fields in the request.
    user = None
    if req.user_id:
        from personalization import user_store
        if not user_store.exists(req.user_id):
            raise HTTPException(404, f"unknown user_id {req.user_id!r}")
        user = user_store.load(req.user_id)
        profile = user_store.to_profile(user)
    else:
        profile = _to_profile(req)

    user_context = ""
    if user is not None:
        from personalization import user_store
        user_context = user_store.context_summary(user)

    use_llm = req.use_llm
    if use_llm is None:  # auto: use an LLM backend iff a key is configured
        from orchestration.orchestrate import active_backend
        use_llm = active_backend() != "deterministic"

    from orchestration.context import current_user
    tok = current_user.set(user)
    try:
        if use_llm:
            from orchestration.orchestrate import route_llm
            answer: Answer = route_llm(req.question, profile, user_context=user_context)
        else:
            answer = route(req.question, profile)
    finally:
        current_user.reset(tok)

    personalization = {}
    if user is not None:
        from personalization.compare import average_vs_you
        from personalization import user_store
        try:
            personalization = average_vs_you(answer, user)
        except Exception:
            personalization = {}
        user_store.record_result(user, req.question, answer)  # track over time

    return AskResponse(
        tool=answer.tool, evidence=answer.evidence, headline=answer.headline,
        text=answer.text, explanation=answer.explanation, indicators=answer.indicators,
        citations=answer.citations, personalization=personalization,
        user_id=req.user_id,
    )


class ChatRequest(BaseModel):
    messages: list[dict]          # [{role: 'user'|'assistant', content: str}, ...]
    user_id: Optional[str] = None
    weight_kg: float = 80
    height_cm: float = 180
    age: float = 35
    sex: str = "male"
    total_chol: Optional[float] = None
    hdl: Optional[float] = None
    sbp: Optional[float] = None
    smoker: bool = False
    diabetic: bool = False


@app.post("/chat")
def chat(req: ChatRequest):
    """Conversational follow-ups. The LLM routes every turn and calls the ML tools for
    any number; prior turns give context. Returns the assistant reply + the updated
    message list the client should keep."""
    from personalization import user_store
    from orchestration.orchestrate import chat_llm

    user, user_context = None, ""
    if req.user_id:
        if not user_store.exists(req.user_id):
            raise HTTPException(404, f"unknown user_id {req.user_id!r}")
        user = user_store.load(req.user_id)
        profile = user_store.to_profile(user)
        user_context = user_store.context_summary(user)
    else:
        profile = UserProfile(req.weight_kg, req.height_cm, req.age, req.sex,
                              req.total_chol, req.hdl, req.sbp, req.smoker, req.diabetic)

    from orchestration.context import current_user
    tok = current_user.set(user)
    try:
        answer: Answer = chat_llm(req.messages, profile, user_context=user_context)
    finally:
        current_user.reset(tok)
    reply = {"role": "assistant", "content": answer.text}
    if user is not None:
        last_user = next((m["content"] for m in reversed(req.messages)
                          if m.get("role") == "user"), "")
        user_store.record_result(user, last_user, answer)
    return {
        "reply": answer.text, "headline": answer.headline, "evidence": answer.evidence,
        "tool": answer.tool, "indicators": answer.indicators, "citations": answer.citations,
        "messages": req.messages + [reply],
    }


@app.get("/users")
def users():
    from personalization import user_store
    return {"users": user_store.list_users()}


@app.get("/users/{user_id}")
def get_user(user_id: str):
    from personalization import user_store
    if not user_store.exists(user_id):
        raise HTTPException(404, "unknown user")
    u = user_store.load(user_id)
    if "profile" not in u:                      # reserved/non-user file (e.g. _auth)
        raise HTTPException(404, "not a user record")
    logs = u.get("logs", {})
    daily = u.get("wearable", {}).get("daily", [])
    return {
        "account": u.get("account", {}),
        "profile": u["profile"], "medical": u.get("medical", {}),
        "notes": u.get("notes", []), "derived": u.get("derived", {}),
        "routines": u.get("routines", []),
        "records": list(reversed(u.get("records", [])))[:20],
        "daily_log": list(reversed(u.get("daily_log", [])))[:30],
        "wearable_recent": daily[-7:],           # last week of daily wearable
        "data_summary": {"weigh_ins": len(logs.get("weighins", [])),
                         "drinks_logged": len(logs.get("drinks", [])),
                         "wearable_days": len(daily),
                         "surveys": len(u.get("surveys", []))},
        "reports": list(reversed(u.get("metrics_history", [])))[:15],  # most recent first
    }


class UserEdit(BaseModel):
    profile: Optional[dict] = None
    medical: Optional[dict] = None    # replaces {allergies, conditions, medications}
    notes: Optional[list] = None      # replaces the whole notes list


@app.put("/users/{user_id}")
def edit_user(user_id: str, edit: UserEdit):
    """Replace-semantics edit (for the on-site editor). Recomputes personalization."""
    from personalization import user_store
    if not user_store.exists(user_id):
        raise HTTPException(404, "unknown user")
    u = user_store.set_fields(user_id, profile=edit.profile, medical=edit.medical,
                              notes=edit.notes)
    return {"ok": True, "profile": u["profile"], "medical": u.get("medical", {}),
            "notes": u.get("notes", []), "derived": u.get("derived", {})}


class UserPatch(BaseModel):
    profile: Optional[dict] = None
    medical: Optional[dict] = None       # {allergies:[], conditions:[], medications:[]}
    notes: Optional[list] = None         # free-text English facts to append
    logs: Optional[dict] = None          # {weighins:[...], drinks:[...]} to append
    routines: Optional[list] = None      # replaces the routines list
    records: Optional[list] = None       # health records to append
    daily_log: Optional[list] = None     # daily log entries to append


@app.patch("/users/{user_id}")
def patch_user(user_id: str, patch: UserPatch):
    """The user can modify any of their data. Lists (notes/medical/logs) append."""
    from personalization import user_store
    if not user_store.exists(user_id):
        raise HTTPException(404, "unknown user")
    body = {k: v for k, v in patch.model_dump().items() if v is not None}
    u = user_store.update(user_id, body)
    return {"ok": True, "medical": u.get("medical", {}), "notes": u.get("notes", []),
            "derived": u.get("derived", {})}


@app.get("/users/{user_id}/systems")
def user_systems(user_id: str):
    """What the model knows about the user, per physiological system (real module
    outputs run on their data). Powers the home dashboard + system detail views."""
    from personalization import user_store
    from personalization.systems import compute_systems
    if not user_store.exists(user_id):
        raise HTTPException(404, "unknown user")
    u = user_store.load(user_id)
    if u.get("systems"):
        return {"systems": u["systems"]}
    systems = compute_systems(u)
    u["systems"] = systems
    user_store.save(u)
    return {"systems": systems}


@app.get("/users/{user_id}/advice")
def user_advice(user_id: str, refresh: bool = False):
    """Always-on personalized advice. Cached in the user record and regenerated only
    when their data changes or `refresh=true` — so the app can just load it, no button."""
    from personalization import user_store
    from orchestration.orchestrate import route_llm
    if not user_store.exists(user_id):
        raise HTTPException(404, "unknown user")
    u = user_store.load(user_id)
    cached = u.get("advice")
    if cached and not refresh:
        return cached

    profile = user_store.to_profile(u)
    ctx = user_store.context_summary(u)
    ans = route_llm(
        "Give me 3 short, concrete, personalized health recommendations based on my "
        "full profile, wearable data, records, logs, medical info and notes. Number "
        "them 1-3, one line each, and tie each to something specific about me.",
        profile, user_context=ctx)
    advice = {"text": ans.text, "ts": user_store._now()}
    u["advice"] = advice
    user_store.save(u)
    return advice


@app.get("/questionnaires")
def questionnaires():
    """Catalogue of inference questionnaires the app can offer (menu metadata)."""
    from personalization.questionnaires import list_questionnaires
    return {"questionnaires": list_questionnaires()}


@app.get("/questionnaires/{qid}")
def questionnaire(qid: str):
    """Full definition (questions + options) to render one questionnaire's form."""
    from personalization.questionnaires import get_questionnaire
    q = get_questionnaire(qid)
    if q is None:
        raise HTTPException(404, "unknown questionnaire")
    return q


class QuestionnaireSubmit(BaseModel):
    responses: dict     # question_id -> answer (number, or an option `value`)


@app.post("/users/{user_id}/questionnaires/{qid}")
def submit_questionnaire(user_id: str, qid: str, body: QuestionnaireSubmit):
    """Score the user's answers into health indicators and store them on the user.
    Returns the scored indicators (constipation flag, hydration, perceived stress, ...)."""
    from personalization import user_store
    from personalization.questionnaires import get_questionnaire
    if not user_store.exists(user_id):
        raise HTTPException(404, "unknown user")
    if get_questionnaire(qid) is None:
        raise HTTPException(404, "unknown questionnaire")
    scored = user_store.submit_questionnaire(user_id, qid, body.responses)
    return scored


@app.get("/users/{user_id}/questionnaires")
def user_questionnaires(user_id: str):
    """Catalogue + this user's latest scored answers per questionnaire, so the app can
    show which check-ins are done and their most recent result."""
    from personalization import user_store
    from personalization.questionnaires import list_questionnaires
    if not user_store.exists(user_id):
        raise HTTPException(404, "unknown user")
    u = user_store.load(user_id)
    latest: dict = {}
    for s in u.get("surveys", []):
        if s.get("id"):
            latest[s["id"]] = {"date": s.get("date"), "indicators": s.get("indicators", [])}
    return {"catalogue": list_questionnaires(), "answers": latest}


@app.get("/users/{user_id}/optimize")
def user_optimize(user_id: str, goal: str = ""):
    """Ranked interventions that most improve a goal for this user (counterfactual sim)."""
    from personalization import user_store
    from personalization.optimize import optimize
    if not user_store.exists(user_id):
        raise HTTPException(404, "unknown user")
    return optimize(user_store.load(user_id), goal)


@app.get("/users/{user_id}/accuracy")
def user_accuracy(user_id: str):
    """The Accuracy Journal: how well the model predicts THIS user's day-to-day metrics
    (MAE, bias, coverage) and how much better than assuming the population average."""
    from personalization import user_store
    from personalization.accuracy_journal import accuracy_summary
    if not user_store.exists(user_id):
        raise HTTPException(404, "unknown user")
    u = user_store.load(user_id)
    return {"metrics": accuracy_summary(u),
            "learned_baselines": (u.get("derived", {}) or {}).get("wearable", {})}


@app.get("/users/{user_id}/differential")
def user_differential(user_id: str, symptom: str):
    """Ranked, personalized causes of a symptom for a stored user (grounded, no LLM)."""
    from personalization import user_store
    from pipeline.causal import explain_symptom
    if not user_store.exists(user_id):
        raise HTTPException(404, "unknown user")
    return explain_symptom(symptom, user_store.load(user_id))


@app.get("/symptoms")
def symptoms():
    """Symptoms the causal graph can currently reason about (for menus)."""
    from pipeline.causal import list_symptoms
    return {"symptoms": list_symptoms()}


@app.post("/users/seed_demo")
def seed_demo_user():
    from personalization import user_store
    u = user_store.seed_demo()
    return {"user_id": u["user_id"], "profile": u["profile"], "derived": u["derived"]}


# ============================ ML LAB / DEBUG =================================
# Lightweight introspection for testing the model ALONE (served at /lab). These
# endpoints expose exactly what the model receives and produces — the assembled
# profile, the user-context string injected into the prompt, the system instruction,
# every tool call + its raw grounded result, and timing — so the pipeline is fully
# transparent while iterating on accuracy.

class DebugAskRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    profile: Optional[dict] = None      # raw profile if no user_id
    use_llm: Optional[bool] = None


@app.post("/debug/ask")
def debug_ask(req: DebugAskRequest):
    """Run one /ask and return the FULL trace: inputs sent to the model + outputs."""
    import time
    from dataclasses import asdict
    from personalization import user_store
    from orchestration.orchestrate import active_backend, route_llm
    from orchestration.router import UserProfile, route as det_route

    user = None
    if req.user_id:
        if not user_store.exists(req.user_id):
            raise HTTPException(404, f"unknown user_id {req.user_id!r}")
        user = user_store.load(req.user_id)
        profile = user_store.to_profile(user)
        user_context = user_store.context_summary(user)
    else:
        p = req.profile or {}
        profile = UserProfile(
            weight_kg=p.get("weight_kg", 80), height_cm=p.get("height_cm", 180),
            age=p.get("age", 35), sex=p.get("sex", "male"),
            total_chol=p.get("total_chol"), hdl=p.get("hdl"), sbp=p.get("sbp"),
            smoker=p.get("smoker", False), diabetic=p.get("diabetic", False))
        user_context = ""

    backend = active_backend()
    use_llm = req.use_llm if req.use_llm is not None else (backend != "deterministic")

    # Reconstruct exactly what the LLM backend would be sent (for transparency).
    system_instruction = None
    user_message = None
    if use_llm and backend != "deterministic":
        from orchestration.llm_gemini import _SYSTEM_INSTRUCTION
        system_instruction = _SYSTEM_INSTRUCTION
        if user_context:
            system_instruction += (f"\n\nThis user's saved medical info & notes: {user_context}")
        user_message = f"User profile: {asdict(profile)}\n\nQuestion: {req.question}"

    t0 = time.perf_counter()
    if use_llm:
        answer = route_llm(req.question, profile, user_context=user_context)
    else:
        answer = det_route(req.question, profile)
    dt_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "request": req.model_dump(),
        "resolved_backend": backend if use_llm else "deterministic",
        "used_llm": bool(use_llm),
        "model_inputs": {
            "profile_sent": asdict(profile),
            "user_context_injected": user_context,
            "system_instruction": system_instruction,
            "user_message": user_message,
        },
        "answer": asdict(answer),          # includes .raw = {backend, tools_used, results}
        "timing_ms": dt_ms,
    }


class DebugChatRequest(BaseModel):
    messages: list[dict]                 # [{role:'user'|'assistant', content:str}, ...]
    user_id: Optional[str] = None


@app.post("/debug/chat")
def debug_chat(req: DebugChatRequest):
    """Conversational turn WITH the full reasoning trace — every simulation the model
    ran (args + result), its planning between steps, what was sent to it, and timing.
    This powers the ML lab's single chat box."""
    import time
    from dataclasses import asdict
    from personalization import user_store
    from orchestration.orchestrate import active_backend, chat_llm
    from orchestration.router import UserProfile
    from orchestration.context import current_user

    user, user_context = None, ""
    if req.user_id:
        if not user_store.exists(req.user_id):
            raise HTTPException(404, f"unknown user_id {req.user_id!r}")
        user = user_store.load(req.user_id)
        if not user.get("systems"):        # cache the body simulation so it's in-context
            from personalization.systems import compute_systems
            user["systems"] = compute_systems(user)
            user_store.save(user)
        profile = user_store.to_profile(user)
        user_context = user_store.context_summary(user)
    else:
        profile = UserProfile(weight_kg=80, height_cm=180, age=35, sex="male")

    question = next((m["content"] for m in reversed(req.messages)
                     if m.get("role") == "user"), "")
    backend = active_backend()
    system_instruction = None
    if backend != "deterministic":
        from orchestration.llm_gemini import _SYSTEM_INSTRUCTION
        system_instruction = _SYSTEM_INSTRUCTION + (
            f"\n\nThis user's saved medical info & notes: {user_context}" if user_context else "")

    tok = current_user.set(user)
    t0 = time.perf_counter()
    try:
        answer: Answer = chat_llm(req.messages, profile, user_context=user_context)
    finally:
        current_user.reset(tok)
    dt_ms = round((time.perf_counter() - t0) * 1000, 1)

    if user is not None:
        user_store.record_result(user, question, answer)

    raw = answer.raw or {}
    return {
        "reply": answer.text, "headline": answer.headline, "evidence": answer.evidence,
        "tool": answer.tool, "citations": answer.citations, "indicators": answer.indicators,
        "trace": raw.get("trace", []),
        "backend": raw.get("backend", backend),
        "model_inputs": {
            "profile_sent": asdict(profile),
            "user_context_injected": user_context,
            "system_instruction": system_instruction,
            "question": question,
        },
        "messages": req.messages + [{"role": "assistant", "content": answer.text}],
        "timing_ms": dt_ms,
    }


@app.get("/debug/user/{user_id}")
def debug_user(user_id: str):
    """The COMPLETE raw stored record for a user (untrimmed) — powers the lab's
    'full record' panel so you can see literally everything on file."""
    from personalization import user_store
    if not user_store.exists(user_id):
        raise HTTPException(404, "unknown user")
    u = user_store.load(user_id)
    if "profile" not in u:
        raise HTTPException(404, "not a user record")
    return u


class DebugPlanRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    use_llm: Optional[bool] = None
    messages: Optional[list[dict]] = None      # prior conversation turns (memory)


@app.post("/debug/plan")
def debug_plan(req: DebugPlanRequest):
    """Plan -> execute -> synthesize for a question: returns the approach (plan), each
    grounded step's result, and the composed answer. The reasoning layer, laid bare."""
    import time
    from personalization import user_store
    from orchestration.planner import plan_and_answer
    from orchestration.orchestrate import active_backend
    from orchestration.router import UserProfile

    if req.user_id:
        if not user_store.exists(req.user_id):
            raise HTTPException(404, f"unknown user_id {req.user_id!r}")
        user = user_store.load(req.user_id)
        if not user.get("systems"):
            from personalization.systems import compute_systems
            user["systems"] = compute_systems(user)
            user_store.save(user)
        profile = user_store.to_profile(user)
        ctx = user_store.context_summary(user)
    else:
        user = {"profile": {"sex": "male", "age": 35, "height_cm": 178, "weight_kg": 80}}
        profile = UserProfile(weight_kg=80, height_cm=178, age=35, sex="male")
        ctx = ""

    use_llm = req.use_llm if req.use_llm is not None else (active_backend() != "deterministic")
    t0 = time.perf_counter()
    out = plan_and_answer(req.question, profile, user, user_context=ctx, use_llm=bool(use_llm),
                          history=(req.messages or [])[:-1] if req.messages else None)
    out["timing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    # if a fact was saved, tell the client to refresh the subject
    out["saved_fact"] = any(s.get("capability") == "remember"
                            and (s.get("result") or {}).get("saved") for s in out.get("steps", []))
    return out


class CGMRequest(BaseModel):
    readings: list[dict] = Field(default_factory=list)     # [{ts, glucose}]


class MealLogRequest(BaseModel):
    ts: str
    carbs_g: float
    protein_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    source: str = "manual"


def _si_status(user: dict) -> dict:
    isf = (user.get("derived", {}) or {}).get("insulin_sensitivity") or {}
    return {"insulin_sensitivity": isf.get("value"), "source": isf.get("source"),
            "n_meal_responses": len(user.get("meal_responses", []))}


@app.post("/users/{user_id}/cgm")
def log_cgm(user_id: str, req: CGMRequest):
    """Ingest a CGM stream ([{ts, glucose}]). Rebuilds meal responses + refits Si."""
    from personalization import user_store
    from personalization.meal_logging import add_cgm
    if not user_store.exists(user_id):
        raise HTTPException(404, f"unknown user_id {user_id!r}")
    user = user_store.load(user_id)
    n = add_cgm(user, req.readings)
    user_store.refresh_derived(user)          # rebuild meal_responses + refit Si
    user_store.save(user)
    return {"cgm_added": n, "cgm_total": len(user.get("wearable", {}).get("cgm", [])),
            **_si_status(user)}


@app.post("/users/{user_id}/meals")
def log_meal(user_id: str, req: MealLogRequest):
    """Log a meal (from a photo or manual entry). If CGM covers its window, it becomes a
    fittable meal response and Si is re-learned."""
    from personalization import user_store
    from personalization.meal_logging import add_meal
    if not user_store.exists(user_id):
        raise HTTPException(404, f"unknown user_id {user_id!r}")
    user = user_store.load(user_id)
    entry = add_meal(user, req.model_dump())
    user_store.refresh_derived(user)
    user_store.save(user)
    return {"logged": entry, **_si_status(user)}


@app.post("/debug/analyze_image")
async def debug_analyze_image(file: UploadFile = File(...)):
    """Upload an image (meal photo, lab report, wearable screenshot). A vision model EXTRACTS
    structured data — it computes no physiological number. Returns the extraction, a
    confirmation prompt (the user must confirm before it drives the engine), and, for a meal,
    the `simulate` args it would run once confirmed."""
    from orchestration.vision import (analyze_image, confirmation_prompt,
                                      extraction_to_sim_args, image_bytes_to_b64)
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(413, "image too large (max 12 MB)")
    mime = file.content_type or "image/jpeg"
    ext = analyze_image(image_bytes_to_b64(data), mime=mime)
    return {"extraction": ext, "confirm": confirmation_prompt(ext),
            "sim_args": extraction_to_sim_args(ext)}


@app.post("/debug/seed_cohort")
def debug_seed_cohort(n_days: int = 21):
    """Create the 5 rich longitudinal demo users (Maya/Raj/Elena/Tom/Grace) with intraday CGM,
    logged meals, labs and months of wearable — data-dense enough to demo predictions, warnings
    and patterns. Slow (~1 min) — a one-time setup."""
    from personalization.cohort import make_cohort
    users = make_cohort(n_days=max(7, min(n_days, 30)))
    return {"created": [{"user_id": u["user_id"], "name": u["account"]["name"],
                         "insulin_sensitivity": (u.get("derived", {}).get("insulin_sensitivity") or {})}
                        for u in users]}


@app.post("/debug/generate_user")
def debug_generate_user():
    """Generate + save ONE complete mock human (full labs, months of wearable/weigh-ins,
    medical history, records, answered questionnaires, personalization) and return the
    whole record plus their computed body-system snapshot. Gives the lab a realistic,
    long-time-user subject to reason about."""
    from personalization import user_store
    from personalization.synthetic import make_full_user
    from personalization.systems import compute_systems
    u = make_full_user()
    user_store.save(u)
    systems = compute_systems(u)
    return {"user_id": u["user_id"], "user": u, "systems": systems}


class DebugSimulateRequest(BaseModel):
    profile: dict                          # sex, age, height_cm, weight_kg, labs, ...
    wearable_daily: Optional[list] = None  # [{resting_hr, hrv_rmssd, steps, sleep_efficiency}]
    weighins: Optional[list] = None        # [{day, weight_kg, mean_daily_intake_kcal}]
    surveys: Optional[list] = None         # scored survey entries to fold in


@app.post("/debug/simulate")
def debug_simulate(req: DebugSimulateRequest):
    """Run the WHOLE simulator (compute_systems) on an arbitrary in-memory body without
    saving anything — the fastest way to test the ML on any inputs, incl. extreme ones."""
    from personalization.systems import compute_systems
    from personalization.user_store import refresh_derived
    user = {
        "user_id": "debug", "profile": req.profile,
        "wearable": {"daily": req.wearable_daily or []},
        "logs": {"weighins": req.weighins or [], "drinks": [], "bac_readings": []},
        "surveys": req.surveys or [], "medical": {}, "notes": [],
    }
    refresh_derived(user)
    systems = compute_systems(user)
    return {"systems": systems, "derived": user.get("derived", {})}


class DebugScoreRequest(BaseModel):
    qid: str
    responses: dict


@app.post("/debug/score")
def debug_score(req: DebugScoreRequest):
    """Score a questionnaire's raw answers (no user needed) — see the exact indicators."""
    from personalization.questionnaires import score, get_questionnaire
    if get_questionnaire(req.qid) is None:
        raise HTTPException(404, "unknown questionnaire")
    return score(req.qid, req.responses)


class DebugDifferentialRequest(BaseModel):
    symptom: str
    user_id: Optional[str] = None
    user: Optional[dict] = None      # a raw user dict (profile/notes/surveys) if no id


@app.post("/debug/differential")
def debug_differential(req: DebugDifferentialRequest):
    """Causal differential for a symptom on a stored user OR an arbitrary in-memory user
    dict — the lab's way to probe the causal graph on any body."""
    from pipeline.causal import explain_symptom
    from personalization import user_store
    user = req.user
    if req.user_id:
        if not user_store.exists(req.user_id):
            raise HTTPException(404, "unknown user")
        user = user_store.load(req.user_id)
    return explain_symptom(req.symptom, user)


@app.get("/debug/population")
def debug_population(n: int = 12, seed: int = 11):
    """A small synthetic population sample + a couple of aggregate stats, for eyeballing
    the mock-user generator that backs the accuracy harness."""
    from personalization.synthetic import make_population
    users = make_population(n=min(n, 60), seed=seed)
    sample = [{"user_id": u["user_id"], "profile": u["profile"],
               "n_weighins": len(u["logs"]["weighins"]),
               "n_wearable": len(u["wearable"]["daily"]),
               "surveys": [s.get("id") for s in u.get("surveys", [])]} for u in users]
    bmis = [u["profile"]["weight_kg"] / ((u["profile"]["height_cm"] / 100) ** 2) for u in users]
    return {"n": len(users),
            "bmi_range": [round(min(bmis), 1), round(max(bmis), 1)],
            "users": sample}


@app.get("/health")
def health():
    from orchestration.orchestrate import active_backend
    return {"status": "ok", "backend": active_backend()}


@app.get("/", response_class=HTMLResponse)
@app.get("/lab", response_class=HTMLResponse)
def lab():
    """The ML lab is now the primary interface (the old test console is retired)."""
    return (_STATIC_DIR / "lab.html").read_text(encoding="utf-8")
