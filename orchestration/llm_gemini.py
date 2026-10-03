"""Gemini-backed orchestrator for Layer 4 — the "brain".

Design (per product direction): SIMULATION/EQUATIONS/CITED-FACTS ARE THE CENTRAL
SOURCE. The LLM's job is to understand any question, decide what to ground it in, call
the grounded tools (validated modules + cited-formula/fact registry), and compose a
clear, definitive answer. It runs a real multi-step tool-use loop, so it can chain
several tools for one question (e.g. convert real drinks -> standard drinks -> BAC
now -> compare to legal limit).

Hard rules given to the model:
  1. Never state a physiological NUMBER you computed yourself — always get it from a
     tool. Prefer simulation/equations/cited facts for everything.
  2. State any assumption you had to make (e.g. "assuming you go to bed ~2 h after
     drinking").
  3. Give a definitive answer to yes/no questions (e.g. "can I drive").
  4. You MAY use your own general knowledge for an established fact no tool covers,
     but you MUST flag it as lower-confidence.
  5. If a question genuinely can't be grounded or reasoned, say so honestly.

Numbers therefore stay 100% grounded (reliability), while the LLM supplies flexible
understanding + presentation (generality). Falls back to the deterministic router if
no API key/SDK is available.

Setup: pip install google-genai ; put GEMINI_API_KEY in ml/.env
"""

from __future__ import annotations

from dataclasses import asdict

from orchestration.router import Answer, UserProfile, route as deterministic_route
from orchestration.tools import TOOL_SCHEMAS, dispatch

DEFAULT_MODEL = "gemini-2.0-flash"
_MAX_TOOL_ROUNDS = 5

_SYSTEM_INSTRUCTION = (
    "You are Horizon, a personal physiological simulation assistant. Answer the user's "
    "question using the provided tools, which run validated simulations, equations, and "
    "cited medical facts on THIS user's data.\n"
    "RULES:\n"
    "1. Never state a physiological number you worked out yourself. Every number "
    "(BAC, weight, risk, BMI, sleep %, etc.) MUST come from a tool result. Call tools "
    "for everything you can — that is the whole point.\n"
    "2. You may call several tools in sequence for one question (e.g. figure out "
    "standard drinks, then estimate_bac with hours_since_drinking, then judge driving).\n"
    "3. Convert real-world amounts to tool inputs yourself (1 beer or 1 shot of spirits "
    "~= 1 US standard drink; 1 UK pint ~= 1.4; a double ~= 2).\n"
    "4. State any assumption you had to make (timing, activity level, etc.).\n"
    "5. Give a DEFINITIVE answer to yes/no questions (e.g. 'No, you're likely still "
    "over the limit' — grounded in the tool's over_legal_limit_now field).\n"
    "6. If no tool covers a needed established fact, you may use your own knowledge, but "
    "say explicitly that that part is lower-confidence and not from the simulation.\n"
    "7. For a legitimate health question that no simulation/tool covers (e.g. 'possible "
    "reasons my hair is falling out'), DO give a helpful answer: list the main "
    "evidence-based possibilities from general medical knowledge and/or web_search, "
    "personalise using the user's medical info/notes (e.g. a vegan with low iron -> "
    "flag iron/protein deficiency as a relevant candidate, but never the ONLY cause), "
    "and clearly flag the whole answer as general-knowledge / lower-confidence, not a "
    "validated simulation. Suggest seeing a professional for a real diagnosis. Only "
    "fully refuse when the question is nonsensical or has no real basis (e.g. 'does "
    "standing on my head fix my eyesight'). Never reply 'I couldn't produce an answer'.\n"
    "8. SAFETY: for driving/impairment questions, being UNDER the legal BAC limit does "
    "NOT mean it is safe to drive — impairment begins well below the legal limit. Never "
    "tell someone they can 'drive safely' after any significant drinking; advise waiting "
    "until fully sober.\n"
    "9. EVIDENCE: strengthen advice with concrete supporting facts and cite them. Prefer "
    "the modules' own citations and the knowledge base; when a needed FACT (e.g. a "
    "recommended daily intake, a drug interaction, a guideline value) isn't available "
    "from a tool, call web_search and clearly mark those facts as web-sourced.\n"
    "10. Personalise to the user's saved medical info/notes when given — e.g. warn about "
    "an allergy, or tailor advice to a noted condition — and say you're doing so.\n"
    "11. You are a BODY SIMULATOR first. Prefer running simulations over generic advice: "
    "for any question about how the user's body is doing, is trending, or would respond, "
    "call simulate_body_systems (whole-body state) and the specific simulation tools "
    "(estimate_bac, project_weight, estimate_cvd_risk, alcohol_effect_on_sleep) before "
    "falling back to general knowledge. Ground the answer in the simulated numbers.\n"
    "12. MEMORY: you see the prior turns of this conversation — USE them (never re-ask "
    "for facts already given, and never ignore something the user just told you). When "
    "the user reveals a durable fact about themselves (diet, condition, medication, "
    "allergy, habit, goal, a lab value), you MUST call remember_about_user to save it. "
    "If they state they HAVE a condition/diagnosis (e.g. 'I have ADHD', 'I'm diabetic'), "
    "save it to medical.conditions and treat it as real in later answers; if it reads as "
    "self-reported/unconfirmed, save it AND gently note that confirming a formal clinical "
    "diagnosis is worthwhile (self-diagnosis can be wrong). Briefly confirm what you saved.\n"
    "FORMAT: Start with a single **bold** headline line — the key takeaway in under 12 "
    "words (e.g. '**Sober in about 5 hours**'). Then present the details as SHORT bullet "
    "points (markdown '- '), one idea per line — NOT a dense paragraph. When you list "
    "multiple causes, factors, or options, give EACH its own bullet (or a small markdown "
    "table if genuinely tabular), and include ALL of them, not just the first. Bold the "
    "key term in each bullet. Keep every line short and plain-language."
)

_PROFILE_KEYS = ("weight_kg", "height_cm", "age", "sex", "total_chol", "hdl", "sbp",
                 "smoker", "diabetic")

_EVIDENCE_RANK = {"strong": 2, "weak": 1, "none": 0}


def build_gemini_tools():
    """Convert TOOL_SCHEMAS into Gemini function declarations (lazy import)."""
    from google.genai import types
    decls = [types.FunctionDeclaration(name=s["name"], description=s["description"],
                                       parameters=s["parameters"]) for s in TOOL_SCHEMAS]
    return [types.Tool(function_declarations=decls)]


def _merge_profile(args: dict, profile: UserProfile) -> dict:
    """Fill missing tool args from the profile (safety net if the LLM omits one)."""
    prof = asdict(profile)
    merged = dict(args)
    for k in _PROFILE_KEYS:
        if k not in merged and prof.get(k) is not None:
            merged[k] = prof[k]
    return merged


def _aggregate_evidence(results: list[dict]) -> str:
    """Overall evidence = weakest grounded tool used; 'weak' if the model answered from
    its own knowledge (no tools)."""
    levels = [r.get("evidence") for r in results if isinstance(r, dict) and r.get("evidence")]
    if not levels:
        return "weak"  # model-knowledge answer, flagged
    return min(levels, key=lambda e: _EVIDENCE_RANK.get(e, 0))


def route_with_gemini(
    question: str,
    profile: UserProfile,
    *,
    history: list | None = None,
    user_context: str = "",
    client=None,
    model: str = DEFAULT_MODEL,
    fallback_to_rules: bool = True,
) -> Answer:
    """Answer a question via the Gemini tool-use loop. Falls back to the deterministic
    router when the SDK/key is unavailable (so the system always responds)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        if fallback_to_rules:
            return deterministic_route(question, profile)
        raise

    if client is None:
        from orchestration.config import gemini_api_key, gemini_model
        api_key = gemini_api_key()
        if not api_key:
            if fallback_to_rules:
                return deterministic_route(question, profile)
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
        client = genai.Client(api_key=api_key)
        model = gemini_model(model)

    system = _SYSTEM_INSTRUCTION
    if user_context:
        system += (f"\n\nThis user's saved medical info & notes (use to personalise "
                   f"advice and flag risks like allergies): {user_context}")
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=build_gemini_tools(),
        temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = []
    for m in (history or [])[-8:]:   # token cap
        role = "model" if m.get("role") == "assistant" else "user"
        contents.append(types.Content(role=role,
                                      parts=[types.Part.from_text(text=m.get("content", ""))]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(
        text=f"User profile: {asdict(profile)}\n\nQuestion: {question}")]))

    tools_used: list[str] = []
    results: list[dict] = []
    indicators: dict = {}
    citations: list[str] = []

    try:
        return _tool_loop(client, model, config, contents, profile, types,
                          tools_used, results, indicators, citations)
    except Exception as e:
        # Transient LLM/API failure (rate limit, 503, network) must not break the
        # app. Fall back to the deterministic router so the user still gets an answer.
        if fallback_to_rules:
            ans = deterministic_route(question, profile)
            ans.explanation = ([f"(LLM temporarily unavailable — {type(e).__name__}; "
                                f"answered with the deterministic engine.)"]
                               + ans.explanation)
            return ans
        raise


def _tool_loop(client, model, config, contents, profile, types,
               tools_used, results, indicators, citations) -> Answer:
    for _ in range(_MAX_TOOL_ROUNDS):
        resp = client.models.generate_content(model=model, contents=contents, config=config)
        calls = resp.function_calls
        if not calls:
            text = (getattr(resp, "text", None) or "I couldn't produce an answer.").strip()
            return Answer(
                text=text, tool=(tools_used[0] if tools_used else None),
                evidence=_aggregate_evidence(results),
                explanation=_grounding_bullets(tools_used, results),
                indicators=indicators,
                citations=list(dict.fromkeys(citations)),
                raw={"tools_used": tools_used, "results": results},
            )

        contents.append(resp.candidates[0].content)  # model's function-call turn
        response_parts = []
        for fc in calls:
            args = _merge_profile(dict(fc.args), profile)
            try:
                result = dispatch(fc.name, args)
            except Exception as e:  # surface tool errors to the model to recover
                result = {"error": str(e)}
            tools_used.append(fc.name)
            results.append(result)
            indicators[fc.name] = result
            for c in (result.get("citations") or []):
                citations.append(c)
            response_parts.append(types.Part.from_function_response(
                name=fc.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=response_parts))

    # Tool-round budget exhausted — return best-effort grounded summary.
    return Answer(
        text="I gathered the simulations but couldn't finish composing an answer; "
             "please rephrase.",
        tool=(tools_used[0] if tools_used else None),
        evidence=_aggregate_evidence(results),
        explanation=_grounding_bullets(tools_used, results),
        indicators=indicators, citations=list(dict.fromkeys(citations)),
        raw={"tools_used": tools_used, "results": results},
    )


_TOOL_LABEL = {
    "estimate_bac": "blood-alcohol simulation (Widmark pharmacokinetics)",
    "compute_bmi": "BMI formula (exact)",
    "project_weight": "weight-trajectory simulation (energy balance)",
    "estimate_cvd_risk": "cardiovascular risk model (Framingham)",
    "alcohol_effect_on_sleep": "alcohol→sleep cross-system simulation",
}


def _grounding_bullets(tools_used: list[str], results: list[dict]) -> list[str]:
    """Plain-language note of what the answer was grounded in (transparency)."""
    if not tools_used:
        return ["This answer is from the assistant's general knowledge, not a "
                "validated simulation — treat it as lower-confidence."]
    seen = []
    for t in tools_used:
        label = _TOOL_LABEL.get(t, t)
        if label not in seen:
            seen.append(label)
    return [f"Grounded in: {', '.join(seen)}."]
