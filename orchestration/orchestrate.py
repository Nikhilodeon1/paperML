"""Unified LLM-orchestration entry point — picks the available backend.

Priority: Groq (if GROQ_API_KEY set) -> Gemini (if GEMINI/GOOGLE key set) ->
deterministic router. Numbers are grounded identically in every backend; only the
reasoning/phrasing model differs. The API and any caller should use `route_llm`
rather than importing a specific backend.
"""

from __future__ import annotations

from orchestration.config import gemini_api_key, groq_api_key
from orchestration.router import Answer, UserProfile, route as deterministic_route


def active_backend() -> str:
    if groq_api_key():
        return "groq"
    if gemini_api_key():
        return "gemini"
    return "deterministic"


def route_llm(question: str, profile: UserProfile, *, user_context: str = "",
              history: list | None = None) -> Answer:
    backend = active_backend()
    if backend == "groq":
        from orchestration.llm_groq import route_with_groq
        return route_with_groq(question, profile, history=history, user_context=user_context)
    if backend == "gemini":
        from orchestration.llm_gemini import route_with_gemini
        return route_with_gemini(question, profile, history=history, user_context=user_context)
    return deterministic_route(question, profile)


def chat_llm(messages: list, profile: UserProfile, *, user_context: str = "") -> Answer:
    """Conversational turn: the latest user message answered with tool-grounded numbers,
    aware of the prior turns. The LLM does all routing; deterministic fallback handles
    only the latest message (no memory) when no backend is available."""
    if not messages:
        return deterministic_route("", profile)
    question = messages[-1].get("content", "")
    history = messages[:-1]
    return route_llm(question, profile, user_context=user_context, history=history)
