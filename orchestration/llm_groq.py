"""Groq-backed orchestrator (OpenAI-compatible) — e.g. openai/gpt-oss-20b.

Same design and guarantees as the Gemini orchestrator (numbers only from grounded
tools; states assumptions; definitive answers; low-confidence flag for own-knowledge)
but over Groq's OpenAI-compatible chat-completions API, so it works with gpt-oss and
other open models Groq serves.

Reuses the shared system prompt + helpers from llm_gemini (plain functions, no Google
import at load time). Falls back to the deterministic router if the SDK/key is missing.

Setup: pip install openai ; put GROQ_API_KEY (+ optional GROQ_MODEL) in ml/.env
"""

from __future__ import annotations

import json
from dataclasses import asdict

from orchestration.llm_gemini import (
    _SYSTEM_INSTRUCTION, _aggregate_evidence, _grounding_bullets, _merge_profile,
)
from orchestration.router import Answer, UserProfile, route as deterministic_route
from orchestration.tools import TOOL_SCHEMAS, dispatch

DEFAULT_MODEL = "openai/gpt-oss-20b"
_BASE_URL = "https://api.groq.com/openai/v1"
_MAX_TOOL_ROUNDS = 5


def _openai_tools():
    """TOOL_SCHEMAS -> OpenAI tools format (Groq is OpenAI-compatible)."""
    return [{"type": "function", "function": s} for s in TOOL_SCHEMAS]


def route_with_groq(
    question: str,
    profile: UserProfile,
    *,
    history: list | None = None,
    user_context: str = "",
    client=None,
    model: str = DEFAULT_MODEL,
    fallback_to_rules: bool = True,
) -> Answer:
    try:
        from openai import OpenAI
    except ImportError:
        if fallback_to_rules:
            return deterministic_route(question, profile)
        raise

    if client is None:
        from orchestration.config import groq_api_key, groq_model
        api_key = groq_api_key()
        if not api_key:
            if fallback_to_rules:
                return deterministic_route(question, profile)
            raise RuntimeError("Set GROQ_API_KEY in ml/.env")
        client = OpenAI(api_key=api_key, base_url=_BASE_URL)
        model = groq_model(model)

    system = _SYSTEM_INSTRUCTION
    if user_context:
        system += (f"\n\nThis user's saved medical info & notes (use to personalise "
                   f"advice and flag risks like allergies): {user_context}")
    messages = [{"role": "system", "content": system}]
    if history:
        messages += history[-8:]   # token cap: keep the last few turns only
    messages.append({"role": "user",
                     "content": f"User profile: {asdict(profile)}\n\nQuestion: {question}"})

    try:
        return _tool_loop(client, model, messages, profile)
    except Exception as e:
        if fallback_to_rules:
            ans = deterministic_route(question, profile)
            ans.explanation = ([f"(Groq LLM unavailable — {type(e).__name__}; answered "
                                f"with the deterministic engine.)"] + ans.explanation)
            return ans
        raise


def _tool_loop(client, model, messages, profile) -> Answer:
    tools_used, results, indicators, citations = [], [], {}, []
    trace: list[dict] = []   # ordered record of how the model reasoned (for the ML lab)
    nudged = False           # whether we've re-prompted for a final answer once

    def _finish(text, tool_val):
        return Answer(
            text=text, tool=tool_val, evidence=_aggregate_evidence(results),
            explanation=_grounding_bullets(tools_used, results),
            indicators=indicators, citations=list(dict.fromkeys(citations)),
            raw={"backend": "groq", "tools_used": tools_used, "results": results,
                 "trace": trace, "rounds": len([t for t in trace if t["kind"] == "think"])})

    for round_i in range(_MAX_TOOL_ROUNDS):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=_openai_tools(),
                tool_choice="auto", temperature=0, reasoning_effort="medium")
        except TypeError:
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=_openai_tools(),
                tool_choice="auto", temperature=0)
        msg = resp.choices[0].message

        # Capture the model's planning/reasoning for this round, if the model exposes it.
        think = (getattr(msg, "reasoning", None) or msg.content or "").strip()
        if think and (msg.tool_calls or round_i == 0):
            trace.append({"kind": "think", "round": round_i + 1, "text": think})

        if not msg.tool_calls:
            text = (msg.content or "").strip()
            # gpt-oss sometimes returns EMPTY final content (everything went to the
            # reasoning channel). Re-prompt once for a clean final answer rather than
            # emitting a dead-end message.
            if not text and not nudged:
                nudged = True
                messages.append({"role": "assistant", "content": think or ""})
                messages.append({"role": "user", "content":
                    "Now write your final answer for the user in clear, plain language, "
                    "using everything above. Use short bullet points."})
                continue
            if not text:
                text = ("I've gathered the relevant information but had trouble phrasing a "
                        "final answer — could you rephrase, or ask about one thing at a time?")
            trace.append({"kind": "answer", "text": text})
            return _finish(text, tools_used[0] if tools_used else None)

        # Echo the assistant tool-call turn, then answer each call.
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = _merge_profile(json.loads(tc.function.arguments or "{}"), profile)
                result = dispatch(tc.function.name, args)
            except Exception as e:
                result = {"error": str(e)}
            tools_used.append(tc.function.name)
            results.append(result)
            indicators[tc.function.name] = result
            for c in (result.get("citations") or []):
                citations.append(c)
            trace.append({"kind": "tool", "name": tc.function.name, "args": args,
                          "result": result, "evidence": result.get("evidence")})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result)})

    text = ("I gathered the simulations but couldn't finish composing an answer; "
            "please rephrase.")
    trace.append({"kind": "answer", "text": text})
    return _finish(text, tools_used[0] if tools_used else None)
