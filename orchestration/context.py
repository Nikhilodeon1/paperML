"""Request-scoped user context for grounded, personalized tools.

Some tools (e.g. the causal-differential) need the FULL user record — diet, notes, labs,
questionnaire indicators — not just the numeric `UserProfile`. Threading the whole user
through every orchestration signature would be noisy, so the API sets it here for the
duration of a request and the relevant tool handler reads it. Deterministic and cheap;
resets after each call so nothing leaks between users.
"""

from __future__ import annotations

from contextvars import ContextVar

current_user: ContextVar[dict | None] = ContextVar("current_user", default=None)
