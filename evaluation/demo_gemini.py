"""Demo: Layer 4 routing via the Gemini API (falls back to rules without a key).

Setup:
    pip install google-genai
    set GEMINI_API_KEY=...        # Windows: setx GEMINI_API_KEY "..."

Then:  python -m evaluation.demo_gemini

Without a key it transparently uses the deterministic router, so this always runs.
"""

from __future__ import annotations

import os

from orchestration.llm_gemini import route_with_gemini
from orchestration.router import UserProfile

PROFILE = UserProfile(weight_kg=80, height_cm=180, age=45, sex="male",
                      total_chol=230, hdl=42, sbp=138, smoker=True)

QUESTIONS = [
    "If I down 5 pints tonight, when am I safe to drive?",
    "Trying to cut to 1800 calories a day for 3 months — what'll my weight do?",
    "Should I be worried about my heart?",
    "How badly will a few beers wreck my sleep tonight?",
]


def main():
    using = "Gemini API" if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) \
        else "deterministic fallback (no GEMINI_API_KEY set)"
    print(f"Routing via: {using}\n")
    for q in QUESTIONS:
        a = route_with_gemini(q, PROFILE)
        print(f"Q: {q}")
        print(f"   [{a.evidence}] via {a.tool}: {a.text}\n")


if __name__ == "__main__":
    main()
