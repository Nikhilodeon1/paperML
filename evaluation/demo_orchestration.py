"""Demo: Layer 4 routing a natural-language question to the right module(s).

Deterministic router (no LLM) — shows the end-to-end interface. Run:
    python -m evaluation.demo_orchestration
"""

from __future__ import annotations

from orchestration.router import UserProfile, route

PROFILE = UserProfile(weight_kg=80, height_cm=180, age=45, sex="male",
                      total_chol=230, hdl=42, sbp=138, smoker=True, diabetic=False)

QUESTIONS = [
    "If I have 4 beers tonight, when will I be sober?",
    "How will 3 drinks before bed affect my sleep?",
    "If I eat 2000 kcal a day for 6 months, what happens to my weight?",
    "What's my heart disease risk?",
    "Will standing on my head improve my eyesight?",   # third outcome: no model
]


def main():
    print(f"User: {PROFILE.sex}, {PROFILE.age}y, {PROFILE.weight_kg}kg\n")
    for q in QUESTIONS:
        a = route(q, PROFILE)
        print(f"Q: {q}")
        print(f"   [{a.evidence}] {a.text}\n")


if __name__ == "__main__":
    main()
