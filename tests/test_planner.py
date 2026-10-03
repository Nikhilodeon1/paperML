"""Planner (plan -> execute -> synthesize) tests — deterministic path.

The deterministic planner + synthesizer must fully work with NO LLM, so these are
reproducible. They lock in that questions decompose into the right grounded capabilities,
that every capability executes and is graded, and that a complete answer is always
produced (the robust floor the LLM refines on top of).
"""

from __future__ import annotations

import pytest

from orchestration.planner import (plan_and_answer, deterministic_plan, execute,
                                   CAPABILITIES, Step)
from orchestration.router import UserProfile


@pytest.fixture()
def user(tmp_path, monkeypatch):
    import personalization.user_store as us
    monkeypatch.setattr(us, "USER_DIR", tmp_path)
    u = us.seed_demo("u_plan")
    return u


def _prof(u):
    from personalization import user_store as us
    return us.to_profile(u)


def test_deterministic_plan_decomposes_multipart(user):
    steps = deterministic_plan("what is my bmi and my heart risk", user)
    caps = {s.capability for s in steps}
    assert "bmi" in caps and "heart_risk" in caps


def test_symptom_question_plans_differential(user):
    steps = deterministic_plan("why do I keep getting sick", user)
    assert any(s.capability == "symptom_causes" for s in steps)


def test_health_question_grounds_in_body_state(user):
    steps = deterministic_plan("what do you know about my health and what am I at risk for", user)
    assert steps[0].capability == "body_state"


def test_alcohol_amount_is_extracted(user):
    steps = deterministic_plan("if I have 4 beers can I drive", user)
    bac = next(s for s in steps if s.capability == "alcohol_bac")
    assert bac.args["standard_drinks"] == 4.0


def test_execute_grounds_every_step(user):
    steps = deterministic_plan("what is my bmi and heart risk", user)
    results = execute(steps, user)
    assert results and all(r.evidence in ("strong", "moderate", "weak", "none") for r in results)
    bmi = next(r for r in results if r.capability == "bmi")
    assert "bmi" in bmi.result


def test_plan_and_answer_deterministic_is_complete(user):
    d = plan_and_answer("what's my BMI, heart risk, and why might I keep getting sick",
                        _prof(user), user, use_llm=False)
    assert d["planner"] == "deterministic"
    assert d["synthesizer"] == "deterministic"
    caps = {s["capability"] for s in d["plan"]}
    assert {"bmi", "heart_risk", "symptom_causes"} <= caps
    assert d["answer"] and "**" in d["answer"]           # structured headline
    assert d["evidence"] in ("strong", "moderate", "weak")


def test_novel_question_still_answers(user):
    # a question with no dedicated capability must still ground in body state, not error
    d = plan_and_answer("is my body basically doing okay overall?", _prof(user), user, use_llm=False)
    assert any(s["capability"] == "body_state" for s in d["steps"])
    assert d["answer"]


def test_capabilities_all_runnable(user):
    # every registered capability executes without raising (grounded on the demo user)
    argmap = {"symptom_causes": {"symptom": "fatigue"}, "alcohol_bac": {"standard_drinks": 2},
              "simulate": {"duration_min": 60, "meals": [{"t_min": 0, "carbs_g": 40}]}}
    for name in CAPABILITIES:
        if name == "fact_lookup":
            continue                                     # network-dependent; skip offline
        results = execute([Step(name, "test", argmap.get(name, {}))], user)
        assert results[0].result is not None
