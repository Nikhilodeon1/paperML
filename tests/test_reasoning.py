"""T3 reasoning — lab interpretation from the user's record, provenance-tagged."""

from __future__ import annotations

from orchestration.planner import deterministic_plan
from personalization.reasoning import (demographic_priors, interpret_labs, reason_about)


def _user(labs=None, **med):
    m = {"allergies": [], "conditions": [], "medications": [], "labs": labs or {}}
    m.update(med)
    return {"profile": {"age": 40, "sex": "male", "weight_kg": 80, "height_cm": 178},
            "medical": m}


def test_low_wbc_is_flagged_with_immune_implication():
    facts = interpret_labs(_user({"wbc": 3.2}))
    assert facts and "immune" in facts[0]["systems"]
    assert "3.2" in facts[0]["fact"] and facts[0]["source"] == "your bloodwork"


def test_in_range_lab_produces_no_fact():
    assert interpret_labs(_user({"wbc": 6.5})) == []      # normal -> nothing to flag


def test_alias_resolves_to_marker():
    facts = interpret_labs(_user({"white blood cell": 3.0}))   # alias, not canonical id
    assert facts and "White blood cell count" in facts[0]["fact"]


def test_high_hba1c_touches_multiple_systems():
    facts = interpret_labs(_user({"hba1c": 6.6}))
    assert facts and {"metabolic", "immune"} <= set(facts[0]["systems"])


def test_everything_is_provenance_tagged():
    r = reason_about("why do I keep getting sick",
                     _user({"wbc": 3.4, "vitamin_d": 15}, conditions=["asthma"]))
    assert r["facts"] and all(f["source"] in
        ("your bloodwork", "your history", "general pattern (age/sex)") for f in r["facts"])
    assert "not a diagnosis" in r["disclaimer"].lower()


def test_reason_links_labs_to_simulated_systems():
    r = reason_about("why am I getting sick", _user({"vitamin_d": 12}))
    # vitamin D touches sleep/metabolic which the engine DOES simulate -> flagged for the caller
    assert set(r["simulated_systems_touched"]) & {"sleep", "metabolic"}


def test_demographic_prior_for_menopause():
    facts = demographic_priors({"profile": {"age": 51, "sex": "female"}})
    assert facts and facts[0]["evidence"] == "weak" and "menopause" in facts[0]["fact"].lower()


def test_no_labs_no_history_gives_nothing():
    r = reason_about("why do I get sick", _user())
    assert r["facts"] == [] and r["evidence"] == "none"


def test_planner_routes_immune_and_lab_questions_to_t3():
    u = _user({"wbc": 3.0})
    for q in ("why do I keep getting sick", "what does my low white blood cell count mean",
              "should I worry about my thyroid"):
        caps = {s.capability for s in deterministic_plan(q, u)}
        assert "health_reasoning" in caps, q
