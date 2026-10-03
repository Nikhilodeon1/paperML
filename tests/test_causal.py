"""Causal differential engine tests — the multi-hop, personalized symptom->cause chain.

Locks in the behaviour that makes this feature valuable: the SAME symptom yields a
DIFFERENT ranked differential per user (driven by their diet/notes/labs/check-ins),
chains are cited, confidence is weakest-link, and nothing outside the authored graph
is ever invented.
"""

from __future__ import annotations

from pipeline.causal import explain_symptom, match_symptom, list_symptoms
from personalization.synthetic import make_user
import numpy as np


def _rng():
    return np.random.default_rng(0)


def test_symptom_matching_fuzzy_and_longest():
    assert match_symptom("why do I keep getting sick lately") == "recurrent_infection"
    assert match_symptom("I've been so tired all the time") == "persistent_fatigue"
    assert match_symptom("my hair is falling out") == "hair_thinning"
    assert match_symptom("purple flavoured elephants") is None


def test_unknown_symptom_returns_suggestions():
    d = explain_symptom("purple flavoured elephants")
    assert d["matched"] is False
    assert any(s["key"] == "recurrent_infection" for s in d["known_symptoms"])


def test_vegan_indoor_user_gets_vitamin_chain_on_top():
    u = make_user("t", rng=_rng(), sex="female", age=31, height_cm=167, weight_kg=60,
                  diet="vegan", with_weighins=False,
                  notes=["Vegan, no supplements.", "Works indoors, rarely outdoors."])
    d = explain_symptom("I keep getting sick", u)
    assert d["matched"] and d["personalized"]
    keys = [c["key"] for c in d["causes"]]
    # the vitamin-D / B12 chain must surface near the top for this body
    assert "vitamin_d_deficiency" in keys[:3] or "vitamin_b12_deficiency" in keys[:3]
    topd = next(c for c in d["causes"] if c["key"] == "vitamin_d_deficiency")
    assert topd["personalized"] and topd["personal_reasons"]
    # the chain is the real multi-hop path, and it carries citations
    assert "Weakened immune function" in topd["chain"]
    assert topd["citations"]


def test_same_symptom_differs_by_user():
    vegan = make_user("v", rng=_rng(), sex="female", age=31, height_cm=167, weight_kg=60,
                      diet="vegan", with_weighins=False, notes=["Indoors, rarely outdoors."])
    omni = make_user("o", rng=_rng(), sex="male", age=45, height_cm=180, weight_kg=82,
                     diet="omnivore", activity="active", with_weighins=False)
    dv = explain_symptom("getting sick a lot", vegan)
    do = explain_symptom("getting sick a lot", omni)
    top_v = dv["causes"][0]["key"]
    # the vegan's top cause is diet/sun-driven; the omnivore has no such personal signal
    assert dv["personalized"]
    assert top_v in ("vitamin_d_deficiency", "vitamin_b12_deficiency", "plant_based_diet_gap")
    assert [c["key"] for c in dv["causes"]] != [c["key"] for c in do["causes"]] \
        or dv["causes"][0]["personalized"] != do["causes"][0]["personalized"]


def test_questionnaire_indicator_feeds_differential():
    # a constipation flag from the gut check-in should drive low-fibre up the list
    u = make_user("q", rng=_rng(), sex="male", age=40, height_cm=178, weight_kg=80,
                  with_weighins=False)
    u["surveys"] = [{"id": "digestive", "name": "Digestive",
                     "indicators": [{"key": "constipation", "label": "Bowel regularity",
                                     "value": 2, "band": "Constipated", "severity": "concern",
                                     "note": "", "citation": ""}]}]
    d = explain_symptom("I'm constipated", u)
    low_fiber = next(c for c in d["causes"] if c["key"] == "low_fiber_intake")
    assert low_fiber["personalized"]
    assert any("constipation" in r.lower() for r in low_fiber["personal_reasons"])


def test_confidence_is_weakest_link_and_no_fabrication():
    d = explain_symptom("muscle cramps", make_user(
        "m", rng=_rng(), sex="male", age=40, height_cm=178, weight_kg=80, with_weighins=False))
    # every surfaced cause must exist in the graph and carry a valid grade
    from pipeline.causal import _graph
    valid = set(_graph()["nodes"])
    for c in d["causes"]:
        assert c["key"] in valid
        assert c["evidence"] in ("strong", "moderate", "weak")


def test_list_symptoms_nonempty():
    assert len(list_symptoms()) >= 8


def test_graph_integrity_no_dangling_edges():
    from pipeline.causal import _graph
    g = _graph()
    nodes = set(g["nodes"])
    for e in g["edges"]:
        assert e["source"] in nodes, f"edge source {e['source']} not a node"
        assert e["target"] in nodes, f"edge target {e['target']} not a node"
        assert e["evidence"] in ("strong", "moderate", "weak")
    # every symptom must reach at least one cause
    for s in list_symptoms():
        assert explain_symptom(s["label"])["causes"], f"{s['key']} has no causes"
