"""Inference-questionnaire scoring + storage tests.

These lock in the *inference* behaviour (answers -> clinically-anchored indicators), not
just that answers round-trip: the whole point is that "3 bowel movements a week" comes
back as a flagged constipation signal, and that the flag reaches the LLM context and a
system tile.
"""

from __future__ import annotations

import numpy as np
import pytest

from personalization.questionnaires import (
    list_questionnaires, get_questionnaire, score, indicator_facts)


def _ind(scored, key):
    return next(i for i in scored["indicators"] if i["key"] == key)


def test_catalogue_and_definitions_load():
    cat = {q["id"] for q in list_questionnaires()}
    assert {"digestive", "hydration", "stress_pss4"} <= cat
    q = get_questionnaire("digestive")
    assert q and len(q["questions"]) == 3


@pytest.mark.parametrize("freq,severity,band_word", [
    (2, "concern", "Constipated"),
    (3, "watch", "Low-normal"),
    (10, "ok", "Regular"),
    (25, "watch", "Frequent"),
])
def test_bowel_regularity_thresholds(freq, severity, band_word):
    s = score("digestive", {"bowel_freq_per_week": freq})
    reg = _ind(s, "constipation")
    assert reg["severity"] == severity
    assert band_word.lower() in reg["band"].lower()


def test_bristol_stool_consistency():
    assert _ind(score("digestive", {"bristol_type": "t1"}), "stool_consistency")["severity"] == "watch"
    assert _ind(score("digestive", {"bristol_type": "t4"}), "stool_consistency")["severity"] == "ok"
    assert _ind(score("digestive", {"bristol_type": "t7"}), "stool_consistency")["severity"] == "watch"


def test_pss4_scale_sum_and_bands():
    # all "very often" on stress items, "never" on the (reverse-scored) positive items = max 16
    hi = score("stress_pss4", {"pss_unable_control": "4", "pss_confident": "0",
                               "pss_going_your_way": "0", "pss_difficulties_piling": "4"})
    ps = _ind(hi, "perceived_stress")
    assert ps["value"] == 16 and ps["severity"] == "concern"
    # calm answers -> low
    lo = score("stress_pss4", {"pss_unable_control": "0", "pss_confident": "4",
                               "pss_going_your_way": "4", "pss_difficulties_piling": "0"})
    assert _ind(lo, "perceived_stress")["value"] == 0
    assert _ind(lo, "perceived_stress")["severity"] == "ok"


def test_missing_answers_are_skipped_not_guessed():
    # only one PSS item answered -> the sum indicator is skipped, not fabricated
    s = score("stress_pss4", {"pss_unable_control": "4"})
    assert s["indicators"] == []


def test_hydration_dark_urine_flagged():
    assert _ind(score("hydration", {"urine_color": "c1"}), "hydration_status")["severity"] == "ok"
    assert _ind(score("hydration", {"urine_color": "c7"}), "hydration_status")["severity"] == "concern"


def test_indicator_facts_only_surface_notable():
    survey = {"name": "Digestive & Gut Health",
              "indicators": score("digestive", {"bowel_freq_per_week": 2})["indicators"]}
    facts = indicator_facts(survey)
    assert any("constipated" in f.lower() for f in facts)
    # a healthy answer produces no noise
    ok_survey = {"name": "x", "indicators": score("digestive", {"bowel_freq_per_week": 10})["indicators"]}
    assert indicator_facts(ok_survey) == []


def test_submit_flows_to_store_context_and_tile(tmp_path, monkeypatch):
    import personalization.user_store as us
    monkeypatch.setattr(us, "USER_DIR", tmp_path)
    us.create("u_q", sex="male", age=40, height_cm=178, weight_kg=82)

    scored = us.submit_questionnaire("u_q", "digestive", {"bowel_freq_per_week": 2,
                                                          "bristol_type": "t1", "straining": "often"})
    assert _ind(scored, "constipation")["severity"] == "concern"

    u = us.load("u_q")
    assert u["surveys"] and u["surveys"][-1]["id"] == "digestive"
    assert u["advice"] is None and u["systems"] is None      # caches cleared

    # flows into the LLM context summary...
    assert "constipated" in us.context_summary(u).lower()

    # ...and into a Digestive system tile flagged 'watch'
    from personalization.systems import compute_systems
    tiles = {t["key"]: t for t in compute_systems(u)}
    assert "digestive" in tiles
    assert tiles["digestive"]["status"] == "watch"
    assert "constipat" in tiles["digestive"]["headline"].lower()
