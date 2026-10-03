"""Tests for the anthropometric module + BAC legal-limit + metabolic age validity
(the fixes behind the four reported bugs)."""

from __future__ import annotations

import pytest

from modules.anthropometric import compute_bmi
from modules.hepatic import EvidenceLevel
from modules.metabolic import project_weight
from orchestration.tools import dispatch


# --- BMI (bug 3: "whats my bmi" used to return no-model) --------------------

def test_bmi_exact_value_and_category():
    r = compute_bmi(weight_kg=80, height_cm=180)
    assert r.bmi == pytest.approx(24.7, abs=0.1)
    assert r.category == "normal weight"
    assert r.evidence is EvidenceLevel.STRONG


def test_bmi_categories():
    assert compute_bmi(50, 180).category == "underweight"
    assert compute_bmi(85, 180).category == "overweight"
    assert compute_bmi(110, 180).category == "obese"


def test_bmi_under_18_is_weak_with_note():
    r = compute_bmi(weight_kg=86, height_cm=190, age=17)
    assert r.evidence is EvidenceLevel.WEAK
    assert "18" in r.note


def test_bmi_healthy_range_and_citation():
    r = compute_bmi(80, 180)
    lo, hi = r.healthy_weight_kg_range
    assert lo < 80 < hi
    assert any("WHO" in c for c in r.citations)


# --- BAC legal limit (bug 1: "can I drive after 3 shots an hour ago") -------

def test_bac_reports_legal_limit_status_with_elapsed_time():
    r = dispatch("estimate_bac", dict(standard_drinks=3, weight_kg=80, sex="male",
                                      hours_since_drinking=1.0))
    assert "bac_now" in r and "over_legal_limit_now" in r
    assert r["legal_driving_limit"] == 0.08
    assert isinstance(r["over_legal_limit_now"], bool)


def test_bac_over_limit_when_heavy_recent_drinking():
    r = dispatch("estimate_bac", dict(standard_drinks=8, weight_kg=70, sex="male",
                                      hours_since_drinking=0.5))
    assert r["over_legal_limit_now"] is True
    assert r["hours_until_under_legal_limit"] > 0


# --- Metabolic age validity (bug 4: 17yo weight projection "completely wrong") --

def test_adolescent_weight_projection_is_weak_evidence():
    teen = project_weight(86, 190, 17, "male", daily_intake_kcal=2000, horizon_days=365)
    adult = project_weight(86, 190, 35, "male", daily_intake_kcal=2000, horizon_days=90)
    assert teen.evidence is EvidenceLevel.WEAK
    assert "18" in teen.confidence_label
    assert adult.evidence is EvidenceLevel.STRONG
