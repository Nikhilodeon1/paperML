"""Tests for the cardiovascular (Framingham CVD) module."""

from __future__ import annotations

import pytest

from modules.cardiovascular import compute_cvd_risk
from modules.hepatic import EvidenceLevel

HEALTHY = dict(total_chol=180, hdl=55, sbp=118, smoker=False, diabetic=False)


def _risk(**kw):
    return compute_cvd_risk(**{"sex": "male", "age": 50, **HEALTHY, **kw}).risk_10yr


def test_valid_probability():
    assert 0.0 < _risk() < 1.0


def test_monotonic_in_risk_factors():
    assert _risk(age=65) > _risk(age=45)
    assert _risk(sbp=160) > _risk(sbp=110)
    assert _risk(total_chol=280) > _risk(total_chol=160)
    assert _risk(hdl=70) < _risk(hdl=35)
    assert _risk(smoker=True) > _risk(smoker=False)
    assert _risk(diabetic=True) > _risk(diabetic=False)


def test_treated_bp_at_least_untreated():
    t = compute_cvd_risk(age=50, sex="male", **HEALTHY, treated_bp=True).risk_10yr
    u = compute_cvd_risk(age=50, sex="male", **HEALTHY, treated_bp=False).risk_10yr
    assert t >= u


def test_archetype_plausibility():
    healthy40 = compute_cvd_risk(age=40, sex="male", **HEALTHY).risk_10yr
    highrisk70 = compute_cvd_risk(age=70, sex="male", total_chol=260, hdl=35, sbp=165,
                                  smoker=True, diabetic=True).risk_10yr
    assert healthy40 < 0.08
    assert highrisk70 > 0.30


def test_ci_brackets_point():
    r = compute_cvd_risk(age=55, sex="male", total_chol=240, hdl=40, sbp=145, smoker=True)
    assert r.risk_ci[0] <= r.risk_10yr <= r.risk_ci[1]


def test_out_of_range_age_is_weak_evidence():
    assert compute_cvd_risk(age=25, sex="male", **HEALTHY).evidence is EvidenceLevel.WEAK
    assert compute_cvd_risk(age=50, sex="male", **HEALTHY).evidence is EvidenceLevel.STRONG


def test_drivers_ranked_and_reduce_risk():
    r = compute_cvd_risk(age=55, sex="male", total_chol=260, hdl=35, sbp=160,
                         smoker=True, diabetic=False)
    assert r.drivers
    impacts = [v for _, v in r.drivers]
    assert impacts == sorted(impacts, reverse=True)
    assert all(v >= 0 for v in impacts)  # each suggested change reduces risk


def test_invalid_sex_rejected():
    with pytest.raises(ValueError):
        compute_cvd_risk(age=50, sex="other", **HEALTHY)


def test_citation_present():
    r = compute_cvd_risk(age=50, sex="male", **HEALTHY)
    assert r.citations and "D'Agostino" in r.citations[0]
