"""Unit tests for the metabolic module."""

from __future__ import annotations

import numpy as np
import pytest

from knowledge_base import load_system
from modules.hepatic import EvidenceLevel
from modules.metabolic import (
    MetabolicPersonalParams,
    _rmr_mifflin,
    project_weight,
)


def test_knowledge_base_all_cited():
    for p in load_system("metabolic").values():
        assert p.citation.strip()


def test_rmr_canonical_value():
    kb = load_system("metabolic")
    # 30 y, 80 kg, 180 cm male => 1780 kcal/day.
    assert _rmr_mifflin(80, 180, 30, "male", kb) == pytest.approx(1780.0, abs=0.5)


def test_surplus_gains_deficit_loses():
    kb = load_system("metabolic")
    maint = kb["pal_sedentary"].value * _rmr_mifflin(80, 180, 30, "male", kb)
    gain = project_weight(80, 180, 30, "male", daily_intake_kcal=maint + 500,
                          horizon_days=60, n_samples=1)
    loss = project_weight(80, 180, 30, "male", daily_intake_kcal=maint - 500,
                          horizon_days=60, n_samples=1)
    assert gain.final_weight > 80
    assert loss.final_weight < 80


def test_ci_band_ordering():
    res = project_weight(80, 180, 30, "male", daily_intake_kcal=2000, horizon_days=90)
    assert np.all(res.weight_lower <= res.weight_median + 1e-9)
    assert np.all(res.weight_median <= res.weight_upper + 1e-9)


def test_ci_widens_with_horizon():
    a = project_weight(80, 180, 30, "male", daily_intake_kcal=1800, horizon_days=30)
    b = project_weight(80, 180, 30, "male", daily_intake_kcal=1800, horizon_days=365)
    wa = a.final_weight_ci[1] - a.final_weight_ci[0]
    wb = b.final_weight_ci[1] - b.final_weight_ci[0]
    assert wb > wa


def test_long_horizon_downgraded_to_weak():
    short = project_weight(80, 180, 30, "male", daily_intake_kcal=1800, horizon_days=90)
    long = project_weight(80, 180, 30, "male", daily_intake_kcal=1800, horizon_days=365)
    assert short.evidence is EvidenceLevel.STRONG
    assert long.evidence is EvidenceLevel.WEAK


def test_personalization_narrows_interval():
    naive = project_weight(80, 180, 30, "male", daily_intake_kcal=1800, horizon_days=90,
                           personal=MetabolicPersonalParams(n_observations=0))
    seasoned = project_weight(80, 180, 30, "male", daily_intake_kcal=1800, horizon_days=90,
                              personal=MetabolicPersonalParams(n_observations=30))
    wn = float(np.max(naive.weight_upper - naive.weight_lower))
    ws = float(np.max(seasoned.weight_upper - seasoned.weight_lower))
    assert ws < wn


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        project_weight(80, 180, 30, "other", daily_intake_kcal=2000)
    with pytest.raises(ValueError):
        project_weight(80, 180, 30, "male", daily_intake_kcal=2000, activity="extreme")
