"""Unit tests for the hepatic module and Layer 1 loader."""

from __future__ import annotations

import numpy as np
import pytest

from knowledge_base import load_system
from modules.hepatic import Drink, EvidenceLevel, PersonalParams, compute_bac


def test_knowledge_base_all_cited():
    """Every Layer 1 parameter must carry a citation (brief 3)."""
    params = load_system("hepatic")
    assert params  # non-empty
    for p in params.values():
        assert p.citation.strip()


def test_no_drinks_returns_none_outcome():
    """No exposure => honest 'none' outcome, never a fabricated number (brief 8)."""
    res = compute_bac([], weight_kg=80, sex="male")
    assert res.evidence is EvidenceLevel.NONE
    assert res.peak_bac == 0.0


def test_strong_evidence_and_citations():
    res = compute_bac([Drink.standard(2)], weight_kg=80, sex="male")
    assert res.evidence is EvidenceLevel.STRONG
    assert len(res.citations) >= 3


def test_more_drinks_higher_peak():
    one = compute_bac([Drink.standard(1)], weight_kg=80, sex="male", seed=1)
    four = compute_bac([Drink.standard(4)], weight_kg=80, sex="male", seed=1)
    assert four.peak_bac > one.peak_bac


def test_female_higher_bac_than_male_same_dose():
    """Lower Widmark r => higher BAC for the same dose/weight."""
    m = compute_bac([Drink.standard(2)], weight_kg=70, sex="male", seed=2)
    f = compute_bac([Drink.standard(2)], weight_kg=70, sex="female", seed=2)
    assert f.peak_bac > m.peak_bac


def test_ci_band_ordering():
    res = compute_bac([Drink.standard(3)], weight_kg=75, sex="male")
    assert np.all(res.bac_lower <= res.bac_median + 1e-9)
    assert np.all(res.bac_median <= res.bac_upper + 1e-9)


def test_personalization_narrows_interval():
    """More observations => tighter posterior => narrower CI (brief 3)."""
    naive = compute_bac([Drink.standard(2)], weight_kg=80, sex="male",
                        personal=PersonalParams(n_observations=0), seed=3)
    seasoned = compute_bac([Drink.standard(2)], weight_kg=80, sex="male",
                           personal=PersonalParams(n_observations=30), seed=3)
    naive_w = float(np.max(naive.bac_upper - naive.bac_lower))
    seasoned_w = float(np.max(seasoned.bac_upper - seasoned.bac_lower))
    assert seasoned_w < naive_w


def test_food_lowers_peak():
    """Food slows absorption (lower ka) => lower, later peak."""
    fasting = compute_bac([Drink.standard(3)], weight_kg=80, sex="male", fed=False, seed=4)
    fed = compute_bac([Drink.standard(3)], weight_kg=80, sex="male", fed=True, seed=4)
    assert fed.peak_bac < fasting.peak_bac


def test_drivers_ranked():
    res = compute_bac([Drink.standard(2)], weight_kg=80, sex="male")
    assert res.drivers
    sens = [s for _, s in res.drivers]
    assert sens == sorted(sens, reverse=True)


def test_invalid_sex_rejected():
    with pytest.raises(ValueError):
        compute_bac([Drink.standard(1)], weight_kg=80, sex="other")
