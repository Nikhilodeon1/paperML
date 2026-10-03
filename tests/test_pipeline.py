"""Tests for the cross-system dependency graph (brief 2.5)."""

from __future__ import annotations

import pytest

from modules.hepatic import Drink, EvidenceLevel
from modules.sleep import SleepModel
from pipeline.graph import get_edge, has_edge, run_alcohol_then_sleep


@pytest.fixture(scope="module")
def model():
    return SleepModel(n_train=2000, seed=0).fit()


def test_declared_edge_exists():
    assert has_edge("hepatic", "sleep")
    edge = get_edge("hepatic", "sleep")
    assert edge.evidence is EvidenceLevel.STRONG
    assert edge.citation


def test_undeclared_edge_refused():
    """The guardrail: no edge => refuse, never fabricate a connection (brief 2.5)."""
    assert not has_edge("hepatic", "cardiovascular")
    with pytest.raises(ValueError):
        get_edge("hepatic", "cardiovascular")


def test_chain_more_alcohol_worsens_sleep(model):
    sober = run_alcohol_then_sleep([], weight_kg=80, sex="male", bedtime_hour=2.0,
                                   age=35, sleep_model=model)
    heavy = run_alcohol_then_sleep([Drink.standard(4, hour=0.0)], weight_kg=80,
                                   sex="male", bedtime_hour=2.0, age=35, sleep_model=model)
    assert heavy.alcohol_gkg_bedtime > sober.alcohol_gkg_bedtime
    assert heavy.sleep.metrics["rem_pct"] < sober.sleep.metrics["rem_pct"]
    assert heavy.sleep.metrics["awakenings"] > sober.sleep.metrics["awakenings"]


def test_chain_zero_drinks_zero_burden(model):
    res = run_alcohol_then_sleep([], weight_kg=80, sex="male", bedtime_hour=2.0,
                                 age=35, sleep_model=model)
    assert res.alcohol_gkg_bedtime == 0.0


def test_chain_carries_grounding_label_and_citations(model):
    res = run_alcohol_then_sleep([Drink.standard(2, hour=0.0)], weight_kg=80,
                                 sex="male", bedtime_hour=2.0, age=35, sleep_model=model)
    # The sleep model's grounding label (real Sleep-EDF or synthetic fallback) and its
    # citations must propagate through the chain.
    assert ("sleep-edf" in res.confidence_label.lower()
            or "synthetic" in res.confidence_label.lower())
    assert len(res.citations) >= 3
    assert res.evidence is EvidenceLevel.STRONG
