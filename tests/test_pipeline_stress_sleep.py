"""Tests for the stress -> sleep cross-system edge."""

from __future__ import annotations

import pytest

from modules.hepatic import EvidenceLevel
from modules.sleep import SleepModel
from modules.stress import StressModel
from pipeline.graph import get_edge, has_edge, run_stress_then_sleep


@pytest.fixture(scope="module")
def models():
    return SleepModel(n_train=2000, seed=0).fit(), StressModel(n_train=2000, seed=0).fit()


def test_edge_declared_and_graded_weak():
    assert has_edge("stress", "sleep")
    assert get_edge("stress", "sleep").evidence is EvidenceLevel.WEAK  # moderate/low conf


def test_higher_stress_lowers_sleep_efficiency(models):
    sleep_m, stress_m = models
    calm = run_stress_then_sleep(heart_rate=60, rmssd=60, eda=1.5, age=35,
                                 sleep_model=sleep_m, stress_model=stress_m)
    tense = run_stress_then_sleep(heart_rate=105, rmssd=18, eda=8.0, age=35,
                                  sleep_model=sleep_m, stress_model=stress_m)
    assert tense.stress_index > calm.stress_index
    assert tense.adjusted_efficiency < calm.adjusted_efficiency
    # The modifier only reduces (or leaves) efficiency, never inflates it.
    assert tense.adjusted_efficiency <= tense.baseline_efficiency + 1e-9


def test_chain_is_low_confidence(models):
    sleep_m, stress_m = models
    res = run_stress_then_sleep(heart_rate=90, rmssd=25, eda=5.0, age=40,
                                sleep_model=sleep_m, stress_model=stress_m)
    assert res.evidence is EvidenceLevel.WEAK
    assert "low confidence" in res.confidence_label.lower()
    assert res.citations
