"""Tests for the sleep module (real Sleep-EDF age baseline + cited modifiers)."""

from __future__ import annotations

import numpy as np
import pytest

from knowledge_base import load_system
from modules.hepatic import EvidenceLevel
from modules.sleep import SleepModel
from modules.sleep_data import FEATURES, TARGETS, generate
from modules.sleep_data_real import is_available


@pytest.fixture(scope="module")
def model():
    return SleepModel(seed=0).fit()


def test_knowledge_base_all_cited():
    for p in load_system("sleep").values():
        assert p.citation.strip()


def test_generate_shapes_and_ranges():
    X, y = generate(n=200, seed=1)
    assert X.shape == (200, len(FEATURES))
    for t in TARGETS:
        assert y[t].shape == (200,)
    assert np.all(y["sleep_efficiency"] <= 99.0)
    assert np.all(y["awakenings"] >= 0.0)


def test_predict_returns_all_metrics_with_intervals(model):
    p = model.predict(age=35)
    assert set(p.metrics) == set(TARGETS)
    for t in TARGETS:
        lo, hi = p.intervals[t]
        assert lo <= p.metrics[t] <= hi
    assert p.evidence is EvidenceLevel.STRONG
    assert p.citations


def test_alcohol_reduces_rem_and_adds_awakenings(model):
    """The alcohol -> sleep cross-system edge must be directionally correct (brief 2.5)."""
    sober = model.predict(age=35, alcohol_gkg_bedtime=0.0)
    drunk = model.predict(age=35, alcohol_gkg_bedtime=0.8)
    assert drunk.metrics["rem_pct"] < sober.metrics["rem_pct"]
    assert drunk.metrics["awakenings"] > sober.metrics["awakenings"]


def test_age_reduces_deep_sleep(model):
    young = model.predict(age=25)
    old = model.predict(age=65)
    assert old.metrics["deep_pct"] < young.metrics["deep_pct"]


def test_drivers_ranked(model):
    p = model.predict(age=40, alcohol_gkg_bedtime=0.5)
    sens = [s for _, s in p.drivers]
    assert sens == sorted(sens, reverse=True)
    assert p.drivers  # named contributors (age baseline + applied modifiers)


def test_caffeine_lowers_efficiency(model):
    base = model.predict(age=35)
    caf = model.predict(age=35, caffeine_mg_afternoon=400)
    assert caf.metrics["sleep_efficiency"] < base.metrics["sleep_efficiency"]


def test_label_reflects_real_data_when_available(model):
    """Honesty: with Sleep-EDF present the label cites real data, not synthetic."""
    p = model.predict(age=30, alcohol_gkg_bedtime=0.3)
    if is_available():
        assert "sleep-edf" in p.confidence_label.lower()
        assert p.evidence is EvidenceLevel.STRONG
    else:
        assert "synthetic" in p.confidence_label.lower()
