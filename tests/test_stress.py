"""Tests for the stress module (supervised WESAD classifier; HR-composite fallback)."""

from __future__ import annotations

import numpy as np
import pytest

from knowledge_base import load_system
from modules.hepatic import EvidenceLevel
from modules.stress import StressModel
from modules.stress_data import FEATURES, TARGET, generate
from modules.stress_data_real import is_available as exam_available
from modules.stress_data_wesad import is_available as wesad_available


@pytest.fixture(scope="module")
def model():
    return StressModel().fit()


def test_knowledge_base_all_cited():
    for p in load_system("stress").values():
        assert p.citation.strip()


def test_generate_shapes_and_range():
    # Synthetic generator retained only as a last-resort reference.
    X, y = generate(n=200, seed=1)
    assert X.shape == (200, len(FEATURES))
    assert np.all((y[TARGET] >= 0) & (y[TARGET] <= 100))


def test_index_in_range_and_within_interval(model):
    p = model.predict(heart_rate=80, rmssd=30, eda=2.0)
    assert 0.0 <= p.stress_index <= 100.0
    lo, hi = p.interval
    assert lo <= p.stress_index <= hi


def test_higher_arousal_higher_stress(model):
    calm = model.predict(heart_rate=63, rmssd=55, eda=1.0)
    tense = model.predict(heart_rate=105, rmssd=20, eda=5.0)
    assert tense.stress_index > calm.stress_index
    assert tense.category in ("moderate", "high")


def test_drivers_ranked_and_hr_or_eda_lead(model):
    p = model.predict(heart_rate=95, rmssd=30, eda=3.0)
    sens = [s for _, s in p.drivers]
    assert sens == sorted(sens, reverse=True)                 # ranked
    top2 = {name for name, _ in p.drivers[:2]}
    assert top2 & {"heart_rate", "eda"}                       # a reliable signal leads


def test_context_reports_raw_signals(model):
    p = model.predict(heart_rate=88, rmssd=33, eda=3.1)
    assert p.context["heart_rate"] == 88 and p.context["rmssd"] == 33


@pytest.mark.skipif(not wesad_available(), reason="WESAD not downloaded")
def test_supervised_when_wesad_present(model):
    """With WESAD present the model is supervised and labelled strong evidence."""
    assert model._supervised
    p = model.predict(heart_rate=90, rmssd=30, eda=4.0)
    assert p.evidence is EvidenceLevel.STRONG
    assert "wesad" in p.confidence_label.lower()


def test_loaders_availability_are_bool():
    assert isinstance(exam_available(), bool)
    assert isinstance(wesad_available(), bool)
