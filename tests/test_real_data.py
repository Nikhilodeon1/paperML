"""Tests against the REAL downloaded datasets (skip cleanly if absent).

Kept light: reads hypnogram annotations (not full PSG signals) and a couple of NHANES
columns, so it runs fast. Heavy validation lives in the evaluation/calibrate_* scripts.
"""

from __future__ import annotations

import itertools

import pytest

from modules.sleep_data_real import _iter_nights, is_available as sleep_available
from modules.stress_data_real import is_available as stress_available


@pytest.mark.skipif(not sleep_available(), reason="Sleep-EDF not downloaded")
def test_sleep_edf_one_night_metrics_sane():
    age, m = next(iter(_iter_nights()))
    assert 18 <= age <= 105
    assert 0 <= m["rem_pct"] <= 40
    assert 0 <= m["deep_pct"] <= 50
    assert 20 <= m["sleep_efficiency"] <= 100
    assert 0 <= m["awakenings"] <= 30


@pytest.mark.skipif(not sleep_available(), reason="Sleep-EDF not downloaded")
def test_sleep_edf_multiple_nights_load():
    nights = list(itertools.islice(_iter_nights(), 5))
    assert len(nights) == 5


@pytest.mark.skipif(not stress_available(), reason="Exam-stress not downloaded")
def test_exam_stress_features_and_arousal_signature():
    import numpy as np
    from modules.stress_data_real import load_exam_features
    X = load_exam_features()
    assert X.shape[0] > 100 and X.shape[1] == 3
    hr, rmssd = X[:, 0], X[:, 1]
    m = ~np.isnan(rmssd)
    # Artifact-rejected RMSSD stays physiological; HR-HRV arousal signature is negative.
    assert rmssd[m].max() <= 150.0
    assert np.corrcoef(hr[m], rmssd[m])[0, 1] < 0


def test_exam_stress_supervised_entry_raises():
    """No stress label in this dataset -> supervised loader must refuse, not fabricate."""
    from modules.stress_data_real import load_exam_stress
    with pytest.raises(NotImplementedError):
        load_exam_stress()


def test_nhanes_loads_and_merges():
    import pathlib
    cdc = pathlib.Path(__file__).resolve().parents[1].parent / "data" / "cdc"
    if not (cdc / "DEMO_J.xpt").exists():
        pytest.skip("NHANES not downloaded")
    import pandas as pd
    demo = pd.read_sas(cdc / "DEMO_J.xpt", format="xport")
    assert {"SEQN", "RIAGENDR", "RIDAGEYR"} <= set(demo.columns)
    assert len(demo) > 5000
