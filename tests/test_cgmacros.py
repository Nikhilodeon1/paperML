"""CGMacros loader + real-human validation plumbing.

Parsing is tested on a tiny in-format CSV so CI never needs the multi-GB dataset. The real
dataset run is `python -m evaluation.cgmacros`; a marker test runs it only if the zip exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.cgmacros import (_ZIP, _pearson, load_bio, load_subject, predict_meal_iauc,
                                 subjects, validate_subject)

# One synthetic subject in the real CGMacros column layout: a meal at 12:00 (60 g carbs) with a
# rising glucose curve after it. Per-minute rows, Dexcom populated.
_HEADER = ("Unnamed: 0,Timestamp,Libre GL,Dexcom GL,HR,Calories (Activity),METs,Meal Type,"
           "Calories,Carbs,Protein,Fat,Fiber,Amount Consumed ,Image path")


def _row(i, t, g, carbs=""):
    meal = f"Lunch,600,{carbs},20,10,3,100,photos/x.jpg" if carbs else ",,,,,,,"
    return f"{i},2020-05-01 {t},{g},{g},70,1.0,10,{meal}"


def _synthetic_csv() -> bytes:
    rows = [_HEADER]
    # 11:30-11:59 flat ~90, meal at 12:00 (60g), then a rise to ~160 and decay over 3h
    for i, mm in enumerate(range(-30, 181)):
        hh, m = divmod((12 * 60 + mm), 60)
        t = f"{hh:02d}:{m % 60:02d}:00"
        g = 90 if mm < 0 else 90 + 70 * (mm / 45) if mm < 45 else 160 - 70 * ((mm - 45) / 135)
        rows.append(_row(i, t, round(g, 1), carbs=60 if mm == 0 else ""))
    return "\n".join(rows).encode()


def test_loader_extracts_a_meal_with_its_glucose_window():
    meals = load_subject(_synthetic_csv(), "TEST-001")
    assert len(meals) == 1
    m = meals[0]
    assert m.carbs_g == 60 and m.image == "photos/x.jpg"
    assert min(m.times_min) < 0 < max(m.times_min)       # window spans pre + post meal


def test_real_iauc_is_positive_for_a_rising_curve():
    m = load_subject(_synthetic_csv(), "TEST-001")[0]
    iauc = m.real_iauc()
    assert iauc is not None and iauc > 1000              # a real excursion, above baseline


def test_engine_prediction_responds_to_insulin_sensitivity():
    """Lower Si (more resistant) must give a LARGER predicted iAUC — the mechanism the
    per-subject fit relies on."""
    resistant, _ = predict_meal_iauc(60, insulin_sensitivity=0.5)
    sensitive, _ = predict_meal_iauc(60, insulin_sensitivity=1.3)
    assert resistant > sensitive


def test_validate_subject_needs_enough_meals():
    assert validate_subject(load_subject(_synthetic_csv(), "TEST-001")) is None   # only 1 meal


def test_pearson_handles_direction_and_missing():
    r, n = _pearson([1, 2, 3, 4], [4, 3, 2, 1])     # perfect negative
    assert r == pytest.approx(-1.0) and n == 4
    r, n = _pearson([1, 2, None, 4], [1, 2, 3, None])  # drops incomplete pairs
    assert n == 2


@pytest.mark.skipif(not _ZIP.exists(), reason="CGMacros dataset not present")
def test_bio_parses_demographics_and_homa_ir():
    bio = load_bio()
    assert len(bio) >= 40
    b = bio["CGMacros-001"]
    assert b["sex"] in ("male", "female") and b["weight_kg"] and b["height_cm"]
    assert b["status"] in ("normal", "prediabetic", "diabetic", "unknown")
    if b["fasting_glucose"] and b["fasting_insulin"]:
        assert b["homa_ir"] == pytest.approx(
            b["fasting_glucose"] * b["fasting_insulin"] / 405.0, abs=0.05)


@pytest.mark.skipif(not _ZIP.exists(), reason="CGMacros dataset not present")
def test_real_dataset_loads_and_scores():
    subs = subjects(limit=2)
    assert subs and any(meals for _, meals in subs)
    v = next((validate_subject(m) for _, m in subs if len(m) >= 4), None)
    if v:
        assert v["n_meals"] >= 4 and 0.2 <= v["fitted_si"] <= 1.6
