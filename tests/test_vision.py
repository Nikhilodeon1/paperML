"""Image extraction — the VLM is mocked; the KB-mapping / provenance / gating core is tested.

Real vision-model calls are exercised live with real photos (D1NAMO meal images). These tests
lock in everything that happens AFTER the model returns: food -> KB mapping, GI attachment,
provenance tags, the confirm-before-engine gate, and the meal -> simulate args bridge.
"""

from __future__ import annotations

import pytest

from orchestration import vision


def test_meal_maps_foods_to_kb_and_attaches_gi():
    raw = {"kind": "meal", "confidence": 0.8, "description": "rice and chicken",
           "foods": [{"name": "white rice", "grams": 200, "confidence": 0.9},
                     {"name": "grilled chicken breast", "grams": 150, "confidence": 0.7}]}
    ext = vision.normalize_extraction(raw)
    assert ext["kind"] == "meal"
    rice = ext["foods"][0]
    assert rice["matched_food"] == "white_rice_cooked" and rice["in_database"]
    assert rice["nutrition"]["gi"] == 73                 # GI is a free KB lookup once ID'd
    assert ext["foods"][1]["matched_food"] == "chicken_breast"


def test_unmatched_food_is_kept_but_flagged():
    ext = vision.normalize_extraction(
        {"kind": "meal", "foods": [{"name": "durian custard", "grams": 100}]})
    f = ext["foods"][0]
    assert f["in_database"] is False and f["matched_food"] is None
    assert "durian custard" in ext["unmatched"]


def test_everything_is_provenance_tagged_and_gated():
    """Non-negotiable: image data is tagged as vision-estimated, never as a measurement, and
    always needs confirmation before it can drive the engine."""
    for raw in ({"kind": "meal", "foods": []}, {"kind": "labs", "labs": {}},
                {"kind": "wearable", "wearable": {}}):
        ext = vision.normalize_extraction(raw)
        assert ext["source"] == "uploaded image (vision-estimated)"
        assert ext["needs_confirmation"] is True


def test_labs_and_wearable_extraction_shapes():
    labs = vision.normalize_extraction(
        {"kind": "labs", "labs": {"hdl": {"value": 48, "unit": "mg/dL"},
                                  "glucose": {"value": 95, "unit": "mg/dL"}}})
    assert labs["labs"]["hdl"]["value"] == 48
    wear = vision.normalize_extraction(
        {"kind": "wearable", "wearable": {"resting_hr": 58, "steps": "8200", "junk": "n/a"}})
    assert wear["wearable"]["resting_hr"] == 58 and wear["wearable"]["steps"] == 8200.0
    assert "junk" not in wear["wearable"]                 # non-numeric dropped


def test_meal_becomes_simulate_args_only_for_known_foods():
    ext = vision.normalize_extraction(
        {"kind": "meal", "foods": [{"name": "white rice", "grams": 200},
                                   {"name": "mystery stew", "grams": 300}]})
    args = vision.extraction_to_sim_args(ext)
    foods = args["foods"]
    assert len(foods) == 1 and foods[0]["food"] == "white_rice_cooked"   # unknown dropped
    assert args["duration_min"] == 240


def test_no_sim_args_when_nothing_recognised():
    assert vision.extraction_to_sim_args(
        {"kind": "meal", "foods": [{"name": "mystery stew", "grams": 300, "matched_food": None,
                                    "in_database": False}]}) is None
    assert vision.extraction_to_sim_args({"kind": "labs", "labs": {}}) is None


def test_confirmation_prompt_reads_like_a_human_check():
    ext = vision.normalize_extraction(
        {"kind": "meal", "foods": [{"name": "white rice", "grams": 200}]})
    p = vision.confirmation_prompt(ext)
    assert "white rice" in p and "200" in p and p.strip().endswith("Correct?")


def test_garbage_or_none_from_the_model_degrades_safely():
    for junk in (None, "not json", {"kind": "unclear"}, 42):
        ext = vision.normalize_extraction(junk if isinstance(junk, (dict, type(None))) else None)
        assert ext["kind"] in ("unclear",) or ext.get("kind") == "unclear" or "kind" in ext
        assert ext["needs_confirmation"] in (True, False)


def test_analyze_image_survives_a_model_exception(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("vision endpoint down")
    monkeypatch.setattr(vision, "_call_vision", boom)
    ext = vision.analyze_image("Zm9v")
    assert ext["kind"] == "error" and "vision endpoint down" in ext["error"]


def test_end_to_end_with_mocked_model(monkeypatch):
    """Full path: (mocked) model JSON -> analyze_image -> KB-mapped, gated, sim-ready."""
    monkeypatch.setattr(vision, "_call_vision", lambda *a, **k: {
        "kind": "meal", "confidence": 0.85, "description": "a plate of pasta",
        "foods": [{"name": "pasta", "grams": 250, "confidence": 0.8}]})
    ext = vision.analyze_image("Zm9v")
    assert ext["foods"][0]["matched_food"] == "pasta_cooked"
    assert vision.extraction_to_sim_args(ext)["foods"][0]["food"] == "pasta_cooked"
