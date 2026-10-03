"""Food resolver — unknown foods -> LLM nutrition -> learned into the DB (LLM mocked)."""

from __future__ import annotations

import json

import pytest

import orchestration.food_resolver as fr
from simulation import foods


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the food cache at a temp file so tests never touch real user_data."""
    cache = tmp_path / "_food_cache.json"
    monkeypatch.setattr(foods, "_CACHE_PATH", cache)
    monkeypatch.setattr(fr, "_CACHE_PATH", cache)
    yield


def test_known_food_comes_from_database_no_llm(monkeypatch):
    monkeypatch.setattr(fr, "_llm_estimate", lambda f: pytest.fail("should not call LLM"))
    r = fr.resolve_nutrition("banana", 120)
    assert r["source"] == "database" and r["gi"] == 51


def test_api_is_used_before_the_llm(monkeypatch):
    """USDA (no LLM) is the primary resolver; the LLM must not be called when the API answers."""
    monkeypatch.setattr(fr, "_api_estimate", lambda f: {"carbs_g": 7, "protein_g": 2, "fat_g": 15,
                        "fibre_g": 5, "gi": 40, "serving_g": 100, "source": "usda"})
    monkeypatch.setattr(fr, "_llm_estimate", lambda f: pytest.fail("LLM must not run when API works"))
    r = fr.resolve_nutrition("avocado")
    assert r["source"] == "usda" and r["kcal"] == round(4 * 7 + 4 * 2 + 9 * 15)


def test_llm_is_the_fallback_when_api_fails(monkeypatch):
    monkeypatch.setattr(fr, "_api_estimate", lambda f: None)      # API down / no match
    monkeypatch.setattr(fr, "_llm_estimate", lambda f: {"carbs_g": 0, "protein_g": 20, "fat_g": 10,
                        "fibre_g": 0, "gi": 0, "serving_g": 150, "source": "estimated"})
    r = fr.resolve_nutrition("mystery fish")
    assert r["source"] == "estimated"


def test_gi_heuristic_reasonable():
    assert fr._estimate_gi(carbs=0, fibre=0, fat=20) == 0        # ~no carbs
    assert 20 <= fr._estimate_gi(carbs=30, fibre=1, fat=1) <= 95


def test_unknown_food_is_estimated_then_cached_and_learned(monkeypatch):
    monkeypatch.setattr(fr, "_api_estimate", lambda f: None)      # force the LLM path
    calls = []

    def fake(f):
        calls.append(f)
        return {"carbs_g": 0, "protein_g": 20, "fat_g": 13, "fibre_g": 0, "gi": 0,
                "serving_g": 150, "source": "estimated"}
    monkeypatch.setattr(fr, "_llm_estimate", fake)

    r = fr.resolve_nutrition("salmon")
    assert r["source"] == "estimated" and r["kcal"] == round(4 * 0 + 4 * 30 + 9 * 19.5)
    # learned into the DB -> the simulator's food resolver now knows it
    assert foods.resolve_food("salmon") == "salmon"
    # second call is served from cache -> no second LLM call
    fr.resolve_nutrition("salmon")
    assert len(calls) == 1


def test_empty_or_junk_never_calls_llm(monkeypatch):
    monkeypatch.setattr(fr, "_llm_estimate", lambda f: pytest.fail("should not call LLM"))
    assert fr.resolve_nutrition("") is None
    assert fr.resolve_nutrition("x") is None


def test_a_whole_sentence_is_not_estimated_as_a_food(monkeypatch):
    monkeypatch.setattr(fr, "_llm_estimate", lambda f: pytest.fail("should not call LLM"))
    # 5+ words, no known food in it -> None, not a junk estimate
    assert fr.resolve_nutrition("what is the best way to eat") is None


def test_sentence_containing_a_known_food_resolves_it(monkeypatch):
    monkeypatch.setattr(fr, "_llm_estimate", lambda f: pytest.fail("should not call LLM"))
    r = fr.resolve_nutrition("how many calories are in a banana please")
    assert r and r["source"] == "database" and r["food"] == "banana"


def test_cache_persists_to_disk(monkeypatch):
    monkeypatch.setattr(fr, "_api_estimate", lambda f: None)     # no network in tests
    monkeypatch.setattr(fr, "_llm_estimate", lambda f: {"carbs_g": 10, "protein_g": 5,
                        "fat_g": 2, "fibre_g": 1, "gi": 40, "serving_g": 100, "source": "estimated"})
    fr.resolve_nutrition("dragonfruit")
    on_disk = json.loads(foods._CACHE_PATH.read_text())
    assert "dragonfruit" in on_disk and on_disk["dragonfruit"]["carbs_g"] == 10
