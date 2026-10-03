"""Data-driven food resolution for the simulation.

Turns a named food + grams into the effective GLYCEMIC carbohydrate load the glucose model
should see: carbohydrate scaled by glycemic index (relative to a GI-70 reference) and slowed
by fibre. This is what makes white rice and lentils produce very different glucose curves
from the same grams of carbohydrate. Foods live in `knowledge_base/foods.json` — add one
there, no code change.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "knowledge_base" / "foods.json"
# Foods resolved on the fly (LLM nutrition estimation, orchestration.food_resolver) are cached
# here so they become permanent DB entries — any food the user mentions is learned ONCE and is
# then available to both the calorie lookup and the glucose simulator.
_CACHE_PATH = Path(__file__).resolve().parents[1] / "user_data" / "_food_cache.json"


@lru_cache(maxsize=1)
def _base_defs() -> dict:
    return json.loads(_PATH.read_text(encoding="utf-8"))["foods"]


def food_defs() -> dict:
    """Curated KB foods + any foods resolved/cached at runtime (cache wins nothing over KB)."""
    defs = dict(_base_defs())
    if _CACHE_PATH.exists():
        try:
            cached = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            for k, v in cached.items():
                defs.setdefault(k, v)                  # never override a curated entry
        except (ValueError, OSError):
            pass
    return defs


def known_foods() -> list[str]:
    return sorted(food_defs())


# Everyday names -> KB keys, so "a coke" / "a slice of pizza" resolve without the LLM
# having to know the internal key. Extend freely; unknown foods simply return None.
_ALIASES = {
    "coke": "coca_cola", "cola": "coca_cola", "coca cola": "coca_cola", "soda": "coca_cola",
    "rice": "white_rice_cooked", "bread": "white_bread", "oatmeal": "oats_cooked",
    "porridge": "oats_cooked", "chicken": "chicken_breast", "oj": "orange_juice",
    "potato": "potato_boiled", "lentils": "lentils_cooked", "pasta": "pasta_cooked",
    "spaghetti": "pasta_cooked", "chocolate": "chocolate_bar", "candy bar": "chocolate_bar",
}


def resolve_food(text: str) -> str | None:
    """Best-effort map of free text to a KB food key (exact key, alias, or name mentioned)."""
    t = str(text or "").strip().lower()
    if not t:
        return None
    defs = food_defs()
    if t in defs:
        return t
    if t in _ALIASES:
        return _ALIASES[t]
    # longest match first so "orange juice" beats "orange"-less keys, "coca cola" beats "cola"
    for key in sorted(defs, key=len, reverse=True):
        if key.replace("_", " ") in t:
            return key
    for alias in sorted(_ALIASES, key=len, reverse=True):
        if alias in t:
            return _ALIASES[alias]
    return None


def find_foods(text: str) -> list[str]:
    """Every KB food mentioned in free text (by name or alias), longest match first.
    Lets the deterministic planner build a real scenario without needing an LLM."""
    t = str(text or "").lower()
    found: list[str] = []
    for name, key in ([(k.replace("_", " "), k) for k in food_defs()]
                      + list(_ALIASES.items())):
        if re.search(rf"\b{re.escape(name)}\b", t) and key not in found:
            found.append(key)
    return found


def nutrition(food: str, grams: float = 0.0) -> dict | None:
    """Macros + calories for a serving of `food` (None if unknown).

    Calories use the Atwater factors (4 kcal/g carbohydrate and protein, 9 kcal/g fat) —
    the standard way nutrition labels are derived — so a calorie question is answered from
    our own food data rather than guessed from a web snippet.
    """
    key = resolve_food(food)
    d = food_defs().get(key) if key else None
    if not d:
        return None
    g = grams if grams and grams > 0 else d.get("serving_g", 100)
    scale = g / 100.0
    carbs = d.get("carbs_g", 0) * scale
    protein = d.get("protein_g", 0) * scale
    fat = d.get("fat_g", 0) * scale
    fibre = d.get("fibre_g", 0) * scale
    return {"food": key, "grams": round(g, 1), "kcal": round(4 * carbs + 4 * protein + 9 * fat),
            "carbs_g": round(carbs, 1), "protein_g": round(protein, 1),
            "fat_g": round(fat, 1), "fibre_g": round(fibre, 1), "gi": d.get("gi"),
            "serving_g": d.get("serving_g")}


def effective_glycemic_carbs(food: str, grams: float = 0.0) -> float:
    """Effective glucose-raising carbohydrate (g) for a serving of `food`. 0 if unknown."""
    d = food_defs().get(str(food).strip().lower())
    if not d:
        return 0.0
    g = grams if grams and grams > 0 else d.get("serving_g", 100)
    carbs = d.get("carbs_g", 0) * g / 100.0
    gi = d.get("gi", 55) or 0
    fibre = d.get("fibre_g", 0) * g / 100.0
    fibre_factor = 1.0 / (1.0 + fibre / 20.0)          # fibre slows/blunts the response
    return carbs * (gi / 70.0) * fibre_factor
