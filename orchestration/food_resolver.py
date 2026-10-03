"""Resolve ANY food to its nutrition, learn it once, store the INFO (not the name).

The curated food DB only has ~15 foods, so "salmon"/"avocado" used to fail. There's no need for
a big hand-built database: a food is resolved to its macros by a real nutrition API (USDA
FoodData Central — free, authoritative, NO LLM), and the result is cached to
`user_data/_food_cache.json`, which `simulation.foods.food_defs` merges in. So a food is looked
up ONCE and becomes a permanent entry usable by BOTH the calorie lookup and the glucose
simulator. The only thing actually "stored" about the user is what they ate (their meal log) —
the nutrition itself comes from the API on demand and is cached.

Resolution order: curated KB (fast, offline) -> cache -> USDA API -> LLM (last-resort estimate).
Every result is source-tagged (database / usda / estimated) so it's never over-trusted, and
calories are derived from macros via Atwater factors. Glycemic index (which USDA doesn't
provide) is estimated with a simple macro heuristic.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.parse
import urllib.request
from pathlib import Path

from simulation.foods import _CACHE_PATH, nutrition as kb_nutrition, resolve_food

_LOCK = threading.Lock()
_ESTIMATE_SYS = ("You are a nutrition database. Reply with ONLY strict JSON, numbers only, "
                 "no prose.")
_ESTIMATE_USER = (
    "Give typical nutrition for the food '{food}' PER 100 GRAMS as JSON: "
    "{{\"carbs_g\": <g>, \"protein_g\": <g>, \"fat_g\": <g>, \"fibre_g\": <g>, "
    "\"gi\": <glycemic index 0-110; use 0 for foods with ~no carbs>, "
    "\"serving_g\": <typical single-serving grams>}}. Base it on standard nutrition data. "
    "If '{food}' is not a real food, reply {{}}.")


def _read_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
    return {}


def _write_cache(entry_key: str, defn: dict) -> None:
    with _LOCK:
        cache = _read_cache()
        cache[entry_key] = defn
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _key(food_text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(food_text).strip().lower()).strip("_")


def _estimate_gi(carbs: float, fibre: float, fat: float) -> int:
    """USDA gives no glycemic index; estimate it from macros (fibre + fat lower GI)."""
    if carbs < 4:
        return 0                                            # ~no carbs -> negligible GI
    return int(max(20, min(95, 72 - 1.5 * fibre - 0.25 * fat)))


def _api_estimate(food_text: str) -> dict | None:
    """USDA FoodData Central lookup (no LLM). Per-100g macros for a generic food. None on any
    failure so resolution falls through to the LLM."""
    try:
        url = "https://api.nal.usda.gov/fdc/v1/foods/search?" + urllib.parse.urlencode({
            "query": food_text, "pageSize": 1, "api_key": os.getenv("USDA_API_KEY", "DEMO_KEY"),
            "dataType": "Foundation,SR Legacy,Survey (FNDDS)"})
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    foods = data.get("foods") or []
    if not foods:
        return None
    nutr = [(str(n.get("nutrientName", "")).lower(), n.get("value"))
            for n in foods[0].get("foodNutrients", [])]

    def g(*needles):
        for name, val in nutr:
            if val is not None and any(nd in name for nd in needles):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return 0.0
    carbs, protein = g("carbohydrate"), g("protein")
    fat, fibre = g("total lipid", "total fat"), g("fiber", "fibre")
    if carbs == protein == fat == 0:
        return None                                         # empty match -> not useful
    return {"carbs_g": carbs, "protein_g": protein, "fat_g": fat, "fibre_g": fibre,
            "gi": _estimate_gi(carbs, fibre, fat), "serving_g": 100.0, "source": "usda"}


def _llm_estimate(food_text: str) -> dict | None:
    from orchestration.orchestrate import active_backend
    from orchestration.planner import _chat_once, _extract_json
    if active_backend() != "groq":
        return None
    try:
        raw = _chat_once("groq", _ESTIMATE_SYS, _ESTIMATE_USER.format(food=food_text),
                         json_mode=True)
        d = _extract_json(raw)
    except Exception:
        return None
    if not d or not all(k in d for k in ("carbs_g", "protein_g", "fat_g")):
        return None
    try:
        return {"carbs_g": float(d["carbs_g"]), "protein_g": float(d["protein_g"]),
                "fat_g": float(d["fat_g"]), "fibre_g": float(d.get("fibre_g", 0) or 0),
                "gi": int(d["gi"]) if d.get("gi") not in (None, "") else 40,
                "serving_g": float(d.get("serving_g", 100) or 100),
                "source": "estimated"}
    except (TypeError, ValueError):
        return None


def resolve_nutrition(food_text: str, grams: float = 0.0) -> dict | None:
    """Nutrition for any food: curated DB -> cache -> LLM estimate (cached for next time).
    Returns the same shape as `simulation.foods.nutrition` plus a `source` field, or None."""
    if not food_text or len(str(food_text).strip()) < 2:
        return None
    # 1) curated DB / alias (also pulls a known food out of a longer phrase)
    if resolve_food(food_text):
        n = kb_nutrition(food_text, grams)
        if n:
            n["source"] = "database"
            return n
    # a whole sentence is not a food name — only estimate short food-like phrases, so we never
    # LLM-estimate "how many calories in a salmon dinner" as if it were one food.
    if len(str(food_text).split()) > 4:
        from simulation.foods import find_foods
        hit = find_foods(food_text)
        if hit:
            n = kb_nutrition(hit[0], grams); n["source"] = "database"; return n
        return None
    # 2) cache (already learned)
    key = _key(food_text)
    d = _read_cache().get(key)
    # 3) USDA API (no LLM) -> 4) LLM estimate. First hit is cached -> permanent DB entry.
    if d is None:
        d = _api_estimate(food_text) or _llm_estimate(food_text)
        if d:
            _write_cache(key, d)
    if not d:
        return None
    g = grams if grams and grams > 0 else d.get("serving_g", 100)
    s = g / 100.0
    carbs, protein, fat = d["carbs_g"] * s, d["protein_g"] * s, d["fat_g"] * s
    return {"food": key, "grams": round(g, 1),
            "kcal": round(4 * carbs + 4 * protein + 9 * fat),
            "carbs_g": round(carbs, 1), "protein_g": round(protein, 1),
            "fat_g": round(fat, 1), "fibre_g": round(d.get("fibre_g", 0) * s, 1),
            "gi": d.get("gi"), "serving_g": d.get("serving_g"),
            "source": d.get("source", "estimated")}
