"""Image understanding — a vision model EXTRACTS structured facts; the engine still computes.

Same rule as the rest of the system: the model turns pixels into structured data, it never
produces a physiological number. A meal photo becomes {foods: [...]}, a lab photo becomes
{labs: {...}} — then the existing engine / planner take over exactly as if the user had typed
those facts.

Why this is the highest-value input upgrade (not a bigger router): the logging-sensitivity
analysis showed carb TYPE dominates postprandial error (~95%), and a photo naturally IDENTIFIES
the food -> its glycemic index is a free KB lookup. Getting the food right matters far more
than the portion (+/-50% grams costs only ~18%).

Discipline enforced here:
  - every extracted value is tagged source="uploaded image (vision-estimated)" — never posed
    as a measurement of the body,
  - `needs_confirmation` is always True: the model hallucinates foods/portions, so the user
    must confirm before anything drives the engine,
  - foods are mapped to real KB keys; anything unmatched is returned as-is for the user to fix.

The vision model is configurable (GROQ_VISION_MODEL); the extraction is patchable so the
downstream mapping is testable without a live API. Verify the current Groq vision catalog —
model ids churn.
"""

from __future__ import annotations

import base64
import json
import os
import re

from simulation.foods import known_foods, nutrition, resolve_food

_DEFAULT_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"   # a Groq multimodal id


def vision_model() -> str:
    return os.getenv("GROQ_VISION_MODEL", _DEFAULT_VISION_MODEL)


_EXTRACT_SYS = (
    "You are an EXTRACTOR for a health app. You look at an uploaded image and return STRUCTURED "
    "DATA only — you never diagnose, never give advice, never invent a value you cannot see. "
    "Decide the image KIND and extract accordingly:\n"
    "  - meal/food photo -> list each distinct food and your best PORTION estimate in grams. "
    "Prefer names from this known-food list when they match: {foods}. If a food is not in the "
    "list, still name it plainly.\n"
    "  - lab report / blood test -> extract each named value you can read (e.g. glucose, hdl, "
    "total cholesterol, wbc, hba1c) with its number and unit.\n"
    "  - wearable / app screenshot -> extract each metric shown (e.g. resting_hr, hrv, "
    "sleep_efficiency, steps).\n"
    "Give a confidence 0-1 for the extraction as a whole. If the image is unclear or none of "
    "these, say so.\n"
    "Reply with ONLY JSON: {\"kind\": \"meal|labs|wearable|unclear\", "
    "\"foods\": [{\"name\": str, \"grams\": number, \"confidence\": number}], "
    "\"labs\": {<name>: {\"value\": number, \"unit\": str}}, "
    "\"wearable\": {<metric>: number}, \"confidence\": number, \"description\": str}."
)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _call_vision(image_b64: str, mime: str = "image/jpeg") -> dict | None:
    """Raw vision-model call. Patched out in tests; exercised live with real images."""
    from openai import OpenAI
    from orchestration.config import groq_api_key
    client = OpenAI(api_key=groq_api_key(), base_url="https://api.groq.com/openai/v1")
    sys = _EXTRACT_SYS.replace("{foods}", ", ".join(known_foods()))
    r = client.chat.completions.create(
        model=vision_model(), temperature=0,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": [
                      {"type": "text", "text": "Extract structured data from this image."},
                      {"type": "image_url",
                       "image_url": {"url": f"data:{mime};base64,{image_b64}"}}]}])
    return _extract_json(r.choices[0].message.content or "")


def analyze_image(image_b64: str, mime: str = "image/jpeg") -> dict:
    """Image -> structured, KB-mapped, provenance-tagged extraction ready for the pipeline.

    Never raises on model junk: a failed/garbage extraction returns kind='unclear'.
    """
    try:
        raw = _call_vision(image_b64, mime)
    except Exception as e:
        return {"kind": "error", "error": f"{type(e).__name__}: {e}", "source": _SRC,
                "needs_confirmation": False}
    return normalize_extraction(raw)


_SRC = "uploaded image (vision-estimated)"


def normalize_extraction(raw: dict | None) -> dict:
    """Clean + KB-map a raw vision JSON. Pure function -> the testable core."""
    if not isinstance(raw, dict):
        return {"kind": "unclear", "source": _SRC, "needs_confirmation": True,
                "description": "could not read the image"}
    kind = str(raw.get("kind", "unclear")).lower()
    out: dict = {"kind": kind, "source": _SRC, "needs_confirmation": True,
                 "confidence": _num(raw.get("confidence"), 0.5),
                 "description": str(raw.get("description", ""))[:300]}

    if kind == "meal":
        foods = []
        for f in raw.get("foods", []) or []:
            if not isinstance(f, dict):
                continue
            name = str(f.get("name", "")).strip()
            if not name:
                continue
            key = resolve_food(name)                      # map to a KB food if we know it
            grams = _num(f.get("grams"), 0.0)
            item = {"name": name, "matched_food": key, "grams": round(grams, 0) or None,
                    "confidence": _num(f.get("confidence"), out["confidence"]),
                    "in_database": key is not None}
            if key:                                       # attach macros so GI is already known
                item["nutrition"] = nutrition(key, grams or 0.0)
            foods.append(item)
        out["foods"] = foods
        out["unmatched"] = [f["name"] for f in foods if not f["in_database"]]

    elif kind == "labs":
        out["labs"] = {str(k): v for k, v in (raw.get("labs") or {}).items()
                       if isinstance(v, dict)}
    elif kind == "wearable":
        out["wearable"] = {str(k): _num(v, None) for k, v in (raw.get("wearable") or {}).items()
                           if _num(v, None) is not None}
    return out


def extraction_to_sim_args(extraction: dict) -> dict | None:
    """A confirmed meal extraction -> `simulate` args (foods at t=0). None if not a meal or no
    KB-matched foods (the engine can only run foods it knows)."""
    if extraction.get("kind") != "meal":
        return None
    foods = [{"t_min": 0, "food": f["matched_food"], "grams": f.get("grams") or 0.0}
             for f in extraction.get("foods", []) if f.get("matched_food")]
    if not foods:
        return None
    return {"foods": foods, "duration_min": 240}


def confirmation_prompt(extraction: dict) -> str:
    """Human-readable 'is this right?' before anything touches the engine."""
    k = extraction.get("kind")
    if k == "meal":
        parts = []
        for f in extraction.get("foods", []):
            g = f"~{int(f['grams'])} g" if f.get("grams") else "amount?"
            tag = "" if f["in_database"] else " (not in food DB)"
            parts.append(f"{f['name']} {g}{tag}")
        return "I see: " + "; ".join(parts) + ". Correct?" if parts else "No food recognized."
    if k == "labs":
        return "I read these lab values: " + ", ".join(
            f"{n} {v.get('value')}{v.get('unit','')}" for n, v in extraction.get("labs", {}).items()
        ) + ". Save them?"
    if k == "wearable":
        return "I read: " + ", ".join(f"{n} {v}" for n, v in extraction.get("wearable", {}).items()) \
               + ". Save them?"
    return "I couldn't confidently read that image — try a clearer photo."


def _num(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def image_bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
