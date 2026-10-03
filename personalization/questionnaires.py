"""Inference questionnaires — turn self-reported answers into health indicators.

The point of these questionnaires is not to store raw answers but to *infer* something
clinically meaningful from them. "How many times a week do you go?" -> a bowel-regularity
indicator that flags constipation against the Rome IV threshold; four PSS questions -> a
validated perceived-stress band. The mapping is fully declarative in
`knowledge_base/questionnaires.json`, so new instruments are added as data, not code.

Scoring model (see the JSON `schema_note`): each indicator pulls a numeric value either
`from` one question (a number answer, or the chosen option's `score`) or as the `sum_of`
several questions (a validated scale total), then walks ordered `bands` to the first one
whose [min, max] contains the value, yielding {label, severity, note, citation}.

`severity` is one of ok | watch | concern and drives both the app tile status and the
one-line English facts we hand the LLM, so advice can react to "you reported constipation"
without the LLM ever having to score anything itself.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_QPATH = Path(__file__).resolve().parents[1] / "knowledge_base" / "questionnaires.json"

# severity -> app tile status (concern and watch both surface as "watch"; the note carries
# the nuance). Kept here so the mapping is single-sourced.
SEVERITY_STATUS = {"ok": "good", "watch": "watch", "concern": "watch"}
_SEVERITY_RANK = {"ok": 0, "watch": 1, "concern": 2}


@lru_cache(maxsize=1)
def _load() -> dict:
    data = json.loads(_QPATH.read_text(encoding="utf-8"))
    return {q["id"]: q for q in data["questionnaires"]}


def list_questionnaires() -> list[dict]:
    """Lightweight catalogue for the app to render a menu."""
    return [{"id": q["id"], "title": q["title"], "description": q.get("description", ""),
             "icon": q.get("icon"), "n_questions": len(q["questions"])}
            for q in _load().values()]


def get_questionnaire(qid: str) -> dict | None:
    """The full definition (questions + options) for rendering the form."""
    return _load().get(qid)


def _question_map(qdef: dict) -> dict:
    return {q["id"]: q for q in qdef["questions"]}


def _value_of(question: dict, answer) -> float:
    """Numeric value of a single answer: the number itself for number questions, or the
    chosen option's `score` for choice questions."""
    if question["type"] == "number":
        return float(answer)
    for opt in question.get("options", []):
        if str(opt["value"]) == str(answer):
            return float(opt["score"])
    raise ValueError(f"answer {answer!r} is not a valid option for {question['id']}")


def _pick_band(value: float, bands: list[dict]) -> dict:
    for b in bands:
        if (b.get("min") is None or value >= b["min"]) and (b.get("max") is None or value <= b["max"]):
            return b
    return bands[-1]  # values below the first band's min fall through to the last as a safe default


def score(qid: str, responses: dict) -> dict:
    """Score a set of answers into indicators.

    `responses` maps question id -> answer (a number, or an option `value`). Questions may
    be omitted; indicators needing a missing answer are skipped rather than guessed."""
    qdef = get_questionnaire(qid)
    if qdef is None:
        raise KeyError(f"unknown questionnaire {qid!r}")
    qmap = _question_map(qdef)

    indicators: list[dict] = []
    for ind in qdef["indicators"]:
        try:
            if "sum_of" in ind:
                needed = ind["sum_of"]
                if any(responses.get(q) is None for q in needed):
                    continue
                value = sum(_value_of(qmap[q], responses[q]) for q in needed)
            else:
                src = ind["from"]
                if responses.get(src) is None:
                    continue
                value = _value_of(qmap[src], responses[src])
        except (ValueError, KeyError):
            continue

        band = _pick_band(value, ind["bands"])
        indicators.append({
            "key": ind["key"],
            "label": ind["label"],
            "value": round(value, 2),
            "band": band["label"],
            "severity": band["severity"],
            "note": band["note"],
            "citation": ind.get("citation"),
        })

    return {
        "id": qid,
        "name": qdef["title"],
        "indicators": indicators,
        "worst_severity": _worst_severity(indicators),
    }


def _worst_severity(indicators: list[dict]) -> str:
    return max((i["severity"] for i in indicators),
               key=lambda s: _SEVERITY_RANK.get(s, 0), default="ok")


def indicator_facts(survey: dict) -> list[str]:
    """One-line English facts for the LLM context from a stored, scored survey entry.

    Only surfaces the ones that actually matter (watch/concern) so we don't pad the prompt
    with 'everything is fine'."""
    facts: list[str] = []
    for ind in survey.get("indicators", []):
        if ind.get("severity") in ("watch", "concern"):
            facts.append(f"{ind['label']}: {ind['band'].lower()} ({survey.get('name', '')}).")
    return facts
