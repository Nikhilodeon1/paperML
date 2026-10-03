"""Abductive causal reasoning — symptom -> ranked, personalized differential.

The qualitative sibling of `pipeline/graph.py`. Where that module propagates NUMBERS
forward along authored edges, this one walks the causal graph BACKWARD from a symptom
("I keep getting sick") to candidate root causes ("vitamin D deficiency"), then ranks
those causes by how well each fits THIS user's data (diet, notes, labs, questionnaire
indicators). The chain and the personalization together produce the essential behaviour:

    recurrent_infection  <-  impaired_immune_function  <-  vitamin_d_deficiency
    ...ranked to the top because the user is vegan, rarely outdoors, has no D level.

Guarantees mirroring the rest of the engine:
  - Only authored edges exist, so no causal link can be fabricated (the graph is the
    guardrail — same principle as `has_edge`).
  - Confidence per cause is the WEAKEST LINK along its path, downgraded again if nothing
    personal matched, so long/speculative chains read as weak — honest by construction.
  - This is explicitly a DIFFERENTIAL, not a diagnosis; the disclaimer travels with it.

Grow the graph by editing `knowledge_base/causal_graph.json`; no code change needed.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "knowledge_base" / "causal_graph.json"

_EVIDENCE_RANK = {"strong": 3, "moderate": 2, "weak": 1}
_RANK_EVIDENCE = {3: "strong", 2: "moderate", 1: "weak"}
# a node is a surfaceable "cause" (actionable root) if it is one of these types
_CAUSE_TYPES = {"deficiency", "behavior", "lifestyle", "state"}


@lru_cache(maxsize=1)
def _graph() -> dict:
    return json.loads(_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _incoming() -> dict:
    """target -> list of (source, edge) — the adjacency we traverse backward."""
    adj: dict[str, list] = {}
    for e in _graph()["edges"]:
        adj.setdefault(e["target"], []).append((e["source"], e))
    return adj


def list_symptoms() -> list[dict]:
    return [{"key": k, "label": n["label"]}
            for k, n in _graph()["nodes"].items() if n["type"] == "symptom"]


def match_symptom(query: str) -> str | None:
    """Fuzzy-match free text to a symptom node via its label/aliases (longest alias wins,
    so 'keep getting sick' beats a bare 'sick')."""
    q = re.sub(r"[^a-z0-9 ]", " ", query.lower())
    q = re.sub(r"\s+", " ", q).strip()
    best, best_len = None, 0
    for key, n in _graph()["nodes"].items():
        if n["type"] != "symptom":
            continue
        for phrase in [key.replace("_", " "), n["label"].lower(), *n.get("aliases", [])]:
            p = re.sub(r"[^a-z0-9 ]", " ", phrase.lower()).strip()
            if p and p in q and len(p) > best_len:
                best, best_len = key, len(p)
    return best


# --- personal-signal evaluation ---------------------------------------------

def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _context(user: dict | None) -> dict:
    """Flatten the parts of a user we match signals against."""
    user = user or {}
    p = user.get("profile", {})
    med = user.get("medical", {})
    notes = [str(x).lower() for x in user.get("notes", [])]
    notes += [str(x).lower() for x in med.get("conditions", [])]
    notes += [str(e.get("text", "")).lower() for e in user.get("daily_log", [])[-6:]]
    indicators: dict[str, str] = {}
    for s in user.get("surveys", []):
        for ind in s.get("indicators", []):
            # keep the most severe seen per indicator key
            cur = indicators.get(ind["key"])
            rank = {"ok": 0, "watch": 1, "concern": 2}
            if cur is None or rank.get(ind["severity"], 0) > rank.get(cur, 0):
                indicators[ind["key"]] = ind["severity"]
    return {"profile": p, "diet": str(p.get("diet", "")).lower(),
            "alcohol": str(p.get("alcohol_freq", "")).lower(),
            "notes": notes, "indicators": indicators}


def _signal_matches(sig: dict, ctx: dict) -> bool:
    t = sig["type"]
    if t == "diet_in":
        return any(v in ctx["diet"] for v in sig["value"])
    if t == "note_contains":
        return any(any(v in n for n in ctx["notes"]) for v in sig["value"])
    if t == "lab_missing":
        return ctx["profile"].get(sig["value"]) in (None, "")
    if t == "lab_below":
        v = _num(ctx["profile"].get(sig["field"]))
        return v is not None and v < sig["value"]
    if t == "lab_above":
        v = _num(ctx["profile"].get(sig["field"]))
        return v is not None and v > sig["value"]
    if t == "profile_flag":
        return bool(ctx["profile"].get(sig["value"]))
    if t == "activity_in":
        return ctx["profile"].get("activity") in sig["value"]
    if t == "sex_is":
        return ctx["profile"].get("sex") == sig["value"]
    if t == "age_over":
        v = _num(ctx["profile"].get("age"))
        return v is not None and v >= sig["value"]
    if t == "age_under":
        v = _num(ctx["profile"].get("age"))
        return v is not None and v <= sig["value"]
    if t == "indicator":
        return ctx["indicators"].get(sig["value"]) in ("watch", "concern")
    if t == "alcohol_freq_in":
        return any(v in ctx["alcohol"] for v in sig["value"])
    return False


# --- traversal + ranking ----------------------------------------------------

def _best_paths(symptom: str) -> dict[str, list]:
    """Backward DFS from the symptom. For each reachable cause node, keep the path with
    the strongest weakest-link (most generous but still honest: 'a pathway graded X exists')."""
    nodes = _graph()["nodes"]
    incoming = _incoming()
    reached: dict[str, list] = {}
    stack: list[tuple[str, list]] = [(symptom, [])]
    while stack:
        cur, path = stack.pop()
        if len(path) > 5:                      # depth guard
            continue
        for src, edge in incoming.get(cur, []):
            if src == symptom or any(e["source"] == src for e in path):
                continue                       # avoid trivial cycles
            new_path = path + [edge]
            if nodes.get(src, {}).get("type") in _CAUSE_TYPES:
                prev = reached.get(src)
                if prev is None or _path_rank(new_path) > _path_rank(prev):
                    reached[src] = new_path
            stack.append((src, new_path))
    return reached


def _path_rank(path: list) -> int:
    return min(_EVIDENCE_RANK[e["evidence"]] for e in path) if path else 0


def explain_symptom(query: str, user: dict | None = None, top_k: int = 6) -> dict:
    """Return a ranked, personalized differential for a free-text symptom."""
    g = _graph()
    nodes = g["nodes"]
    symptom = match_symptom(query)
    if symptom is None:
        return {"matched": False, "query": query,
                "message": "I don't have a causal map for that symptom yet.",
                "known_symptoms": list_symptoms()}

    ctx = _context(user)
    reached = _best_paths(symptom)

    causes = []
    for key, path in reached.items():
        n = nodes[key]
        signals = n.get("personal_signals", [])
        matched = [s for s in signals if _signal_matches(s, ctx)]
        reasons = [s["reason"] for s in matched]
        weight = sum(s.get("weight", 1) for s in matched)
        rank = _path_rank(path)
        personalized = bool(matched)
        if not personalized:
            rank = max(1, rank - 1)            # no personal fit -> one grade weaker
        # blend how well it fits the user (personal weight) with how strong the pathway
        # evidence is, so a moderate-evidence cause outranks a weak one at equal fit.
        evidence_bonus = {3: 1.5, 2: 0.75, 1: 0.0}[rank]
        score = round(n.get("base_prior", 0.4) + weight + evidence_bonus, 2)
        # path shown outermost (symptom side) first
        chain = [symptom] + [e["source"] for e in path]
        causes.append({
            "key": key, "label": n["label"], "type": n["type"],
            "category": n.get("category"),
            "score": score,
            "personalized": personalized,
            "evidence": _RANK_EVIDENCE[rank],
            "personal_reasons": reasons,
            "chain": [nodes[c]["label"] for c in chain],
            "advice": n.get("advice"),
            "citations": list(dict.fromkeys(e["citation"] for e in path)),
        })

    # rank: personalized first, then the blended score
    causes.sort(key=lambda c: (c["personalized"], c["score"]), reverse=True)
    any_personal = any(c["personalized"] for c in causes)

    return {
        "matched": True,
        "symptom": symptom,
        "symptom_label": nodes[symptom]["label"],
        "personalized": any_personal,
        "causes": causes[:top_k],
        "disclaimer": g["disclaimer"],
    }
