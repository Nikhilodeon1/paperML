"""T3 reasoning — answer questions about systems the ENGINE can't simulate, grounded in the
user's own record + known science, honestly tagged by provenance.

The ODE engine covers ~5 systems. For everything else (immune, thyroid, general lab
interpretation) we do NOT simulate — we REASON from what we already know about the user:
their stored labs, conditions, family history, and demographics, matched against a
data-driven reference (knowledge_base/lab_reference.json). This is what lets the app answer
"my white cell count is low, why do I keep getting sick" with a grounded, cited implication
instead of the LLM improvising.

Discipline (the whole point): every fact carries a `source` so an answer never blurs
  - "your bloodwork"        (a value in the user's record)   -> evidence moderate
  - "your history"          (a stated condition / family)     -> evidence moderate
  - "general pattern"       (age/sex prior)                   -> evidence weak
None of it is a diagnosis; the planner degrades to the weakest link as always.

Labs live in `user["medical"]["labs"]` as {marker: value} or {marker: {value, unit}}, plus a
few in `profile` (total_chol, hdl, sbp). A new marker is a JSON entry — no code change.
"""

from __future__ import annotations

import re

from knowledge_base import load_raw


def _markers() -> dict:
    return load_raw("lab_reference")["markers"]


def _num(v):
    if isinstance(v, dict):
        v = v.get("value")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _user_labs(user: dict) -> dict:
    """Collect the user's lab values keyed by canonical marker id (labs dict + profile fields)."""
    out: dict = {}
    med = user.get("medical", {}) or {}
    labs = med.get("labs", {}) or {}
    marks = _markers()
    # direct + alias resolution from the medical.labs dict
    for k, v in labs.items():
        key = str(k).strip().lower()
        val = _num(v)
        if val is None:
            continue
        if key in marks:
            out[key] = val
        else:
            for mid, m in marks.items():
                if key == mid or key in [a.lower() for a in m.get("aliases", [])]:
                    out[mid] = val
                    break
    # a few labs live on the profile
    p = user.get("profile", {}) or {}
    for pfield, mid in (("total_chol", "ldl"), ("hdl", "hdl"), ("sbp", None)):
        if mid and _num(p.get(pfield)) is not None and mid not in out:
            # total_chol isn't LDL; only map hdl directly. (kept minimal + honest)
            if pfield == "hdl":
                out["hdl"] = _num(p.get("hdl"))
    return out


def interpret_labs(user: dict) -> list[dict]:
    """Tagged implications for any out-of-range lab the user has on file."""
    marks = _markers()
    facts = []
    for mid, val in _user_labs(user).items():
        m = marks.get(mid)
        if not m:
            continue
        low, high = m.get("low"), m.get("high")
        if low is not None and val < low:
            imp = m.get("low_implication")
        elif high is not None and val > high:
            imp = m.get("high_implication")
        else:
            continue                                    # in range -> nothing to flag
        if imp and imp not in ("in range", "in the normal range"):
            facts.append({"fact": f"Your {m['name']} is {val} {m.get('unit','')} — {imp}.".strip(),
                          "source": "your bloodwork", "evidence": "moderate",
                          "systems": m.get("systems", []), "citation": m.get("citation", "")})
    return facts


def demographic_priors(user: dict) -> list[dict]:
    """Age/sex-based directional priors (hormonal state etc.) — weak, clearly a general pattern."""
    p = user.get("profile", {}) or {}
    age, sex = _num(p.get("age")), str(p.get("sex", "")).lower()
    out = []
    if age is None:
        return out
    if sex.startswith("f") and 45 <= age <= 58:
        out.append({"fact": "Around your age, the menopause transition is common and can itself "
                            "affect sleep, mood, temperature regulation, and cardiovascular risk "
                            "— worth keeping in mind when interpreting symptoms.",
                    "source": "general pattern (age/sex)", "evidence": "weak",
                    "systems": ["sleep", "stress", "cardiovascular"], "citation": ""})
    if sex.startswith("m") and age >= 45:
        out.append({"fact": "Testosterone gradually declines with age in men, which can subtly "
                            "affect energy, muscle, and mood — a general pattern, not a diagnosis.",
                    "source": "general pattern (age/sex)", "evidence": "weak",
                    "systems": ["metabolic", "activity"], "citation": ""})
    return out


def history_facts(user: dict) -> list[dict]:
    """Stated conditions / family history as tagged facts the reasoning can build on."""
    med = user.get("medical", {}) or {}
    out = []
    if med.get("conditions"):
        out.append({"fact": "On your record: " + ", ".join(map(str, med["conditions"])) + ".",
                    "source": "your history", "evidence": "moderate", "systems": [], "citation": ""})
    if med.get("family_history"):
        out.append({"fact": "Family history you've shared: " + ", ".join(map(str, med["family_history"])) + ".",
                    "source": "your history", "evidence": "moderate", "systems": [], "citation": ""})
    return out


# systems the ODE engine actually simulates — used to tell the caller when to add live sim data
_SIMULATED = {"metabolic", "cardiovascular", "sleep", "stress", "activity", "hepatic", "digestive"}


# Question keywords -> the body systems whose labs are relevant. Lets an "HDL" question pull
# the lipid labs and a "flu"/"immune" question pull the immune labs, instead of every lab.
_QUERY_SYSTEMS = {
    "immune": ("sick", "infection", "immune", "cold", "flu", "vaccine", "vaccination", "virus",
               "wbc", "white blood", "neutrophil", "crp", "inflammation"),
    "cardiovascular": ("hdl", "ldl", "cholesterol", "triglyceride", "lipid", "heart", "cardio",
                       "blood pressure", "stroke"),
    "metabolic": ("glucose", "sugar", "hba1c", "a1c", "diabet", "insulin", "weight", "metabolic"),
    "thyroid": ("thyroid", "tsh", "cold all the time", "hair"),
    "sleep": ("sleep", "vitamin d", "tired", "fatigue"),
}
# question intents that make family/condition history or age/sex priors actually relevant
_RISK_Q = ("risk", "develop", "could i", "chance", "inherit", "genetic", "prone", "family",
           "history", "at risk", "what do you know", "conditions", "hereditary")
_DEMO_Q = ("hormone", "menopause", "testosterone", "aging", "age", "energy", "mood", "libido",
           "hot flash", "period", "cycle", "puberty")


def reason_about(query: str, user: dict) -> dict:
    """Assemble grounded, provenance-tagged facts RELEVANT to the question. Irrelevant record
    items (a family-diabetes note on an HDL question, a menopause prior on a flu question) are
    left out — if nothing on file is relevant, facts is empty and the caller falls back to
    general knowledge rather than dumping unrelated record data.
    """
    q = (query or "").lower()
    asked = {sysname for sysname, kws in _QUERY_SYSTEMS.items() if any(k in q for k in kws)}

    # labs: only those whose systems intersect what the question is about (or the marker is
    # named directly). No asked-system match -> no lab facts (don't dump the whole panel).
    labs = []
    for f in interpret_labs(user):
        fsys = set(f.get("systems", []))
        if (asked and fsys & asked) or _marker_named(q, f):
            labs.append(f)

    facts = list(labs)
    if any(k in q for k in _RISK_Q):            # family/condition history only for risk questions
        facts += history_facts(user)
    if any(k in q for k in _DEMO_Q):            # age/sex priors only for hormone/aging questions
        facts += demographic_priors(user)

    touched = sorted({s for f in labs for s in f.get("systems", []) if s in _SIMULATED})
    ranks = {"strong": 3, "moderate": 2, "weak": 1}
    ev = max((f["evidence"] for f in facts), key=lambda e: ranks.get(e, 0), default="none")
    cites = [f["citation"] for f in facts if f.get("citation")]
    return {"facts": facts, "simulated_systems_touched": touched,
            "evidence": ev if facts else "none",
            "citations": list(dict.fromkeys(cites)),
            "disclaimer": "Grounded in your record + general medical knowledge, tagged by "
                          "source. Not a diagnosis."}


def _marker_named(q: str, fact: dict) -> bool:
    """True if the question directly names this lab marker (by name or alias)."""
    name = fact.get("fact", "").lower()
    for mid, m in _markers().items():
        if mid in name or m["name"].lower() in name:
            if mid in q or any(a in q for a in m.get("aliases", [])):
                return True
    return False
