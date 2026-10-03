"""Deterministic reference router for Layer 4.

Maps a natural-language question + the user's profile to a tool call, dispatches it,
and renders a plain-language explanation of the numbers the module returned. This is
the offline, testable reference implementation; an LLM can replace the *routing* step
(see `route` docstring) but must call the same `orchestration.tools.dispatch`.

This is the PRIMARY/default router (ML/mechanistic, no LLM call, deterministic,
free, and reproducible) — the Gemini router in llm_gemini.py is an optional upgrade
for messy natural-language phrasing, not the default computation path.

Hard rule (brief 8): if no tool/edge supports the question, return the third outcome —
say so honestly. Never fabricate an answer. The router never invents numbers; it only
formats numbers a module computed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from orchestration.tools import dispatch

# Words that indicate an alcohol-related question. Includes spirits/cocktails so
# "2 shots of vodka" is recognized, not just "drink"/"beer".
_ALCOHOL_WORDS = (
    "drink", "beer", "drunk", "bac", "alcohol", "vodka", "whiskey", "whisky",
    "rum", "gin", "tequila", "liquor", "spirit", "cocktail", "wine", "shot",
)


@dataclass
class UserProfile:
    weight_kg: float
    height_cm: float
    age: float
    sex: str
    total_chol: float | None = None
    hdl: float | None = None
    sbp: float | None = None
    smoker: bool = False
    diabetic: bool = False


@dataclass
class Answer:
    text: str
    tool: str | None
    evidence: str                    # strong | weak | none
    headline: str = ""               # compact one-line takeaway (shown bold)
    explanation: list[str] = field(default_factory=list)  # plain-language driver bullets
    indicators: dict = field(default_factory=dict)         # key computed numbers, labelled
    citations: list[str] = field(default_factory=list)
    personalization: dict = field(default_factory=dict)    # average-human vs you
    raw: dict | None = None


def _has_alcohol_word(q: str) -> bool:
    return any(w in q for w in _ALCOHOL_WORDS)


def _drinks_in(q: str) -> float | None:
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:standard\s+)?"
        r"(?:drink|beer|glass|shot|wine|vodka|whiskey|whisky|rum|gin|tequila)", q)
    return float(m.group(1)) if m else None


def route(question: str, profile: UserProfile) -> Answer:
    """Route a question to a module and explain the result.

    LLM swap-in point: replace this function's intent detection with an LLM given
    `orchestration.tools.TOOL_SCHEMAS`; take the tool name + args it returns, call
    `dispatch(name, args)`, and pass the structured result to `_explain_*`. The LLM
    must not compute numbers — only choose the tool and phrase the explanation.
    """
    q = question.lower()

    # BMI (exact formula; foundational + heavily used).
    if "bmi" in q or "body mass" in q:
        res = dispatch("compute_bmi", dict(
            weight_kg=profile.weight_kg, height_cm=profile.height_cm, age=profile.age))
        return _explain_bmi(res)

    # Cross-system: alcohol -> sleep (check before single-system alcohol).
    if ("sleep" in q or "rem" in q) and _has_alcohol_word(q):
        n = _drinks_in(q) or 0
        res = dispatch("alcohol_effect_on_sleep", dict(
            standard_drinks=n, weight_kg=profile.weight_kg, sex=profile.sex,
            age=profile.age))
        return _explain_alcohol_sleep(n, res)

    # Single-system: BAC.
    if _has_alcohol_word(q):
        n = _drinks_in(q)
        if n is None:
            return Answer("How many drinks (or shots/glasses) do you mean? I need a "
                          "number to estimate BAC.", tool=None, evidence="none")
        res = dispatch("estimate_bac", dict(
            standard_drinks=n, weight_kg=profile.weight_kg, sex=profile.sex))
        return _explain_bac(n, res)

    # Weight / diet. Fire on weight keywords OR any explicit calorie amount.
    cal_match = re.search(r"(\d{3,5})\s*(?:kcal|calories?|cals?)\b", q)
    if cal_match or any(w in q for w in ("weight", "lose", "gain", "calorie", "diet")):
        intake = float(cal_match.group(1)) if cal_match else profile.weight_kg * 30
        days = 365
        dm = re.search(r"(\d+)\s*(year|month|week)", q)
        if dm:
            unit = {"year": 365, "month": 30, "week": 7}[dm.group(2)]
            days = int(dm.group(1)) * unit
        res = dispatch("project_weight", dict(
            weight_kg=profile.weight_kg, height_cm=profile.height_cm, age=profile.age,
            sex=profile.sex, daily_intake_kcal=intake, horizon_days=days))
        return _explain_weight(intake, days, res)

    # Cardiovascular.
    if "heart" in q or "cvd" in q or "cardiovascular" in q or "risk" in q:
        if None in (profile.total_chol, profile.hdl, profile.sbp):
            return Answer("I need your total cholesterol, HDL, and systolic blood "
                          "pressure to estimate cardiovascular risk.",
                          tool=None, evidence="none")
        res = dispatch("estimate_cvd_risk", dict(
            age=profile.age, sex=profile.sex, total_chol=profile.total_chol,
            hdl=profile.hdl, sbp=profile.sbp, treated_bp=False,
            smoker=profile.smoker, diabetic=profile.diabetic))
        return _explain_cvd(res)

    # Third outcome: no supporting module/edge.
    return Answer(
        "I don't have an evidence-backed model for that question yet, so I won't "
        "guess. I can currently project BAC, body weight, cardiovascular risk, and "
        "how alcohol affects sleep.",
        tool=None, evidence="none")


def explain(tool: str, args: dict, res: dict) -> Answer:
    """Render a grounded explanation for a dispatched tool result. Shared by the
    deterministic router and the LLM router so the numbers/phrasing are identical and
    always come from the module (never the LLM)."""
    if tool == "estimate_bac":
        return _explain_bac(args.get("standard_drinks", 0), res)
    if tool == "project_weight":
        return _explain_weight(args.get("daily_intake_kcal", 0),
                               args.get("horizon_days", 365), res)
    if tool == "estimate_cvd_risk":
        return _explain_cvd(res)
    if tool == "alcohol_effect_on_sleep":
        return _explain_alcohol_sleep(args.get("standard_drinks", 0), res)
    if tool == "compute_bmi":
        return _explain_bmi(res)
    raise ValueError(f"No explainer for tool {tool!r}")


def _explain_bmi(res: dict) -> Answer:
    lo, hi = res["healthy_weight_kg_range"]
    txt = (f"Your BMI is {res['bmi']} ({res['category']}). A healthy weight for your "
           f"height is about {lo}-{hi} kg." + (f" Note: {res['note']}" if res.get("note") else ""))
    return Answer(
        txt, tool="compute_bmi", evidence=res["evidence"],
        headline=f"BMI {res['bmi']} — {res['category']}",
        explanation=[f"BMI = weight ÷ height² — an exact formula, {res['confidence']}."],
        indicators={"bmi": res["bmi"], "category": res["category"],
                    "healthy_weight_kg_range": res["healthy_weight_kg_range"]},
        citations=res["citations"], raw=res,
    )


# --- Explanation formatters (deterministic; numbers come from modules) ------
#
# Every formatter builds: (1) a one-line summary `text`, (2) a list of plain-language
# `explanation` bullets derived from the module's ranked `drivers` (never invented),
# and (3) a labelled `indicators` dict of the raw numbers behind the answer — this is
# the explainability contract: the user can always see WHAT indicator caused WHAT
# effect, in simple language, traceable back to a real computed number.

def _conf(res: dict) -> str:
    return f" (confidence: {res['confidence']})"


def _driver_bullets(drivers: list, unit_label: str = "impact") -> list[str]:
    """Turn a module's ranked (name, magnitude) driver list into plain bullets."""
    if not drivers:
        return []
    total = sum(v for _, v in drivers) or 1.0
    bullets = []
    for name, v in drivers[:3]:
        pct = 100.0 * v / total
        bullets.append(f"{name} — about {pct:.0f}% of the {unit_label} you see here")
    return bullets


def _explain_bac(n, res) -> Answer:
    lo, hi = res["time_to_sober_ci"]
    headline = f"Peak BAC ~{res['peak_bac']:.3f}, sober in ~{res['time_to_sober_h']:.1f} h"
    txt = (f"{n:g} standard drink(s): peak BAC ~{res['peak_bac']:.3f} g/100mL, sober "
           f"in ~{res['time_to_sober_h']:.1f} h ({lo:.1f}-{hi:.1f} h). "
           f"Main driver: {res['drivers'][0][0]}." + _conf(res))
    return Answer(
        txt, tool="estimate_bac", evidence=res["evidence"], headline=headline,
        explanation=_driver_bullets(res["drivers"], "difference in your BAC"),
        indicators={
            "standard_drinks": n, "peak_bac_g_per_100ml": round(res["peak_bac"], 4),
            "peak_bac_90pct_interval": [round(v, 4) for v in res["peak_bac_ci"]],
            "hours_to_sober": round(res["time_to_sober_h"], 2),
        },
        citations=res["citations"], raw=res,
    )


def _explain_weight(intake, days, res) -> Answer:
    lo, hi = res["delta_kg_ci"]
    direction = "lose" if res["delta_kg"] < 0 else "gain"
    headline = f"{direction.capitalize()} ~{abs(res['delta_kg']):.1f} kg over {days} days"
    txt = (f"At {intake:.0f} kcal/day for {days} days you'd {direction} "
           f"~{abs(res['delta_kg']):.1f} kg ({lo:+.1f} to {hi:+.1f} kg). "
           f"Maintenance is ~{res['tdee_estimate']:.0f} kcal/day." + _conf(res))
    return Answer(
        txt, tool="project_weight", evidence=res["evidence"], headline=headline,
        explanation=_driver_bullets(res["drivers"], "uncertainty in this projection"),
        indicators={
            "daily_intake_kcal": intake, "horizon_days": days,
            "maintenance_tdee_kcal": round(res["tdee_estimate"]),
            "projected_change_kg": round(res["delta_kg"], 2),
            "projected_change_90pct_interval_kg": [round(v, 2) for v in res["delta_kg_ci"]],
        },
        citations=res["citations"], raw=res,
    )


def _explain_cvd(res) -> Answer:
    lo, hi = res["risk_ci"]
    drv = f" Biggest lever: {res['drivers'][0][0]}." if res["drivers"] else ""
    headline = f"10-year heart-disease risk ~{res['risk_10yr']*100:.1f}%"
    txt = (f"10-year cardiovascular risk ~{res['risk_10yr']*100:.1f}% "
           f"({lo*100:.1f}-{hi*100:.1f}%). {res['heart_age']}.{drv}" + _conf(res))
    bullets = [f"{name} — changing this would lower your risk by about {impact*100:.1f} "
              f"percentage points" for name, impact in res["drivers"][:3]]
    return Answer(
        txt, tool="estimate_cvd_risk", evidence=res["evidence"], headline=headline,
        explanation=bullets,
        indicators={
            "risk_10yr_pct": round(res["risk_10yr"] * 100, 1),
            "risk_90pct_interval_pct": [round(v * 100, 1) for v in res["risk_ci"]],
            "heart_age": res["heart_age"],
        },
        citations=res["citations"], raw=res,
    )


def _explain_alcohol_sleep(n, res) -> Answer:
    m = res["sleep_metrics"]
    headline = f"REM ~{m['rem_pct']:.0f}%, ~{m['awakenings']:.1f} awakenings tonight"
    txt = (f"{n:g} drink(s) before bed -> bedtime BAC ~{res['bedtime_bac']:.3f}, "
           f"REM ~{m['rem_pct']:.0f}% and ~{m['awakenings']:.1f} awakenings." + _conf(res))
    explanation = [
        f"Alcohol at bedtime (BAC ~{res['bedtime_bac']:.3f}, "
        f"{res['alcohol_gkg_bedtime']:.2f} g per kg body weight) suppresses REM sleep "
        f"and increases fragmentation — this is the alcohol-to-sleep effect, not a "
        f"direct measurement of your sleep tonight.",
    ]
    return Answer(
        txt, tool="alcohol_effect_on_sleep", evidence=res["evidence"], headline=headline,
        explanation=explanation,
        indicators={
            "standard_drinks": n, "bedtime_bac": round(res["bedtime_bac"], 4),
            "alcohol_g_per_kg_at_bedtime": round(res["alcohol_gkg_bedtime"], 3),
            "predicted_rem_pct": round(m["rem_pct"], 1),
            "predicted_deep_pct": round(m["deep_pct"], 1),
            "predicted_sleep_efficiency_pct": round(m["sleep_efficiency"], 1),
            "predicted_awakenings": round(m["awakenings"], 1),
        },
        citations=res["citations"], raw=res,
    )
