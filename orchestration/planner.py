"""Plan -> Execute -> Synthesize orchestrator — the reasoning layer.

The free-form tool loop asks a small LLM to answer, pick tools, and format all at once,
which is where it drops parts of a question or stops at the first tool. This planner
separates the three jobs so complex / unexpected questions are handled thoroughly:

  1. PLAN     — decompose the question into grounded sub-analyses ("capabilities"), each
                with a rationale. A deterministic planner always produces a useful plan
                from the question + the user's data; an LLM planner refines it and can add
                argument-heavy steps (e.g. simulate a specific scenario).
  2. EXECUTE  — run every planned capability against real modules / the simulation engine.
                Numbers are 100% grounded here; the LLM never computes them.
  3. SYNTHESIZE — compose the executed evidence into one answer. Works fully without an
                LLM (structured deterministic write-up); the LLM makes it read naturally.

Every capability carries its own evidence grade and citations, and the overall answer is
the weakest link — so novel questions with only partial coverage degrade honestly instead
of bluffing. The whole plan + each step's result is returned for inspection (the ML lab).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from orchestration.router import UserProfile

# ---------------------------------------------------------------------------
# Capability registry — the grounded things the planner can decide to run. Each wraps
# existing modules / the causal graph / the simulation engine. `needs_args` capabilities
# depend on values that must be extracted from the question (LLM planner supplies them);
# the rest run purely on the user's stored data (the deterministic planner can use them).
# ---------------------------------------------------------------------------

_SYSTEM_KEYS = ("metabolic", "cardiovascular", "sleep", "stress", "activity", "hepatic", "digestive")


def _cap_body_state(args, user):
    """The user's current simulated state. `args["systems"]` scopes it to only the systems
    the question is about (the default is NOT the whole body — the planner passes the
    relevant subset so an answer isn't padded with seven unrelated tiles)."""
    from personalization.systems import compute_systems
    want = [str(s).lower() for s in (args.get("systems") or []) if str(s).lower() in _SYSTEM_KEYS]
    tiles = compute_systems(user)
    if want:
        tiles = [t for t in tiles if t.get("key") in want]
    return {
        "scope": want or "whole_body",
        "summary": "; ".join(f"{t['title']}: {t['headline']} ({t['status']})"
                             for t in tiles if t.get("status") != "none"),
        "systems": [{"system": t["key"], "status": t["status"], "headline": t["headline"],
                     "metrics": {m["label"]: f"{m['value']} {m.get('sub','')}".strip()
                                 for m in t.get("metrics", [])}} for t in tiles],
        "evidence": "strong",
        "citations": ["Horizon system modules (metabolic, cardiovascular, sleep, stress, hepatic)"],
    }


def _cap_symptom_causes(args, user):
    from pipeline.causal import explain_symptom
    return explain_symptom(args.get("symptom") or args.get("query") or "", user)


def _cap_patterns(args, user):
    """Find real correlations/trends in the user's OWN longitudinal data (e.g. glucose worse
    on poor-sleep days). Statistical, reported with strength + n, no causation claimed."""
    from personalization.patterns import detect_patterns
    r = detect_patterns(user or {})
    if not r.get("patterns"):
        return {"unavailable": "not enough logged data yet to surface a reliable pattern",
                "evidence": "none", "citations": []}
    return r


def _cap_health_reasoning(args, user):
    """T3: reason about systems the engine can't simulate (immune, thyroid, labs) from the
    user's own record + known science, provenance-tagged. Not a simulation, not a diagnosis."""
    from personalization.reasoning import reason_about
    r = reason_about(args.get("query") or args.get("symptom") or "", user or {})
    if not r.get("facts"):
        return {"unavailable": "no relevant labs/history on file to reason from",
                "evidence": "none", "citations": []}
    return r


def _cap_heart_risk(args, user):
    from modules.cardiovascular import compute_cvd_risk
    p = user.get("profile", {})
    if not all(p.get(k) is not None for k in ("total_chol", "hdl", "sbp")):
        return {"unavailable": "needs total cholesterol, HDL and blood pressure on file",
                "evidence": "none", "citations": []}
    if not (30 <= float(p.get("age", 0)) <= 74):
        return {"unavailable": "Framingham applies to ages 30-74", "evidence": "weak", "citations": []}
    r = compute_cvd_risk(age=float(p["age"]), sex=p.get("sex", "male"),
                         total_chol=float(p["total_chol"]), hdl=float(p["hdl"]),
                         sbp=float(p["sbp"]), smoker=bool(p.get("smoker")),
                         diabetic=bool(p.get("diabetic")))
    return {"risk_10yr_pct": round(r.risk_10yr * 100, 1), "heart_age": r.heart_age_note,
            "top_lever": r.drivers[0][0] if r.drivers else None,
            "evidence": r.evidence.value, "citations": list(r.citations)}


def _cap_bmi(args, user):
    from modules.anthropometric import compute_bmi
    p = user.get("profile", {})
    res = compute_bmi(weight_kg=float(p.get("weight_kg", 75)),
                      height_cm=float(p.get("height_cm", 175)), age=p.get("age"))
    return {"bmi": res.bmi, "category": res.category,
            "healthy_weight_kg": list(res.healthy_weight_kg_range),
            "evidence": res.evidence.value, "citations": res.citations}


def _cap_food_lookup(args, user):
    """Calories + macros for a named food. Curated DB first; unknown foods are resolved via the
    LLM nutrition estimator and LEARNED (cached into the DB) so next time is free + simulatable."""
    from orchestration.food_resolver import resolve_nutrition
    q = str(args.get("food") or args.get("query") or "")
    grams = args.get("grams")
    try:
        grams = float(grams) if grams else 0.0
    except (TypeError, ValueError):
        grams = 0.0
    if not grams:                       # pull a quantity out of the text: "12 oz", "330 ml", "150 g"
        m = re.search(r"(\d+(?:\.\d+)?)\s*(oz|ounce|ml|millilitre|milliliter|g\b|gram)", q.lower())
        if m:
            n, unit = float(m.group(1)), m.group(2)
            grams = n * 29.6 if unit.startswith(("oz", "ounce")) else n   # ml/g ~= 1 g for drinks
    res = resolve_nutrition(q, grams)
    if not res:
        return {"unavailable": f"couldn't identify a food in '{q[:40]}'",
                "evidence": "none", "citations": []}
    src = res.pop("source", "database")
    cite = {"database": "Horizon food database + Atwater factors",
            "usda": "USDA FoodData Central + Atwater factors",
            "estimated": "AI nutrition estimate + Atwater factors"}.get(src, "nutrition estimate")
    return {**res, "estimated": src == "estimated",
            "evidence": "weak" if src == "estimated" else "moderate", "citations": [cite]}


def _cap_daily_energy(args, user):
    """Estimate the user's maintenance calories/day from their body (Mifflin-St Jeor RMR x
    activity), plus their logged intake if they have any. Answers 'how many calories do I eat /
    need per day'."""
    from modules.metabolic import _rmr_mifflin, _PAL_PARAM
    from knowledge_base import load_system
    p = user.get("profile", {})
    weight = float(p.get("weight_kg", 75)); height = float(p.get("height_cm", 175))
    age = float(p.get("age", 35)); sex = p.get("sex", "male") if p.get("sex") in ("male", "female") else "male"
    activity = p.get("activity") if p.get("activity") in ("sedentary", "moderate", "active") else "sedentary"
    kb = load_system("metabolic")
    rmr = _rmr_mifflin(weight, height, age, sex, kb)
    pal = kb[_PAL_PARAM[activity]].value
    maintenance = rmr * pal
    weighins = user.get("logs", {}).get("weighins", [])
    logged = next((w.get("mean_daily_intake_kcal") for w in reversed(weighins)
                   if isinstance(w.get("mean_daily_intake_kcal"), (int, float))), None)
    return {"maintenance_kcal": round(maintenance), "rmr_kcal": round(rmr),
            "activity": activity, "logged_intake_kcal": round(logged) if logged else None,
            "evidence": "moderate",
            "citations": ["Mifflin-St Jeor RMR x FAO/WHO activity factor (maintenance estimate)"]}


def _cap_project_weight(args, user):
    """Predict the user's weight forward under their current (or a given) lifestyle — the
    energy-balance projection, NOT the minute-by-minute ODE engine. This is the right tool
    for 'what will I weigh in a year', 'am I going to gain/lose weight' questions."""
    from modules.metabolic import project_weight
    from personalization.user_store import metabolic_params
    p = user.get("profile", {})
    weight_kg = float(p.get("weight_kg", 75))
    height_cm = float(p.get("height_cm", 175))
    age = float(p.get("age", 35))
    sex = p.get("sex", "male") if p.get("sex") in ("male", "female") else "male"
    activity = p.get("activity") if p.get("activity") in ("sedentary", "moderate", "active") else "sedentary"
    # prefer a horizon parsed from the user's own words ("a month" -> 30) over the LLM's
    # imprecise guess (it returned ~36 days for "a month"); fall back to arg, then 1 year.
    horizon = horizon_from_text(args.get("query", ""))
    if horizon is None:
        try:
            horizon = int(args.get("horizon_days", 365))
        except (TypeError, ValueError):
            horizon = 365
    horizon = max(7, min(horizon, 5 * 365))
    personal = metabolic_params(user)   # Bayesian RMR from their weigh-ins (may be None)

    # Daily intake: explicit arg > their logged mean intake > assume they eat at
    # maintenance (so "same lifestyle" -> weight is ~stable, and we say so honestly).
    intake = args.get("daily_intake_kcal")
    assumed_maintenance = False
    if intake is None:
        weighins = user.get("logs", {}).get("weighins", [])
        intakes = [w.get("mean_daily_intake_kcal") for w in weighins
                   if isinstance(w.get("mean_daily_intake_kcal"), (int, float))]
        intake = intakes[-1] if intakes else None
    if intake is None:
        probe = project_weight(weight_kg, height_cm, age, sex, 2000.0, activity,
                               personal=personal, horizon_days=7)
        intake = probe.tdee_estimate           # eat at maintenance -> stable
        assumed_maintenance = True

    res = project_weight(weight_kg, height_cm, age, sex, float(intake), activity,
                         personal=personal, horizon_days=horizon)
    return {
        "current_weight_kg": round(weight_kg, 1),
        "final_weight_kg": round(res.final_weight, 1),
        "final_weight_ci_kg": [round(res.final_weight_ci[0], 1), round(res.final_weight_ci[1], 1)],
        "delta_kg": round(res.delta_kg, 1),
        "horizon_days": horizon,
        "daily_intake_kcal": round(float(intake)),
        "tdee_estimate_kcal": round(res.tdee_estimate),
        "assumed_eating_at_maintenance": assumed_maintenance,
        "top_driver": res.drivers[0][0] if res.drivers else None,
        "evidence": res.evidence.value, "citations": list(res.citations),
    }


def _cap_simulate(args, user):
    from orchestration.tools import dispatch
    from orchestration.context import current_user
    tok = current_user.set(user)
    try:
        return dispatch("simulate_scenario", args or {})
    finally:
        current_user.reset(tok)


def _cap_bac(args, user):
    from modules.hepatic import Drink, compute_bac
    p = user.get("profile", {})
    n = float(args.get("standard_drinks", 0) or 0)
    res = compute_bac([Drink.standard(n)], weight_kg=float(p.get("weight_kg", 75)),
                      sex=p.get("sex", "male"))
    return {"standard_drinks": n, "peak_bac": round(res.peak_bac, 3),
            "time_to_sober_h": round(res.time_to_sober_h, 1), "evidence": res.evidence.value,
            "citations": list(res.citations)}


def _cap_fact(args, user):
    from orchestration.websearch import search_facts
    return search_facts(args.get("query", ""), max_results=int(args.get("max_results", 3)))


def _cap_optimize(args, user):
    from personalization.optimize import optimize
    return optimize(user, args.get("goal") or args.get("query") or "")


def _cap_remember(args, user):
    from personalization import user_store
    uid = (user or {}).get("user_id")
    if not uid or not user_store.exists(uid):
        return {"saved": False, "reason": "no stored user"}
    patch = {}
    if args.get("note"):
        patch["notes"] = [str(args["note"])]
    if isinstance(args.get("profile"), dict) and args["profile"]:
        patch["profile"] = args["profile"]
    if isinstance(args.get("medical"), dict) and args["medical"]:
        patch["medical"] = args["medical"]
    if not patch:
        return {"saved": False, "reason": "nothing durable to save"}
    user_store.update(uid, patch)
    return {"saved": True, **patch, "evidence": "strong"}


CAPABILITIES: dict[str, dict] = {
    "body_state": {"label": "simulate current body systems", "needs_args": False,
                   "desc": "the user's current simulated state. Pass args {systems:[...]} to "
                           "scope it to only the RELEVANT systems (any of: metabolic, "
                           "cardiovascular, sleep, stress, activity, hepatic, digestive) — do "
                           "this, don't pull the whole body unless the question is genuinely "
                           "whole-body ('how am I doing overall'). Include this only when the "
                           "question is about the user's current bodily state.",
                   "run": _cap_body_state},
    "symptom_causes": {"label": "differential (symptom -> causes)", "needs_args": True,
                       "desc": "ranked likely causes of a symptom the user reports, "
                               "personalized to their data. args: {symptom}",
                       "run": _cap_symptom_causes},
    "patterns": {"label": "patterns in your data", "needs_args": False,
                 "desc": "find real correlations/trends in the user's own longitudinal data "
                         "(e.g. 'your glucose runs higher on poor-sleep days', 'your resting HR "
                         "has risen'). Use for 'what patterns / have you noticed / any trends "
                         "in my data'.", "run": _cap_patterns},
    "health_reasoning": {"label": "reason from your record (labs, history)", "needs_args": True,
                         "desc": "for questions about systems the simulator does NOT model "
                                 "(immune / getting sick, thyroid, hormones, interpreting a "
                                 "lab value): reason from the user's stored labs, conditions, "
                                 "family history and age/sex against known science, each fact "
                                 "tagged by source. Not a simulation. args: {query}",
                         "run": _cap_health_reasoning},
    "heart_risk": {"label": "cardiovascular risk", "needs_args": False,
                   "desc": "10-year cardiovascular risk from the user's labs (Framingham)",
                   "run": _cap_heart_risk},
    "bmi": {"label": "BMI", "needs_args": False,
            "desc": "body-mass index + category + healthy weight range", "run": _cap_bmi},
    "daily_energy": {"label": "daily calories (maintenance)", "needs_args": False,
                     "desc": "estimate the user's maintenance calories/day from their body + "
                             "activity (and their logged intake if any). Use for 'how many "
                             "calories do I eat / need per day'.", "run": _cap_daily_energy},
    "project_weight": {"label": "project future weight (energy balance)", "needs_args": False,
                       "desc": "predict the user's weight over time under their current or a "
                               "given lifestyle. USE THIS (not `simulate`) for 'what will I "
                               "weigh / gain or lose weight over N months/years' questions. "
                               "args: {horizon_days, daily_intake_kcal (optional)}",
                       "run": _cap_project_weight},
    "simulate": {"label": "run body-simulation engine over time", "needs_args": True,
                 "desc": "evolve the body minute-by-minute for a SHORT scenario made of "
                         "concrete events (hours, not months). Use ONLY when the question "
                         "gives a timeline of things happening (eat/drink/exercise/sleep) and "
                         "wants the physiological time-course. NOT for long-term weight "
                         "(use project_weight). args: {duration_min, meals:[{t_min,carbs_g}], "
                         "drinks:[{t_min,standard_drinks}], caffeine:[{t_min,mg}], "
                         "insulin:[{t_min,units}], exercise:[{start_min,end_min,intensity_mets}], "
                         "doses, stressors, sleep, focus:[...]}",
                 "run": _cap_simulate},
    "alcohol_bac": {"label": "blood-alcohol / time to sober", "needs_args": True,
                    "desc": "peak BAC and time to sober. args: {standard_drinks}",
                    "run": _cap_bac},
    "food_lookup": {"label": "food calories / macros", "needs_args": True,
                    "desc": "calories and macros for a named food or drink from the built-in "
                            "food database. USE THIS (not fact_lookup) for 'how many calories "
                            "/ how much sugar / macros in <food>'. args: {food, grams}",
                    "run": _cap_food_lookup},
    "fact_lookup": {"label": "web fact lookup", "needs_args": True,
                    "desc": "look up an established external fact/guideline the models and "
                            "food database don't cover. args: {query}",
                    "run": _cap_fact},
    "optimize": {"label": "optimize for a goal", "needs_args": True,
                 "desc": "rank the interventions that most improve a goal (heart risk, "
                         "weight, glucose) by counterfactual simulation. args: {goal}",
                 "run": _cap_optimize},
    "remember": {"label": "save a durable fact to the profile", "needs_args": True,
                 "desc": "when the user states a durable fact about themselves (a "
                         "condition, medication, diet, allergy, goal), save it. args: "
                         "{note, profile:{...}, medical:{conditions/medications/allergies}}",
                 "run": _cap_remember},
}


# ---------------------------------------------------------------------------

@dataclass
class Step:
    capability: str
    why: str
    args: dict = field(default_factory=dict)


@dataclass
class StepResult:
    capability: str
    label: str
    why: str
    args: dict
    result: dict
    evidence: str
    citations: list[str]


# --- deterministic planner (always available, no LLM) -----------------------

_ALCOHOL = ("drink", "beer", "wine", "vodka", "whiskey", "alcohol", "shot", "pint", "drunk", "drive")
_HEART = ("heart", "cardiovascular", "cardiac", "cholesterol", "blood pressure", "stroke", "cvd", "risk")
_BMI = ("bmi", "overweight", "obese", "underweight", "body mass", "how much should i weigh")
_SIM = ("simulate", "walk me through", "what happens", "over the next", "if i eat", "if i drink",
        "if i take", "after i", "throughout the day", "hour by hour", "trajectory")
_WHOLE_BODY = ("how am i", "how's my body", "hows my body", "am i healthy", "my health",
               "overall", "how am i doing", "everything", "all my systems", "full picture",
               "my body", "at risk", "risk for", "what do you know about", "should i worry")

# Which body systems a question is actually about -> so body_state is scoped to those,
# not padded with seven unrelated tiles. Keyword hit => that system is relevant.
_SYSTEM_KEYWORDS = {
    "metabolic": ("glucose", "blood sugar", "sugar", "insulin", "weight", "bmi", "metabolism",
                  "metabolic", "triglyceride", "diabet", "carb", "calorie", "overweight", "obese"),
    "cardiovascular": ("heart", "cardiac", "cardiovascular", "blood pressure", "cholesterol",
                       "stroke", "cvd", "hypertension", "pulse", "circulation"),
    "sleep": ("sleep", "insomnia", "rem", "nap", "tired", "fatigue", "drowsy", "rested", "wake"),
    "stress": ("stress", "anxiet", "cortisol", "overwhelm", "burnout", "tense", "relax"),
    "activity": ("steps", "exercise", "workout", "fitness", "walk", "run", "gym", "training", "active"),
    # NB: no bare "drink" — it matches "drink 3 coffees" and made a caffeine question report
    # blood-alcohol. Only genuinely alcohol-specific words belong here.
    "hepatic": ("alcohol", "liquor", "liver", "beer", "wine", "hangover", "bac", "drunk",
                "standard drink"),
    "digestive": ("digest", "gut", "stomach", "bloat", "constipat", "bowel", "poop", "ibs", "nausea"),
}

_WEIGHT_FUTURE = ("weight going to be", "will i weigh", "gonna weigh", "future weight",
                  "gain weight", "lose weight", "put on weight", "weight be in", "weight after")
_OPT = ("what should i change", "biggest change", "one thing", "most impact",
        "how do i improve", "how can i lower", "how can i improve", "best way to")

# "On-demand" analyses — heavy/specific tools that must only run when the QUESTION calls for
# them. The LLM planner (a small model) tends to pile these on unprompted (e.g. a weight
# projection when the user only stated a fact), so every planned on-demand step is checked
# against the same relevance signals the deterministic planner uses, and dropped otherwise.
_ONDEMAND = {"project_weight", "heart_risk", "bmi", "optimize", "alcohol_bac", "simulate",
             "food_lookup"}
_FOOD_Q = ("calorie", "kcal", "how much sugar", "macros", "how many carbs", "how much protein",
           "how much fat", "nutrition", "carbs in", "sugar in", "fibre", "fiber")


def _weight_future(q: str) -> bool:
    return any(k in q for k in _WEIGHT_FUTURE) or ("weight" in q and any(
        k in q for k in ("year", "month", "week", "future", "going to", "will i", "projected", "in a")))


def _horizon_phrase(days) -> str:
    """Human time span: days/weeks/months/years — not '0.1 yr'."""
    d = int(days)
    if d <= 24:
        return f"{d} days"
    if 25 <= d <= 34:
        return "~1 month"
    if d < 75:
        return f"~{round(d / 7)} weeks"
    if d < 340:
        return f"~{round(d / 30)} months"
    yrs = d / 365.0
    return "~1 year" if abs(yrs - 1) < 0.15 else f"~{yrs:.1f} years"


def horizon_from_text(q: str) -> int | None:
    """Parse a time horizon from the question -> days, or None if none stated.
    'a month'->30, '6 weeks'->42, '3 months'->90, etc."""
    ql = (q or "").lower()
    m = re.search(r"(\d+)\s*(day|week|month|year)", ql)
    if m:
        return int(m.group(1)) * {"day": 1, "week": 7, "month": 30, "year": 365}[m.group(2)]
    if re.search(r"\bmonth\b", ql):
        return 30
    if re.search(r"\bweek\b", ql):
        return 7
    if re.search(r"\byear\b", ql):
        return 365
    return None


def _ondemand_relevant(cap: str, q: str, args: dict) -> bool:
    """Is this on-demand analysis actually warranted by the question? Keeps the LLM from
    running analyses nobody asked for, without touching legitimately-scoped ones."""
    if cap == "project_weight":
        return _weight_future(q)
    if cap == "heart_risk":
        return any(k in q for k in _HEART)
    if cap == "bmi":
        return any(k in q for k in _BMI)
    if cap == "optimize":
        return any(k in q for k in _OPT)
    if cap == "alcohol_bac":
        return any(k in q for k in _ALCOHOL)
    if cap == "simulate":
        # a real scenario: either time-course phrasing, or the LLM supplied concrete events
        has_events = any(args.get(k) for k in ("meals", "foods", "drinks", "caffeine", "water",
                                               "doses", "exercise", "stressors", "sleep", "ambient"))
        return has_events or any(k in q for k in _SIM)
    if cap == "food_lookup":
        # only when the QUESTION asks about a food's nutrition — stops the LLM looking up
        # example foods it invented ("HDL-boosting salmon") on an unrelated question
        return any(k in q for k in _FOOD_Q)
    return True


def _relevant_systems(q: str) -> list[str]:
    return [k for k, kws in _SYSTEM_KEYWORDS.items() if any(w in q for w in kws)]


# Rough MET intensities for exercise the user names in plain words (ACSM compendium bands).
_EXERCISE_METS = {"running": 9.0, "run": 9.0, "jog": 7.0, "jogging": 7.0, "sprint": 12.0,
                  "swim": 8.0, "swimming": 8.0, "cycle": 8.0, "cycling": 8.0, "bike": 8.0,
                  "gym": 6.0, "workout": 6.0, "lift": 5.0, "exercise": 6.0,
                  "walk": 3.5, "walking": 3.5, "hike": 6.0, "hiking": 6.0}


def scenario_args(question: str) -> dict | None:
    """Build simulate() args from plain English — foods, coffee, drinks, exercise, duration.

    Without this the engine could only ever be driven by the LLM planner (`simulate` needs
    args), so whenever the LLM was unavailable or rate-limited the flagship simulator
    silently degraded to a body-stats dump. Returns None when nothing is actually happening
    in the question (an event-free run has nothing to evolve)."""
    from simulation.foods import find_foods
    ql = question.lower()
    args: dict = {}

    foods = find_foods(question)
    if foods:
        args["foods"] = [{"t_min": 0, "food": f} for f in foods]

    m = re.search(r"(\d+)\s*(?:cups?\s*of\s*)?(?:coffees?|espressos?|lattes?)", ql)
    if m:
        args["caffeine"] = [{"t_min": 0, "mg": 95.0 * min(int(m.group(1)), 20)}]
    elif re.search(r"\b(coffee|espresso|latte|caffeine)\b", ql):
        args["caffeine"] = [{"t_min": 0, "mg": 95.0}]

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:standard\s+)?(?:drinks?|beers?|shots?|"
                  r"glass(?:es)?\s+of\s+wine|pints?)", ql)
    if m:
        args["drinks"] = [{"t_min": 0, "standard_drinks": float(m.group(1))}]

    for word, mets in _EXERCISE_METS.items():
        if re.search(rf"\b{word}\b", ql):
            args["exercise"] = [{"start_min": 60, "end_min": 90, "intensity_mets": mets}]
            break

    if not args:
        return None
    m = re.search(r"next\s+(\d+)\s*hour", ql)
    args["duration_min"] = min(int(m.group(1)) * 60, 24 * 60) if m else 240
    return args


# --- durable-fact capture (deterministic; doesn't depend on the small LLM) --------------
# Family relations whose "had/has X" we should remember as family history.
_KIN = (r"mom|mother|dad|father|sister|brother|son|daughter|grandmother|grandma|grandfather|"
        r"grandpa|parent|parents|aunt|uncle|cousin|sibling|family")
_STOP_TAIL = re.compile(r"[,.;!?]|\b(and|but|so|because|which|that|what|how|why|when|do|does|"
                        r"could|would|should|is|are|can|will)\b")


def _clip(text: str, n: int = 60) -> str:
    """Take the condition phrase up to the first clause boundary / stop word."""
    text = text.strip().strip(".,;!? ")
    m = _STOP_TAIL.search(text)
    return (text[:m.start()] if m else text).strip().strip(".,;!? ")[:n]


_BAD_COND = ("beer", "wine", "drink", "coffee", "soda", "eat", "ate", "meal", "snack",
             "water", "question", "idea", "time", "issue", "cold", "headache")


def durable_facts(question: str) -> dict | None:
    """Detect stable self/family facts worth persisting. Returns args for the `remember`
    capability ({note, medical/profile}) or None. Conservative — only clear statements.

    Collects EVERY kind of fact in the message, not just the first: a real user writes "I
    have type 2 diabetes ... and an allergy to penicillin" in one breath, and an if/elif
    chain would silently keep only one of them."""
    ql = question.strip().lower()
    notes: list[str] = []
    medical: dict = {}
    profile: dict = {}

    def _add(key, value, note):
        if value and value not in medical.setdefault(key, []):
            medical[key].append(value)
            notes.append(note)

    for m in re.finditer(rf"\b(?:my\s+)?({_KIN})\b\s+(?:had|has|have|got|developed|"
                         rf"suffered from|was diagnosed with)\s+(.+)", ql):
        kin, cond = m.group(1), _clip(m.group(2))
        if cond and cond not in ("it", "one", "the same"):
            _add("family_history", f"{kin}: {cond}", f"Family history - {kin} had {cond}.")

    # Allergies first, so "I have a peanut allergy" is an allergy rather than a condition.
    for pat in (r"\bi(?:'m| am)?\s+allergic to\s+([^,.;]+)",
                r"\bi have (?:a|an)\s+(.+?)\s+allergy",
                r"\ban? allergy to\s+([^,.;]+)"):
        for m in re.finditer(pat, ql):
            allg = _clip(m.group(1))
            if allg:
                _add("allergies", allg, f"Allergic to {allg}.")

    # Explicit diagnosis phrasings are safe; a bare "I have X" is only a condition when X
    # isn't a quantity ("I have 4 beers") or an everyday object ("I have a question").
    # NB: the allergy check is scoped to the MATCHED phrase — checking the whole message
    # meant one stray "allergy" anywhere silently disabled all condition capture.
    for pat in (r"\bdiagnosed with\s+(.+)",                 # "just got diagnosed with diabetes"
                r"\bi(?:'m| am)?\s+(?:was\s+)?(?:suffer from)\s+(.+)",
                r"\bi\s+have\s+(?!\s*\d)(.+)"):
        m = re.search(pat, ql)
        if not m:
            continue
        cond = _clip(m.group(1))
        if (cond and len(cond) > 2 and not cond[0].isdigit()
                and "allerg" not in cond and not any(b in cond for b in _BAD_COND)):
            _add("conditions", cond, f"Has condition: {cond}.")
            break

    m = re.search(r"\bi(?:'m| am)?\s+(?:currently\s+)?(?:on|taking|take)\s+(.+)", ql)
    if m:
        med = _clip(m.group(1))
        if med and len(med) > 2 and "allerg" not in med:
            _add("medications", med, f"Takes {med}.")
    m = re.search(r"\bi(?:'m| am)?\s+(?:a\s+)?(vegetarian|vegan|pescatarian|"
                  r"keto|carnivore|gluten[- ]free|lactose intolerant)\b", ql)
    if m:
        profile["diet"] = m.group(1)
        notes.append(f"Diet: {m.group(1)}.")

    if not notes:
        return None
    args = {"note": " ".join(notes)}
    if medical:
        args["medical"] = {k: v for k, v in medical.items() if v}
    if profile:
        args["profile"] = profile
    return args


def deterministic_plan(question: str, user: dict) -> list[Step]:
    from pipeline.causal import match_symptom
    q = question.lower()
    p = user.get("profile", {})
    steps: list[Step] = []

    fact = durable_facts(question)
    if fact:
        steps.append(Step("remember", "the user stated a durable fact about themselves", fact))

    sym = match_symptom(question)
    if sym:
        steps.append(Step("symptom_causes", f"the question describes a symptom ('{sym}')",
                          {"symptom": question}))
    if any(k in q for k in ("pattern", "have you noticed", "noticed anything", "any trend",
                            "trends in my", "in my data", "correlat", "what do you see in my")):
        steps.append(Step("patterns", "the question asks about patterns/trends in the user's data"))
    # T3: questions about systems the engine doesn't model (immune, thyroid, labs, hormones)
    if any(k in q for k in ("immune", "get sick", "getting sick", "infection", "sick a lot",
                            "thyroid", "hormone", "white blood", "wbc", "vitamin d", "ferritin",
                            "iron", "cholesterol", "hba1c", "a1c", "my labs", "blood test",
                            "lab result", "crp", "inflammation", "menopause")):
        steps.append(Step("health_reasoning", "the question is about a lab/immune/hormone topic "
                          "the engine doesn't simulate", {"query": question}))
    if any(k in q for k in ("calorie", "kcal", "how much sugar", "macros", "how many carbs")):
        from simulation.foods import resolve_food
        if resolve_food(question):
            steps.append(Step("food_lookup", "the question asks for a food's calories/macros",
                              {"food": question}))
    if any(k in q for k in ("how many calories do i", "calories do i eat", "calories i eat",
                            "calories i need", "calories should i", "my daily calories",
                            "how much do i eat", "maintenance calorie", "how many calories a day")):
        steps.append(Step("daily_energy", "the question asks about the user's daily calories"))
    if _weight_future(q):
        steps.append(Step("project_weight", "the question asks to predict future weight",
                          {"query": question}))
    if any(k in q for k in _HEART) and all(p.get(k) is not None for k in ("total_chol", "hdl", "sbp")):
        steps.append(Step("heart_risk", "the question touches cardiovascular health and labs are on file"))
    if any(k in q for k in _BMI) and not any(s.capability == "project_weight" for s in steps):
        steps.append(Step("bmi", "the question is about body weight / BMI"))
    if any(k in q for k in _OPT):
        steps.append(Step("optimize", "the question asks what change would help most",
                          {"goal": question}))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:standard\s+)?(?:drink|beer|shot|pint|glass(?:es)? of wine)", q)
    if m and any(k in q for k in _ALCOHOL):
        steps.append(Step("alcohol_bac", "the question involves a specific amount of alcohol",
                          {"standard_drinks": float(m.group(1))}))
    # Drive the engine WITHOUT an LLM when the question describes something happening.
    if any(k in q for k in _SIM):
        sc = scenario_args(question)
        if sc:
            steps.append(Step("simulate", "the question describes a scenario to run forward", sc))

    # Ground in the CURRENT body only when relevant — and scope it to the systems the
    # question is actually about (not a blanket seven-tile dump on every question).
    rel = _relevant_systems(q)
    already = {s.capability for s in steps}
    if any(k in q for k in _WHOLE_BODY):
        steps.insert(0, Step("body_state", "the question is about overall health — whole body"))
    elif rel and not ({"symptom_causes", "food_lookup"} & already):
        steps.insert(0, Step("body_state", f"grounding in the relevant system(s): {', '.join(rel)}",
                             {"systems": rel}))
    elif not steps:
        steps.insert(0, Step("body_state", "grounding the answer in the user's current body"))
    return steps


# --- LLM planner (refines; supplies args for scenario-type steps) -----------

def _capabilities_menu() -> str:
    return "\n".join(f"- {name}: {c['desc']}" for name, c in CAPABILITIES.items())


# NB: uses a %%MENU%% placeholder (not str.format) because the prompt contains literal
# JSON braces that would otherwise be parsed as format fields.
_PLAN_SYSTEM = (
    "You are the PLANNER for a physiological body simulator. You do NOT answer the "
    "question. You output a PLAN: the minimal set of grounded analyses ('capabilities') "
    "needed to answer it thoroughly, each with a one-line reason and any required args. "
    "Decompose multi-part questions so EVERY part is covered. Prefer running the "
    "simulation/analyses over guessing. Available capabilities:\n%%MENU%%\n\n"
    "Reply with ONLY a JSON object: {\"steps\": [{\"capability\": <name>, \"why\": <reason>, "
    "\"args\": {<...>}}]}. Use the exact capability names. Only include 'body_state' when the "
    "question is about the user's CURRENT bodily state, and when you do, scope it with "
    "args {\"systems\":[...]} to just the relevant systems (metabolic, cardiovascular, sleep, "
    "stress, activity, hepatic, digestive) — do not pull the whole body for a narrow "
    "question. For long-term weight use 'project_weight', NOT 'simulate'. If the user states "
    "a durable fact about themselves (condition, family history, medication, allergy, diet), "
    "add a 'remember' step. Omit args (use {}) when not needed."
)


def _history_text(history: list | None) -> str:
    if not history:
        return ""
    turns = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-6:]]
    return "\n\nRecent conversation (for context — resolve pronouns/references from it):\n" + "\n".join(turns)


def llm_plan(question: str, profile: UserProfile, user: dict, user_context: str,
             history: list | None = None) -> list[Step] | None:
    """Ask the active LLM for a structured plan; None if unavailable/failed."""
    from orchestration.orchestrate import active_backend
    backend = active_backend()
    if backend == "deterministic":
        return None
    try:
        from dataclasses import asdict
        system = _PLAN_SYSTEM.replace("%%MENU%%", _capabilities_menu())
        if user_context:
            system += f"\n\nUser context: {user_context}"
        content = f"User profile: {asdict(profile)}{_history_text(history)}\nQuestion: {question}"
        data = None
        for _ in range(2):                       # one retry for flaky JSON
            data = _extract_json(_chat_once(backend, system, content, json_mode=True))
            if data and data.get("steps"):
                break
        raw_steps = (data or {}).get("steps") or []
        steps = []
        for s in raw_steps:
            cap = str(s.get("capability", "")).strip()
            if cap in CAPABILITIES:
                steps.append(Step(cap, str(s.get("why", "")), dict(s.get("args") or {})))
        return steps or None
    except Exception:
        return None


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)   # first {...} block, tolerates prose/fences
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _chat_once(backend: str, system: str, user_msg: str, json_mode: bool = False) -> str:
    """A single, tool-free chat completion on the active backend (for plan + synthesis).
    `json_mode` asks the model for strictly valid JSON (used for planning)."""
    if backend == "groq":
        from openai import OpenAI
        from orchestration.config import groq_api_key, groq_model
        client = OpenAI(api_key=groq_api_key(), base_url="https://api.groq.com/openai/v1")
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        try:
            r = client.chat.completions.create(
                model=groq_model("openai/gpt-oss-20b"), temperature=0,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user_msg}], **kw)
        except Exception:
            if json_mode:   # model/endpoint may not support json_object -> retry plain
                r = client.chat.completions.create(
                    model=groq_model("openai/gpt-oss-20b"), temperature=0,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user_msg}])
            else:
                raise
        return (r.choices[0].message.content or "").strip()
    raise RuntimeError("one-shot chat only wired for groq; use deterministic")


# --- execute + synthesize ---------------------------------------------------

def execute(steps: list[Step], user: dict) -> list[StepResult]:
    out: list[StepResult] = []
    seen: set = set()
    for st in steps:
        key = (st.capability, json.dumps(st.args, sort_keys=True))
        if key in seen or st.capability not in CAPABILITIES:
            continue
        seen.add(key)
        cap = CAPABILITIES[st.capability]
        try:
            res = cap["run"](st.args, user)
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}", "evidence": "none", "citations": []}
        out.append(StepResult(
            capability=st.capability, label=cap["label"], why=st.why, args=st.args,
            result=res, evidence=str(res.get("evidence", "strong") if isinstance(res, dict) else "strong"),
            citations=list(res.get("citations", []) if isinstance(res, dict) else [])))
    return out


_RANK = {"strong": 3, "moderate": 2, "weak": 1, "none": 0}


def aggregate_evidence(results: list[StepResult]) -> str:
    grades = [r.evidence for r in results if r.evidence in _RANK and r.evidence != "none"]
    return min(grades, key=lambda g: _RANK.get(g, 0)) if grades else "weak"


def _summarize(sr: StepResult) -> str:
    r = sr.result
    if not isinstance(r, dict):
        return str(r)
    if r.get("error"):
        return f"(could not run: {r['error']})"
    if r.get("unavailable"):
        return f"(not available: {r['unavailable']})"
    if sr.capability == "body_state":
        return r.get("summary", "")
    if sr.capability == "symptom_causes":
        if not r.get("matched"):
            return "no causal map for that symptom yet"
        cs = r.get("causes", [])[:5]
        return "; ".join(f"{c['label']} ({c['evidence']}"
                         + (f": {', '.join(c.get('personal_reasons', []))}" if c.get('personal_reasons') else "")
                         + ")" for c in cs)
    if sr.capability == "heart_risk":
        return f"{r.get('risk_10yr_pct')}% 10-yr risk; {r.get('heart_age','')}; biggest lever: {r.get('top_lever')}"
    if sr.capability == "bmi":
        return f"BMI {r.get('bmi')} ({r.get('category')})"
    if sr.capability == "daily_energy":
        s = (f"maintenance ~{r.get('maintenance_kcal')} kcal/day (RMR {r.get('rmr_kcal')}, "
             f"{r.get('activity')})")
        if r.get("logged_intake_kcal"):
            s += f"; your logged intake ~{r.get('logged_intake_kcal')} kcal/day"
        return s
    if sr.capability == "project_weight":
        base = (f"{r.get('current_weight_kg')} kg now -> {r.get('final_weight_kg')} kg in "
                f"{_horizon_phrase(r.get('horizon_days', 365))} (change {r.get('delta_kg'):+.1f} kg, "
                f"90% CI {r.get('final_weight_ci_kg')})")
        if r.get("assumed_eating_at_maintenance"):
            base += " — assuming you eat at maintenance (no intake logged)"
        return base
    if sr.capability == "remember":
        if not r.get("saved"):
            return f"(not saved: {r.get('reason', 'nothing durable')})"
        note = r.get("note") or (r.get("notes") or [None])[0]
        med = r.get("medical") or {}
        detail = note or "; ".join(f"{k}: {', '.join(map(str, v))}" for k, v in med.items()
                                   if isinstance(v, list)) or "profile updated"
        return f"saved to your profile: {detail}"
    if sr.capability == "alcohol_bac":
        return f"{r.get('standard_drinks')} drinks -> peak BAC {r.get('peak_bac')}, sober in ~{r.get('time_to_sober_h')} h"
    if sr.capability == "simulate":
        s = r.get("summary", {})
        if isinstance(s, dict) and s:
            # Report the variables that actually MOVED, not the first five: a drinks scenario
            # would otherwise show flat glucose/insulin and hide the blood-alcohol entirely.
            def _swing(v):
                try:
                    lo, hi = float(v["min"]), float(v["max"])
                except (TypeError, ValueError, KeyError):
                    return 0.0
                return (hi - lo) / max(abs(hi), 1e-6)
            # energy is cumulative (always rises from 0) so it would always win — rank it last
            ranked = sorted(((k, v) for k, v in s.items() if k != "energy_expended_kcal"),
                            key=lambda kv: _swing(kv[1]), reverse=True)
            shown = [kv for kv in ranked if _swing(kv[1]) > 1e-3][:5] or ranked[:3]
            body = "; ".join(f"{k.replace('_',' ')}: {v.get('min')}-{v.get('max')} (end {v.get('end')})"
                             for k, v in shown)
        else:
            body = "simulated"
        prefix = "".join(f"[{r[k]}] " for k in ("horizon_note", "insulin_note") if r.get(k))
        return prefix + body
    if sr.capability == "patterns":
        return "; ".join(p["pattern"] for p in r.get("patterns", [])[:4]) or "no clear patterns yet"
    if sr.capability == "health_reasoning":
        facts = r.get("facts", [])
        return " | ".join(f"[{f['source']}] {f['fact']}" for f in facts[:4]) or "no relevant record"
    if sr.capability == "food_lookup":
        return (f"{r.get('food')} ({r.get('grams')} g): {r.get('kcal')} kcal, "
                f"{r.get('carbs_g')} g carbs, {r.get('protein_g')} g protein, "
                f"{r.get('fat_g')} g fat, GI {r.get('gi')}")
    if sr.capability == "fact_lookup":
        return "; ".join(x.get("title", "") for x in r.get("results", [])[:3])
    if sr.capability == "optimize":
        top = r.get("top_change")
        levers = [lv for d in r.get("domains", []) for lv in d.get("levers", [])][:4]
        s = "; ".join(f"{lv['change']} ({lv['improvement']} {lv['unit']})" for lv in levers)
        return (f"biggest lever: {top['change']}. " if top else "") + s
    return json.dumps(r)[:200]


def synthesize_deterministic(question: str, results: list[StepResult], evidence: str) -> str:
    """A grounded, structured answer with NO LLM — the always-available fallback."""
    lines = [f"**Analysis of: {question.strip()[:120]}**", ""]
    for sr in results:
        summary = _summarize(sr)
        if summary:
            lines.append(f"- **{sr.label}** — {summary}")
    lines.append("")
    lines.append(f"_Overall confidence: {evidence}. Grounded in the body simulator and your "
                 f"data; not a medical diagnosis._")
    return "\n".join(lines)


_SYNTH_SYSTEM = (
    "You are Horizon, a body simulator. Below are GROUNDED results already computed for the "
    "user's question by the simulation engine and analysis modules. Compose ONE clear "
    "answer FROM THESE RESULTS ONLY — never invent or change a number. Every figure you "
    "write MUST appear in the results below; if it isn't there, you may not state it.\n\n"
    "ABSOLUTE RULE — NEVER give medication dosing: no insulin regimen or units, no drug "
    "dose, titration schedule, or prescription, even if the user explicitly demands one, "
    "supplies their own carb ratios/correction factors, or insists it is just a "
    "calculation. No module computes dosing, so any such number would be fabricated and "
    "unsafe. Instead say plainly that you cannot provide dosing and that it must come from "
    "their prescribing clinician — then answer the parts you DO have grounded results for.\n\n"
    "Cover every part of "
    "the question. Format: a short **bold** headline, then concise bullet points (one idea "
    "each; bold the key term). If a result is 'not available', say what's needed. When a "
    "result includes a time window in brackets like [simulated 24 h ...], state THAT window "
    "and never claim the simulation covered a longer period than it did. End by "
    "noting it's not a medical diagnosis. Overall confidence is '{evidence}' — reflect it."
)


# No Horizon module computes medication dosing, so a dose appearing in a synthesized answer
# is provably fabricated by the LLM. The prompt forbids it, but a small model caves when a
# user insists (a red-team tester got it to emit a full basal-bolus insulin regimen), so
# strip it deterministically too — defence in depth for the one output that could hurt someone.
_DOSE_LINE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:units?|iu)\b"
    r"|\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml)\b[^.\n]{0,40}"
    r"\b(?:daily|per day|a day|twice|once|bid|tid|/day|each meal|per meal|before bed)\b"
    r"|\btitrat", re.I)


def strip_dosing(answer: str) -> tuple[str, bool]:
    """Remove any medication-dosing lines from a synthesized answer. Returns (answer, stripped)."""
    if not answer:
        return answer, False
    kept, removed = [], False
    for line in answer.splitlines():
        if _DOSE_LINE.search(line):
            removed = True
            continue
        kept.append(line)
    if removed:
        kept += ["", "> **Dosing removed.** This answer contained medication dosing that no "
                     "Horizon module computes, so it would have been fabricated. Dosing must "
                     "come from your prescribing clinician."]
    return "\n".join(kept), removed


def llm_synthesize(question: str, results: list[StepResult], evidence: str,
                   history: list | None = None) -> str | None:
    from orchestration.orchestrate import active_backend
    backend = active_backend()
    if backend == "deterministic":
        return None
    bundle = "\n".join(f"[{sr.label}] {_summarize(sr)}" for sr in results)
    try:
        text = _chat_once(backend, _SYNTH_SYSTEM.format(evidence=evidence),
                          f"Question: {question}{_history_text(history)}\n\nComputed results:\n{bundle}")
        return text or None
    except Exception:
        return None


# --- top-level entry --------------------------------------------------------

def plan_and_answer(question: str, profile: UserProfile, user: dict, *,
                    user_context: str = "", use_llm: bool = True,
                    history: list | None = None) -> dict:
    """Full pipeline: plan -> execute -> synthesize. Returns the plan, every step's grounded
    result, and the composed answer. Fully functional without an LLM (deterministic plan +
    synthesis); the LLM refines the plan and prose when available. `history` = prior
    conversation turns, so the planner is memory-aware."""
    if not (question or "").strip():        # an empty box shouldn't dump a whole body report
        return {"question": question, "planner": "none", "synthesizer": "none", "plan": [],
                "steps": [], "answer": "Ask me something about your body and I'll simulate it.",
                "evidence": "none", "citations": [], "dosing_stripped": False}
    det_steps = deterministic_plan(question, user)
    llm_steps = llm_plan(question, profile, user, user_context, history) if use_llm else None
    if llm_steps:
        # honour the LLM's richer plan, but never drop the deterministic grounding
        # essentials (current body state; a symptom differential) it may have missed.
        planner_kind = "llm+deterministic"
        have = {s.capability for s in llm_steps}
        # Never drop deterministic essentials the small model tends to miss: a symptom
        # differential, saving a stated durable fact, and the correct weight-projection
        # tool. body_state is only carried if it's SCOPED to relevant systems — the
        # unscoped whole-body fallback is dropped when the LLM already has a real plan, so
        # narrow questions don't get padded with a seven-tile dump.
        _essential = ("symptom_causes", "remember", "project_weight", "health_reasoning",
                      "optimize", "patterns")
        carry = [s for s in det_steps if s.capability in _essential and s.capability not in have]
        carry += [s for s in det_steps if s.capability == "body_state" and s.args.get("systems")
                  and "body_state" not in have]
        steps = list(llm_steps) + carry
    else:
        planner_kind = "deterministic"
        steps = det_steps

    # Prune on-demand analyses the question doesn't warrant (stops the LLM volunteering an
    # unrequested weight projection / heart-risk / simulation on a simple statement).
    ql = question.lower()
    steps = [s for s in steps
             if s.capability not in _ONDEMAND or _ondemand_relevant(s.capability, ql, s.args)]

    results = execute(steps, user)
    evidence = aggregate_evidence(results)

    answer = llm_synthesize(question, results, evidence, history) if use_llm else None
    synth_kind = "llm"
    if not answer:
        answer = synthesize_deterministic(question, results, evidence)
        synth_kind = "deterministic"
    answer, dosing_stripped = strip_dosing(answer)   # safety net: never emit fabricated dosing

    citations: list[str] = []
    for sr in results:
        for c in sr.citations:
            if c not in citations:
                citations.append(c)

    return {
        "question": question,
        "planner": planner_kind,
        "synthesizer": synth_kind,
        "plan": [{"capability": s.capability, "label": CAPABILITIES[s.capability]["label"],
                  "why": s.why, "args": s.args} for s in steps if s.capability in CAPABILITIES],
        "steps": [{"capability": sr.capability, "label": sr.label, "why": sr.why,
                   "args": sr.args, "summary": _summarize(sr), "evidence": sr.evidence,
                   "result": sr.result} for sr in results],
        "answer": answer,
        "evidence": evidence,
        "citations": citations,
        "dosing_stripped": dosing_stripped,   # true = the LLM tried to emit fabricated dosing
    }
