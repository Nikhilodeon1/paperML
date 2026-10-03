"""Tool schemas + deterministic dispatcher for Layer 4.

Each module is exposed as a function-calling "tool": a JSON schema (the contract an
LLM would target) plus a Python handler that actually invokes Layer 2/3 and returns a
plain-dict result. The schemas are the same shape used by mainstream tool-use APIs, so
wiring an LLM later is just: give it TOOL_SCHEMAS, let it pick a tool + args, then call
`dispatch(name, args)`.

Crucially, every handler returns the module's own evidence level and confidence label
unchanged — the three-outcome rule (brief 8) is enforced by the modules, and the
orchestration layer must not paper over it.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from knowledge_base import load_raw
from modules.anthropometric import compute_bmi
from modules.cardiovascular import compute_cvd_risk
from modules.hepatic import Drink, compute_bac
from modules.metabolic import project_weight
from modules.sleep import SleepModel
from pipeline.graph import has_edge, run_alcohol_then_sleep

# A single shared sleep model (training is cheap but not free; reuse across calls).
_sleep_model: SleepModel | None = None


def _get_sleep_model() -> SleepModel:
    global _sleep_model
    if _sleep_model is None:
        _sleep_model = SleepModel().fit()
    return _sleep_model


# --- Handlers ---------------------------------------------------------------

def _h_bac(args: dict) -> dict:
    drinks = [Drink.standard(n=args.get("standard_drinks", 0), hour=0.0)]
    res = compute_bac(drinks, weight_kg=args["weight_kg"], sex=args["sex"],
                      fed=args.get("fed", False))
    out = {
        "peak_bac": res.peak_bac, "peak_bac_ci": res.peak_bac_ci,
        "time_to_sober_h": res.time_to_sober_h, "time_to_sober_ci": res.time_to_sober_ci,
        "drivers": res.drivers, "evidence": res.evidence.value,
        "confidence": res.confidence_label, "citations": list(res.citations),
    }

    # Elapsed-time + legal-limit status (grounds "can I drive?" type questions).
    hours_since = args.get("hours_since_drinking")
    limit = load_raw("facts")["facts"]["legal_bac_limit_driving_us"]
    if hours_since is not None:
        t = float(hours_since)
        bac_now = float(np.interp(t, res.times_h, res.bac_median))
        # First time at/after `t` where the curve drops below the legal limit.
        after = res.times_h >= t
        under = after & (res.bac_median < limit["value"])
        hrs_until_legal = (float(res.times_h[np.argmax(under)] - t)
                           if under.any() else float("inf"))
        out.update({
            "hours_since_drinking": t,
            "bac_now": round(bac_now, 4),
            "legal_driving_limit": limit["value"],
            "over_legal_limit_now": bac_now >= limit["value"],
            "hours_until_under_legal_limit": (round(hrs_until_legal, 1)
                                              if hrs_until_legal != float("inf") else None),
        })
        out["citations"].append(limit["citation"])
    return out


def _h_web_search(args: dict) -> dict:
    from orchestration.websearch import search_facts
    return search_facts(args["query"], max_results=int(args.get("max_results", 3)))


def _h_bmi(args: dict) -> dict:
    res = compute_bmi(weight_kg=args["weight_kg"], height_cm=args["height_cm"],
                      age=args.get("age"))
    return {
        "bmi": res.bmi, "category": res.category,
        "healthy_weight_kg_range": list(res.healthy_weight_kg_range),
        "evidence": res.evidence.value, "confidence": res.confidence_label,
        "note": res.note, "citations": res.citations,
    }


def _h_weight(args: dict) -> dict:
    res = project_weight(
        weight_kg=args["weight_kg"], height_cm=args["height_cm"], age=args["age"],
        sex=args["sex"], daily_intake_kcal=args["daily_intake_kcal"],
        activity=args.get("activity", "sedentary"),
        horizon_days=args.get("horizon_days", 365),
    )
    return {
        "final_weight": res.final_weight, "final_weight_ci": res.final_weight_ci,
        "delta_kg": res.delta_kg, "delta_kg_ci": res.delta_kg_ci,
        "tdee_estimate": res.tdee_estimate, "drivers": res.drivers,
        "evidence": res.evidence.value, "confidence": res.confidence_label,
        "citations": res.citations,
    }


def _h_cvd(args: dict) -> dict:
    res = compute_cvd_risk(
        age=args["age"], sex=args["sex"], total_chol=args["total_chol"],
        hdl=args["hdl"], sbp=args["sbp"], treated_bp=args.get("treated_bp", False),
        smoker=args.get("smoker", False), diabetic=args.get("diabetic", False),
    )
    return {
        "risk_10yr": res.risk_10yr, "risk_ci": res.risk_ci,
        "heart_age": res.heart_age_note, "drivers": res.drivers,
        "evidence": res.evidence.value, "confidence": res.confidence_label,
        "citations": res.citations,
    }


def _h_simulate_scenario(args: dict) -> dict:
    """Run the coupled body-simulation ENGINE forward over time for a scenario of events
    (meals, drinks, coffees, exercise, stress, sleep) on the current user's physiology.
    This is the true body-simulator: it evolves glucose, insulin, heart rate, cortisol,
    BAC, glycogen and sleep pressure minute-by-minute with the systems coupled."""
    from simulation import (Simulator, PhysioParams, Schedule, Meal, Food, Drink, Caffeine,
                            Water, Dose, Insulin, Exercise, Stressor, Sleep, Ambient)
    from orchestration.context import current_user
    u = current_user.get()
    profile = (u or {}).get("profile", {}) if u else {
        "weight_kg": args.get("weight_kg", 80), "height_cm": args.get("height_cm", 178),
        "age": args.get("age", 35), "sex": args.get("sex", "male")}
    learned_over = {}
    if u:
        from personalization.wearable_learning import learned_params
        learned_over = learned_params(u)               # resting HR / HRV from wearable
    p = PhysioParams.from_profile(profile, learned=learned_over)
    derived = ((u or {}).get("derived", {}) or {}) if u else {}
    metab = derived.get("metabolic")
    if metab and metab.get("rmr_multiplier"):
        p.rmr_kcal_min *= float(metab["rmr_multiplier"])
    # personal insulin sensitivity, but ONLY when the fit is trustworthy (gated in
    # learn_insulin_sensitivity) — otherwise keep the population/demographic default.
    isf = derived.get("insulin_sensitivity")
    if isf and isf.get("source") == "fitted" and isf.get("value"):
        p.insulin_sensitivity = float(isf["value"])

    # Defensive against malformed LLM args: coerce each field to a list of dicts, and
    # read numbers safely, so a stray int/string can never crash the simulation.
    def _items(key):
        v = args.get(key)
        if isinstance(v, dict):
            return [v]
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    def _num(d, key, default):
        try:
            return float(d.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    sch = Schedule()
    for m in _items("meals"):
        sch.add(Meal(_num(m, "t_min", 0), carbs_g=_num(m, "carbs_g", 60),
                     protein_g=_num(m, "protein_g", 0), fat_g=_num(m, "fat_g", 0)))
    for fd in _items("foods"):
        sch.add(Food(_num(fd, "t_min", 0), str(fd.get("food", "")), grams=_num(fd, "grams", 0)))
    for d in _items("drinks"):
        sch.add(Drink(_num(d, "t_min", 0), standard_drinks=_num(d, "standard_drinks", 1)))
    for cf in _items("caffeine"):
        sch.add(Caffeine(_num(cf, "t_min", 0), mg=_num(cf, "mg", 95)))
    for wt in _items("water"):
        sch.add(Water(_num(wt, "t_min", 0), ml=_num(wt, "ml", 250)))
    for ds in _items("doses"):
        sch.add(Dose(_num(ds, "t_min", 0), str(ds.get("substance", "")),
                     mg=_num(ds, "mg", 0)))
    for ins in _items("insulin"):
        sch.add(Insulin(_num(ins, "t_min", 0), units=_num(ins, "units", 0)))
    for e in _items("exercise"):
        sch.add(Exercise(_num(e, "start_min", 0), _num(e, "end_min", 30),
                         intensity_mets=_num(e, "intensity_mets", 6)))
    for st in _items("stressors"):
        sch.add(Stressor(_num(st, "start_min", 0), _num(st, "end_min", 60),
                         level=_num(st, "level", 0.6)))
    for sl in _items("sleep"):
        sch.add(Sleep(_num(sl, "start_min", 0), _num(sl, "end_min", 480)))
    for am in _items("ambient"):
        sch.add(Ambient(_num(am, "start_min", 0), _num(am, "end_min", 120),
                        temp_c=_num(am, "temp_c", 21)))

    try:
        requested = max(float(args.get("duration_min", 240)), 5.0)
    except (TypeError, ValueError):
        requested = 240.0
    # The minute-by-minute coupled engine is only trustworthy for about a day: a realistic
    # multi-day run needs repeated meals/sleep/hydration the caller won't fully specify, so a
    # sparse long run drifts into simulated starvation/dehydration (bogus HR/temperature).
    # Cap at 24h and DISCLOSE the real horizon so an answer can't claim "over a week".
    MAX_MIN = 24 * 60
    duration = min(requested, MAX_MIN)
    n_events = sum(len(getattr(sch, k)) for k in ("meals", "foods", "drinks", "caffeine",
                   "water", "doses", "insulin", "exercise", "stressors", "sleep", "ambient"))
    if n_events == 0:                        # nothing to evolve -> keep it short
        duration = min(duration, 8 * 60)
    _hrs = duration / 60.0
    horizon_note = f"simulated {_hrs:.0f} h" + (f" ({_hrs/24:.1f} days)" if _hrs > 36 else "")
    if requested > duration:
        horizon_note += (f" — requested ~{requested/60:.0f} h was capped: the minute-by-minute "
                         f"engine is reliable only up to ~24 h, so treat this as a representative "
                         f"day, not a longer-term forecast")
    # strategic: only simulate the systems the question needs (default = whole body)
    _FOCUS = {"glucose": "glucose_mg_dl", "insulin": "insulin_uU_ml",
              "heart_rate": "heart_rate_bpm", "blood_pressure": "sbp_mmhg",
              "cortisol": "cortisol_ug_dl", "stress": "cortisol_ug_dl",
              "alcohol": "bac_g_dl", "bac": "bac_g_dl", "hydration": "water_deficit_ml",
              "sleep": "sleep_pressure", "energy": "energy_expended_kcal",
              "glycogen": "glycogen_g", "ketones": "ketones_mmol_l", "fasting": "ketones_mmol_l",
              "temperature": "core_temp_c", "core_temp": "core_temp_c", "heat": "core_temp_c",
              "hrv": "hrv_rmssd_ms", "recovery": "hrv_rmssd_ms", "fitness": "hrv_rmssd_ms"}
    _focus_in = args.get("focus") or []
    if not isinstance(_focus_in, list):
        _focus_in = [_focus_in]
    focus = [f for f in (_FOCUS.get(str(x).lower()) for x in _focus_in) if f]
    _CHART = ("glucose_mg_dl", "insulin_uU_ml", "heart_rate_bpm", "cortisol_ug_dl",
              "bac_g_dl", "ketones_mmol_l", "core_temp_c")
    _CITE = ["Horizon body-simulation engine: Bergman minimal glucose-insulin model, "
             "Widmark alcohol PK, two-process sleep (Borbely), HPA/autonomic coupling"]

    # --- uncertainty mode: Monte-Carlo over parameter uncertainty -> confidence bands ---
    if args.get("uncertainty"):
        from simulation.uncertainty import run_ensemble, ParamUncertainty
        unc = ParamUncertainty.from_user(u) if u else ParamUncertainty()
        try:
            n = int(args.get("n_samples", 120) or 120)
        except (TypeError, ValueError):
            n = 120
        ens = run_ensemble(p, sch, duration, n=n, unc=unc, outputs=focus or None)
        step = max(1, len(ens.times_min) // 24)
        idx = list(range(0, len(ens.times_min), step))
        bands = {v: {k: [ens.bands[v][k][i] for i in idx] for k in ("median", "lo", "hi")}
                 for v in _CHART if v in ens.bands}
        bands["t_min"] = [ens.times_min[i] for i in idx]
        return {"summary_ci": ens.summary(), "bands": bands, "n_samples": ens.n,
                "duration_min": duration, "horizon_note": horizon_note,
                "personalized": bool(u and u.get("derived")), "evidence": "strong",
                "citations": _CITE + ["confidence bands = 5-95% Monte-Carlo over parameter uncertainty"]}

    traj = Simulator(p).run(sch, duration_min=duration, outputs=focus or None)
    summary = traj.summary()
    # a downsampled series for context/plotting (~24 points)
    step = max(1, len(traj.times_min) // 24)
    idx = list(range(0, len(traj.times_min), step))
    series = {v: [traj.series[v][i] for i in idx] for v in _CHART}
    series["t_min"] = [traj.times_min[i] for i in idx]
    out = {"summary": summary, "series": series, "duration_min": duration,
           "horizon_note": horizon_note, "evidence": "strong", "citations": _CITE}
    if sch.insulin:      # exogenous-insulin PK is a simplified model -> flag it honestly
        out["evidence"] = "moderate"
        out["insulin_note"] = ("exogenous insulin modelled with simplified rapid-acting "
                               "kinetics — direction and rough magnitude only, not a dosing guide")
    return out


def _h_simulate_body(args: dict) -> dict:
    """Run the FULL physiological simulation of the current user's body across every
    system (metabolic, cardiovascular, sleep, stress, activity, liver, digestive) on
    their real stored data. This is the core body-simulator call — use it to ground any
    'how is my body / how am I doing / simulate my X' question in actual per-user numbers."""
    from orchestration.context import current_user
    from personalization.systems import compute_systems
    u = current_user.get()
    if not u:
        return {"error": "no user loaded to simulate"}
    systems = compute_systems(u)
    focus = (args.get("system") or "").strip().lower()
    out = []
    for t in systems:
        if focus and focus not in (t["key"], t["title"].lower()):
            continue
        out.append({
            "system": t["key"], "status": t["status"], "headline": t["headline"],
            "metrics": {m["label"]: f"{m['value']} {m.get('sub','')}".strip() for m in t.get("metrics", [])},
            "note": t.get("note"),
        })
    return {"simulated_systems": out, "evidence": "strong",
            "citations": ["Horizon system modules: metabolic (Mifflin-St Jeor), "
                          "cardiovascular (Framingham D'Agostino 2008), sleep (GBM on "
                          "Sleep-EDF), stress (WESAD), hepatic (Widmark)"]}


def _h_remember(args: dict) -> dict:
    """Persist a durable, valuable fact the user shared in conversation to their stored
    profile so every future answer uses it."""
    from orchestration.context import current_user
    from personalization import user_store
    u = current_user.get()
    if not u or not u.get("user_id"):
        return {"saved": False, "reason": "no stored user in context"}
    patch: dict = {}
    if args.get("note"):
        patch["notes"] = [str(args["note"])]
    if isinstance(args.get("profile"), dict) and args["profile"]:
        patch["profile"] = args["profile"]
    if isinstance(args.get("medical"), dict) and args["medical"]:
        patch["medical"] = args["medical"]
    if not patch:
        return {"saved": False, "reason": "nothing durable to save"}
    user_store.update(u["user_id"], patch)
    return {"saved": True, "note": args.get("note"),
            "profile": args.get("profile"), "medical": args.get("medical")}


def _h_optimize(args: dict) -> dict:
    """Rank the interventions that most improve a goal for the current user, by
    counterfactual simulation (heart risk, weight, glucose)."""
    from personalization.optimize import optimize
    from orchestration.context import current_user
    u = current_user.get()
    if not u:
        return {"error": "no user loaded to optimize for"}
    res = optimize(u, args.get("goal", ""))
    res["evidence"] = "strong"
    res["citations"] = list({c for d in res.get("domains", []) for c in d.get("citations", [])})
    return res


def _h_differential(args: dict) -> dict:
    """Abductive: rank the likely CAUSES of a reported symptom, personalized to the
    current user (diet, notes, labs, questionnaire indicators). Reads the request-scoped
    user set by the API; still returns a (non-personalized) differential without one."""
    from pipeline.causal import explain_symptom
    from orchestration.context import current_user
    return explain_symptom(args["symptom"], current_user.get())


def _h_alcohol_sleep(args: dict) -> dict:
    drinks = ([Drink.standard(n=args["standard_drinks"], hour=0.0)]
              if args.get("standard_drinks") else [])
    res = run_alcohol_then_sleep(
        drinks=drinks, weight_kg=args["weight_kg"], sex=args["sex"],
        bedtime_hour=args.get("bedtime_hour", 2.0), age=args["age"],
        sleep_model=_get_sleep_model(),
    )
    return {
        "bedtime_bac": res.bedtime_bac, "alcohol_gkg_bedtime": res.alcohol_gkg_bedtime,
        "sleep_metrics": res.sleep.metrics, "sleep_intervals": res.sleep.intervals,
        "evidence": res.evidence.value, "confidence": res.confidence_label,
        "citations": res.citations,
    }


# --- Schemas + registry -----------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "estimate_bac",
        "description": ("Blood alcohol concentration from drinks: peak BAC and time to "
                        "sober. If the drinks were consumed some time ago, pass "
                        "hours_since_drinking to get the CURRENT BAC, whether the person "
                        "is over the legal driving limit right now, and hours until "
                        "they're under it (use this to answer 'can I drive?'). Convert "
                        "real drinks to US standard drinks first (1 beer/1 shot ~= 1; "
                        "1 UK pint ~= 1.4; a double ~= 2)."),
        "parameters": {
            "type": "object",
            "properties": {
                "standard_drinks": {"type": "number", "description": "US standard drinks (14 g each)"},
                "weight_kg": {"type": "number"},
                "sex": {"type": "string", "enum": ["male", "female"]},
                "fed": {"type": "boolean", "description": "true if drinking with food"},
                "hours_since_drinking": {"type": "number", "description":
                    "hours elapsed since the drinks were consumed; omit if 'now'/just drank"},
            },
            "required": ["standard_drinks", "weight_kg", "sex"],
        },
    },
    {
        "name": "web_search",
        "description": ("Search the web for an established FACT the simulations and "
                        "knowledge base don't cover (e.g. recommended daily calcium, a "
                        "drug's common side effects, a guideline value). Use ONLY for "
                        "facts, never to compute a physiological number. Results are "
                        "web-sourced and lower-confidence — say so and cite the titles."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "compute_bmi",
        "description": "Body Mass Index from weight and height: exact value, WHO "
                       "category, and the healthy weight range for that height.",
        "parameters": {
            "type": "object",
            "properties": {
                "weight_kg": {"type": "number"}, "height_cm": {"type": "number"},
                "age": {"type": "number", "description": "used to flag <18y (BMI categories are adult-only)"},
            },
            "required": ["weight_kg", "height_cm"],
        },
    },
    {
        "name": "project_weight",
        "description": "Project body-weight trajectory under a daily calorie intake and activity level.",
        "parameters": {
            "type": "object",
            "properties": {
                "weight_kg": {"type": "number"}, "height_cm": {"type": "number"},
                "age": {"type": "number"}, "sex": {"type": "string", "enum": ["male", "female"]},
                "daily_intake_kcal": {"type": "number"},
                "activity": {"type": "string", "enum": ["sedentary", "moderate", "active"]},
                "horizon_days": {"type": "integer"},
            },
            "required": ["weight_kg", "height_cm", "age", "sex", "daily_intake_kcal"],
        },
    },
    {
        "name": "estimate_cvd_risk",
        "description": "10-year cardiovascular disease risk (Framingham General CVD).",
        "parameters": {
            "type": "object",
            "properties": {
                "age": {"type": "number"}, "sex": {"type": "string", "enum": ["male", "female"]},
                "total_chol": {"type": "number", "description": "mg/dL"},
                "hdl": {"type": "number", "description": "mg/dL"},
                "sbp": {"type": "number", "description": "systolic BP, mmHg"},
                "treated_bp": {"type": "boolean"}, "smoker": {"type": "boolean"},
                "diabetic": {"type": "boolean"},
            },
            "required": ["age", "sex", "total_chol", "hdl", "sbp"],
        },
    },
    {
        "name": "simulate_scenario",
        "description": ("Run the coupled BODY-SIMULATION ENGINE forward over time for a "
                        "scenario — the true simulator. Give it a timeline of events (in "
                        "minutes from t=0) and it evolves glucose, insulin, heart rate, "
                        "blood pressure, cortisol, blood-alcohol, glycogen, hydration and "
                        "sleep/alertness minute by minute with the systems COUPLED (a "
                        "stressor raises cortisol which raises glucose and heart rate; "
                        "exercise lowers glucose and dehydrates; a drink raises BAC; a "
                        "drug shifts HR/BP/alertness). Handles meals, drinks, coffee, "
                        "water, EXERCISE, stress, sleep, and DRUGS/SUBSTANCES (nicotine, "
                        "melatonin, pseudoephedrine, l_theanine, diphenhydramine) via "
                        "`doses`. Use for any 'simulate / walk me through / what happens "
                        "to my body if / over the next N hours' question. Convert real "
                        "amounts (1 slice bread ~15g carbs; 1 beer=1 drink; 1 coffee~95mg)."),
        "parameters": {
            "type": "object",
            "properties": {
                "duration_min": {"type": "number", "description": "how long to simulate (minutes)"},
                "uncertainty": {"type": "boolean", "description": "true = also return 5-95% "
                                "confidence bands (Monte-Carlo over parameter uncertainty). "
                                "Use when the user wants a range / how confident, or for "
                                "important predictions. Bands narrow for well-personalized users."},
                "meals": {"type": "array", "items": {"type": "object", "properties": {
                    "t_min": {"type": "number"}, "carbs_g": {"type": "number"}}}},
                "foods": {"type": "array", "description": "named foods (looked up in the food "
                          "DB for macros + glycemic index); e.g. white_rice_cooked, banana, "
                          "lentils_cooked, coca_cola, pizza. Prefer this over `meals` when the "
                          "user names a food.", "items": {"type": "object", "properties": {
                          "t_min": {"type": "number"}, "food": {"type": "string"},
                          "grams": {"type": "number"}}}},
                "drinks": {"type": "array", "items": {"type": "object", "properties": {
                    "t_min": {"type": "number"}, "standard_drinks": {"type": "number"}}}},
                "caffeine": {"type": "array", "items": {"type": "object", "properties": {
                    "t_min": {"type": "number"}, "mg": {"type": "number"}}}},
                "water": {"type": "array", "items": {"type": "object", "properties": {
                    "t_min": {"type": "number"}, "ml": {"type": "number"}}}},
                "doses": {"type": "array", "description": "drugs/substances taken; substance "
                          "must be one of: nicotine, melatonin, pseudoephedrine, l_theanine, "
                          "diphenhydramine (caffeine/alcohol use the caffeine/drinks fields)",
                          "items": {"type": "object", "properties": {
                          "t_min": {"type": "number"}, "substance": {"type": "string"},
                          "mg": {"type": "number"}}}},
                "insulin": {"type": "array", "description": "exogenous (injected) rapid-acting "
                            "insulin boluses in IU — use this when the user mentions taking "
                            "insulin (e.g. '10 units at breakfast'). Lowers glucose via the "
                            "modelled insulin pathway.", "items": {"type": "object", "properties": {
                            "t_min": {"type": "number"}, "units": {"type": "number"}}}},
                "focus": {"type": "array", "items": {"type": "string"}, "description":
                    "optional: only simulate these systems for speed (glucose, insulin, "
                    "heart_rate, blood_pressure, cortisol, alcohol, hydration, sleep, "
                    "energy). Omit to simulate the whole body."},
                "exercise": {"type": "array", "items": {"type": "object", "properties": {
                    "start_min": {"type": "number"}, "end_min": {"type": "number"},
                    "intensity_mets": {"type": "number"}}}},
                "stressors": {"type": "array", "items": {"type": "object", "properties": {
                    "start_min": {"type": "number"}, "end_min": {"type": "number"},
                    "level": {"type": "number"}}}},
                "sleep": {"type": "array", "items": {"type": "object", "properties": {
                    "start_min": {"type": "number"}, "end_min": {"type": "number"}}}},
                "ambient": {"type": "array", "description": "environmental temperature "
                            "exposure (heat/cold), °C", "items": {"type": "object", "properties": {
                            "start_min": {"type": "number"}, "end_min": {"type": "number"},
                            "temp_c": {"type": "number"}}}},
            },
        },
    },
    {
        "name": "simulate_body_systems",
        "description": ("Run the FULL physiological simulation of the user's body on their "
                        "real data — returns per-system state with concrete numbers: "
                        "metabolic (BMI, RMR, maintenance kcal), cardiovascular (10-yr "
                        "risk, heart age), sleep, stress, activity, liver, digestive. This "
                        "is the primary body-simulator tool: call it for any 'how is my "
                        "body / how am I doing / what's my current X / simulate me' "
                        "question, and to ground whole-body or multi-system reasoning in "
                        "the user's actual simulated physiology. Optionally focus on one "
                        "system."),
        "parameters": {
            "type": "object",
            "properties": {
                "system": {"type": "string", "description":
                    "optional: focus on one system (metabolic|cardiovascular|sleep|stress|activity|hepatic|digestive)"},
            },
        },
    },
    {
        "name": "remember_about_user",
        "description": ("Save a durable, valuable fact the user reveals in conversation to "
                        "their stored profile so future answers use it. Call this whenever "
                        "they share stable information about themselves: a diet, a "
                        "diagnosed condition, a medication, an allergy, a habit/lifestyle, "
                        "a goal, or a lab value. Save concise facts, NOT transient state "
                        "(e.g. save 'is vegetarian' or 'takes metformin', not 'had a "
                        "sandwich'). Put a plain-English fact in `note`; use `profile` for "
                        "structured fields (e.g. {\"diet\":\"vegetarian\"}) and `medical` "
                        "for {\"conditions\":[...],\"medications\":[...],\"allergies\":[...]}."),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "one concise English fact to remember"},
                "profile": {"type": "object", "description": "structured profile fields to set"},
                "medical": {"type": "object", "description": "medical lists to add (conditions/medications/allergies)"},
            },
        },
    },
    {
        "name": "optimize_for_goal",
        "description": ("For 'what's the ONE thing / biggest change / how do I best improve "
                        "X' questions. Ranks candidate interventions by how much each "
                        "improves the user's goal, computed by counterfactual simulation on "
                        "THEIR data (heart risk via Framingham, weight via energy balance, "
                        "glucose via the engine). Returns levers with from->to numbers. "
                        "Present the top few ranked, with the quantified improvement."),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description":
                    "the user's goal, e.g. 'lower heart risk', 'lose weight', 'flatten blood sugar'"},
            },
        },
    },
    {
        "name": "differential_for_symptom",
        "description": ("For 'why do I keep... / what could be causing my...' symptom "
                        "questions (e.g. getting sick a lot, fatigue, brain fog, hair "
                        "loss, constipation, cramps). Walks the causal knowledge graph "
                        "backward from the symptom to likely root causes (nutrient "
                        "deficiencies, lifestyle) and ranks them by THIS user's data — "
                        "diet, notes, labs, check-ins. Returns a cited, evidence-graded "
                        "differential (NOT a diagnosis). Present ALL the returned causes, "
                        "ranked, EACH on its own line with its reason and evidence grade — "
                        "not only the top one — then the disclaimer. Do not invent causes "
                        "beyond the ones returned."),
        "parameters": {
            "type": "object",
            "properties": {
                "symptom": {"type": "string", "description":
                    "the symptom in plain words, e.g. 'getting sick a lot' or 'fatigue'"},
            },
            "required": ["symptom"],
        },
    },
    {
        "name": "alcohol_effect_on_sleep",
        "description": ("Cross-system: effect of drinking before bed on tonight's sleep "
                        "architecture (REM, awakenings). Chains hepatic -> sleep."),
        "parameters": {
            "type": "object",
            "properties": {
                "standard_drinks": {"type": "number"}, "weight_kg": {"type": "number"},
                "sex": {"type": "string", "enum": ["male", "female"]}, "age": {"type": "number"},
                "bedtime_hour": {"type": "number", "description":
                    "hours between the FIRST drink and going to bed (default 2 if unknown)"},
            },
            "required": ["standard_drinks", "weight_kg", "sex", "age"],
        },
    },
]

_HANDLERS: dict[str, Callable[[dict], dict]] = {
    "estimate_bac": _h_bac,
    "web_search": _h_web_search,
    "compute_bmi": _h_bmi,
    "project_weight": _h_weight,
    "estimate_cvd_risk": _h_cvd,
    "simulate_scenario": _h_simulate_scenario,
    "simulate_body_systems": _h_simulate_body,
    "remember_about_user": _h_remember,
    "optimize_for_goal": _h_optimize,
    "differential_for_symptom": _h_differential,
    "alcohol_effect_on_sleep": _h_alcohol_sleep,
}


def dispatch(name: str, args: dict) -> dict:
    """Execute a tool call. Raises on unknown tools — the LLM may only call declared
    tools, mirroring the dependency-graph guardrail (brief 2.5)."""
    if name not in _HANDLERS:
        raise ValueError(f"Unknown tool {name!r}. Allowed: {list(_HANDLERS)}")
    # Cross-system tools require a declared edge.
    if name == "alcohol_effect_on_sleep" and not has_edge("hepatic", "sleep"):
        raise ValueError("alcohol->sleep edge is not declared; refusing to chain.")
    return _HANDLERS[name](args)
