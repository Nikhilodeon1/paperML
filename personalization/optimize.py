"""Optimization engine — 'the single change that helps you most'.

Given a user and a goal, this evaluates a set of candidate interventions by COUNTERFACTUAL
simulation: perturb one lever at a time, recompute the outcome with the validated models
(Framingham for heart risk, the metabolic projection for weight, the body-simulation engine
for glucose), and rank the levers by how much each improves the target. So instead of
generic advice, the user gets a ranked, quantified list of what actually moves their numbers
most — grounded, personalized, and specific to their body.
"""

from __future__ import annotations


# --- cardiovascular risk levers (Framingham) --------------------------------

def optimize_cardiovascular(user: dict) -> dict | None:
    from modules.cardiovascular import compute_cvd_risk
    p = user.get("profile", {})
    if not all(p.get(k) is not None for k in ("total_chol", "hdl", "sbp")):
        return None
    age = float(p.get("age", 0))
    if not (30 <= age <= 74):
        return None
    base_kw = dict(age=age, sex=p.get("sex", "male"), total_chol=float(p["total_chol"]),
                   hdl=float(p["hdl"]), sbp=float(p["sbp"]),
                   smoker=bool(p.get("smoker")), diabetic=bool(p.get("diabetic")))
    base = compute_cvd_risk(**base_kw).risk_10yr

    candidates = []
    if base_kw["smoker"]:
        candidates.append(("Quit smoking", {**base_kw, "smoker": False}))
    if base_kw["sbp"] > 120:
        candidates.append((f"Lower blood pressure by 15 (to {base_kw['sbp']-15:.0f})",
                           {**base_kw, "sbp": base_kw["sbp"] - 15}))
    if base_kw["hdl"] < 60:
        candidates.append(("Raise HDL by 15 (exercise/diet)", {**base_kw, "hdl": base_kw["hdl"] + 15}))
    if base_kw["total_chol"] > 180:
        candidates.append(("Lower total cholesterol by 40",
                           {**base_kw, "total_chol": base_kw["total_chol"] - 40}))
    if base_kw["diabetic"]:
        candidates.append(("Bring blood sugar into range", {**base_kw, "diabetic": False}))

    levers = []
    for label, kw in candidates:
        new = compute_cvd_risk(**kw).risk_10yr
        levers.append({"change": label,
                       "from": round(base * 100, 1), "to": round(new * 100, 1),
                       "improvement": round((base - new) * 100, 1), "unit": "% 10-yr risk"})
    levers.sort(key=lambda x: x["improvement"], reverse=True)
    return {"target": "10-year cardiovascular risk", "baseline": round(base * 100, 1),
            "levers": levers,
            "citations": ["Framingham General CVD (D'Agostino 2008)"]}


# --- weight-goal levers (energy balance) ------------------------------------

def optimize_weight(user: dict) -> dict | None:
    from modules.metabolic import project_weight
    p = user.get("profile", {})
    w = user.get("logs", {}).get("weighins", [])
    if len(w) < 2:
        intake = None
    else:
        intake = w[-1].get("mean_daily_intake_kcal")
    if not intake:
        return None
    kw = dict(weight_kg=float(p.get("weight_kg", 80)), height_cm=float(p.get("height_cm", 175)),
              age=float(p.get("age", 35)), sex=p.get("sex", "male"),
              activity=p.get("activity", "sedentary"), horizon_days=90)
    base = project_weight(daily_intake_kcal=float(intake), **kw).delta_kg
    levers = []
    for label, dkw, dintake in [
            ("Eat 300 fewer kcal/day", {}, -300),
            ("Eat 500 fewer kcal/day", {}, -500),
            ("Move from sedentary to moderate activity", {"activity": "moderate"}, 0),
            ("Move to active", {"activity": "active"}, 0)]:
        if dkw.get("activity") and p.get("activity") in ("active",):
            continue
        new = project_weight(daily_intake_kcal=float(intake) + dintake, **{**kw, **dkw}).delta_kg
        levers.append({"change": label, "from": round(base, 1), "to": round(new, 1),
                       "improvement": round(base - new, 1), "unit": "kg change over 90 days"})
    levers.sort(key=lambda x: x["improvement"], reverse=True)
    return {"target": "weight change over 90 days", "baseline": round(base, 1),
            "levers": levers, "citations": ["Energy-balance projection (Mifflin-St Jeor)"]}


# --- glucose levers (body-simulation engine) --------------------------------

def optimize_glucose(user: dict) -> dict:
    from simulation import Simulator, PhysioParams, Schedule, Food, Exercise
    from personalization.wearable_learning import learned_params
    p = PhysioParams.from_profile(user.get("profile", {}), learned=learned_params(user))
    metab = (user.get("derived", {}) or {}).get("metabolic") or {}
    if metab.get("rmr_multiplier"):
        p.rmr_kcal_min *= float(metab["rmr_multiplier"])

    def peak(sch):
        return max(Simulator(p).run(sch, 180, outputs=["glucose_mg_dl"]).series["glucose_mg_dl"])

    base = peak(Schedule().add(Food(0, "white_rice_cooked", 200)))
    levers = []
    for label, sch in [
            ("Swap white rice for lentils", Schedule().add(Food(0, "lentils_cooked", 200))),
            ("Swap white rice for brown rice", Schedule().add(Food(0, "brown_rice_cooked", 200))),
            ("Take a 30-min walk after the meal",
             Schedule().add(Food(0, "white_rice_cooked", 200)).add(Exercise(15, 45, 4)))]:
        new = peak(sch)
        levers.append({"change": label, "from": round(base), "to": round(new),
                       "improvement": round(base - new), "unit": "mg/dL lower glucose peak"})
    levers.sort(key=lambda x: x["improvement"], reverse=True)
    return {"target": "post-meal glucose peak (200g white-rice reference meal)",
            "baseline": round(base), "levers": levers,
            "citations": ["Horizon body-simulation engine"]}


# --- dispatcher -------------------------------------------------------------

def optimize(user: dict, goal: str = "") -> dict:
    """Pick the relevant domain from the goal (or default to whatever data supports) and
    return the ranked levers."""
    g = (goal or "").lower()
    domains = []
    if any(k in g for k in ("heart", "cardio", "cholesterol", "blood pressure", "stroke", "risk",
                            "hdl", "ldl", "lipid", "triglyceride")):
        domains = ["cardiovascular"]
    elif any(k in g for k in ("weight", "lose", "fat", "slim")):
        domains = ["weight"]
    elif any(k in g for k in ("sugar", "glucose", "diabet", "spike", "carb")):
        domains = ["glucose"]
    else:
        domains = ["cardiovascular", "weight", "glucose"]     # general: try all we can

    fns = {"cardiovascular": optimize_cardiovascular, "weight": optimize_weight,
           "glucose": optimize_glucose}
    results = []
    for d in domains:
        r = fns[d](user)
        if r and r.get("levers"):
            results.append(r)
    top = None
    if results:
        # the single best lever across the surfaced domains
        best = max((lv for r in results for lv in r["levers"]),
                   key=lambda lv: lv["improvement"], default=None)
        top = best
    return {"goal": goal, "domains": results, "top_change": top,
            "note": "Ranked by counterfactual simulation on your data — not medical advice."}
