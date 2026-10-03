"""Average-human vs this-specific-user comparison.

Shows two grounded numbers side by side: the prediction for an average person on the
population priors, and the prediction personalized to THIS user using their Bayesian
posteriors (from their logged data in the user store). The `why` line states what about
them drives the difference. Concise by design — one metric per query.

Returns {} when there's no personalization yet (so the answer just omits the section
and, ideally, invites the user to log more data).
"""

from __future__ import annotations

from modules.hepatic import Drink, compute_bac
from modules.metabolic import project_weight
from personalization.user_store import hepatic_params, metabolic_params


def average_vs_you(answer, user: dict) -> dict:
    tool = getattr(answer, "tool", None)
    ind = getattr(answer, "indicators", {}) or {}
    p = user["profile"]

    if tool == "project_weight":
        mp = metabolic_params(user)
        if mp is None:
            return {}
        intake = ind.get("daily_intake_kcal")
        days = int(ind.get("horizon_days", 365))
        if intake is None:
            return {}
        common = dict(weight_kg=p["weight_kg"], height_cm=p["height_cm"], age=p["age"],
                      sex=p["sex"], daily_intake_kcal=intake, horizon_days=days)
        avg = project_weight(**common)
        you = project_weight(**common, personal=mp)
        return {
            "metric": f"projected weight change over {days} days (kg)",
            "average_person": round(avg.delta_kg, 1),
            "you": round(you.delta_kg, 1),
            "why": (f"your calibrated metabolism runs ~{mp.rmr_multiplier*100:.0f}% of the "
                    f"population average (learned from {mp.n_observations} of your weigh-ins)"),
        }

    if tool == "estimate_bac":
        hp = hepatic_params(user)
        if hp is None:
            return {}
        n = ind.get("standard_drinks", 0) or 0
        drinks = [Drink.standard(n=n, hour=0.0)]
        avg = compute_bac(drinks, weight_kg=p["weight_kg"], sex=p["sex"])
        you = compute_bac(drinks, weight_kg=p["weight_kg"], sex=p["sex"], personal=hp)
        return {
            "metric": "time to sober (hours)",
            "average_person": round(avg.time_to_sober_h, 1),
            "you": round(you.time_to_sober_h, 1),
            "why": (f"your measured alcohol elimination rate differs from average "
                    f"(learned from {hp.n_observations} of your BAC readings)"),
        }

    return {}
