"""Five rich, longitudinal demo users — data-dense enough to actually demo predictions,
early warnings, and pattern detection.

Each user gets: full demographics + labs (some deliberately out of range so T3 has something
to warn about), medical/family history, ~90 days of daily wearable, ~21 days of intraday CGM
built from LOGGED MEALS (3/day with macros) generated from a HIDDEN true insulin sensitivity,
weigh-ins, questionnaires and records. Because the CGM comes from a known true Si, refreshing
derived params RECOVERS that Si — so personalization is real per user, not a stub.

One persona (Tom) has a real embedded PATTERN: his glucose response is worse on poor-sleep
days, and his wearable records those poor-sleep days — so pattern-detection has a true signal
to find. The others span the clinical range (healthy / prediabetic / diabetic / menopausal /
immune-flagged) so warnings and reasoning differ meaningfully between them.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from personalization import user_store
from simulation import Simulator, PhysioParams, Schedule, Meal
from simulation.observation import observe_series, spec_for

# each persona: hidden true Si + clinical picture. labs chosen to trigger the right warnings.
_PERSONAS = [
    dict(uid="u_maya", name="Maya Chen", age=26, sex="female", weight=58, height=165,
         activity="active", true_si=1.4, resting_hr=54, hrv=68, sleep_eff=93, steps=11000,
         conditions=[], family=[], meds=[], diet="pescatarian",
         labs=dict(wbc=6.2, vitamin_d=42, ferritin=65, hba1c=5.1, hdl=68, ldl=95,
                   triglycerides=70, tsh=1.8, crp=0.4),
         total_chol=170, sbp=112, story="healthy, athletic; baseline 'all good'"),
    dict(uid="u_raj", name="Raj Patel", age=52, sex="male", weight=94, height=176,
         activity="sedentary", true_si=0.6, resting_hr=72, hrv=32, sleep_eff=84, steps=4200,
         conditions=["prediabetes"], family=["type 2 diabetes", "heart disease"], meds=[],
         diet="mixed",
         labs=dict(wbc=6.8, vitamin_d=22, ferritin=140, hba1c=6.1, hdl=38, ldl=155,
                   triglycerides=210, tsh=2.4, crp=3.5),
         total_chol=230, sbp=138, story="prediabetic, overweight, sedentary; CV + glucose warnings"),
    dict(uid="u_elena", name="Elena Rossi", age=58, sex="female", weight=78, height=162,
         activity="sedentary", true_si=0.45, resting_hr=70, hrv=28, sleep_eff=79, steps=5100,
         conditions=["type 2 diabetes", "hypertension"], family=["type 2 diabetes"],
         meds=["metformin", "lisinopril"], diet="mixed",
         labs=dict(wbc=5.4, vitamin_d=16, ferritin=90, hba1c=7.2, hdl=45, ldl=130,
                   triglycerides=180, tsh=3.1, crp=4.2),
         total_chol=205, sbp=142, story="T2 diabetic on metformin, menopausal, low vit D"),
    dict(uid="u_tom", name="Tom Becker", age=34, sex="male", weight=79, height=181,
         activity="moderate", true_si=1.1, resting_hr=60, hrv=45, sleep_eff=82, steps=8800,
         conditions=[], family=["high cholesterol"], meds=[], diet="mixed",
         labs=dict(wbc=6.0, vitamin_d=28, ferritin=24, hba1c=5.5, hdl=52, ldl=118,
                   triglycerides=120, tsh=2.0, crp=1.1),
         total_chol=190, sbp=124, story="fit but high stress + poor sleep; low iron; SLEEP->GLUCOSE pattern"),
    dict(uid="u_grace", name="Grace Adeyemi", age=45, sex="female", weight=68, height=168,
         activity="moderate", true_si=0.9, resting_hr=64, hrv=41, sleep_eff=88, steps=7600,
         conditions=["mild hypertension"], family=["stroke"], meds=[],
         diet="vegetarian",
         labs=dict(wbc=3.4, vitamin_d=19, ferritin=18, hba1c=5.6, hdl=58, ldl=110,
                   triglycerides=95, tsh=2.2, crp=1.8),
         total_chol=185, sbp=134, story="low WBC + low iron -> immune/T3 flags; mild HTN"),
]

_MEAL_SLOTS = [  # (hour, carbs, protein, fat, fibre) typical
    (8, 45, 15, 10, 5), (13, 72, 32, 22, 8), (19, 60, 26, 26, 6)]


def _meal_cgm(profile, si, carbs, protein, fat, fibre, meal_dt, rng, noise=12.0):
    p = PhysioParams.from_profile(profile)
    p.insulin_sensitivity = float(np.clip(si, 0.1, 2.0))
    s = Schedule()
    s.add(Meal(30, carbs_g=carbs, protein_g=protein, fat_g=fat, fiber_g=fibre))
    tr = Simulator(p).run(s, duration_min=210, dt=1.0, record_every=5, outputs=["glucose_mg_dl"])
    obs = observe_series(tr, spec_for("cgm", "glucose"), step_min=5.0)
    start = meal_dt - timedelta(minutes=30)
    return [{"ts": (start + timedelta(minutes=5 * i)).isoformat(),
             "glucose": round(float(v + rng.normal(0, noise)), 1)} for i, v in enumerate(obs["values"])]


def make_rich_user(persona: dict, n_days: int = 21, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    prof = {"weight_kg": persona["weight"], "height_cm": persona["height"],
            "age": persona["age"], "sex": persona["sex"]}
    u = user_store.create(persona["uid"], sex=persona["sex"], age=persona["age"],
                          height_cm=persona["height"], weight_kg=persona["weight"],
                          total_chol=persona["total_chol"], hdl=persona["labs"]["hdl"],
                          sbp=persona["sbp"])
    u["account"] = {"name": persona["name"], "email": f"{persona['uid']}@demo.com",
                    "signup_source": "cohort"}
    u["profile"].update({"activity": persona["activity"], "diet": persona["diet"],
                         "goal": "understand and improve my health"})
    u["medical"] = {"allergies": [], "conditions": list(persona["conditions"]),
                    "medications": list(persona["meds"]), "family_history": list(persona["family"]),
                    "labs": dict(persona["labs"])}
    u["notes"] = [persona["story"]]

    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=n_days)
    cgm, meals, daily, weighins = [], [], [], []
    # 90 days of daily wearable (only last n_days also have CGM/meals)
    for d in range(90):
        day = base - timedelta(days=90 - n_days - d) if d < 90 - n_days else base + timedelta(days=d - (90 - n_days))
        poor_sleep = rng.random() < 0.30
        sleep_eff = persona["sleep_eff"] - (rng.normal(9, 3) if poor_sleep else 0) + rng.normal(0, 2)
        daily.append({"date": day.date().isoformat(),
                      "resting_hr": round(float(rng.normal(persona["resting_hr"], 3))),
                      "hrv_rmssd": round(float(rng.normal(persona["hrv"], 6))),
                      "steps": int(max(0, rng.normal(persona["steps"], 1800))),
                      "sleep_efficiency": round(float(np.clip(sleep_eff, 60, 99)), 1),
                      "poor_sleep": bool(poor_sleep)})

    # last n_days: intraday CGM + logged meals from the hidden true Si (+ Tom's sleep pattern)
    for d in range(n_days):
        day = base + timedelta(days=d)
        poor_sleep = daily[90 - n_days + d]["poor_sleep"]
        # PATTERN: Tom's glucose control drops markedly on poor-sleep days (a real signal for
        # the pattern detector to find; other users have no such day-dependence).
        si_today = persona["true_si"] * (0.6 if (poor_sleep and persona["uid"] == "u_tom") else 1.0)
        si_today *= rng.normal(1.0, 0.06)                      # day-to-day biological wobble
        for (hour, c, pr, f, fb) in _MEAL_SLOTS:
            mt = day + timedelta(hours=hour, minutes=int(rng.integers(-25, 25)))
            cj = max(10, c + rng.integers(-12, 12))
            cgm += _meal_cgm(prof, si_today, cj, pr, f, fb, mt, rng)
            meals.append({"ts": mt.isoformat(), "carbs_g": float(cj), "protein_g": float(pr),
                          "fat_g": float(f), "fiber_g": float(fb), "source": "logged"})
        # a weekly weigh-in
        if d % 7 == 0:
            weighins.append({"day": d, "weight_kg": round(persona["weight"] + rng.normal(0, 0.5), 1),
                             "mean_daily_intake_kcal": 2200 if persona["sex"] == "male" else 1850})

    u["wearable"]["daily"] = daily
    u["wearable"]["cgm"] = cgm
    u.setdefault("logs", {}).update({"meals": meals, "weighins": weighins})
    u["records"] = [{"ts": user_store._now(), "kind": "blood_test", "title": "Full panel",
                     "summary": f"HbA1c {persona['labs']['hba1c']}, HDL {persona['labs']['hdl']}, "
                                f"LDL {persona['labs']['ldl']}, vit D {persona['labs']['vitamin_d']}, "
                                f"WBC {persona['labs']['wbc']}, ferritin {persona['labs']['ferritin']}."}]
    u["surveys"] = [{"date": user_store._now(), "name": "onboarding",
                     "responses": {"activity_level": persona["activity"], "diet": persona["diet"],
                                   "goal": "understand my health"}}]
    user_store.refresh_derived(u)     # builds meal_responses + fits Si from the CGM
    user_store.save(u)
    return u


def make_cohort(n_days: int = 21) -> list[dict]:
    return [make_rich_user(p, n_days=n_days, seed=i) for i, p in enumerate(_PERSONAS)]
