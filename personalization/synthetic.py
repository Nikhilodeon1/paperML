"""Synthetic user population — the mock-user factory for testing the simulator.

Builds realistic *user dicts* (same open schema as `user_store`) so we can exercise
`compute_systems` and the whole personalization stack over hundreds of bodies without
touching disk or real people. Two entry points:

  - `make_population(n)`  -> n users sampled from correlated, documented population
    distributions (US-adult-ish: age, sex, height, BMI, SBP, lipids, smoking, diabetes,
    plus ~12 weeks of wearable + weigh-ins carrying a genuine hidden metabolic signal so
    personalization has something real to recover).

  - `archetypes()`  -> a fixed roster of *named edge cases* — the deliberately extreme /
    abnormal / boundary bodies (morbidly obese, very elderly, low-HRV high-stress, heavy
    drinker, endurance athlete, teenager, missing-labs) that a body simulator must handle
    without crashing or emitting nonsense.

Nothing here writes to `user_data/`; these are in-memory dicts. `make_user(...)` mirrors
the shape `user_store.create` produces so downstream code (systems, context_summary,
metabolic_params, hepatic_params) consumes them unchanged.

Dependency-light on purpose (numpy only): the population validator runs as part of the
accuracy report and must stay fast and deterministic under a fixed seed.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import numpy as np

from knowledge_base import load_system
from modules.metabolic import _simulate_weight


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- one user ---------------------------------------------------------------

def make_user(
    user_id: str,
    *,
    rng: np.random.Generator,
    sex: str,
    age: float,
    height_cm: float,
    weight_kg: float,
    sbp: float | None = None,
    total_chol: float | None = None,
    hdl: float | None = None,
    smoker: bool = False,
    diabetic: bool = False,
    activity: str = "sedentary",
    diet: str | None = None,
    alcohol_freq: str | None = None,
    true_rmr_mult: float = 1.0,
    n_weeks: int = 12,
    n_wearable_days: int = 28,
    resting_hr: float = 62.0,
    hrv_rmssd: float = 46.0,
    steps: float = 8000.0,
    sleep_efficiency: float = 88.0,
    drinks: int = 0,
    with_wearable: bool = True,
    with_weighins: bool = True,
    notes: list[str] | None = None,
    conditions: list[str] | None = None,
) -> dict:
    """Assemble a full user dict. Weigh-ins are generated from an energy-balance
    simulation with a hidden `true_rmr_mult`, so Layer-3 personalization can recover it.
    Wearable dailies are noisy draws around the given per-user means."""
    kb = load_system("metabolic")
    pal, kcal = kb["pal_sedentary"].value, kb["kcal_per_kg_fat"].value

    # Weigh-ins carrying a real hidden metabolic rate (same mechanism as seed_demo).
    weighins: list[dict] = []
    if with_weighins:
        # Intake chosen to sit slightly off maintenance so the trajectory actually moves.
        rmr0 = (10 * weight_kg + 6.25 * height_cm - 5 * age
                + (5 if sex == "male" else -161))
        intake = pal * rmr0 * true_rmr_mult + float(rng.normal(0, 120))
        days = np.arange(1, n_weeks * 14 + 1)
        traj = _simulate_weight(weight_kg, height_cm, age, sex, intake,
                                true_rmr_mult, pal, kcal, days, kb)
        weighins.append({"day": 0,
                         "weight_kg": round(weight_kg + float(rng.normal(0, 0.4)), 1),
                         "mean_daily_intake_kcal": round(intake)})
        for i in range(1, n_weeks):
            d = i * 14
            weighins.append({"day": d,
                             "weight_kg": round(float(traj[d - 1] + rng.normal(0, 0.4)), 1),
                             "mean_daily_intake_kcal": round(intake)})
        weight_kg = weighins[-1]["weight_kg"]  # profile weight = latest weigh-in

    # Wearable dailies.
    wearable: list[dict] = []
    if with_wearable:
        for day in range(n_wearable_days):
            wearable.append({
                "date": f"day_{day:02d}",
                "resting_hr": round(float(np.clip(rng.normal(resting_hr, 4), 35, 120)), 0),
                "hrv_rmssd": round(float(np.clip(rng.normal(hrv_rmssd, 8), 3, 180)), 0),
                "steps": max(0, int(rng.normal(steps, steps * 0.25))),
                "sleep_efficiency": round(float(np.clip(rng.normal(sleep_efficiency, 4), 40, 99)), 1),
            })

    user = {
        "user_id": user_id, "created_at": _now(),
        "account": {"name": user_id, "signup_source": "synthetic"},
        "profile": {
            "sex": sex, "age": float(age), "height_cm": float(height_cm),
            "weight_kg": float(weight_kg), "smoker": bool(smoker), "diabetic": bool(diabetic),
            "total_chol": total_chol, "hdl": hdl, "sbp": sbp, "activity": activity,
            "diet": diet, "alcohol_freq": alcohol_freq,
        },
        "notes": notes or [],
        "medical": {"allergies": [], "conditions": conditions or [], "medications": []},
        "surveys": [], "uploads": [],
        "logs": {"weighins": weighins,
                 "drinks": [{"hour": 0.0, "standard_drinks": drinks}] if drinks else [],
                 "bac_readings": []},
        "wearable": {"daily": wearable}, "derived": {}, "metrics_history": [],
        "routines": [], "records": [], "daily_log": [], "advice": None,
    }
    return user


# --- random questionnaire answers -------------------------------------------

def random_survey(qid: str, rng: np.random.Generator) -> dict:
    """A valid, randomly-answered + scored survey entry for a questionnaire, so the mock
    population also exercises the questionnaire-inference path (digestive tile, PSS, etc.)."""
    from personalization.questionnaires import get_questionnaire, score
    qdef = get_questionnaire(qid)
    responses: dict = {}
    for q in qdef["questions"]:
        if q["type"] == "number":
            lo, hi = q.get("min", 0), q.get("max", 20)
            responses[q["id"]] = int(rng.integers(int(lo), int(hi) + 1))
        else:
            responses[q["id"]] = str(rng.choice([o["value"] for o in q["options"]]))
    scored = score(qid, responses)
    return {"date": _now(), "id": qid, "name": scored["name"],
            "responses": responses, "indicators": scored["indicators"]}


# --- a whole population ------------------------------------------------------

_ACTIVITY = ["sedentary", "moderate", "active"]
_QUESTIONNAIRES = ["digestive", "hydration", "stress_pss4"]


def make_population(n: int = 200, seed: int = 11) -> list[dict]:
    """Sample `n` plausible adults with correlated demographics + lifestyle.

    Correlations encoded (documented, not fitted): SBP rises with age; diabetes and
    smoking prevalence rise with age/BMI; HDL is sex-specific; resting HR rises and HRV
    falls with age and BMI. BMI is lognormal around the US adult mean, clipped to a broad
    but non-pathological band (archetypes cover the tails)."""
    rng = np.random.default_rng(seed)
    users: list[dict] = []
    for i in range(n):
        sex = "male" if rng.random() < 0.5 else "female"
        age = float(np.clip(rng.normal(45, 16), 18, 88))
        height = float(rng.normal(176 if sex == "male" else 163, 7))
        bmi = float(np.clip(rng.lognormal(np.log(26.5), 0.18), 16.5, 44))
        weight = bmi * (height / 100) ** 2

        sbp = float(np.clip(105 + 0.45 * (age - 20) + rng.normal(0, 11), 90, 200))
        total_chol = float(np.clip(rng.normal(192, 34), 120, 330))
        hdl = float(np.clip(rng.normal(48 if sex == "male" else 58, 13), 22, 100))
        # prevalence-style Bernoullis nudged by age/BMI
        smoker = rng.random() < 0.15
        p_diab = min(0.55, 0.02 + 0.004 * max(0, age - 30) + 0.02 * max(0, bmi - 25))
        diabetic = rng.random() < p_diab

        # wearable means shift with age + adiposity
        rhr = 58 + 0.15 * (age - 40) + 0.6 * max(0, bmi - 25) + rng.normal(0, 3)
        hrv = max(8.0, 55 - 0.4 * (age - 30) - 0.5 * max(0, bmi - 25) + rng.normal(0, 6))
        steps = float(np.clip(rng.normal(8500 - 25 * max(0, age - 40), 1500), 1500, 16000))
        sleff = float(np.clip(rng.normal(88 - 0.1 * max(0, age - 50), 4), 65, 98))
        activity = _ACTIVITY[int(np.clip(rng.integers(0, 3), 0, 2))]
        true_mult = float(np.clip(rng.normal(1.0, 0.08), 0.78, 1.25))
        # diet + lifestyle so the causal graph has real signal to reason from
        diet = str(rng.choice(["omnivore", "omnivore", "omnivore", "omnivore",
                               "vegetarian", "vegan", "pescatarian"]))
        alcohol_freq = str(rng.choice(["never", "rarely", "weekly", "weekly", "daily"]))
        notes = []
        if diet in ("vegan", "vegetarian"):
            notes.append("Rarely eats meat; mostly plant-based.")
        if activity == "sedentary" and rng.random() < 0.5:
            notes.append("Works indoors at a desk, little sun exposure.")

        u = make_user(
            f"synth_{i:04d}", rng=rng, sex=sex, age=age, height_cm=height,
            weight_kg=weight, sbp=sbp, total_chol=total_chol, hdl=hdl,
            smoker=smoker, diabetic=diabetic, activity=activity, diet=diet,
            alcohol_freq=alcohol_freq, true_rmr_mult=true_mult,
            resting_hr=rhr, hrv_rmssd=hrv, steps=steps, sleep_efficiency=sleff,
            drinks=int(rng.integers(0, 5)), notes=notes)
        # ~60% of users have answered a random subset of questionnaires
        for qid in _QUESTIONNAIRES:
            if rng.random() < 0.6:
                u["surveys"].append(random_survey(qid, rng))
        users.append(u)
    return users


# --- one fully-populated, realistic "power user" ----------------------------

def _days_ago(n: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat(timespec="seconds")


def make_full_user(user_id: str | None = None, seed: int | None = None) -> dict:
    """Generate ONE complete human as if they'd used the app for ~6 months: full profile +
    a big vitals/labs panel, ~120 days of wearable, ~6 months of weigh-ins, drink logs,
    medical history, a long list of health records, a rich daily log, all questionnaires
    answered, several notes, routines, past app interactions, and computed personalization.
    This is the data-heavy subject the ML lab reasons about."""
    from personalization.user_store import refresh_derived
    rng = np.random.default_rng(seed if seed is not None else int(np.random.default_rng().integers(1, 1_000_000)))

    # correlated demographics + lifestyle from one population draw
    sex = "male" if rng.random() < 0.5 else "female"
    age = float(np.clip(rng.normal(44, 15), 20, 82))
    height = float(rng.normal(176 if sex == "male" else 163, 7))
    bmi = float(np.clip(rng.lognormal(np.log(26.5), 0.18), 17, 42))
    weight = bmi * (height / 100) ** 2
    diet = str(rng.choice(["omnivore", "omnivore", "vegetarian", "vegan", "pescatarian"]))
    alcohol_freq = str(rng.choice(["never", "rarely", "weekly", "weekly", "daily"]))
    activity = str(rng.choice(_ACTIVITY))
    diabetic = rng.random() < min(0.4, 0.02 + 0.02 * max(0, bmi - 25))
    smoker = rng.random() < 0.15
    uid = user_id or f"mock_{int(rng.integers(0, 1 << 28)):07x}"

    rhr_mean = 58 + 0.15 * (age - 40) + 0.6 * max(0, bmi - 25) + rng.normal(0, 3)
    hrv_mean = max(9.0, 55 - 0.4 * (age - 30) - 0.5 * max(0, bmi - 25) + rng.normal(0, 6))
    steps_mean = float(np.clip(rng.normal(8500 - 25 * max(0, age - 40), 1500), 1800, 15000))
    sleff_mean = float(np.clip(rng.normal(88 - 0.1 * max(0, age - 50), 4), 66, 97))

    notes = []
    if diet in ("vegan", "vegetarian"):
        notes.append(f"{diet.capitalize()} for years; no B12 or vitamin D supplement.")
    if activity == "sedentary":
        notes.append("Works indoors at a desk, little sun exposure.")
    notes.append(str(rng.choice(["Trains fasted in the mornings.",
                                 "Two coffees before noon, none after.",
                                 "High stress period at work recently.",
                                 "Rarely eats dairy (low calcium)."])))

    base = make_user(
        uid, rng=rng, sex=sex, age=age, height_cm=height, weight_kg=weight,
        smoker=smoker, diabetic=diabetic, activity=activity, diet=diet,
        alcohol_freq=alcohol_freq, n_weeks=24, n_wearable_days=120,
        resting_hr=rhr_mean, hrv_rmssd=hrv_mean, steps=steps_mean, sleep_efficiency=sleff_mean,
        true_rmr_mult=float(np.clip(rng.normal(1.0, 0.08), 0.78, 1.25)),
        drinks=int(rng.integers(0, 4)), notes=notes)
    # date the wearable series across the last ~120 days so it reads like real history
    for i, d in enumerate(base["wearable"]["daily"]):
        d["date"] = _days_ago(len(base["wearable"]["daily"]) - i)[:10]

    p = base["profile"]
    hdl = float(np.clip(rng.normal(52 if sex == "male" else 60, 12), 25, 95))
    chol = float(np.clip(rng.normal(195, 32), 130, 310))
    trig = float(np.clip(rng.normal(130, 55), 45, 380))
    p.update({
        "total_chol": round(chol), "hdl": round(hdl),
        "ldl": round(max(40, chol - hdl - trig / 5)), "triglycerides": round(trig),
        "sbp": round(float(np.clip(105 + 0.45 * (age - 20) + rng.normal(0, 11), 95, 190))),
        "dbp": round(float(np.clip(rng.normal(78, 8), 60, 112))),
        "hba1c": round(float(np.clip(rng.normal(6.4 if diabetic else 5.4, 0.4), 4.6, 9.5)), 1),
        "fasting_glucose": round(float(np.clip(rng.normal(120 if diabetic else 92, 10), 70, 210))),
        "resting_hr": round(_avg_or(base["wearable"]["daily"], "resting_hr", 62)),
        "vitamin_d": round(float(np.clip(rng.normal(26, 11), 8, 72))),
        "b12": round(float(np.clip(rng.normal(400, 150), 120, 950))),
        "ferritin": round(float(np.clip(rng.normal(85, 55), 8, 320))),
        "tsh": round(float(np.clip(rng.normal(2.2, 1.0), 0.3, 6.5)), 2),
        "crp": round(float(np.clip(rng.normal(1.6, 1.4), 0.1, 12)), 1),
        "vo2max": round(float(np.clip(rng.normal(42 - 0.3 * (age - 30), 6), 20, 60))),
        "sleep_hours": round(float(np.clip(rng.normal(7, 1), 4.5, 9)), 1),
        "caffeine_mg_day": int(np.clip(rng.normal(220, 120), 0, 600)),
        "goal": str(rng.choice(["lose weight", "build muscle", "improve sleep",
                                "maintain weight & energy", "lower cholesterol", "run a 10k"])),
    })

    base["account"] = {"name": _NAME(rng), "email": f"{uid}@example.com",
                       "signup_source": "generated", "member_since": _days_ago(185)[:10]}
    hypertensive = p["sbp"] > 140
    base["medical"] = {
        "allergies": [a for a in rng.choice(["penicillin", "peanuts", "pollen", "shellfish"],
                                            size=int(rng.integers(0, 3)), replace=False)],
        "conditions": [c for c in [("hypertension" if hypertensive else None),
                                   ("type 2 diabetes" if diabetic else None),
                                   (str(rng.choice(["seasonal allergies", "mild anxiety",
                                                    "IBS", "migraines"])) if rng.random() < 0.5 else None)] if c],
        "medications": (["metformin"] if diabetic else []) + (["lisinopril"] if hypertensive else [])
                       + (["vitamin D3"] if rng.random() < 0.3 else []),
        "family_history": [f for f in rng.choice(["heart disease", "type 2 diabetes",
                                                  "high cholesterol", "stroke"],
                                                 size=int(rng.integers(0, 3)), replace=False)],
        # structured labs -> T3 reasoning has real per-user values to interpret. Some are
        # drawn out of range on purpose so immune/thyroid/lab questions have something to say.
        "labs": {"wbc": round(float(rng.normal(6.0, 2.2)), 1),
                 "vitamin_d": round(float(rng.normal(28, 12)), 0),
                 "ferritin": round(float(rng.normal(90, 55)), 0),
                 "hba1c": round(6.7 if diabetic else float(rng.normal(5.4, 0.4)), 1),
                 "hdl": p.get("hdl") or round(float(rng.normal(52, 12)), 0),
                 "triglycerides": round(float(rng.normal(130, 55)), 0),
                 "tsh": round(float(rng.normal(2.2, 1.3)), 1)},
    }

    # a long-time user accrues a stack of health records over months
    base["records"] = [
        {"ts": _days_ago(170), "kind": "blood_test", "title": "Baseline lipid + metabolic panel",
         "summary": f"Total chol {p['total_chol']+12}, HDL {p['hdl']-3}, LDL {p['ldl']+10}, "
                    f"trig {p['triglycerides']+20}, HbA1c {p['hba1c']+0.2:.1f}%."},
        {"ts": _days_ago(150), "kind": "doctor", "title": "New-patient visit",
         "summary": f"BP {p['sbp']+6}/{p['dbp']+4}. Advised lifestyle changes, recheck in 3 months."},
        {"ts": _days_ago(95), "kind": "vaccination", "title": "Influenza vaccine", "summary": "Seasonal flu shot administered."},
        {"ts": _days_ago(60), "kind": "blood_test", "title": "Micronutrient panel",
         "summary": f"Vitamin D {p['vitamin_d']} ng/mL, B12 {p['b12']} pg/mL, ferritin {p['ferritin']} ng/mL, TSH {p['tsh']}."},
        {"ts": _days_ago(30), "kind": "doctor", "title": "Follow-up",
         "summary": f"BP {p['sbp']}/{p['dbp']}, CRP {p['crp']} mg/L. {'Started medication. ' if base['medical']['medications'] else ''}Trending in the right direction."},
        {"ts": _days_ago(7), "kind": "news", "title": "Read: fibre & gut health",
         "summary": "Saved an article on dietary fibre and the gut microbiome."},
    ]

    log_lines = [
        "Slept {sl}h, woke {wk}x.", "Ran {km}k, felt {feel}.", "Gym: push day, {sets} sets.",
        "Ate ~{kcal} kcal, {meals} meals.", "{drinks} drinks with dinner.",
        "Skipped breakfast (fasted until noon).", "Long day, high stress at work.",
        "Walked {steps} steps.", "Rest day, lots of water.", "Headache in the afternoon.",
        "Meditated 10 min before bed.", "Big salad + tofu for lunch.",
    ]
    base["daily_log"] = []
    for d in sorted(rng.choice(range(45), size=14, replace=False), reverse=True):
        t = str(rng.choice(log_lines)).format(
            sl=round(float(rng.uniform(5, 9)), 1), wk=int(rng.integers(0, 3)),
            km=int(rng.integers(3, 12)), feel=rng.choice(["strong", "sluggish", "great", "tired"]),
            sets=int(rng.integers(12, 24)), kcal=int(rng.integers(1600, 2800)),
            meals=int(rng.integers(2, 5)), drinks=int(rng.integers(1, 4)),
            steps=int(rng.integers(3000, 14000)))
        base["daily_log"].append({"ts": _days_ago(int(d)), "text": t})

    # spread drink logs + a couple of BAC calibration readings
    base["logs"]["drinks"] = [{"hour": float(h), "standard_drinks": int(rng.integers(1, 4))}
                              for h in range(0, int(rng.integers(2, 5)))]
    base["logs"]["bac_readings"] = [{"hour": 1.0, "bac": round(float(rng.uniform(0.03, 0.06)), 3)},
                                    {"hour": 3.0, "bac": round(float(rng.uniform(0.01, 0.03)), 3)}]

    base["routines"] = [
        {"title": "Morning weigh-in", "icon": "scale", "done": bool(rng.integers(0, 2))},
        {"title": "10k steps", "icon": "steps", "done": bool(rng.integers(0, 2))},
        {"title": "8 glasses water", "icon": "water", "done": bool(rng.integers(0, 2))},
        {"title": "Lights out by 11pm", "icon": "sleep", "done": False},
        {"title": "Take supplements", "icon": "pill", "done": bool(rng.integers(0, 2))},
    ]

    # all questionnaires answered, some more than once (a history)
    base["surveys"] = []
    for qid in _QUESTIONNAIRES:
        s_old = random_survey(qid, rng); s_old["date"] = _days_ago(40)
        s_new = random_survey(qid, rng); s_new["date"] = _days_ago(3)
        base["surveys"] += [s_old, s_new]

    # a few past app interactions
    base["metrics_history"] = [
        {"ts": _days_ago(20), "question": "What's my 10-year heart risk?", "tool": "estimate_cvd_risk",
         "headline": "Heart risk estimated", "evidence": "strong", "indicators": {}},
        {"ts": _days_ago(9), "question": "If I drink 3 beers can I drive in 2 hours?",
         "tool": "estimate_bac", "headline": "BAC + time-to-sober computed", "evidence": "strong", "indicators": {}},
        {"ts": _days_ago(2), "question": "Why am I so tired?", "tool": "differential_for_symptom",
         "headline": "Fatigue differential", "evidence": "weak", "indicators": {}},
    ]

    refresh_derived(base)
    return base


_FIRST = ["Alex", "Sam", "Jordan", "Taylor", "Maya", "Liam", "Noah", "Ava", "Priya",
          "Diego", "Chen", "Fatima", "Omar", "Sofia", "Ravi", "Nina"]
_LAST = ["Rivera", "Kim", "Patel", "Nguyen", "Garcia", "Hassan", "Silva", "Cohen",
         "Okafor", "Rossi", "Larsson", "Mbeki"]


def _NAME(rng) -> str:
    return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"


def _avg_or(daily, key, default):
    vals = [d.get(key) for d in daily if isinstance(d.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else default


# --- named edge cases -------------------------------------------------------

def archetypes(seed: int = 3) -> list[dict]:
    """Deliberately extreme / boundary bodies the simulator must survive gracefully.

    These are the 'abnormal/unrealistic scenario' probes: each pushes one or more inputs
    to a physiological extreme (or omits data) to confirm the model degrades sensibly
    (widened intervals, 'watch' status, downgraded evidence) instead of crashing or
    printing absurdities."""
    rng = np.random.default_rng(seed)
    A: list[dict] = []

    A.append(make_user("arch_obese", rng=rng, sex="male", age=52, height_cm=178,
                        weight_kg=155, sbp=158, total_chol=270, hdl=32, smoker=True,
                        diabetic=True, activity="sedentary", true_rmr_mult=1.05,
                        resting_hr=82, hrv_rmssd=14, steps=2500, sleep_efficiency=71,
                        conditions=["obesity", "type 2 diabetes"]))
    A.append(make_user("arch_elderly", rng=rng, sex="female", age=88, height_cm=158,
                        weight_kg=52, sbp=168, total_chol=245, hdl=61, smoker=False,
                        diabetic=False, activity="sedentary", true_rmr_mult=0.9,
                        resting_hr=74, hrv_rmssd=12, steps=1800, sleep_efficiency=76))
    A.append(make_user("arch_athlete", rng=rng, sex="male", age=27, height_cm=182,
                        weight_kg=74, sbp=112, total_chol=170, hdl=66, smoker=False,
                        diabetic=False, activity="active", true_rmr_mult=1.18,
                        resting_hr=44, hrv_rmssd=105, steps=14500, sleep_efficiency=95,
                        notes=["Trains twice daily."]))
    A.append(make_user("arch_teen", rng=rng, sex="female", age=18, height_cm=165,
                        weight_kg=55, sbp=110, total_chol=155, hdl=60, smoker=False,
                        diabetic=False, activity="moderate", true_rmr_mult=1.1,
                        resting_hr=68, hrv_rmssd=70, steps=9500, sleep_efficiency=90))
    A.append(make_user("arch_heavy_drinker", rng=rng, sex="male", age=41, height_cm=180,
                        weight_kg=88, sbp=140, total_chol=225, hdl=38, smoker=True,
                        diabetic=False, activity="sedentary", true_rmr_mult=1.0,
                        resting_hr=76, hrv_rmssd=20, steps=5000, sleep_efficiency=78,
                        drinks=8, conditions=["heavy alcohol use"]))
    A.append(make_user("arch_underweight", rng=rng, sex="female", age=24, height_cm=170,
                        weight_kg=45, sbp=100, total_chol=150, hdl=70, smoker=False,
                        diabetic=False, activity="moderate", true_rmr_mult=0.95,
                        resting_hr=64, hrv_rmssd=60, steps=8000, sleep_efficiency=88))
    A.append(make_user("arch_low_hrv_stress", rng=rng, sex="male", age=38, height_cm=175,
                        weight_kg=90, sbp=135, total_chol=210, hdl=40, smoker=False,
                        diabetic=False, activity="sedentary", true_rmr_mult=1.0,
                        resting_hr=88, hrv_rmssd=9, steps=4000, sleep_efficiency=72,
                        conditions=["chronic stress"]))
    # Data-poor: no wearable, no labs, no weigh-ins — must still return safe tiles.
    A.append(make_user("arch_no_data", rng=rng, sex="male", age=33, height_cm=177,
                        weight_kg=80, sbp=None, total_chol=None, hdl=None,
                        with_wearable=False, with_weighins=False))
    # Boundary age just outside Framingham's validated 30-74 window.
    A.append(make_user("arch_young_labs", rng=rng, sex="male", age=25, height_cm=180,
                        weight_kg=78, sbp=120, total_chol=185, hdl=52,
                        with_weighins=False))
    # Causal-reasoning probe: the canonical vegan + indoors + sedentary "keeps getting
    # sick" body — exercises the multi-hop differential (immune <- vit D/B12 <- diet/sun).
    A.append(make_user("arch_vegan_indoor", rng=rng, sex="female", age=31, height_cm=167,
                        weight_kg=60, sbp=112, total_chol=175, hdl=58, activity="sedentary",
                        diet="vegan", alcohol_freq="rarely", with_weighins=False,
                        notes=["Vegan for years, no B12 or vitamin D supplement.",
                               "Works from home indoors, rarely outdoors in the sun."]))
    return A
