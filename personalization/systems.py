"""Per-user system snapshots — what the model actually KNOWS about this user.

For each physiological system we run the real Layer-2 module on the user's stored data
and return concrete, quantified, user-specific numbers (their BMI, their personalized
metabolic rate, their 10-yr cardiovascular risk, their predicted sleep, their current
stress index, etc.) plus how they compare to the population where relevant. This is the
"user-specific layer on top of the human simulation" surfaced for the app.

Kept module-level singletons for the ML models so repeated calls are fast.
"""

from __future__ import annotations

import numpy as np

_sleep_model = None
_stress_model = None


def _get_sleep():
    global _sleep_model
    if _sleep_model is None:
        from modules.sleep import SleepModel
        _sleep_model = SleepModel().fit()
    return _sleep_model


def _get_stress():
    global _stress_model
    if _stress_model is None:
        from modules.stress import StressModel
        _stress_model = StressModel().fit()
    return _stress_model


def _avg(daily, key):
    vals = [d.get(key) for d in daily if isinstance(d.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None


def _latest_survey(user: dict, qid: str) -> dict | None:
    """Most recent scored answers for a given questionnaire, or None."""
    for s in reversed(user.get("surveys", [])):
        if s.get("id") == qid and s.get("indicators"):
            return s
    return None


def _ind(survey: dict, key: str) -> dict | None:
    return next((i for i in survey["indicators"] if i["key"] == key), None) if survey else None


def compute_systems(user: dict) -> list[dict]:
    from knowledge_base import load_system
    from modules.metabolic import _rmr_mifflin, MetabolicPersonalParams, project_weight
    from modules.cardiovascular import compute_cvd_risk
    from modules.hepatic import Drink, compute_bac
    from personalization.user_store import metabolic_params, hepatic_params

    p = user.get("profile", {})
    daily = user.get("wearable", {}).get("daily", [])
    sex = p.get("sex", "male")
    age = float(p.get("age", 30))
    w = float(p.get("weight_kg", 75))
    h = float(p.get("height_cm", 175))
    out: list[dict] = []

    # ---- Metabolic --------------------------------------------------------
    kb = load_system("metabolic")
    rmr = _rmr_mifflin(w, h, age, sex, kb)
    mp = metabolic_params(user)
    mult = mp.rmr_multiplier if mp else 1.0
    tdee = kb["pal_sedentary"].value * rmr * mult
    bmi = w / ((h / 100) ** 2)
    cat = ("underweight" if bmi < 18.5 else "normal weight" if bmi < 25
           else "overweight" if bmi < 30 else "obese")
    metab = {
        "key": "metabolic", "title": "Metabolic", "icon": "fire",
        "status": "good" if 18.5 <= bmi < 25 else "watch",
        "headline": f"BMI {bmi:.1f} · {cat}",
        "metrics": [
            {"label": "BMI", "value": f"{bmi:.1f}", "sub": cat},
            {"label": "Resting metabolic rate", "value": f"{rmr*mult:.0f}", "sub": "kcal/day"},
            {"label": "Maintenance calories", "value": f"{tdee:.0f}", "sub": "kcal/day (sedentary)"},
        ],
        "note": (f"Your calibrated metabolism runs {mult*100:.0f}% of the population "
                 f"average, learned from {mp.n_observations} of your weigh-ins."
                 if mp else "Log a few weigh-ins and I'll personalize your metabolic rate."),
        "ask": "How is my metabolism and weight trending, and what should I change?",
    }
    out.append(metab)

    # ---- Cardiovascular ---------------------------------------------------
    chol, hdl, sbp = p.get("total_chol"), p.get("hdl"), p.get("sbp")
    if all(v is not None for v in (chol, hdl, sbp)) and 30 <= age <= 74:
        r = compute_cvd_risk(age=age, sex=sex, total_chol=float(chol), hdl=float(hdl),
                             sbp=float(sbp), smoker=bool(p.get("smoker")),
                             diabetic=bool(p.get("diabetic")))
        out.append({
            "key": "cardiovascular", "title": "Heart", "icon": "heart",
            "status": "good" if r.risk_10yr < 0.1 else "watch",
            "headline": f"{r.risk_10yr*100:.0f}% 10-year risk",
            "metrics": [
                {"label": "10-year CVD risk", "value": f"{r.risk_10yr*100:.1f}%",
                 "sub": f"{r.risk_ci[0]*100:.0f}–{r.risk_ci[1]*100:.0f}% range"},
                {"label": "Heart age", "value": r.heart_age_note.split("(")[0].strip(), "sub": ""},
                {"label": "Biggest lever", "value": r.drivers[0][0] if r.drivers else "—",
                 "sub": f"-{r.drivers[0][1]*100:.1f} pts" if r.drivers else ""},
            ],
            "note": "Framingham model (D'Agostino 2008), computed from your labs.",
            "ask": "What's driving my heart risk and how do I lower it?",
        })
    else:
        out.append({
            "key": "cardiovascular", "title": "Heart", "icon": "heart", "status": "none",
            "headline": "Add labs to unlock",
            "metrics": [], "note": "Add total cholesterol, HDL and blood pressure in "
            "your data to get a real 10-year cardiovascular risk.",
            "ask": "What labs do you need for my heart risk?",
        })

    # ---- Sleep ------------------------------------------------------------
    sp = _get_sleep().predict(age=age)
    eff_wear = _avg(daily, "sleep_efficiency")
    out.append({
        "key": "sleep", "title": "Sleep", "icon": "sleep",
        "status": "good" if (eff_wear or sp.metrics["sleep_efficiency"]) >= 85 else "watch",
        "headline": f"~{sp.metrics['sleep_efficiency']:.0f}% efficiency expected",
        "metrics": [
            {"label": "Sleep efficiency (recent)",
             "value": f"{eff_wear:.0f}%" if eff_wear else f"{sp.metrics['sleep_efficiency']:.0f}%",
             "sub": "from your wearable" if eff_wear else "model estimate"},
            {"label": "REM", "value": f"{sp.metrics['rem_pct']:.0f}%", "sub": "of the night"},
            {"label": "Deep sleep", "value": f"{sp.metrics['deep_pct']:.0f}%", "sub": "of the night"},
        ],
        "note": "Baseline trained on real PhysioNet Sleep-EDF; alcohol/caffeine shift it.",
        "ask": "How is my sleep and what's hurting it?",
    })

    # ---- Stress -----------------------------------------------------------
    hr = _avg(daily, "resting_hr")
    rmssd = _avg(daily, "hrv_rmssd")
    if hr and rmssd:
        st = _get_stress().predict(heart_rate=hr, rmssd=rmssd, eda=2.5)
        out.append({
            "key": "stress", "title": "Stress", "icon": "calm",
            "status": "good" if st.stress_index < 40 else "watch",
            "headline": f"Stress index {st.stress_index:.0f}/100 ({st.category})",
            "metrics": [
                {"label": "Stress index", "value": f"{st.stress_index:.0f}", "sub": "/100"},
                {"label": "Resting HR", "value": f"{hr:.0f}", "sub": "bpm avg"},
                {"label": "HRV (RMSSD)", "value": f"{rmssd:.0f}", "sub": "ms avg"},
            ],
            "note": "Supervised on WESAD wrist data (87.9% balanced accuracy).",
            "ask": "What does my heart-rate variability say about my stress?",
        })
    else:
        out.append({"key": "stress", "title": "Stress", "icon": "calm", "status": "none",
                    "headline": "No wearable data", "metrics": [],
                    "note": "Connect a wearable to read your stress from HR + HRV.",
                    "ask": "How do you measure stress?"})

    # ---- Activity ---------------------------------------------------------
    steps = _avg(daily, "steps")
    out.append({
        "key": "activity", "title": "Activity", "icon": "walk",
        "status": "good" if (steps or 0) >= 7000 else "watch",
        "headline": f"{(steps or 0)/1000:.1f}k steps/day" if steps else "No data",
        "metrics": [
            {"label": "Daily steps", "value": f"{steps:.0f}" if steps else "—", "sub": "7-day avg"},
            {"label": "Goal", "value": "10,000", "sub": "steps/day"},
        ],
        "note": "From your wearable.",
        "ask": "Am I active enough for my goals?",
    })

    # ---- Hepatic ----------------------------------------------------------
    hp = hepatic_params(user)
    res = compute_bac([Drink.standard(3)], weight_kg=w, sex=sex,
                      personal=hp if hp else None)
    out.append({
        "key": "hepatic", "title": "Liver / Alcohol", "icon": "science",
        "status": "good",
        "headline": f"3 drinks → sober in ~{res.time_to_sober_h:.1f} h",
        "metrics": [
            {"label": "Peak BAC (3 drinks)", "value": f"{res.peak_bac:.3f}", "sub": "g/100mL"},
            {"label": "Time to sober", "value": f"{res.time_to_sober_h:.1f} h",
             "sub": f"{res.time_to_sober_ci[0]:.1f}–{res.time_to_sober_ci[1]:.1f} h"},
        ],
        "note": ("Personalized to your logged elimination rate."
                 if hp else "Widmark pharmacokinetics for your weight & sex."),
        "ask": "If I drink tonight, when am I safe to drive?",
    })

    # ---- Digestive (from the digestive questionnaire) ---------------------
    from personalization.questionnaires import SEVERITY_STATUS
    dig = _latest_survey(user, "digestive")
    if dig:
        reg = _ind(dig, "constipation")
        cons = _ind(dig, "stool_consistency")
        worst = max((i["severity"] for i in dig["indicators"]),
                    key=lambda s: {"ok": 0, "watch": 1, "concern": 2}.get(s, 0))
        metrics = []
        if reg:
            metrics.append({"label": "Bowel regularity", "value": f"{reg['value']:.0f}/wk",
                            "sub": reg["band"].lower()})
        if cons:
            metrics.append({"label": "Stool type", "value": cons["band"].split(" ")[0],
                            "sub": "Bristol scale"})
        note = (reg or cons or {}).get("note", "")
        out.append({
            "key": "digestive", "title": "Digestive", "icon": "gut",
            "status": SEVERITY_STATUS.get(worst, "watch"),
            "headline": reg["band"] if reg else "Logged",
            "metrics": metrics,
            "note": note or "From your gut-health questionnaire.",
            "ask": "What does my gut-health questionnaire say I should change?",
        })
    else:
        out.append({
            "key": "digestive", "title": "Digestive", "icon": "gut", "status": "none",
            "headline": "Take the 30-sec quiz", "metrics": [],
            "note": "Answer 3 quick gut-health questions and I'll flag things like "
                    "constipation against the clinical (Rome IV) threshold.",
            "ask": "Why does gut health matter for me?",
        })

    # If the user did the PSS-4, fold perceived stress into the stress tile's note.
    pss = _latest_survey(user, "stress_pss4")
    ps = _ind(pss, "perceived_stress")
    if ps:
        stress_tile = next((t for t in out if t["key"] == "stress"), None)
        if stress_tile:
            stress_tile["note"] = (stress_tile["note"] + f" You also self-report "
                                   f"{ps['band'].lower()} perceived stress (PSS-4 {ps['value']:.0f}/16).")
            if ps["severity"] == "concern" and stress_tile["status"] == "good":
                stress_tile["status"] = "watch"

    return out
