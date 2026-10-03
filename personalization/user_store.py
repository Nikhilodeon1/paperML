"""User data store — the single home for EVERYTHING we know about a user.

One JSON file per user under `ml/user_data/<user_id>.json` (gitignored). The schema is
deliberately open: any data we ever collect about a user goes here — account info,
survey answers, self-reported logs, file uploads, wearable time-series, and the derived
personalization parameters + a running history of results/metrics over time. Extra keys
are always allowed; nothing about personalization is capped.

Model-friendly layout (flat, labelled, units in the key names) so a module or the LLM
can consume it directly:

{
  "user_id", "created_at",
  "account":  {email, name, signup_source, ...},
  "profile":  {sex, age, height_cm, weight_kg, smoker, diabetic, total_chol, hdl, sbp},
  "surveys":  [{date, name, responses:{...}}],
  "uploads":  [{date, kind, summary, ...}],
  "logs":     {weighins:[{day, weight_kg, mean_daily_intake_kcal}],
               drinks:[{hour, standard_drinks}],
               bac_readings:[{hour, bac}]},
  "wearable": {daily:[{date, resting_hr, hrv_rmssd, steps, sleep_efficiency, ...}]},
  "derived":  {metabolic:{rmr_multiplier, rmr_multiplier_sd, n}, hepatic:{...}},  # auto
  "metrics_history": [{ts, question, tool, headline, evidence}],
  "notes":    "..."
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modules.metabolic import MetabolicPersonalParams
from modules.hepatic import PersonalParams
from orchestration.router import UserProfile
from personalization.metabolic import Weighin, personal_params_from_logs
from personalization.hepatic import BacReading, personal_params_from_readings

USER_DIR = Path(__file__).resolve().parents[1] / "user_data"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path(user_id: str) -> Path:
    return USER_DIR / f"{user_id}.json"


def list_users() -> list[str]:
    if not USER_DIR.exists():
        return []
    # Skip underscore-prefixed reserved files (e.g. `_auth.json`, the auth store) —
    # they live in this dir but are not user records.
    return sorted(p.stem for p in USER_DIR.glob("*.json") if not p.stem.startswith("_"))


def exists(user_id: str) -> bool:
    return _path(user_id).exists()


def load(user_id: str) -> dict:
    return json.loads(_path(user_id).read_text(encoding="utf-8"))


def save(user: dict) -> None:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    user["updated_at"] = _now()
    _path(user["user_id"]).write_text(json.dumps(user, indent=2), encoding="utf-8")


def create(user_id: str, **profile) -> dict:
    """Create a blank user with an open schema. Any profile fields can be passed."""
    prof = {"sex": "male", "age": 30, "height_cm": 175.0, "weight_kg": 75.0,
            "smoker": False, "diabetic": False,
            "total_chol": None, "hdl": None, "sbp": None}
    prof.update(profile)
    user = {"user_id": user_id, "created_at": _now(), "account": {},
            "profile": prof,
            # Free-text English facts the user can add/edit, e.g. "very low calcium
            # consumption", "trains fasted in the mornings". The model reads these.
            "notes": [],
            # Structured medical context — surfaced to the model on every query so it
            # can personalise advice (allergy warnings, condition-aware caveats).
            "medical": {"allergies": [], "conditions": [], "medications": []},
            "surveys": [], "uploads": [],
            "logs": {"weighins": [], "drinks": [], "bac_readings": []},
            "wearable": {"daily": []}, "derived": {}, "metrics_history": [],
            # User-managed daily routines: [{title, icon, done}]
            "routines": [],
            # Health records / updates: blood tests, doctor visits, health news.
            #   [{ts, kind, title, summary}]
            "records": [],
            # Free-text daily log entries: [{ts, text}]
            "daily_log": [],
            # Cached always-on advice {text, ts}; cleared when data changes.
            "advice": None}
    save(user)
    return user


def update(user_id: str, patch: dict) -> dict:
    """Shallow-merge a patch into a user (the user can edit anything). `notes` and
    medical list fields append; other keys replace. Recomputes derived params."""
    user = load(user_id)
    user["advice"] = None    # data changed -> regenerate advice + system snapshot
    user["systems"] = None
    for k, v in patch.items():
        if k in ("records", "daily_log") and isinstance(v, list):
            user.setdefault(k, []).extend(v)
        elif k == "notes" and isinstance(v, (list, str)):
            items = v if isinstance(v, list) else [v]
            user.setdefault("notes", []).extend(items)
        elif k == "medical" and isinstance(v, dict):
            med = user.setdefault("medical", {"allergies": [], "conditions": [], "medications": []})
            for mk, mv in v.items():
                if isinstance(mv, list):
                    med.setdefault(mk, []).extend(mv)
                else:
                    med[mk] = mv
        elif k == "profile" and isinstance(v, dict):
            user.setdefault("profile", {}).update(v)
        elif k == "logs" and isinstance(v, dict):
            for lk, lv in v.items():
                user.setdefault("logs", {}).setdefault(lk, []).extend(lv)
        else:
            user[k] = v
    refresh_derived(user)
    save(user)
    return user


def set_fields(user_id: str, profile: dict | None = None, medical: dict | None = None,
               notes: list | None = None) -> dict:
    """REPLACE semantics for an edit form (vs. `update` which appends). Profile fields
    are merged; medical and notes are replaced wholesale with what the user submitted."""
    user = load(user_id)
    user["advice"] = None    # data changed -> regenerate advice + system snapshot
    user["systems"] = None
    if profile:
        user.setdefault("profile", {}).update(profile)
    if medical is not None:
        user["medical"] = medical
    if notes is not None:
        user["notes"] = notes
    refresh_derived(user)
    save(user)
    return user


def context_summary(user: dict) -> str:
    """Short English summary of everything we know about the user (medical, lifestyle,
    recent records + log) for the LLM prompt. Kept concise to control tokens."""
    med = user.get("medical", {})
    p = user.get("profile", {})
    bits = []
    for key, label in (("diet", "Diet"), ("activity", "Activity level"),
                       ("goal", "Goal"), ("sleep_hours", "Usual sleep (h)"),
                       ("alcohol_freq", "Alcohol")):
        if p.get(key):
            bits.append(f"{label}: {p[key]}.")
    if med.get("allergies"):
        bits.append("Allergies: " + ", ".join(map(str, med["allergies"])) + ".")
    if med.get("conditions"):
        bits.append("Conditions: " + ", ".join(map(str, med["conditions"])) + ".")
    if med.get("medications"):
        bits.append("Medications: " + ", ".join(map(str, med["medications"])) + ".")
    if med.get("family_history"):
        bits.append("Family history: " + ", ".join(map(str, med["family_history"])) + ".")
    if user.get("notes"):
        bits.append("Notes: " + "; ".join(map(str, user["notes"])) + ".")
    recs = user.get("records", [])[-3:]
    if recs:
        bits.append("Recent records: "
                    + "; ".join((r.get("summary") or r.get("title", "")) for r in recs) + ".")
    log = user.get("daily_log", [])[-4:]
    if log:
        bits.append("Recent log: " + "; ".join(e.get("text", "") for e in log) + ".")
    # Notable questionnaire indicators (constipation, dehydration, high stress, ...).
    from personalization.questionnaires import indicator_facts
    seen: set[str] = set()
    for s in reversed(user.get("surveys", [])):    # newest first; one fact per indicator
        for f in indicator_facts(s):
            if f not in seen:
                seen.add(f)
                bits.append("Self-report: " + f)
    # Current simulated body state (from the cached per-system snapshot, if computed) so
    # every answer is grounded in the user's actual simulation without recomputing.
    systems = user.get("systems")
    if systems:
        snap = "; ".join(f"{t['title']}: {t['headline']} ({t['status']})"
                         for t in systems if t.get("headline") and t.get("status") != "none")
        if snap:
            bits.append("Current simulated body — " + snap + ".")
    return " ".join(bits)


def submit_questionnaire(user_id: str, qid: str, responses: dict) -> dict:
    """Score a questionnaire's answers into indicators and store them on the user.

    Answers land in `surveys` as {date, id, name, responses, indicators}. Scored
    indicators (e.g. constipation, dehydration, high perceived stress) then flow into the
    LLM context and system tiles. Clears the advice/systems cache so both regenerate with
    the new signal."""
    from personalization.questionnaires import score
    scored = score(qid, responses)
    user = load(user_id)
    user.setdefault("surveys", []).append({
        "date": _now(), "id": qid, "name": scored["name"],
        "responses": responses, "indicators": scored["indicators"],
    })
    user["advice"] = None
    user["systems"] = None
    save(user)
    return scored


# --- turning stored data into what the modules consume ----------------------

def to_profile(user: dict) -> UserProfile:
    p = user["profile"]
    return UserProfile(
        weight_kg=p["weight_kg"], height_cm=p["height_cm"], age=p["age"], sex=p["sex"],
        total_chol=p.get("total_chol"), hdl=p.get("hdl"), sbp=p.get("sbp"),
        smoker=p.get("smoker", False), diabetic=p.get("diabetic", False))


def metabolic_params(user: dict) -> MetabolicPersonalParams | None:
    """Bayesian RMR personalization from the user's weigh-in logs (None if too few)."""
    w = user.get("logs", {}).get("weighins", [])
    if len(w) < 2:
        return None
    logs = [Weighin(day=e["day"], weight_kg=e["weight_kg"],
                    mean_daily_intake_kcal=e["mean_daily_intake_kcal"]) for e in w]
    p = user["profile"]
    return personal_params_from_logs(logs, p["height_cm"], p["age"], p["sex"],
                                     p.get("activity", "sedentary"))


def hepatic_params(user: dict) -> PersonalParams | None:
    """Personal alcohol elimination rate from logged BAC readings (None if too few)."""
    r = user.get("logs", {}).get("bac_readings", [])
    if len(r) < 2:
        return None
    return personal_params_from_readings([BacReading(hour=e["hour"], bac=e["bac"]) for e in r])


def refresh_derived(user: dict) -> dict:
    """Recompute + cache the Bayesian personalization params from current logs."""
    mp, hp = metabolic_params(user), hepatic_params(user)
    user["derived"] = {}
    if mp is not None:
        user["derived"]["metabolic"] = {
            "rmr_multiplier": round(mp.rmr_multiplier, 4),
            "rmr_multiplier_sd": round(mp.rmr_multiplier_sd, 4),
            "n_observations": mp.n_observations}
    if hp is not None:
        user["derived"]["hepatic"] = {
            "elimination_beta": round(hp.elimination_beta, 5),
            "elimination_beta_sd": round(hp.elimination_beta_sd, 5),
            "n_observations": hp.n_observations}
    # learned wearable baselines (resting HR, HRV, sleep) — feed the twin's params
    from personalization.wearable_learning import learn_wearable
    w = learn_wearable(user)
    if w:
        user["derived"]["wearable"] = w
    # rebuild meal responses from logged meals + CGM stream, then fit Si from them (gated:
    # only trusted when the fit beats the population prior — see insulin_sensitivity)
    from personalization.meal_logging import build_meal_responses
    build_meal_responses(user)
    from personalization.insulin_sensitivity import learn_insulin_sensitivity
    isf = learn_insulin_sensitivity(user)
    if isf:
        user["derived"]["insulin_sensitivity"] = isf
    return user


def record_result(user: dict, question: str, answer) -> None:
    """Append a result to the user's metric history (tracks outcomes over time).

    Re-loads the latest record from disk before appending so we don't clobber changes a
    tool made DURING this request (e.g. `remember_about_user` writing to the profile).
    """
    entry = {
        "ts": _now(), "question": question, "tool": getattr(answer, "tool", None),
        "headline": getattr(answer, "headline", "") or getattr(answer, "text", "")[:80],
        "evidence": getattr(answer, "evidence", None),
        "indicators": getattr(answer, "indicators", {})}
    uid = user.get("user_id")
    latest = load(uid) if uid and exists(uid) else user
    latest.setdefault("metrics_history", []).append(entry)
    user["metrics_history"] = latest["metrics_history"]   # keep the caller's dict in sync
    save(latest)


# --- demo user with MOCK wearable + logs (to exercise personalization) ------

def seed_demo(user_id: str = "u_demo", seed: int = 7) -> dict:
    """Create a demo user with ~12 weeks of mock wearable data + weigh-ins whose trend
    encodes a real (hidden) metabolic rate, so personalization has genuine signal."""
    from knowledge_base import load_system
    from modules.metabolic import _simulate_weight

    rng = np.random.default_rng(seed)
    height, age, sex = 178.0, 34.0, "male"
    true_rmr_mult, start_w, intake = 0.90, 84.0, 2350.0

    kb = load_system("metabolic")
    pal, kcal = kb["pal_sedentary"].value, kb["kcal_per_kg_fat"].value
    days = np.arange(1, 12 * 14 + 1)
    traj = _simulate_weight(start_w, height, age, sex, intake, true_rmr_mult, pal, kcal, days, kb)

    weighins = [{"day": 0, "weight_kg": round(start_w + float(rng.normal(0, 0.4)), 1),
                 "mean_daily_intake_kcal": intake}]
    wearable = []
    for i in range(1, 12):  # weekly-ish weigh-ins
        d = i * 14
        weighins.append({"day": d, "weight_kg": round(float(traj[d - 1] + rng.normal(0, 0.4)), 1),
                         "mean_daily_intake_kcal": intake})
    for day in range(0, 84):  # daily mock wearable
        wearable.append({
            "date": f"day_{day:02d}",
            "resting_hr": round(float(rng.normal(62, 4)), 0),
            "hrv_rmssd": round(float(rng.normal(48, 10)), 0),
            "steps": int(rng.normal(8200, 2200)),
            "sleep_efficiency": round(float(np.clip(rng.normal(89, 4), 70, 99)), 1)})

    user = create(user_id, sex=sex, age=age, height_cm=height,
                  weight_kg=weighins[-1]["weight_kg"], total_chol=205, hdl=48, sbp=128)
    user["profile"].update({"activity": "sedentary", "diet": "vegan",
                            "alcohol_freq": "weekly", "sleep_hours": 7,
                            "goal": "maintain weight & improve sleep"})
    user["account"] = {"name": "Alex Rivera", "email": "demo@example.com",
                       "signup_source": "seed"}
    user["records"] = [
        {"ts": _now(), "kind": "blood_test", "title": "Lipid panel",
         "summary": "Total chol 205, HDL 48, LDL 130, triglycerides 120 mg/dL."},
        {"ts": _now(), "kind": "doctor", "title": "GP visit",
         "summary": "BP 128/82, advised to reduce sodium and monitor weight."}]
    user["daily_log"] = [
        {"ts": _now(), "text": "Slept 6.5h, woke twice."},
        {"ts": _now(), "text": "Hiked 3 miles, ~500ft elevation gain."}]
    user["logs"]["weighins"] = weighins
    user["logs"]["drinks"] = [{"hour": 0.0, "standard_drinks": 2}]
    user["wearable"]["daily"] = wearable
    user["surveys"] = [{"date": _now(), "name": "onboarding",
                        "responses": {"activity_level": "sedentary", "alcohol_freq": "weekly",
                                      "caffeine_mg_day": 200, "goal": "maintain weight"}}]
    user["notes"] = ["Very low calcium consumption (rarely eats dairy).",
                     "Trains fasted in the mornings."]
    user["medical"] = {"allergies": ["penicillin"], "conditions": ["mild hypertension"],
                       "medications": [],
                       # structured labs power T3 reasoning (immune/thyroid/lab questions)
                       "labs": {"wbc": 3.6, "vitamin_d": 17, "ferritin": 22, "hdl": 48,
                                "triglycerides": 120, "hba1c": 5.4, "tsh": 2.1}}
    user["routines"] = [
        {"title": "Log morning weight", "icon": "scale", "done": False},
        {"title": "10k steps", "icon": "steps", "done": False},
        {"title": "Take vitamin D", "icon": "pill", "done": False},
        {"title": "Lights out by 11pm", "icon": "sleep", "done": False}]
    refresh_derived(user)
    save(user)
    return user
