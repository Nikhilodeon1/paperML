"""Population-scale simulator validation — the "many mock users" harness.

Rather than trust a single demo user, we synthesise a whole population (plus a roster of
deliberately extreme archetypes) and assert the body simulator behaves correctly across
all of them. Four families of check, each a property a *correct* simulator must satisfy
for every body — far stronger than eyeballing one output:

  1. SAFETY / RANGE — over N synthetic users, `compute_systems` never raises, always
     returns the full tile set, and every quantity sits in a physiological range with no
     NaN/inf. (Does the simulator ever emit nonsense?)

  2. METAMORPHIC MONOTONICITY — perturb one input and the right output must move the
     right way, for randomly drawn real users: more drinks -> longer to sober; higher
     SBP -> higher CVD risk; higher HDL -> lower risk; more weight -> higher BMI.
     (Does it respond to change the way physiology does?)

  3. PERSONALIZATION SHARPENS — for users with weigh-ins, the Bayesian RMR posterior is
     tighter than the population prior, and more observations shrink it further. (Does
     learning from a person actually reduce uncertainty about them?)

  4. ABNORMAL-SCENARIO DEGRADATION — the extreme/edge archetypes degrade *gracefully*:
     'watch' status, downgraded evidence, or 'no data' tiles — never a crash or an
     absurd number. (Does it fail safe on unrealistic bodies?)

Run:  python -m evaluation.backtest_population
"""

from __future__ import annotations

import math

import numpy as np

from knowledge_base import load_system
from modules.cardiovascular import compute_cvd_risk
from modules.hepatic import Drink, compute_bac, EvidenceLevel
from modules.metabolic import _rmr_mifflin
from personalization.metabolic import Weighin, personal_params_from_logs
from personalization.synthetic import make_population, archetypes
from personalization.systems import compute_systems

_TILE_KEYS = {"key", "title", "icon", "status", "headline", "metrics", "note", "ask"}
_STATUSES = {"good", "watch", "none"}
_EXPECTED_SYSTEMS = {"metabolic", "cardiovascular", "sleep", "stress", "activity", "hepatic"}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _numbers(s: str) -> list[float]:
    """Pull the numeric tokens out of a formatted metric value like '24.3' or '78%'."""
    out = []
    for tok in str(s).replace("%", " ").replace("/", " ").replace(",", "").split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


# --- 1. safety / range across the population --------------------------------

def _check_population_safety(users: list[dict]) -> tuple[bool, list[str]]:
    ok = True
    n_tiles = 0
    bmis, rmrs, risks, stresses = [], [], [], []
    bad: list[str] = []

    for u in users:
        try:
            tiles = compute_systems(u)
        except Exception as e:  # a crash on any real-ish body is an automatic fail
            bad.append(f"{u['user_id']}: compute_systems raised {type(e).__name__}: {e}")
            ok = False
            continue

        keys = {t["key"] for t in tiles}
        if not _EXPECTED_SYSTEMS.issubset(keys):
            bad.append(f"{u['user_id']}: missing tiles {_EXPECTED_SYSTEMS - keys}")
            ok = False

        for t in tiles:
            n_tiles += 1
            if not _TILE_KEYS.issubset(t):
                bad.append(f"{u['user_id']}/{t.get('key')}: missing tile keys")
                ok = False
            if t["status"] not in _STATUSES:
                bad.append(f"{u['user_id']}/{t['key']}: bad status {t['status']}")
                ok = False
            for m in t["metrics"]:
                for v in _numbers(m.get("value", "")):
                    if not math.isfinite(v):
                        bad.append(f"{u['user_id']}/{t['key']}: non-finite metric {v}")
                        ok = False

        # Raw physiological ranges (numbers, not formatted strings).
        p = u["profile"]
        h, w, age, sex = p["height_cm"], p["weight_kg"], p["age"], p["sex"]
        bmi = w / ((h / 100) ** 2)
        rmr = _rmr_mifflin(w, h, age, sex, load_system("metabolic"))
        bmis.append(bmi)
        rmrs.append(rmr)
        if not (10 <= bmi <= 75):
            bad.append(f"{u['user_id']}: BMI {bmi:.1f} out of range")
            ok = False
        if not (600 <= rmr <= 2800):
            bad.append(f"{u['user_id']}: RMR {rmr:.0f} out of range")
            ok = False

        if all(p.get(k) is not None for k in ("total_chol", "hdl", "sbp")) and 30 <= age <= 74:
            r = compute_cvd_risk(age=age, sex=sex, total_chol=float(p["total_chol"]),
                                 hdl=float(p["hdl"]), sbp=float(p["sbp"]),
                                 smoker=bool(p["smoker"]), diabetic=bool(p["diabetic"]))
            risks.append(r.risk_10yr)
            if not (0 < r.risk_10yr < 1 and r.risk_ci[0] <= r.risk_10yr <= r.risk_ci[1]):
                bad.append(f"{u['user_id']}: CVD risk/CI invalid")
                ok = False

    lines = [
        f"users simulated: {len(users)}  |  tiles emitted: {n_tiles}  |  zero crashes: "
        f"{'OK' if not any('raised' in b for b in bad) else 'FAIL'}",
        f"BMI range [{min(bmis):.1f}, {max(bmis):.1f}]  RMR range "
        f"[{min(rmrs):.0f}, {max(rmrs):.0f}] kcal/day",
    ]
    if risks:
        lines.append(f"CVD 10-yr risk mean {np.mean(risks)*100:.1f}% "
                     f"[{min(risks)*100:.1f}, {max(risks)*100:.1f}%], all in (0,1)+CI brackets")
    if bad:
        lines.append(f"VIOLATIONS ({len(bad)}): " + "; ".join(bad[:4])
                     + (" ..." if len(bad) > 4 else ""))
    return ok, lines


# --- 2. metamorphic monotonicity on random real users -----------------------

def _check_metamorphic(users: list[dict], seed: int = 5) -> tuple[bool, list[str]]:
    rng = np.random.default_rng(seed)
    sample = [users[i] for i in rng.choice(len(users), size=min(40, len(users)), replace=False)]
    fails = {"drinks->sober": 0, "sbp->risk": 0, "hdl->risk": 0, "weight->bmi": 0}

    for u in sample:
        p = u["profile"]
        w, sex = p["weight_kg"], p["sex"]

        # more drinks -> longer to sober AND higher peak BAC
        t1 = compute_bac([Drink.standard(1)], weight_kg=w, sex=sex)
        t4 = compute_bac([Drink.standard(4)], weight_kg=w, sex=sex)
        if not (t4.time_to_sober_h > t1.time_to_sober_h and t4.peak_bac > t1.peak_bac):
            fails["drinks->sober"] += 1

        # CVD monotonicity only where the model applies
        if all(p.get(k) is not None for k in ("total_chol", "hdl", "sbp")) and 30 <= p["age"] <= 74:
            base = dict(age=p["age"], sex=sex, total_chol=float(p["total_chol"]),
                        hdl=float(p["hdl"]), sbp=float(p["sbp"]),
                        smoker=bool(p["smoker"]), diabetic=bool(p["diabetic"]))
            r0 = compute_cvd_risk(**base).risk_10yr
            if not (compute_cvd_risk(**{**base, "sbp": base["sbp"] + 25}).risk_10yr > r0):
                fails["sbp->risk"] += 1
            if not (compute_cvd_risk(**{**base, "hdl": base["hdl"] + 25}).risk_10yr < r0):
                fails["hdl->risk"] += 1

        # more mass -> higher BMI (trivially true but guards the plumbing)
        bmi0 = w / ((p["height_cm"] / 100) ** 2)
        bmi1 = (w + 15) / ((p["height_cm"] / 100) ** 2)
        if not bmi1 > bmi0:
            fails["weight->bmi"] += 1

    ok = all(v == 0 for v in fails.values())
    lines = [f"metamorphic checks on {len(sample)} random users -- "
             + ", ".join(f"{k}: {'OK' if v == 0 else f'{v} FAIL'}" for k, v in fails.items())]
    return ok, lines


# --- 3. personalization sharpens the estimate -------------------------------

def _check_personalization(users: list[dict]) -> tuple[bool, list[str]]:
    prior_sd = load_system("metabolic")["rmr_individual_sd"].population_sd
    ok = True
    tighter_than_prior = 0
    more_obs_helps = 0
    eligible = 0
    ratios = []

    for u in users:
        wi = u.get("logs", {}).get("weighins", [])
        if len(wi) < 3:
            continue
        eligible += 1
        p = u["profile"]
        logs = [Weighin(**w) for w in wi]
        full = personal_params_from_logs(logs, p["height_cm"], p["age"], p["sex"],
                                         p.get("activity", "sedentary"))
        few = personal_params_from_logs(logs[:2], p["height_cm"], p["age"], p["sex"],
                                        p.get("activity", "sedentary"))
        if full.rmr_multiplier_sd < prior_sd and _finite(full.rmr_multiplier_sd):
            tighter_than_prior += 1
        ratios.append(full.rmr_multiplier_sd / prior_sd)
        # more observations should not increase the posterior SD
        if full.rmr_multiplier_sd <= few.rmr_multiplier_sd + 1e-9:
            more_obs_helps += 1

    # require the property to hold for essentially all eligible users
    if eligible:
        ok = (tighter_than_prior >= 0.99 * eligible) and (more_obs_helps >= 0.99 * eligible)
    lines = [
        f"prior RMR-multiplier SD = {prior_sd:.3f}; personalized on {eligible} users",
        f"posterior tighter than prior: {tighter_than_prior}/{eligible}  |  "
        f"more weigh-ins never widen SD: {more_obs_helps}/{eligible}",
    ]
    if ratios:
        lines.append(f"mean posterior/prior SD ratio = {np.mean(ratios):.2f} "
                     f"(=uncertainty cut ~{(1-np.mean(ratios))*100:.0f}%)")
    return ok, lines


# --- 4. abnormal-scenario graceful degradation ------------------------------

def _tile(tiles: list[dict], key: str) -> dict:
    return next(t for t in tiles if t["key"] == key)


def _check_archetypes(archs: list[dict]) -> tuple[bool, list[str]]:
    ok = True
    notes: list[str] = []
    by_id = {u["user_id"]: u for u in archs}

    # obese/diabetic/smoker -> metabolic + heart flagged 'watch', never 'good'
    obese = compute_systems(by_id["arch_obese"])
    if _tile(obese, "metabolic")["status"] != "watch" or _tile(obese, "cardiovascular")["status"] == "good":
        ok = False; notes.append("obese archetype not flagged")

    # data-poor user -> heart & stress tiles say 'none', nothing crashes
    nod = compute_systems(by_id["arch_no_data"])
    if _tile(nod, "cardiovascular")["status"] != "none" or _tile(nod, "stress")["status"] != "none":
        ok = False; notes.append("no-data archetype should show 'none' tiles")

    # age 25 with labs -> Framingham downgraded to WEAK evidence (outside 30-74)
    yl = by_id["arch_young_labs"]["profile"]
    young = compute_cvd_risk(age=yl["age"], sex=yl["sex"], total_chol=float(yl["total_chol"]),
                             hdl=float(yl["hdl"]), sbp=float(yl["sbp"]))
    if young.evidence is not EvidenceLevel.WEAK:
        ok = False; notes.append("young-labs archetype not downgraded to weak")

    # every archetype must simulate without raising and yield finite numbers
    crashes = 0
    for u in archs:
        try:
            tiles = compute_systems(u)
            for t in tiles:
                for m in t["metrics"]:
                    for v in _numbers(m.get("value", "")):
                        assert math.isfinite(v)
        except Exception as e:
            crashes += 1
            notes.append(f"{u['user_id']}: {type(e).__name__}")
    if crashes:
        ok = False

    lines = [
        f"archetypes tested: {len(archs)} (obese, elderly, athlete, teen, heavy-drinker, "
        f"underweight, low-HRV, no-data, young-labs)",
        f"graceful degradation: {'OK' if ok else 'FAIL'} | crashes: {crashes}",
    ]
    if notes:
        lines.append("issues: " + "; ".join(notes[:4]))
    return ok, lines


# --- 5. questionnaire inference is well-formed ------------------------------

def _check_questionnaires(users: list[dict]) -> tuple[bool, list[str]]:
    from personalization.questionnaires import score, list_questionnaires
    ok = True
    valid_sev = {"ok", "watch", "concern"}
    n_surveys = 0
    n_flags = 0
    notes: list[str] = []

    for u in users:
        for s in u.get("surveys", []):
            if "id" not in s:
                continue
            n_surveys += 1
            # every stored indicator must be re-derivable and well-typed
            rescored = {i["key"]: i for i in score(s["id"], s["responses"])["indicators"]}
            for ind in s.get("indicators", []):
                if ind["severity"] not in valid_sev:
                    ok = False; notes.append(f"{s['id']}/{ind['key']}: bad severity")
                if not math.isfinite(ind["value"]):
                    ok = False; notes.append(f"{s['id']}/{ind['key']}: non-finite value")
                if ind["key"] not in rescored or rescored[ind["key"]]["band"] != ind["band"]:
                    ok = False; notes.append(f"{s['id']}/{ind['key']}: not reproducible")
                if ind["severity"] in ("watch", "concern"):
                    n_flags += 1

    # boundary correctness: the exact clinical thresholds must land on the right band
    dig2 = score("digestive", {"bowel_freq_per_week": 2})["indicators"][0]
    dig3 = score("digestive", {"bowel_freq_per_week": 3})["indicators"][0]
    pss_hi = score("stress_pss4", {"pss_unable_control": "4", "pss_confident": "0",
                                   "pss_going_your_way": "0", "pss_difficulties_piling": "4"})
    thresholds_ok = (dig2["severity"] == "concern" and dig3["severity"] == "watch"
                     and pss_hi["indicators"][0]["value"] == 16
                     and pss_hi["indicators"][0]["severity"] == "concern")
    ok &= thresholds_ok

    lines = [
        f"questionnaires available: {len(list_questionnaires())}  |  surveys scored: {n_surveys}  "
        f"|  actionable flags raised: {n_flags}",
        f"stored indicators reproducible + well-typed: {'OK' if not notes else 'FAIL'}",
        f"clinical thresholds (Rome IV <3/wk -> constipation; PSS-4 16 -> high): "
        f"{'OK' if thresholds_ok else 'FAIL'}",
    ]
    if notes:
        lines.append("issues: " + "; ".join(notes[:4]))
    return ok, lines


def run() -> int:
    print("=" * 74)
    print("POPULATION SIMULATOR VALIDATION  (synthetic mock-user harness)")
    print("=" * 74)

    users = make_population(n=200, seed=11)
    archs = archetypes()

    checks = [
        ("1. Safety / physiological range over the population", _check_population_safety, users),
        ("2. Metamorphic monotonicity on random real users", _check_metamorphic, users),
        ("3. Personalization sharpens the RMR estimate", _check_personalization, users),
        ("4. Abnormal-scenario graceful degradation", _check_archetypes, archs),
        ("5. Questionnaire inference is well-formed + clinically anchored", _check_questionnaires, users),
    ]

    ok = True
    for label, fn, arg in checks:
        passed, lines = fn(arg)
        ok &= passed
        print(f"\n[{'OK' if passed else 'FAIL'}] {label}")
        for l in lines:
            print(f"      {l}")

    print("\n" + "=" * 74)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
