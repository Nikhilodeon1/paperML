"""20 hard prompts, validated for SCIENTIFIC ACCURACY.

Each prompt is routed through the deterministic engine — the SAME grounded numbers the
LLM serves (the LLM only phrases them) — so this check is reproducible and token-free.
Each assertion encodes a real physiological expectation (Widmark, energy balance,
Framingham, WHO BMI, alcohol→sleep direction, and the three-outcome rule).

Run:  python -m evaluation.hard_prompts
"""

from __future__ import annotations

from knowledge_base import load_system
from modules.metabolic import _rmr_mifflin
from orchestration.router import UserProfile, route

kb = load_system("metabolic")


def _maint(weight, height, age, sex):
    return kb["pal_sedentary"].value * _rmr_mifflin(weight, height, age, sex, kb)


# (label, question, profile, check(answer) -> (ok, detail))
def _ind(a):
    return a.indicators


CASES = [
    ("6 beers, 70kg female — heavy dose",
     "if I have 6 beers tonight, when will I be sober?",
     UserProfile(70, 165, 30, "female"),
     lambda a: (a.tool == "estimate_bac" and _ind(a)["peak_bac_g_per_100ml"] > 0.13
                and _ind(a)["hours_to_sober"] > 7,
                f"peak {_ind(a).get('peak_bac_g_per_100ml')}, sober {_ind(a).get('hours_to_sober')}h")),

    ("1 glass wine, 90kg male — light dose",
     "1 glass of wine, what's my BAC?",
     UserProfile(90, 182, 40, "male"),
     lambda a: (_ind(a)["peak_bac_g_per_100ml"] < 0.04,
                f"peak {_ind(a).get('peak_bac_g_per_100ml')}")),

    ("8 shots, 55kg female — very high",
     "8 shots of vodka, peak BAC?",
     UserProfile(55, 160, 27, "female"),
     lambda a: (_ind(a)["peak_bac_g_per_100ml"] > 0.22,
                f"peak {_ind(a).get('peak_bac_g_per_100ml')}")),

    ("40 beers, 80kg male — extreme (must stay finite/sane)",
     "40 beers, when am I sober?",
     UserProfile(80, 180, 35, "male"),
     lambda a: (_ind(a)["peak_bac_g_per_100ml"] > 0.4 and _ind(a)["hours_to_sober"] >= 20,
                f"peak {_ind(a).get('peak_bac_g_per_100ml')}, sober {_ind(a).get('hours_to_sober')}h")),

    ("1500 kcal, 100kg male, 6 months — big deficit, long horizon",
     "if I eat 1500 kcal a day for 6 months, what happens to my weight?",
     UserProfile(100, 180, 40, "male"),
     lambda a: (a.tool == "project_weight" and _ind(a)["projected_change_kg"] < -8
                and a.evidence == "weak",
                f"delta {_ind(a).get('projected_change_kg')}kg, evidence {a.evidence}")),

    ("4000 kcal, 60kg female, 3 months — surplus => gain",
     "if I eat 4000 calories a day for 3 months what happens to my weight?",
     UserProfile(60, 165, 30, "female"),
     lambda a: (_ind(a)["projected_change_kg"] > 3,
                f"delta {_ind(a).get('projected_change_kg')}kg")),

    ("exact maintenance, 3 months — weight ~flat",
     f"if I eat {int(_maint(75,178,32,'male'))} kcal a day for 3 months what happens to my weight?",
     UserProfile(75, 178, 32, "male"),
     lambda a: (abs(_ind(a)["projected_change_kg"]) < 1.5,
                f"delta {_ind(a).get('projected_change_kg')}kg (want ~0)")),

    ("1200 kcal, 50kg female, 1 month — modest loss",
     "eat 1200 calories a day for 1 month, weight change?",
     UserProfile(50, 160, 25, "female"),
     lambda a: (-4 < _ind(a)["projected_change_kg"] < -0.3,
                f"delta {_ind(a).get('projected_change_kg')}kg")),

    ("65yo male, worst-case labs — very high CVD risk",
     "what's my heart disease risk?",
     UserProfile(80, 178, 65, "male", total_chol=280, hdl=30, sbp=160, smoker=True, diabetic=True),
     lambda a: (a.tool == "estimate_cvd_risk" and _ind(a)["risk_10yr_pct"] > 40,
                f"risk {_ind(a).get('risk_10yr_pct')}%")),

    ("35yo female, ideal labs — very low CVD risk",
     "what's my cardiovascular risk?",
     UserProfile(62, 168, 35, "female", total_chol=170, hdl=60, sbp=110, smoker=False),
     lambda a: (_ind(a)["risk_10yr_pct"] < 3,
                f"risk {_ind(a).get('risk_10yr_pct')}%")),

    ("55yo male smoker, mid labs — elevated CVD risk",
     "my heart risk?",
     UserProfile(85, 180, 55, "male", total_chol=240, hdl=40, sbp=145, smoker=True),
     lambda a: (_ind(a)["risk_10yr_pct"] > 15,
                f"risk {_ind(a).get('risk_10yr_pct')}%")),

    ("CVD asked with no labs — must ask, not guess (3-outcome rule)",
     "what's my heart disease risk?",
     UserProfile(80, 180, 50, "male"),
     lambda a: (a.tool is None and a.evidence == "none",
                f"tool {a.tool}, evidence {a.evidence}")),

    ("BMI 100kg/170cm — obese",
     "what's my bmi?",
     UserProfile(100, 170, 40, "male"),
     lambda a: (a.tool == "compute_bmi" and _ind(a)["category"] == "obese"
                and 34 < _ind(a)["bmi"] < 35,
                f"bmi {_ind(a).get('bmi')} {_ind(a).get('category')}")),

    ("BMI 55kg/180cm — underweight",
     "bmi?",
     UserProfile(55, 180, 30, "male"),
     lambda a: (_ind(a)["category"] == "underweight",
                f"bmi {_ind(a).get('bmi')} {_ind(a).get('category')}")),

    ("BMI 70kg/175cm — normal",
     "what is my body mass index",
     UserProfile(70, 175, 33, "male"),
     lambda a: (_ind(a)["category"] == "normal weight" and 22 < _ind(a)["bmi"] < 24,
                f"bmi {_ind(a).get('bmi')} {_ind(a).get('category')}")),

    ("adolescent BMI — weak evidence + caveat",
     "what's my bmi?",
     UserProfile(60, 168, 15, "male"),
     lambda a: (a.evidence == "weak" and a.tool == "compute_bmi",
                f"evidence {a.evidence}, tool {a.tool}")),

    ("3 drinks before bed — REM suppressed, more awakenings",
     "how will 3 drinks before bed affect my sleep?",
     UserProfile(80, 180, 35, "male"),
     lambda a: (a.tool == "alcohol_effect_on_sleep"
                and _ind(a)["alcohol_g_per_kg_at_bedtime"] > 0
                and _ind(a)["predicted_rem_pct"] < 24,
                f"gkg {_ind(a).get('alcohol_g_per_kg_at_bedtime')}, REM {_ind(a).get('predicted_rem_pct')}%")),

    ("no drinks before bed — baseline sleep (no alcohol burden)",
     "how will drinking affect my sleep if I have no alcohol?",
     UserProfile(80, 180, 35, "male"),
     lambda a: (_ind(a)["alcohol_g_per_kg_at_bedtime"] == 0,
                f"gkg {_ind(a).get('alcohol_g_per_kg_at_bedtime')}")),

    ("unsupported question — honest refusal, no fabrication",
     "will standing on my head improve my eyesight?",
     UserProfile(80, 180, 35, "male"),
     lambda a: (a.tool is None and a.evidence == "none" and not a.indicators,
                f"tool {a.tool}, evidence {a.evidence}")),

    ("2 whiskeys, 60kg female — mid dose, sober time sane",
     "2 whiskeys, when am I sober?",
     UserProfile(60, 165, 29, "female"),
     lambda a: (a.tool == "estimate_bac" and 2 < _ind(a)["hours_to_sober"] < 8,
                f"sober {_ind(a).get('hours_to_sober')}h")),
]


def run() -> int:
    print("=" * 84)
    print("20 HARD PROMPTS — SCIENTIFIC ACCURACY (grounded numbers, deterministic)")
    print("=" * 84)
    passed = 0
    for i, (label, q, prof, check) in enumerate(CASES, 1):
        a = route(q, prof)
        try:
            ok, detail = check(a)
        except Exception as e:
            ok, detail = False, f"ERROR {type(e).__name__}: {e}"
        passed += ok
        print(f"{i:>2}. [{'PASS' if ok else 'FAIL'}] {label}")
        print(f"      -> {detail}")
    print("=" * 84)
    print(f"SCOREBOARD: {passed}/{len(CASES)} scientifically-valid")
    print("RESULT:", "PASS" if passed == len(CASES) else "FAIL")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(run())
