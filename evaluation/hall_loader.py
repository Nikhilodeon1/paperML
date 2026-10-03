"""Hall 2018 loader — second external validation cohort (standardized-meal CGM).

Hall et al. 2018 (PLOS Biology, "Glucotypes reveal new patterns of glucose dysregulation"):
30 subjects with standardized-meal CGM curves, plus clinical labs including HbA1c and SSPG (steady-
state plasma glucose — a DIRECT insulin-resistance measure, and the better clinical target here
since the cohort is mostly non-diabetic). Three dietitian-standardized meals, each eaten twice:
Bar (protein bar), CF (corn flakes + milk), PB (bread + peanut butter).

Files (PLOS supplementary IDs, in data/hall2018):
  s015.tsv  meal CGM curves (Meal, userID, time, GlucoseValue) — 37 points/meal, -30..+150 min
  s014.db   `clinical` table: A1C, SSPG, diagnosis, Weight, Height, Age (+ more)
  s016.tsv  meal type per subject (userID, meal, mealType Bar/CF/PB)
The userID join (s015 '2133-001' <-> clinical) is EXACT (clinical holds both cohorts' IDs).

CARBS: the data files carry NO macronutrient content, so carb grams per meal type are USDA standard-
serving estimates (justified: meals were dietitian-standardized, so a fixed per-type value is
correct; and gradient Si-vs-labs was shown robust to +/-45% carb error). Corn flakes are the
highest-glycemic standardized meal, bar the lowest.

STANDARDIZED-MEAL NOTE: every subject eats identical meals, so carb input is CONTROLLED and only the
glucose response (-> Si) varies across subjects. This removes the carb-estimation confound that
degraded the Shanghai cohort — the design is a strength for this pipeline.

CAVEAT: Hall curves end at +150 min (2.5 h); the committed gradient fit simulates to +180. The
observed iAUC is computed over 0..150; the +150..180 tail the fit adds is near-baseline and its
(systematic) effect is absorbed into Si scale, preserving the cross-subject correlation we report.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import numpy as np

import data_paths

_DIR = data_paths.dataset("hall2018")
_S6 = _DIR / "pbio.2005143.s015.tsv"
_DB = _DIR / "pbio.2005143.s014.db"
_PRE, _POST, _STEP = 30.0, 150.0, 5.0

# carb grams per meal type (USDA standard servings; see module docstring).
_CARB = {"Bar": 30.0, "CF": 50.0, "PB": 35.0}


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _meal_type(meal_name: str) -> str | None:
    """'Bar 1'/'CF 2'/'PB 1' -> 'Bar'/'CF'/'PB'."""
    key = meal_name.split()[0]
    return key if key in _CARB else None


_NEED = 36          # >=36 pts => covers -30..+145; use a FIXED 0..145 window so iAUC is comparable
_POST_N = 30        # post-meal points 0..145 min (indices 6..35)


def _iauc(curve: list[float]) -> float | None:
    """Baseline-subtracted iAUC over a fixed 0..145 min window (30 post-meal pts, meal at index 6).
    Curves vary 36-38 pts; a fixed window keeps iAUC comparable across meals."""
    if len(curve) < _NEED or any(v is None for v in curve[:_NEED]):
        return None
    base = float(np.mean(curve[:6]))                 # -30..-5 min pre-meal baseline
    post = np.array(curve[6:6 + _POST_N], dtype=float)   # 0..145 min
    incr = np.maximum(0.0, post - base)
    t = np.arange(len(post)) * _STEP
    return float(np.trapezoid(incr, t))


def _load_clinical() -> dict:
    c = sqlite3.connect(_DB)
    out = {}
    for r in c.execute("SELECT userID, A1C, SSPG, diagnosis, Weight, Height, Age FROM clinical"):
        out[r[0]] = {"hba1c": _f(r[1]), "sspg": _f(r[2]),
                     "dx": (r[3] or "").strip() or None,
                     "weight_kg": _f(r[4]) or 78.0, "height_cm": _f(r[5]) or 170.0,
                     "age": _f(r[6]) or 45.0}
    return out


def load_hall(min_meals: int = 3) -> list[dict]:
    """Per-subject standardized-meal cohort in the CGMacros meal schema."""
    rows = list(csv.DictReader(open(_S6, encoding="utf-8"), delimiter="\t"))
    clinical = _load_clinical()

    # group curves by (userID, Meal), ordered by timestamp
    series: dict[tuple, list[tuple]] = {}
    for r in rows:
        g = _f(r["GlucoseValue"])
        series.setdefault((r["userID"], r["Meal"]), []).append((r["time"], g))

    per_subject: dict[str, list[dict]] = {}
    for (uid, meal), pts in series.items():
        mt = _meal_type(meal)
        if mt is None:
            continue
        pts.sort(key=lambda x: x[0])
        curve = [g for _t, g in pts]
        r = _iauc(curve)
        if r is None:
            continue
        per_subject.setdefault(uid, []).append(
            {"carbs_g": _CARB[mt], "fat_g": 0.0, "fiber_g": 0.0, "meal_type": mt,
             "observed_iAUC": r, "cgm_curve": [v for v in curve]})

    out = []
    for uid, meals in per_subject.items():
        bio = clinical.get(uid)
        if not bio or len(meals) < min_meals:
            continue
        if bio["hba1c"] is None and bio["sspg"] is None:
            continue
        out.append({"subject_id": f"Hall-{uid}", "meals": meals,
                    "HbA1c": bio["hba1c"], "SSPG": bio["sspg"], "dx_group": bio["dx"],
                    "demographics": {"age": bio["age"], "sex": "unknown",
                                     "weight_kg": bio["weight_kg"], "height_cm": bio["height_cm"]}})
    return out


def summary(subs: list[dict]) -> dict:
    a1c = [s["HbA1c"] for s in subs if s["HbA1c"] is not None]
    sspg = [s["SSPG"] for s in subs if s["SSPG"] is not None]
    meals = [len(s["meals"]) for s in subs]
    from collections import Counter
    return {"n_subjects": len(subs), "n_with_hba1c": len(a1c), "n_with_sspg": len(sspg),
            "n_meals_total": sum(meals), "meals_per_subject_median": int(np.median(meals)),
            "hba1c_range": (round(min(a1c), 1), round(max(a1c), 1)) if a1c else None,
            "sspg_range": (round(min(sspg), 0), round(max(sspg), 0)) if sspg else None,
            "dx": dict(Counter(s["dx_group"] for s in subs))}


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    subs = load_hall()
    s = summary(subs)
    print("=" * 66)
    print("HALL 2018 LOADED")
    print("=" * 66)
    for k, v in s.items():
        print(f"  {k}: {v}")
    print("=" * 66)


if __name__ == "__main__":
    main()
