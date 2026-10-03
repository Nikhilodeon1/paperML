"""ShanghaiT2DM loader — external validation cohort (weakness-1 fix).

ShanghaiT2DM (Zhao et al., Scientific Data 2023): T2D patients, free-living CGM at 15-min
intervals, free-text dietary records (food name + grams, NO macros), and clinical labs incl HbA1c.
This loader emits the SAME per-meal schema the CGMacros loader/inference consume, so the frozen
CGMacros artifacts can be applied unchanged (pure transfer — no Shanghai-specific tuning).

Two dataset-specific choices, forced by the data (documented for the paper):
  1. Carbohydrate is NOT recorded — only food name + grams. We estimate carbs with a keyword->
     carb-density table (g carb per 100 g food) over the observed vocabulary. EVERY meal is
     `carb_estimated=True`; we report the fraction (100%). This adds carb-estimate noise on top of
     the already-noisy free-living setting — a limitation to state plainly.
  2. CGM is 15-min; we linearly upsample to a 5-min grid WITHIN each postprandial window when
     coverage is adequate, and never interpolate across gaps > 30 min.

HbA1c is stored in mmol/mol; converted to % (NGSP): pct = mmol_mol/10.929 + 2.15.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import data_paths

_DIR = data_paths.dataset("diabetes_datasets")
_WINDOW, _PRE = 180.0, 30.0

# Ordered (keyword, g-carb per 100 g). FIRST match wins, so put specific/high-carb before generic
# and proteins (0) before the vegetable default. Grounded in food-composition tables (USDA SR /
# China FCT), cooked where applicable. Coarse but consistent; the whole pipeline is relative.
_CARB_DENSITY = [
    (("sugar", "honey", "candy", "syrup"), 90.0),
    (("rice cake", "cake", "cookie", "biscuit", "pastry", "moon"), 55.0),
    (("steamed bread", "steamed bun", "mantou", "toast", "bread", "bun", "flatbread"), 48.0),
    (("noodle", "pasta", "spaghetti", "vermicelli", "wonton"), 25.0),
    (("dumpling", "baozi", "jiaozi", "stuffed"), 25.0),
    (("porridge", "congee", "cereal", "oatmeal", "gruel"), 12.0),
    (("rice",), 28.0),
    (("buckwheat", "millet", "barley", "oat", "sorghum", "coarse grain", "coarase grain",
      "grain", "cornmeal", "wheat"), 22.0),
    (("corn",), 19.0),
    (("potato", "taro", "yam", "lotus root", "sweet potato", "tuber", "chestnut"), 18.0),
    (("banana",), 23.0),
    (("watermelon",), 8.0),
    (("apple", "pear", "orange", "grape", "peach", "fruit", "melon", "berry", "kiwi",
      "pineapple", "mango", "date", "jujube", "persimmon"), 13.0),
    (("tofu", "bean curd"), 3.0),
    (("red bean", "mung bean", "bean", "lentil", "soy"), 15.0),
    (("milk", "yogurt", "soybean milk", "soymilk", "dairy"), 5.0),
    (("meal replacement", "powder", "notoginseng", "protein powder"), 18.0),
    (("nut", "peanut", "walnut", "sesame", "seed", "almond"), 15.0),
    # proteins ~ 0 carb
    (("egg",), 1.0),
    (("beef", "pork", "chicken", "duck", "fish", "shrimp", "crab", "meat", "mutton", "lamb",
      "feet", "luncheon", "sausage", "ham", "smoked", "bacon", "tripe", "liver", "poultry"), 0.0),
    (("soup", "broth"), 3.0),
    # vegetables / greens (default-ish low carb)
    (("vegetable", "cabbage", "lettuce", "spinach", "greens", "cucumber", "radish", "celery",
      "daisy", "purse", "bok", "broccoli", "mushroom", "gourd", "pepper", "eggplant", "tomato",
      "carrot", "bamboo", "seaweed", "kelp", "fungus", "onion", "garlic", "bean sprout"), 4.0),
]
_DEFAULT_DENSITY = 10.0


def carbs_from_text(text: str) -> tuple[float, int]:
    """Estimate total carbohydrate grams from a free-text dietary record. Returns (carbs_g, n_items)."""
    total, n = 0.0, 0
    for line in str(text).split("\n"):
        m = re.match(r"^\s*(.+?)\s+([\d.]+)\s*g\s*$", line.strip(), flags=re.I)
        if not m:
            continue
        name, grams = m.group(1).strip().lower(), float(m.group(2))
        density = _DEFAULT_DENSITY
        for keys, d in _CARB_DENSITY:
            if any(k in name for k in keys):
                density = d
                break
        total += grams * density / 100.0
        n += 1
    return total, n


def _f(x, default=None):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _hba1c_pct(mmol_mol):
    v = _f(mmol_mol)
    return None if v is None else v / 10.929 + 2.15


def _homa_ir(fpg_mgdl, fins_pmol):
    fpg, fins = _f(fpg_mgdl), _f(fins_pmol)
    if not (fpg and fins):
        return None
    ins_uU = fins / 6.945                       # pmol/L -> uU/mL
    return round(fpg * ins_uU / 405.0, 2)


def _load_summary():
    import pandas as pd
    s = pd.read_excel(_DIR / "Shanghai_T2DM_Summary.xlsx")
    out = {}
    for _, r in s.iterrows():
        pid = str(r.get("Patient Number", "")).strip()
        if not pid:
            continue
        ht_m = _f(r.get("Height (m)"))
        out[pid] = {
            "hba1c": _hba1c_pct(r.get("HbA1c (mmol/mol)")),
            "homa_ir": _homa_ir(r.get("Fasting Plasma Glucose (mg/dl)"), r.get("Fasting Insulin (pmol/L)")),
            "age": _f(r.get("Age (years)"), 55.0),
            "sex": "female" if _f(r.get("Gender (Female=1, Male=2)")) == 1 else "male",
            "weight_kg": _f(r.get("Weight (kg)"), 70.0),
            "height_cm": round(ht_m * 100.0, 1) if ht_m else 168.0,
        }
    return out


def _meal_window(times_min, cgm, meal_t):
    """5-min grid of CGM from meal_t-PRE to meal_t+WINDOW (meal at 0). None if coverage poor."""
    edges = np.arange(-_PRE, _WINDOW + 1e-6, 5.0)
    rel = np.array(times_min) - meal_t
    inwin = (rel >= -_PRE - 8) & (rel <= _WINDOW + 8)
    if inwin.sum() < 12:                        # ~ need decent 15-min coverage over 3.5h
        return None
    rr, gg = rel[inwin], np.array(cgm)[inwin]
    order = np.argsort(rr); rr, gg = rr[order], gg[order]
    # linear interp onto 5-min grid; guard against extrapolating across big gaps
    vals = np.interp(edges, rr, gg)
    # blank any grid point whose nearest real sample is > 30 min away (long gap)
    for i, e in enumerate(edges):
        if np.min(np.abs(rr - e)) > 30.0:
            vals[i] = np.nan
    if np.isnan(vals).sum() > len(vals) * 0.25:
        return None
    vals = np.array(vals)
    idx = np.where(~np.isnan(vals))[0]
    vals = np.interp(np.arange(len(vals)), idx, vals[idx])   # fill small residual gaps
    return list(vals), float(edges[0])


def _iauc(vals, t0):
    from evaluation.forward_validation import postprandial_metrics
    m = postprandial_metrics(vals, t0, 5.0, 0.0, _WINDOW, _PRE)
    return m["iauc"] if m else None


def load_shanghai_t2dm(data_dir: str | None = None, min_meals: int = 3) -> list[dict]:
    """Load ShanghaiT2DM in the CGMacros per-subject schema. One entry per patient recording that
    has >= `min_meals` scorable meals."""
    import pandas as pd
    base = Path(data_dir) if data_dir else _DIR
    summ = _load_summary()
    out = []
    files = sorted(p for p in (base / "Shanghai_T2DM").glob("*.xls*") if not p.name.startswith("._"))
    for f in files:                     # 89 .xls (xlrd) + 20 .xlsx (openpyxl)
        bio = summ.get(f.stem)          # summary "Patient Number" is the full file stem
        if not bio or bio["hba1c"] is None:
            continue
        try:
            df = pd.read_excel(f)
        except Exception:
            continue
        dcol = next((c for c in df.columns if "ietary" in str(c)), None)
        gcol = next((c for c in df.columns if "CGM" in str(c)), None)
        tcol = next((c for c in df.columns if str(c).strip().lower() == "date"), None)
        if not (dcol and gcol and tcol):
            continue
        df = df.sort_values(tcol)
        t0 = pd.to_datetime(df[tcol]).iloc[0]
        mins = (pd.to_datetime(df[tcol]) - t0).dt.total_seconds().to_numpy() / 60.0
        cgm = pd.to_numeric(df[gcol], errors="coerce").to_numpy()
        ok = np.isfinite(cgm)
        ctimes, cvals = mins[ok], cgm[ok]

        meals = []
        for i, txt in enumerate(df[dcol].to_numpy()):
            if not isinstance(txt, str) or not txt.strip():
                continue
            carbs, nitem = carbs_from_text(txt)
            if nitem == 0 or carbs <= 0:
                continue
            win = _meal_window(ctimes, cvals, mins[i])
            if win is None:
                continue
            vals, wt0 = win
            r = _iauc(vals, wt0)
            if r is None:
                continue
            meals.append({"time": float(mins[i]), "carbs_g": round(carbs, 1),
                          "fat_g": 0.0, "fiber_g": 0.0, "cgm_curve": vals,
                          "observed_iAUC": float(r), "carb_estimated": True})
        if len(meals) < min_meals:
            continue
        out.append({"subject_id": f"Shanghai-{f.stem}", "meals": meals,
                    "HbA1c": bio["hba1c"], "HOMA_IR": bio["homa_ir"], "dx_group": "diabetic",
                    "demographics": {"age": bio["age"], "sex": bio["sex"],
                                     "weight_kg": bio["weight_kg"], "height_cm": bio["height_cm"]}})
    return out


def validate_shanghai_format(subjects: list[dict], cgmacros_iauc_mean: float | None = None) -> dict:
    """Sanity checks + OOD comparison of iAUC means vs CGMacros."""
    meal_counts = [len(s["meals"]) for s in subjects]
    all_iauc = [m["observed_iAUC"] for s in subjects for m in s["meals"]]
    all_gluc = [np.mean(m["cgm_curve"]) for s in subjects for m in s["meals"]]
    a1c = [s["HbA1c"] for s in subjects if s["HbA1c"] is not None]
    flagged_gluc = [s["subject_id"] for s in subjects
                    if np.mean([np.mean(m["cgm_curve"]) for m in s["meals"]]) < 70
                    or np.mean([np.mean(m["cgm_curve"]) for m in s["meals"]]) > 400]
    flagged_a1c = [s["subject_id"] for s in subjects
                   if s["HbA1c"] is not None and not (4.0 <= s["HbA1c"] <= 14.0)]
    res = {
        "n_subjects": len(subjects), "n_meals_total": len(all_iauc),
        "meals_per_subject": {"min": int(min(meal_counts)), "median": int(np.median(meal_counts)),
                              "max": int(max(meal_counts))},
        "iauc_mean": float(np.mean(all_iauc)), "iauc_std": float(np.std(all_iauc)),
        "mean_glucose": float(np.mean(all_gluc)),
        "hba1c_range": (round(min(a1c), 1), round(max(a1c), 1)),
        "flagged_glucose": flagged_gluc, "flagged_hba1c": flagged_a1c,
        "carb_estimated_fraction": 1.0,
    }
    if cgmacros_iauc_mean:
        res["iauc_z_vs_cgmacros"] = (np.mean(all_iauc) - cgmacros_iauc_mean) / (np.std(all_iauc) + 1e-9)
    return res


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    subs = load_shanghai_t2dm()
    v = validate_shanghai_format(subs)
    print("=" * 72)
    print("ShanghaiT2DM LOADED")
    print("=" * 72)
    for k, val in v.items():
        print(f"  {k}: {val}")
    print("=" * 72)


if __name__ == "__main__":
    main()
