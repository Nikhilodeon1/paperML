"""CGMacros loader + FIRST real-human validation of the engine.

CGMacros (PhysioNet 2024): 48 subjects (healthy / pre-diabetic / T2D), one aligned
per-minute CSV each — two CGMs (mg/dL), HR, and meals with REAL carb/protein/fat grams and a
photo. This is the first dataset that lets the engine be checked against actual people instead
of its own synthetic output.

Design choices forced by the real data (read from the file, not assumed):
  - Timestamps start mid-morning, not midnight, and each person's fasting baseline differs.
    So validation is PER MEAL on baseline-subtracted iAUC (0-3h), which cancels the
    engine-vs-real baseline offset. This is exactly why iAUC was made the primary metric.
  - The Libre and Dexcom sensors disagree by up to ~40 mg/dL on the same reading. Dexcom G6 is
    research-grade, so it is preferred; Libre is the fallback when Dexcom is missing.
  - CSVs are read straight from the distributed zip — no multi-GB extraction, photos untouched.

WHAT THIS VALIDATES (unlike the synthetic harness): the physiology against real people. The
honest bar is again persistence (same meal slot's previous response). CGM noise (~10-15 mg/dL)
plus carb-estimate error set the floor — the engine is not expected to be perfect, only to
beat persistence and to have a per-subject insulin sensitivity that tracks their diabetic
status.

Run:  python -m evaluation.cgmacros            # summary over N subjects
"""

from __future__ import annotations

import csv
import io
import statistics
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import data_paths
from simulation import Simulator, PhysioParams, Schedule, Meal
from simulation.observation import observe_series, spec_for
from evaluation.forward_validation import postprandial_metrics

# Resolved through data_paths, not by counting parents: the datasets moved out of the
# project into a shared folder, and the old relative path broke at that point.
_ZIP = data_paths.dataset("cgmacros", "1.0.0", "CGMacros_dateshifted365.zip")
_WINDOW = 180.0        # postprandial window (min)
_PRE = 30.0            # pre-meal baseline window (min)


@dataclass
class Meal_:
    subject: str
    t: datetime
    carbs_g: float
    cgm: str                       # which sensor the window came from
    times_min: list[float]         # minutes relative to the meal (negative = pre-meal)
    glucose: list[float]           # mg/dL, aligned to times_min
    image: str | None
    protein_g: float = 0.0         # CGMacros has these; fat/fibre blunt & delay glucose
    fat_g: float = 0.0
    fiber_g: float = 0.0

    @property
    def clock_hour(self) -> float:
        return self.t.hour + self.t.minute / 60.0

    def real_iauc(self):
        vals, t0 = self._grid()
        m = postprandial_metrics(vals, t0, 5.0, 0.0, _WINDOW, _PRE)
        return m["iauc"] if m else None

    def real_peak(self):
        m = postprandial_metrics(*self._grid(), 5.0, 0.0, _WINDOW, _PRE)
        return m["peak"] if m else None

    def _grid(self):
        """5-min grid of glucose from -PRE to +WINDOW, meal at t=0 -> returns (values, t0)."""
        edges = np.arange(-_PRE, _WINDOW + 1e-6, 5.0)
        vals = []
        for e in edges:
            near = [g for t, g in zip(self.times_min, self.glucose) if e - 2.5 <= t < e + 2.5]
            vals.append(statistics.fmean(near) if near else np.nan)
        # forward/back fill small gaps so a stray missing sample doesn't kill the window
        v = np.array(vals, float)
        idx = np.where(~np.isnan(v))[0]
        if len(idx) >= 5:
            v = np.interp(np.arange(len(v)), idx, v[idx])
        return list(v), float(edges[0])


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_subject(subject_csv_bytes: bytes, subject: str) -> list[Meal_]:
    rows = list(csv.DictReader(io.StringIO(subject_csv_bytes.decode("utf-8", "replace"))))
    if not rows:
        return []
    # pick the CGM column with better coverage for THIS subject
    cols = {"Dexcom GL": 0, "Libre GL": 0}
    for r in rows:
        for c in cols:
            if _f(r.get(c)) is not None:
                cols[c] += 1
    cgm_col = "Dexcom GL" if cols["Dexcom GL"] >= 0.5 * cols["Libre GL"] else "Libre GL"

    ts, gl = [], []
    meals = []
    for r in rows:
        try:
            t = datetime.strptime(r["Timestamp"].strip(), "%Y-%m-%d %H:%M:%S")
        except (ValueError, KeyError):
            continue
        g = _f(r.get(cgm_col))
        if g is None:
            g = _f(r.get("Libre GL")) or _f(r.get("Dexcom GL"))
        if g is not None:
            ts.append(t); gl.append(g)
        carbs = _f(r.get("Carbs"))
        if carbs is not None and carbs > 0:
            meals.append((t, carbs, (r.get("Image path") or "").strip() or None,
                          _f(r.get("Protein")) or 0.0, _f(r.get("Fat")) or 0.0,
                          _f(r.get("Fiber")) or 0.0))

    out = []
    ts_arr = ts
    for mt, carbs, img, prot, fat, fib in meals:
        win_t, win_g = [], []
        for t, g in zip(ts_arr, gl):
            dm = (t - mt).total_seconds() / 60.0
            if -_PRE - 3 <= dm <= _WINDOW + 3:
                win_t.append(dm); win_g.append(g)
        if len(win_g) >= 20:                       # enough coverage to score
            out.append(Meal_(subject, mt, carbs, cgm_col, win_t, win_g, img,
                             protein_g=prot, fat_g=fat, fiber_g=fib))
    return out


def subjects(zip_path: Path = _ZIP, limit: int | None = None) -> list[tuple[str, list[Meal_]]]:
    if not zip_path.exists():
        raise FileNotFoundError(f"CGMacros zip not found at {zip_path}")
    z = zipfile.ZipFile(zip_path)
    csvs = sorted(n for n in z.namelist() if n.endswith(".csv") and "CGMacros-" in n)
    out = []
    for n in csvs[: limit or len(csvs)]:
        sid = n.split("/")[-1].replace(".csv", "")
        out.append((sid, load_subject(z.read(n), sid)))
    return out


# --- engine prediction ------------------------------------------------------------------

def predict_meal_iauc(carbs_g: float, insulin_sensitivity: float,
                      profile: dict | None = None, protein_g: float = 0.0,
                      fat_g: float = 0.0, fiber_g: float = 0.0) -> tuple[float, float]:
    """Simulate one meal from a fasted baseline -> (iAUC, peak) over 0-3h, same metric as real.
    Passing fat/fibre lets the engine distinguish a salad from candy at equal carbs."""
    p = PhysioParams.from_profile(profile or {"weight_kg": 82, "height_cm": 175,
                                              "age": 45, "sex": "male"})
    p.insulin_sensitivity = float(np.clip(insulin_sensitivity, 0.05, 3.0))
    s = Schedule(); s.add(Meal(_PRE, carbs_g=carbs_g, protein_g=protein_g,
                               fat_g=fat_g, fiber_g=fiber_g))   # meal at t=PRE -> a baseline
    traj = Simulator(p).run(s, duration_min=_PRE + _WINDOW, dt=1.0, record_every=5,
                            outputs=["glucose_mg_dl"])
    obs = observe_series(traj, spec_for("cgm", "glucose"), step_min=5.0)
    m = postprandial_metrics(obs["values"], obs["t0_min"], 5.0, _PRE, _WINDOW, _PRE)
    return (m["iauc"], m["peak"]) if m else (float("nan"), float("nan"))


def fit_subject_si(meals: list[Meal_], grid=None, profile: dict | None = None) -> float:
    """Per-subject insulin sensitivity from their real meal responses (1-D grid on iAUC),
    using their REAL demographics so Si isn't compensating for a wrong body size."""
    grid = np.linspace(0.2, 1.6, 15) if grid is None else grid
    reals = [(m, m.real_iauc()) for m in meals]
    reals = [(m, r) for m, r in reals if r is not None]
    if len(reals) < 3:
        return float("nan")
    best, best_err = float(grid[0]), float("inf")
    for si in grid:
        err = sum(abs(predict_meal_iauc(m.carbs_g, si, profile, m.protein_g, m.fat_g,
                                        m.fiber_g)[0] - r) for m, r in reals)
        if err < best_err:
            best, best_err = float(si), err
    return best


# --- real demographics + labs (bio.csv) -------------------------------------------------

def load_bio(zip_path: Path = _ZIP) -> dict[str, dict]:
    """subject id (e.g. 'CGMacros-001') -> real demographics + fasting labs from bio.csv."""
    z = zipfile.ZipFile(zip_path)
    name = next((n for n in z.namelist() if n.lower().endswith("bio.csv")), None)
    if not name:
        return {}
    out = {}
    for r in csv.DictReader(io.StringIO(z.read(name).decode("utf-8", "replace"))):
        sid = str(r.get("subject", "")).strip()
        if not sid:
            continue
        wt_lb, ht_in = _f(r.get("Body weight ")), _f(r.get("Height "))
        a1c = _f(r.get("A1c PDL (Lab)"))
        fgl, fins = _f(r.get("Fasting GLU - PDL (Lab)")), _f(r.get("Insulin "))
        out[f"CGMacros-{int(sid):03d}"] = {
            "age": _f(r.get("Age")), "sex": "female" if str(r.get("Gender", "")).strip().upper() == "F" else "male",
            "weight_kg": round(wt_lb * 0.453592, 1) if wt_lb else None,
            "height_cm": round(ht_in * 2.54, 1) if ht_in else None,
            "bmi": _f(r.get("BMI")), "hba1c": a1c,
            "fasting_glucose": fgl, "fasting_insulin": fins,
            # HOMA-IR = fasting glucose(mg/dL) * fasting insulin(uU/mL) / 405 — the standard
            # insulin-resistance index; HIGHER = more resistant.
            "homa_ir": round(fgl * fins / 405.0, 2) if (fgl and fins) else None,
            # ADA HbA1c cutoffs: >=6.5 diabetic, 5.7-6.4 prediabetic, <5.7 normal
            "status": ("diabetic" if a1c and a1c >= 6.5 else
                       "prediabetic" if a1c and a1c >= 5.7 else "normal") if a1c else "unknown",
        }
    return out


def _profile(bio: dict | None) -> dict | None:
    if not bio or not (bio.get("weight_kg") and bio.get("height_cm")):
        return None
    return {"weight_kg": bio["weight_kg"], "height_cm": bio["height_cm"],
            "age": bio.get("age") or 45, "sex": bio.get("sex", "male")}


def _pearson(xs, ys) -> tuple[float, int]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None
             and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return float("nan"), len(pairs)
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    return float(np.corrcoef(a, b)[0, 1]), len(pairs)


def validate_subject(meals: list[Meal_]) -> dict | None:
    reals = [(m, m.real_iauc()) for m in meals]
    reals = [(m, r) for m, r in reals if r is not None]
    if len(reals) < 4:
        return None
    si = fit_subject_si(meals)
    eng_err, pop_err, pers_err = [], [], []
    prev = None
    for m, r in reals:
        eng_err.append(abs(predict_meal_iauc(m.carbs_g, si)[0] - r))
        pop_err.append(abs(predict_meal_iauc(m.carbs_g, 1.0)[0] - r))
        if prev is not None:
            pers_err.append(abs(prev - r))
        prev = r
    mean_real = statistics.fmean(r for _, r in reals)
    return {"subject": meals[0].subject, "n_meals": len(reals), "fitted_si": round(si, 3),
            "mean_real_iauc": round(mean_real, 0),
            "engine_fitted_mae": round(statistics.fmean(eng_err), 0),
            "engine_population_mae": round(statistics.fmean(pop_err), 0),
            "persistence_mae": round(statistics.fmean(pers_err), 0) if pers_err else None}


def held_out_subject(meals: list[Meal_], profile: dict | None = None,
                     min_meals: int = 8) -> dict | None:
    """Held-out predictive validation for ONE subject: fit Si on the chronological FIRST HALF
    of their meals, then predict the SECOND HALF. Nothing about the test meals touches the fit.

    This tests PREDICTIVE utility (does the personalized twin predict future meals better than
    baselines), distinct from the Si-vs-labs result which only tests physiological meaning. The
    bar that justifies shipping personalization: engine must beat `personal_mean` — the
    subject's own average iAUC — on held-out meals. If it can't, the physiology adds nothing
    over 'assume every meal is your typical one'.
    """
    reals = [(m, m.real_iauc()) for m in meals]
    reals = [(m, r) for m, r in reals if r is not None]     # already chronological
    if len(reals) < min_meals:
        return None
    k = len(reals) // 2
    train, test = reals[:k], reals[k:]
    si = fit_subject_si([m for m, _ in train], profile=profile)
    if not np.isfinite(si):
        return None
    personal_mean = statistics.fmean(r for _, r in train)

    eng, pop, pm, pers = [], [], [], []
    prev = train[-1][1]                                     # last train meal seeds persistence
    for m, r in test:
        macros = (m.protein_g, m.fat_g, m.fiber_g)
        eng.append(abs(predict_meal_iauc(m.carbs_g, si, profile, *macros)[0] - r))
        pop.append(abs(predict_meal_iauc(m.carbs_g, 1.0, profile, *macros)[0] - r))
        pm.append(abs(personal_mean - r))
        pers.append(abs(prev - r))
        prev = r
    return {"subject": meals[0].subject, "fitted_si": round(si, 3),
            "n_train": len(train), "n_test": len(test),
            "engine_fitted_mae": round(statistics.fmean(eng), 0),
            "engine_population_mae": round(statistics.fmean(pop), 0),
            "personal_mean_mae": round(statistics.fmean(pm), 0),
            "persistence_mae": round(statistics.fmean(pers), 0)}


def run_held_out(limit: int | None = None) -> list[dict]:
    bio = load_bio()
    out = []
    for sid, meals in subjects(limit=limit):
        v = held_out_subject(meals, profile=_profile(bio.get(sid)))
        if v:
            out.append(v)
    return out


def si_vs_labs(limit: int | None = None) -> list[dict]:
    """Fit each subject's Si with their REAL demographics, pair with real HbA1c / HOMA-IR.
    The physiological test: a meaningful Si must fall as insulin resistance rises."""
    bio = load_bio()
    out = []
    for sid, meals in subjects(limit=limit):
        b = bio.get(sid)
        si = fit_subject_si(meals, profile=_profile(b))
        n = sum(1 for m in meals if m.real_iauc() is not None)
        if b and np.isfinite(si) and n >= 4:
            out.append({"subject": sid, "si": si, "n": n, "hba1c": b["hba1c"],
                        "homa_ir": b["homa_ir"], "status": b["status"], "bmi": b["bmi"]})
    return out


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=None, help="limit (default: all 45)")
    ap.add_argument("--mode", choices=["heldout", "corr"], default="heldout")
    args = ap.parse_args()

    if args.mode == "heldout":
        return _print_held_out(run_held_out(limit=args.subjects))

    rows = si_vs_labs(limit=args.subjects)
    print("=" * 84)
    print("CGMacros — is fitted engine Si PHYSIOLOGICALLY real? (Si vs real HbA1c / HOMA-IR)")
    print("=" * 84)
    print(f"  {'subject':16}{'Si':>6}{'HbA1c':>7}{'HOMA-IR':>9}{'BMI':>7}  status")
    for r in sorted(rows, key=lambda r: r["si"]):
        print(f"  {r['subject']:16}{r['si']:>6.2f}{_s(r['hba1c']):>7}{_s(r['homa_ir']):>9}"
              f"{_s(r['bmi'], 1):>7}  {r['status']}")

    r_a1c, n_a1c = _pearson([r["si"] for r in rows], [r["hba1c"] for r in rows])
    r_homa, n_homa = _pearson([r["si"] for r in rows], [r["homa_ir"] for r in rows])
    # group means
    def gm(st):
        v = [r["si"] for r in rows if r["status"] == st]
        return (round(statistics.fmean(v), 2), len(v)) if v else ("-", 0)

    print("-" * 84)
    print(f"  n subjects: {len(rows)}")
    print(f"  Pearson r(Si, HbA1c)   = {r_a1c:+.2f}  (n={n_a1c})   expect NEGATIVE")
    print(f"  Pearson r(Si, HOMA-IR) = {r_homa:+.2f}  (n={n_homa})   expect NEGATIVE")
    print(f"  mean Si  normal {gm('normal')}  prediabetic {gm('prediabetic')}  diabetic {gm('diabetic')}")
    verdict = (r_a1c < -0.3 or r_homa < -0.3)
    print(f"\n  VERDICT: fitted Si tracks real insulin resistance: "
          f"{'YES — the personalization is physiologically grounded' if verdict else 'WEAK/NO — Si is a curve-fit knob, not physiology'}")
    print("  NOTE: Si still fit IN-SAMPLE on iAUC; this tests physiological meaning, not")
    print("  held-out predictive accuracy. Both matter.")
    print("=" * 84)


def _print_held_out(rows: list[dict]) -> None:
    print("=" * 84)
    print("CGMacros — HELD-OUT predictive validation (fit Si on 1st half of meals, predict 2nd)")
    print("=" * 84)
    if not rows:
        print("  no subjects with enough meals")
        return
    print(f"  {'subject':16}{'Si':>6}{'test':>6}   iAUC MAE:  {'engine':>8}{'personal':>10}"
          f"{'persist':>9}{'pop':>8}")
    for r in sorted(rows, key=lambda r: r["engine_fitted_mae"]):
        print(f"  {r['subject']:16}{r['fitted_si']:>6.2f}{r['n_test']:>6}   "
              f"{'':11}{r['engine_fitted_mae']:>8.0f}{r['personal_mean_mae']:>10.0f}"
              f"{r['persistence_mae']:>9.0f}{r['engine_population_mae']:>8.0f}")

    def avg(k):
        return statistics.fmean(r[k] for r in rows)
    e, pm, pe, po = (avg("engine_fitted_mae"), avg("personal_mean_mae"),
                     avg("persistence_mae"), avg("engine_population_mae"))
    # per-subject win rate vs the bar that matters
    beats_pm = sum(r["engine_fitted_mae"] < r["personal_mean_mae"] for r in rows)
    print("-" * 84)
    print(f"  n subjects: {len(rows)}   (all predictions HELD OUT)")
    print(f"  mean held-out iAUC MAE:  engine {e:.0f}  |  personal-mean {pm:.0f}  |  "
          f"persistence {pe:.0f}  |  population {po:.0f}")
    print(f"  engine beats 'your own average' on held-out meals: "
          f"{beats_pm}/{len(rows)} subjects ({100*beats_pm/len(rows):.0f}%)")
    win = e < pm
    print(f"\n  VERDICT: personalized twin predicts held-out meals better than a personal "
          f"constant: {'YES — physiology adds predictive value' if win else 'NO — personalization not justified yet'}")
    print("=" * 84)


def _s(x, dp=1):
    return f"{x:.{dp}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "-"


if __name__ == "__main__":
    main()
