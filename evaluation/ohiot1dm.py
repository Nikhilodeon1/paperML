"""OhioT1DM — validate the exogenous-INSULIN pathway on real Type-1 diabetes data.

The insulin bolus support added to the engine (`simulation/inputs.py::Insulin`,
`metabolic.on_impulse`) was only checked on synthetic data. OhioT1DM has what's needed to
check it on real people: per-5-min CGM (`cbg`), meal carbs (`carbInput`), and INSULIN boluses
in units (`bolus`) for Type-1 patients (who make ~no insulin of their own, so their glucose
response is dominated by the bolus — the cleanest possible test of the pathway).

The test is RELATIVE, so it doesn't depend on knowing each patient's demographics: for each
meal that had a bolus, predict the postprandial glucose response WITH the bolus modelled and
WITHOUT it, and check which matches the real curve better. If modelling the bolus reduces
error, the insulin pathway is doing real work on real data. T1D physiology is set by zeroing
endogenous secretion.

Run:  python -m evaluation.ohiot1dm
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import numpy as np

from simulation import Simulator, PhysioParams, Schedule, Meal, Insulin
from simulation.observation import observe_series, spec_for
from evaluation.forward_validation import postprandial_metrics

_DIR = Path(__file__).resolve().parents[2] / "data" / "ohiot1dm-glucose-dataset" / "Ohio Data"
_PRE, _WINDOW = 30.0, 180.0
_STEP = 5.0
# a generic adult T1D profile — the with/without-bolus comparison is relative so this cancels
_PROFILE = {"weight_kg": 75, "height_cm": 175, "age": 40, "sex": "male"}


def _t1d_params():
    p = PhysioParams.from_profile(_PROFILE)
    p.insulin_secretion = 0.01          # Type 1: negligible endogenous insulin
    return p


def _rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            yield r


def load_meals(path: Path) -> list[dict]:
    """Meals with a nearby bolus + their real glucose window, from one processed CSV."""
    ts, gl, carbs, boluses = [], [], [], []
    for r in _rows(path):
        try:
            t = float(r["5minute_intervals_timestamp"]) * 5.0     # -> minutes
        except (ValueError, KeyError):
            continue
        g = r.get("cbg")
        g = float(g) if g not in ("", None) and r.get("missing_cbg", "0") in ("0", "0.0", "") else None
        ts.append(t); gl.append(g)
        c = r.get("carbInput"); b = r.get("bolus")
        if c not in ("", None) and float(c) > 0:
            carbs.append((t, float(c)))
        if b not in ("", None) and float(b) > 0:
            boluses.append((t, float(b)))

    meals = []
    for mt, c in carbs:
        # bolus within +/-20 min of the meal (typical mealtime dosing)
        near = [(bt, bu) for bt, bu in boluses if abs(bt - mt) <= 20.0]
        if not near:
            continue
        bolus_u = sum(bu for _, bu in near)
        win = [(t - mt, g) for t, g in zip(ts, gl)
               if g is not None and -_PRE - 5 <= (t - mt) <= _WINDOW + 5]
        if len(win) >= 20:
            meals.append({"carbs_g": c, "bolus_u": bolus_u, "window": win})
    return meals


def _grid(window):
    """5-min grid of the real glucose window, meal at t=0 -> (values, t0)."""
    edges = np.arange(-_PRE, _WINDOW + 1e-6, _STEP)
    vals = []
    for e in edges:
        near = [g for dt, g in window if e - 2.5 <= dt < e + 2.5]
        vals.append(statistics.fmean(near) if near else np.nan)
    v = np.array(vals, float)
    idx = np.where(~np.isnan(v))[0]
    if len(idx) >= 5:
        v = np.interp(np.arange(len(v)), idx, v[idx])
    return list(v), float(edges[0])


def _real_iauc(window):
    vals, t0 = _grid(window)
    m = postprandial_metrics(vals, t0, _STEP, 0.0, _WINDOW, _PRE)
    return m["iauc"] if m else None


def _pred_iauc(carbs, bolus_u):
    p = _t1d_params()
    s = Schedule(); s.add(Meal(_PRE, carbs_g=carbs))
    if bolus_u > 0:
        s.add(Insulin(_PRE, units=bolus_u))            # bolus at the meal
    traj = Simulator(p).run(s, duration_min=_PRE + _WINDOW, dt=1.0, record_every=5,
                            outputs=["glucose_mg_dl"])
    obs = observe_series(traj, spec_for("cgm", "glucose"), step_min=_STEP)
    m = postprandial_metrics(obs["values"], obs["t0_min"], _STEP, _PRE, _WINDOW, _PRE)
    return m["iauc"] if m else None


def validate() -> dict:
    files = list(_DIR.rglob("*_processed.csv"))
    with_err, without_err, n_meals, n_subj = [], [], 0, 0
    for f in files:
        meals = load_meals(f)
        if not meals:
            continue
        n_subj += 1
        for m in meals:
            real = _real_iauc(m["window"])
            with_b = _pred_iauc(m["carbs_g"], m["bolus_u"])
            without_b = _pred_iauc(m["carbs_g"], 0.0)
            if real is None or with_b is None or without_b is None:
                continue
            with_err.append(abs(with_b - real))
            without_err.append(abs(without_b - real))
            n_meals += 1
    if not with_err:
        return {"n_meals": 0}
    w, wo = statistics.fmean(with_err), statistics.fmean(without_err)
    wins = sum(a < b for a, b in zip(with_err, without_err))
    return {"n_subjects": n_subj, "n_meals": n_meals,
            "with_bolus_mae": round(w, 0), "without_bolus_mae": round(wo, 0),
            "improvement_pct": round(100 * (1 - w / wo)) if wo else 0,
            "meals_better_with_bolus_pct": round(100 * wins / n_meals)}


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if not _DIR.exists():
        print(f"OhioT1DM not found at {_DIR}")
        return
    r = validate()
    print("=" * 76)
    print("OhioT1DM — does modelling the INSULIN BOLUS fit real T1D glucose better?")
    print("=" * 76)
    if not r.get("n_meals"):
        print("  no meals with a paired bolus found")
        return
    print(f"  subjects {r['n_subjects']} | meals-with-bolus {r['n_meals']}")
    print(f"  postprandial iAUC MAE vs real glucose:")
    print(f"    WITH bolus modelled    : {r['with_bolus_mae']:.0f} mg/dL*min")
    print(f"    WITHOUT bolus (ignored): {r['without_bolus_mae']:.0f} mg/dL*min")
    print(f"  modelling the bolus improves the fit by {r['improvement_pct']}% "
          f"on {r['meals_better_with_bolus_pct']}% of meals")
    ok = r["with_bolus_mae"] < r["without_bolus_mae"]
    print(f"\n  VERDICT: insulin pathway helps on real T1D data: "
          f"{'YES — modelling the bolus fits real glucose better' if ok else 'NO — pathway not helping'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
