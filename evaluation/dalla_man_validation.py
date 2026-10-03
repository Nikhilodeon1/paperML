"""Correctness gates for the Dalla Man implementation — run BEFORE any inference.

Gate 1 (`validate_dalla_man_vs_paper`): a 75 g oral glucose load with population parameters must
reproduce the published response — plasma glucose peaking ~160-180 mg/dl at ~40-50 min and back to
approximately basal by 180 min.
Gate 2 (`validate_cgm_lag`): the subcutaneous compartment must trail plasma by ~10-15 min with a
slightly lower peak (this is the whole reason to prefer Dalla Man for CGM data).
Gate 0 (`validate_steady_state`): with no meal the model must not drift off basal — otherwise every
iAUC is corrupted.

IMPORTANT CAVEAT for the paper: this is a REDUCED, CALIBRATED Dalla Man-style model, not a verbatim
reimplementation. Portal insulin / dynamic beta-cell secretion / renal excretion are reduced (see
`simulation/dalla_man.py` docstring), and three parameters (kmin, beta_sec, kcr) were calibrated so
the OGTT matches the published curve — the nominal values available here produced a ~123 mg/dl peak
and a basal undershoot. Verify constants against the paper's supplementary Table I before claiming
a faithful Dalla Man implementation.

Run:  python -m evaluation.dalla_man_validation
"""
from __future__ import annotations

import numpy as np

from simulation.dalla_man import DallaManParams, run_meal


def validate_steady_state(tol: float = 1.0) -> dict:
    p = DallaManParams.defaults(78.0, 100.0)
    _ts, g = run_meal(p, 0.0, duration_min=240.0, plasma=True)
    g = np.asarray(g)
    drift = float(np.abs(g - 100.0).max())
    return {"max_drift_mg_dl": round(drift, 3), "pass": drift < tol}


def validate_dalla_man_vs_paper() -> dict:
    """75 g OGTT (liquid load) vs the published response."""
    p = DallaManParams.defaults(78.0, 100.0)
    ts, g = run_meal(p, 75.0, meal_time=0.0, duration_min=240.0, plasma=True, solid_frac=0.0)
    ts, g = np.asarray(ts), np.asarray(g)
    i = int(g.argmax())
    peak, t_peak, g180 = float(g[i]), float(ts[i]), float(g[36])
    ok = (160.0 <= peak <= 180.0) and (40.0 <= t_peak <= 50.0) and (90.0 <= g180 <= 115.0)
    return {"peak_mg_dl": round(peak, 1), "t_peak_min": t_peak,
            "g_120min": round(float(g[24]), 1), "g_180min": round(g180, 1),
            "target": "peak 160-180 @ 40-50 min, ~basal by 180", "pass": ok}


def validate_cgm_lag() -> dict:
    """Gsc must lag plasma by ~10-15 min and peak slightly lower."""
    p = DallaManParams.defaults(78.0, 100.0)
    ts, gp = run_meal(p, 75.0, meal_time=0.0, duration_min=240.0, plasma=True, solid_frac=0.0)
    _ts, gs = run_meal(p, 75.0, meal_time=0.0, duration_min=240.0, solid_frac=0.0)
    ts, gp, gs = np.asarray(ts), np.asarray(gp), np.asarray(gs)
    ip, is_ = int(gp.argmax()), int(gs.argmax())
    lag = float(ts[is_] - ts[ip])
    damp = float(gp[ip] - gs[is_])
    return {"plasma_peak": round(float(gp[ip]), 1), "gsc_peak": round(float(gs[is_]), 1),
            "lag_min": lag, "peak_damping_mg_dl": round(damp, 2),
            "pass": (5.0 <= lag <= 20.0) and (damp > 0)}


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    print("=" * 74)
    print("DALLA MAN VALIDATION GATES")
    print("=" * 74)
    results = {"steady_state": validate_steady_state(),
               "ogtt_vs_paper": validate_dalla_man_vs_paper(),
               "cgm_lag": validate_cgm_lag()}
    for name, r in results.items():
        status = "PASS" if r["pass"] else "FAIL"
        detail = "  ".join(f"{k}={v}" for k, v in r.items() if k not in ("pass", "target"))
        print(f"  [{status}] {name}: {detail}")
    allpass = all(r["pass"] for r in results.values())
    print("-" * 74)
    print(f"  ALL GATES: {'PASS — cleared to run inference' if allpass else 'FAIL — do not proceed'}")
    print("  NOTE: reduced + calibrated Dalla Man-style model; verify Table I constants against")
    print("        the paper before claiming a verbatim implementation.")
    print("=" * 74)


if __name__ == "__main__":
    main()
