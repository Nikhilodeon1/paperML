"""Calibrate the simulation engine's blood-alcohol against the validated Widmark module.

`modules/hepatic.py` is already validated against the analytic Widmark model (backtest_
hepatic). This checks that the ENGINE's dynamic alcohol module reproduces that same,
trusted BAC — quantified error in real units — across doses, body weights and sex, so the
two alcohol paths in the product agree.

Run:  python -m evaluation.calibrate_engine_bac
"""

from __future__ import annotations

import statistics

from simulation import Simulator, PhysioParams, Schedule, Drink
from modules.hepatic import Drink as HDrink, compute_bac


def run() -> int:
    print("=" * 70)
    print("ENGINE BAC CALIBRATION vs validated Widmark (modules/hepatic)")
    print("=" * 70)
    peak_errs, sober_errs = [], []
    cases = [(80, "male", n) for n in (1, 2, 3, 4)] + \
            [(60, "female", n) for n in (1, 2, 3)] + [(95, "male", 5)]
    print("  weight sex drinks | engine_peak hepatic_peak | engine_sober hepatic_sober")
    for w, sex, n in cases:
        p = PhysioParams(weight_kg=w, sex=sex)
        e = Simulator(p).run(Schedule().add(Drink(0, n)), 600, outputs=["bac_g_dl"])
        eg = e.series["bac_g_dl"]
        e_peak = max(eg)
        # engine time-to-sober: last time BAC >= 0.001
        idx = max((i for i, v in enumerate(eg) if v >= 0.001), default=0)
        e_sober = e.times_min[idx] / 60.0
        h = compute_bac([HDrink.standard(n)], weight_kg=w, sex=sex)
        peak_errs.append(abs(e_peak - h.peak_bac))
        sober_errs.append(abs(e_sober - h.time_to_sober_h))
        print(f"  {w:>4} {sex:<6} {n:>2}    | {e_peak:.3f}       {h.peak_bac:.3f}       "
              f"| {e_sober:.1f}h        {h.time_to_sober_h:.1f}h")

    peak_mae = statistics.mean(peak_errs)
    sober_mae = statistics.mean(sober_errs)
    peak_ok = peak_mae < 0.005            # < 0.005 g/dL
    sober_ok = sober_mae < 0.8            # < ~48 min
    ok = peak_ok and sober_ok
    print("-" * 70)
    print(f"Peak-BAC MAE {peak_mae:.4f} g/dL (tol 0.005): {'OK' if peak_ok else 'FAIL'}")
    print(f"Time-to-sober MAE {sober_mae:.2f} h (tol 0.8): {'OK' if sober_ok else 'FAIL'}")
    print("=" * 70)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
