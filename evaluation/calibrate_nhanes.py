"""Validate the mechanistic modules against a real population (NHANES 2017-2018).

NHANES is cross-sectional (no 10-year follow-up), so this is a face-validity /
distribution check, not an outcome backtest: do our equation modules produce sensible
numbers across ~9,000 real people?

  1. Cardiovascular (Framingham): 10-yr CVD risk must rise monotonically across age
     decades and sit in a plausible population range.
  2. Metabolic (Mifflin-St Jeor): RMR must fall in a plausible physiological range;
     estimated TDEE vs self-reported intake should show the well-documented NHANES
     under-reporting (reported intake below estimated need), not an absurd ratio.

Run:  python -m evaluation.calibrate_nhanes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.cardiovascular import _risk
from modules.metabolic import _rmr_mifflin
from knowledge_base import load_raw, load_system

_CDC = __import__("pathlib").Path(__file__).resolve().parents[1].parent / "data" / "cdc"


def _load():
    def x(f):
        return pd.read_sas(_CDC / f, format="xport")
    demo = x("DEMO_J.xpt")[["SEQN", "RIAGENDR", "RIDAGEYR"]]
    bmx = x("BMX_J.xpt")[["SEQN", "BMXWT", "BMXHT"]]
    bpx = x("BPX_J.xpt")[["SEQN", "BPXSY1", "BPXSY2", "BPXSY3"]]
    tc = x("TCHOL_J.xpt")[["SEQN", "LBXTC"]]
    hdl = x("HDL_J.xpt")[["SEQN", "LBDHDD"]]
    kcal = x("DR1TOT_J.xpt")[["SEQN", "DR1TKCAL"]]
    smq = x("SMQ_J.xpt")[["SEQN", "SMQ040"]]
    diq = x("DIQ_J.xpt")[["SEQN", "DIQ010"]]

    df = demo
    for other in (bmx, bpx, tc, hdl, kcal, smq, diq):
        df = df.merge(other, on="SEQN", how="left")

    df["sex"] = np.where(df["RIAGENDR"] == 1, "male", "female")
    df["age"] = df["RIDAGEYR"]
    df["sbp"] = df[["BPXSY1", "BPXSY2", "BPXSY3"]].replace(0, np.nan).mean(axis=1)
    df["smoker"] = df["SMQ040"].isin([1, 2])          # current every/some days
    df["diabetic"] = df["DIQ010"] == 1
    return df


def run() -> int:
    df = _load()
    coeffs = load_raw("cardiovascular")["coefficients"]
    ok = True

    # --- 1. Cardiovascular ---------------------------------------------------
    cvd = df.dropna(subset=["age", "LBXTC", "LBDHDD", "sbp"]).copy()
    cvd = cvd[(cvd["age"] >= 30) & (cvd["age"] <= 74)
              & (cvd["LBXTC"] > 0) & (cvd["LBDHDD"] > 0)]

    def risk_row(r):
        c = coeffs[r["sex"]]
        return _risk(c, age=float(r["age"]), total_chol=float(r["LBXTC"]),
                     hdl=float(r["LBDHDD"]), sbp=float(r["sbp"]),
                     treated_bp=False, smoker=bool(r["smoker"]), diabetic=bool(r["diabetic"]))

    cvd["risk"] = cvd.apply(risk_row, axis=1)

    print("=" * 66)
    print(f"NHANES 2017-2018 CALIBRATION  (n={len(df)} respondents)")
    print("=" * 66)
    print(f"CARDIOVASCULAR (Framingham) — {len(cvd)} adults 30-74 with full labs")
    bands = [(30, 40), (40, 50), (50, 60), (60, 75)]
    means = []
    for lo, hi in bands:
        sub = cvd[(cvd["age"] >= lo) & (cvd["age"] < hi)]
        m = sub["risk"].mean()
        means.append(m)
        print(f"  age {lo}-{hi}: mean 10-yr risk {m*100:5.1f}%  (n={len(sub)})")
    monotonic = all(means[i] < means[i + 1] for i in range(len(means) - 1))
    overall = cvd["risk"].mean()
    plausible = 0.02 <= overall <= 0.30
    ok &= monotonic and plausible
    print(f"  overall mean {overall*100:.1f}% | monotonic by age: "
          f"{'OK' if monotonic else 'FAIL'} | plausible: {'OK' if plausible else 'FAIL'}")

    # --- 2. Metabolic --------------------------------------------------------
    met = df.dropna(subset=["age", "BMXWT", "BMXHT"]).copy()
    met = met[(met["age"] >= 20) & (met["age"] <= 80)]
    kb = load_system("metabolic")
    met["rmr"] = met.apply(lambda r: _rmr_mifflin(float(r["BMXWT"]), float(r["BMXHT"]),
                                                  float(r["age"]), r["sex"], kb), axis=1)
    met["tdee"] = met["rmr"] * kb["pal_sedentary"].value
    rmr_mean = met["rmr"].mean()
    rmr_ok = 1200 <= rmr_mean <= 1900

    intake = met.dropna(subset=["DR1TKCAL"])
    intake = intake[intake["DR1TKCAL"] > 0]
    ratio = intake["DR1TKCAL"].mean() / intake["tdee"].mean()
    # NHANES self-report is known to under-capture intake (~0.6-0.9 of need).
    ratio_ok = 0.55 <= ratio <= 1.05
    ok &= rmr_ok and ratio_ok

    print(f"\nMETABOLIC (Mifflin-St Jeor) — {len(met)} adults 20-80")
    print(f"  mean RMR {rmr_mean:.0f} kcal/day (plausible 1200-1900): "
          f"{'OK' if rmr_ok else 'FAIL'}")
    print(f"  reported intake / estimated TDEE = {ratio:.2f} "
          f"(under-report expected): {'OK' if ratio_ok else 'FAIL'}")

    print("=" * 66)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
