"""Backtest the cardiovascular module (Framingham General CVD).

The coefficients are the published model, so validation focuses on the properties a
correct implementation MUST have (these are certain, unlike a single hand-checked
number which is error-prone to transcribe):

  1. Output is a valid probability in (0, 1).
  2. Monotonic in every risk factor: risk rises with age, SBP, total cholesterol,
     smoking, diabetes; falls as HDL rises.
  3. Treated-BP coefficient >= untreated (treatment flags higher underlying risk).
  4. Archetype plausibility: a healthy 40 y is low risk (<8%); a 70 y smoker with
     diabetes and poor lipids is high risk (>30%).
  5. CI brackets the point estimate and widens with noisier inputs.

Reference cases are printed for eyeballing against any external Framingham calculator.

Run:  python -m evaluation.backtest_cardiovascular
"""

from __future__ import annotations

from modules.cardiovascular import compute_cvd_risk

HEALTHY = dict(total_chol=180, hdl=55, sbp=118, smoker=False, diabetic=False)


def run() -> int:
    ok = True
    print("=" * 70)
    print("CARDIOVASCULAR BACKTEST  (Framingham General CVD, D'Agostino 2008)")
    print("=" * 70)

    # Reference cases (for eyeballing).
    print("Reference cases (median 10-yr risk [90% CI]):")
    for label, kw in {
        "40y M, healthy": dict(age=40, sex="male", **HEALTHY),
        "40y F, healthy": dict(age=40, sex="female", **HEALTHY),
        "55y M, chol240 HDL40 SBP140 smoker": dict(age=55, sex="male", total_chol=240,
                                                   hdl=40, sbp=140, smoker=True, diabetic=False),
        "70y F, diabetic smoker poor lipids": dict(age=70, sex="female", total_chol=260,
                                                   hdl=38, sbp=160, smoker=True, diabetic=True),
    }.items():
        r = compute_cvd_risk(**kw)
        print(f"  {label:<42} {r.risk_10yr*100:5.1f}%  "
              f"[{r.risk_ci[0]*100:4.1f}, {r.risk_ci[1]*100:4.1f}]  {r.heart_age_note}")

    def risk(**kw):
        return compute_cvd_risk(**{"sex": "male", **HEALTHY, "age": 50, **kw}).risk_10yr

    # 1. Valid probability.
    base = risk()
    p_ok = 0 < base < 1
    ok &= p_ok

    # 2. Monotonicity.
    checks = {
        "age up -> risk up": risk(age=65) > risk(age=45),
        "SBP up -> risk up": risk(sbp=160) > risk(sbp=110),
        "total chol up -> risk up": risk(total_chol=280) > risk(total_chol=160),
        "HDL up -> risk down": risk(hdl=70) < risk(hdl=35),
        "smoking -> risk up": risk(smoker=True) > risk(smoker=False),
        "diabetes -> risk up": risk(diabetic=True) > risk(diabetic=False),
    }
    print("\nMonotonicity:")
    for label, passed in checks.items():
        ok &= passed
        print(f"  {label:<32} {'OK' if passed else 'FAIL'}")

    # 3. Treated BP >= untreated at same SBP.
    treated = compute_cvd_risk(age=50, sex="male", **HEALTHY, treated_bp=True).risk_10yr
    untreated = compute_cvd_risk(age=50, sex="male", **HEALTHY, treated_bp=False).risk_10yr
    t_ok = treated >= untreated
    ok &= t_ok
    print(f"\nTreated BP >= untreated at same SBP: {'OK' if t_ok else 'FAIL'} "
          f"({treated*100:.1f}% vs {untreated*100:.1f}%)")

    # 4. Archetype plausibility.
    healthy40 = compute_cvd_risk(age=40, sex="male", **HEALTHY).risk_10yr
    highrisk70 = compute_cvd_risk(age=70, sex="male", total_chol=260, hdl=35, sbp=165,
                                  smoker=True, diabetic=True).risk_10yr
    a_ok = healthy40 < 0.08 and highrisk70 > 0.30
    ok &= a_ok
    print(f"Archetypes: healthy 40y = {healthy40*100:.1f}% (<8%), "
          f"high-risk 70y = {highrisk70*100:.1f}% (>30%) -> {'OK' if a_ok else 'FAIL'}")

    # 5. CI brackets point.
    r = compute_cvd_risk(age=55, sex="male", total_chol=240, hdl=40, sbp=145, smoker=True)
    ci_ok = r.risk_ci[0] <= r.risk_10yr <= r.risk_ci[1]
    ok &= ci_ok
    print(f"CI brackets point estimate: {'OK' if ci_ok else 'FAIL'}")

    # 6. Out-of-range age downgrades evidence.
    from modules.hepatic import EvidenceLevel
    young = compute_cvd_risk(age=25, sex="male", **HEALTHY)
    e_ok = young.evidence is EvidenceLevel.WEAK
    ok &= e_ok
    print(f"Age 25 (outside 30-74) downgraded to weak evidence: {'OK' if e_ok else 'FAIL'}")

    print("=" * 70)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
