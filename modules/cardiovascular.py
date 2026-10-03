"""Cardiovascular module — 10-year CVD risk (Framingham General CVD, D'Agostino 2008).

Mechanistic/statistical, NOT a trained model (brief 2): a published Cox-model risk
equation whose coefficients live in Layer 1. The linear predictor sums weighted
log-transformed risk factors; risk = 1 - S0^exp(sum - mean).

Confidence interval comes from Monte-Carlo over measurement noise on the inputs
(a single clinic BP / lipid reading is noisy) — the same honest-uncertainty pattern
as the other modules. Outside the model's validated age range (30-74) the result is
downgraded to weak evidence rather than reported as precise (brief 8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from knowledge_base import load_raw, load_system
from modules.hepatic import EvidenceLevel

Sex = str  # "male" | "female"


@dataclass
class CvdRiskResult:
    risk_10yr: float                     # median 10-year CVD probability (0..1)
    risk_ci: tuple[float, float]
    heart_age_note: str
    drivers: list[tuple[str, float]]     # modifiable factors ranked by impact
    evidence: EvidenceLevel
    confidence_label: str
    citations: list[str] = field(default_factory=list)


def _linear_predictor(c: dict, age, total_chol, hdl, sbp, treated_bp, smoker, diabetic) -> float:
    sbp_beta = c["ln_sbp_treated"] if treated_bp else c["ln_sbp_untreated"]
    return (c["ln_age"] * math.log(age)
            + c["ln_total_chol"] * math.log(total_chol)
            + c["ln_hdl"] * math.log(hdl)
            + sbp_beta * math.log(sbp)
            + c["smoker"] * (1.0 if smoker else 0.0)
            + c["diabetes"] * (1.0 if diabetic else 0.0))


def _risk(c: dict, **kw) -> float:
    s = _linear_predictor(c, **kw)
    return 1.0 - c["baseline_survival_s0"] ** math.exp(s - c["mean_sum"])


def compute_cvd_risk(
    age: float,
    sex: Sex,
    total_chol: float,       # mg/dL
    hdl: float,              # mg/dL
    sbp: float,              # mmHg, systolic
    treated_bp: bool = False,
    smoker: bool = False,
    diabetic: bool = False,
    n_samples: int = 2000,
    ci: float = 0.90,
    seed: int = 0,
) -> CvdRiskResult:
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")

    raw = load_raw("cardiovascular")["coefficients"]
    c = raw[sex]
    lo_age, hi_age = raw["_validated_age_range"]

    noise = load_system("cardiovascular")
    rng = np.random.default_rng(seed)

    # MC over measurement noise on the continuous inputs.
    sbp_s = np.clip(rng.normal(sbp, noise["sbp_measurement_sd"].population_sd, n_samples), 80, 250)
    chol_s = np.clip(rng.normal(total_chol, noise["total_chol_measurement_sd"].population_sd, n_samples), 100, 400)
    hdl_s = np.clip(rng.normal(hdl, noise["hdl_measurement_sd"].population_sd, n_samples), 15, 120)

    risks = np.array([
        _risk(c, age=age, total_chol=chol_s[i], hdl=hdl_s[i], sbp=sbp_s[i],
              treated_bp=treated_bp, smoker=smoker, diabetic=diabetic)
        for i in range(n_samples)
    ])

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    point = _risk(c, age=age, total_chol=total_chol, hdl=hdl, sbp=sbp,
                  treated_bp=treated_bp, smoker=smoker, diabetic=diabetic)

    in_range = lo_age <= age <= hi_age
    if in_range:
        evidence = EvidenceLevel.STRONG
        why = "Framingham General CVD model, within its validated age range"
    else:
        evidence = EvidenceLevel.WEAK
        why = f"age {age:.0f} is outside the model's validated range {lo_age}-{hi_age} — extrapolation"

    drivers = _rank_drivers(c, age, total_chol, hdl, sbp, treated_bp, smoker, diabetic, point)

    return CvdRiskResult(
        risk_10yr=float(point),
        risk_ci=(float(np.percentile(risks, lo_q)), float(np.percentile(risks, hi_q))),
        heart_age_note=_heart_age(c, point, sex, treated_bp),
        drivers=drivers,
        evidence=evidence,
        confidence_label=f"population-level model; {why}",
        citations=[load_raw("cardiovascular")["coefficients"]["_citation"]],
    )


def _rank_drivers(c, age, total_chol, hdl, sbp, treated_bp, smoker, diabetic, base):
    """Impact of improving each modifiable factor to a healthy reference value."""
    scenarios = {
        "quit smoking": dict(smoker=False) if smoker else None,
        "lower SBP to 120": dict(sbp=min(sbp, 120.0)) if sbp > 120 else None,
        "raise HDL to 60": dict(hdl=max(hdl, 60.0)) if hdl < 60 else None,
        "lower total chol to 180": dict(total_chol=min(total_chol, 180.0)) if total_chol > 180 else None,
    }
    base_kw = dict(age=age, total_chol=total_chol, hdl=hdl, sbp=sbp,
                   treated_bp=treated_bp, smoker=smoker, diabetic=diabetic)
    out = []
    for label, override in scenarios.items():
        if not override:
            continue
        kw = {**base_kw, **override}
        out.append((label, base - _risk(c, **kw)))  # absolute risk reduction
    out.sort(key=lambda z: z[1], reverse=True)
    return out


def _heart_age(c, risk, sex, treated_bp):
    """Approximate 'heart age': the age of a same-sex person with otherwise ideal
    risk factors who carries this 10-year risk. A communication aid (brief 4 style)."""
    ideal = dict(total_chol=180.0, hdl=50.0, sbp=120.0, treated_bp=False,
                 smoker=False, diabetic=False)
    for a in range(30, 96):
        if _risk(c, age=float(a), **ideal) >= risk:
            return f"~{a} years (heart age vs an otherwise-ideal {sex})"
    return ">95 years (heart age)"
