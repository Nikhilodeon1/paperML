"""Metabolic module — energy balance and weight trajectory.

Mechanistic model (brief 2, Layer 2), built on validated equations from Layer 1:

  - Resting metabolic rate (RMR): Mifflin-St Jeor.
  - Total daily energy expenditure: TDEE = PAL * RMR.
  - Weight trajectory: the energy-balance identity integrated day by day. RMR (and
    thus TDEE) is recomputed as weight changes, so the projection self-corrects
    toward an energy-balance plateau rather than predicting unbounded linear change.

Confidence intervals come from Monte-Carlo over the population priors (per-user RMR
multiplier, PAL). They widen with horizon because the daily energy-balance residual
compounds over time (brief 2). Layer 3 will tighten the RMR prior per user.

Caveat encoded in Layer 1: the fixed 7700 kcal/kg constant overestimates long-run
loss (Hall 2011 dynamic model). v1 uses the simple constant with a documented note;
refine later. This is a 'strong but simplified' evidence case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from knowledge_base import load_system
from modules.hepatic import EvidenceLevel  # shared three-outcome enum

Sex = str  # "male" | "female"

_PAL_PARAM = {
    "sedentary": "pal_sedentary",
    "moderate": "pal_moderate",
    "active": "pal_active",
}


@dataclass(frozen=True)
class MetabolicPersonalParams:
    """Optional Layer 3 overrides. None => population prior.

    `rmr_multiplier_sd`, when supplied, is the Bayesian posterior SD from Layer 3 and
    is used directly as the RMR-multiplier uncertainty (the proper hookup). When it
    is None, the cruder `n_observations` shrink stand-in is used instead.
    """

    rmr_multiplier: float | None = None      # personal RMR / predicted RMR (posterior mean)
    rmr_multiplier_sd: float | None = None    # posterior SD from Layer 3
    pal: float | None = None
    n_observations: int = 0


@dataclass
class WeightResult:
    days: np.ndarray
    weight_median: np.ndarray          # kg
    weight_lower: np.ndarray
    weight_upper: np.ndarray
    final_weight: float                # median kg at horizon
    final_weight_ci: tuple[float, float]
    delta_kg: float                    # median change from start
    delta_kg_ci: tuple[float, float]
    tdee_estimate: float               # median maintenance TDEE at start (kcal/day)
    drivers: list[tuple[str, float]]
    evidence: EvidenceLevel
    confidence_label: str
    citations: list[str] = field(default_factory=list)


def _rmr_mifflin(weight_kg, height_cm, age, sex, kb) -> float:
    s = kb["msj_sex_offset_male"].value if sex == "male" else kb["msj_sex_offset_female"].value
    return (kb["msj_weight_coeff"].value * weight_kg
            + kb["msj_height_coeff"].value * height_cm
            - kb["msj_age_coeff"].value * age
            + s)


def _simulate_weight(weight0, height_cm, age, sex, intake_kcal,
                     rmr_mult, pal, kcal_per_kg, days, kb) -> np.ndarray:
    """Integrate the energy-balance identity day by day. RMR tracks current weight."""
    w = weight0
    out = np.empty(days.size)
    for i, _ in enumerate(days):
        rmr = _rmr_mifflin(w, height_cm, age, sex, kb) * rmr_mult
        tdee = pal * rmr
        balance = intake_kcal - tdee            # kcal/day surplus(+)/deficit(-)
        w = max(30.0, w + balance / kcal_per_kg)  # floor to avoid nonsense
        out[i] = w
    return out


def project_weight(
    weight_kg: float,
    height_cm: float,
    age: float,
    sex: Sex,
    daily_intake_kcal: float,
    activity: str = "sedentary",
    personal: MetabolicPersonalParams | None = None,
    horizon_days: int = 365,
    n_samples: int = 400,
    ci: float = 0.90,
    seed: int = 0,
) -> WeightResult:
    """Project weight trajectory under a constant daily intake / activity level.

    Strong-but-simplified evidence (brief 8): Mifflin-St Jeor + energy balance are
    well validated short-to-medium term; the fixed kcal/kg constant is a documented
    long-horizon simplification, so very long projections are returned at reduced
    confidence rather than as a precise figure.
    """
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")
    if activity not in _PAL_PARAM:
        raise ValueError(f"activity must be one of {list(_PAL_PARAM)}")

    kb = load_system("metabolic")
    personal = personal or MetabolicPersonalParams()
    days = np.arange(1, horizon_days + 1)

    pal_param = kb[_PAL_PARAM[activity]]
    rmr_sd_param = kb["rmr_individual_sd"]
    kcal_param = kb["kcal_per_kg_fat"]

    pal_mu = personal.pal if personal.pal is not None else pal_param.value
    rmr_mult_mu = personal.rmr_multiplier if personal.rmr_multiplier is not None else 1.0

    # RMR-multiplier uncertainty: prefer a real Layer 3 posterior SD; otherwise fall
    # back to the population prior narrowed by the n_observations shrink stand-in.
    shrink = 1.0 / np.sqrt(personal.n_observations + 1.0)
    rmr_mult_sd = (personal.rmr_multiplier_sd if personal.rmr_multiplier_sd is not None
                   else rmr_sd_param.population_sd * shrink)
    rng = np.random.default_rng(seed)

    rmr_mult_s = np.clip(
        rng.normal(rmr_mult_mu, rmr_mult_sd, n_samples), 0.7, 1.3
    )
    pal_s = np.clip(
        rng.normal(pal_mu, pal_param.population_sd * shrink, n_samples), *pal_param.plausible_range
    )
    kcal_s = np.clip(
        rng.normal(kcal_param.value, (kcal_param.value * 0.03), n_samples), *kcal_param.plausible_range
    )

    traj = np.empty((n_samples, days.size))
    for k in range(n_samples):
        traj[k] = _simulate_weight(weight_kg, height_cm, age, sex, daily_intake_kcal,
                                   rmr_mult_s[k], pal_s[k], kcal_s[k], days, kb)

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    final = traj[:, -1]
    delta = final - weight_kg

    base_rmr = _rmr_mifflin(weight_kg, height_cm, age, sex, kb) * rmr_mult_mu
    tdee0 = pal_mu * base_rmr

    drivers = _rank_drivers(weight_kg, height_cm, age, sex, daily_intake_kcal,
                            rmr_mult_mu, pal_mu, kcal_param.value, days, kb,
                            rmr_sd_param, pal_param)

    # Evidence downgrades: adolescents (Mifflin-St Jeor is adult-only and ignores
    # growth), and long horizons (fixed kcal/kg + no metabolic adaptation).
    if age < 18:
        evidence = EvidenceLevel.WEAK
        why = ("age under 18: Mifflin-St Jeor is validated for adults and ignores "
               "growth — an adolescent is still developing, so this is a rough "
               "adult-model estimate, not reliable")
    elif age > 80:
        evidence = EvidenceLevel.WEAK
        why = "age over 80: outside the equation's typical validation range — treat as rough"
    elif horizon_days >= 180:
        evidence = EvidenceLevel.WEAK
        why = ("long horizon (>= 6 months): fixed kcal/kg constant overestimates loss and "
               "ignores metabolic adaptation (Hall 2011) — treat as directional")
    else:
        evidence = EvidenceLevel.STRONG
        why = "Mifflin-St Jeor + energy balance, well validated at this horizon"

    return WeightResult(
        days=days,
        weight_median=np.median(traj, axis=0),
        weight_lower=np.percentile(traj, lo_q, axis=0),
        weight_upper=np.percentile(traj, hi_q, axis=0),
        final_weight=float(np.median(final)),
        final_weight_ci=(float(np.percentile(final, lo_q)), float(np.percentile(final, hi_q))),
        delta_kg=float(np.median(delta)),
        delta_kg_ci=(float(np.percentile(delta, lo_q)), float(np.percentile(delta, hi_q))),
        tdee_estimate=float(tdee0),
        drivers=drivers,
        evidence=evidence,
        confidence_label=_confidence_label(personal.n_observations, why),
        citations=[kb["msj_weight_coeff"].citation, pal_param.citation, kcal_param.citation],
    )


def _rank_drivers(weight0, height_cm, age, sex, intake, rmr_mult_mu, pal_mu,
                  kcal, days, kb, rmr_sd_param, pal_param) -> list[tuple[str, float]]:
    """One-at-a-time sensitivity of final-weight change to each parameter (+1 SD)."""
    def final_delta(rmr_mult, pal):
        w = _simulate_weight(weight0, height_cm, age, sex, intake, rmr_mult, pal, kcal, days, kb)
        return w[-1] - weight0

    base = final_delta(rmr_mult_mu, pal_mu)
    scale = max(abs(base), 1e-6)
    out = [
        ("resting metabolic rate (personal RMR)",
         abs(final_delta(rmr_mult_mu + rmr_sd_param.population_sd, pal_mu) - base) / scale),
        ("activity level (PAL)",
         abs(final_delta(rmr_mult_mu, min(pal_mu + pal_param.population_sd, pal_param.plausible_range[1])) - base) / scale),
    ]
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _confidence_label(n_obs: int, why: str) -> str:
    if n_obs >= 20:
        data = "high (well-personalized)"
    elif n_obs >= 5:
        data = "moderate (some personal data)"
    else:
        data = "population-level only (no personal calibration yet)"
    return f"{data}; {why}"
