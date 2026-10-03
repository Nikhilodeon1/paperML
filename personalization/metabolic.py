"""Layer 3 personalization for the metabolic module.

The latent parameter we personalize is the user's RMR multiplier `m`: their true
resting metabolic rate divided by the Mifflin-St Jeor population prediction. Prior:
N(1.0, sd^2) with sd from Layer 1 (`rmr_individual_sd`, ~10%).

Each pair of consecutive weigh-ins is one observation. Over an interval of `dt` days
with average daily intake `I`, the energy-balance identity gives the user's true
maintenance expenditure, hence an observation of `m`:

    Δw / dt = (I - TDEE_true) / kcal_per_kg
    TDEE_true = PAL * RMR_pred(mid-weight) * m
    => m_obs = (I - (Δw/dt) * kcal_per_kg) / (PAL * RMR_pred)

Day-to-day weight noise (water, glycogen) σ_w propagates to the observation variance
and shrinks with longer intervals, so sparse-but-spaced weigh-ins still inform `m`.
A small process variance lets metabolism drift slowly so the filter keeps tracking.
"""

from __future__ import annotations

from dataclasses import dataclass

from knowledge_base import load_system
from modules.metabolic import MetabolicPersonalParams, _rmr_mifflin
from personalization.gaussian_bayes import GaussianPosterior

_PAL_PARAM = {"sedentary": "pal_sedentary", "moderate": "pal_moderate", "active": "pal_active"}


@dataclass(frozen=True)
class Weighin:
    """One logged weigh-in plus the average daily intake since the previous one."""

    day: float                       # days since tracking start
    weight_kg: float
    mean_daily_intake_kcal: float    # average intake over the interval ending here


def estimate_rmr_multiplier(
    logs: list[Weighin],
    height_cm: float,
    age: float,
    sex: str,
    activity: str = "sedentary",
    weight_noise_kg: float = 0.7,
    drift_sd_per_year: float = 0.05,
) -> GaussianPosterior:
    """Recursively update the posterior over the RMR multiplier from weigh-ins."""
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")
    if activity not in _PAL_PARAM:
        raise ValueError(f"activity must be one of {list(_PAL_PARAM)}")

    kb = load_system("metabolic")
    pal = kb[_PAL_PARAM[activity]].value
    kcal = kb["kcal_per_kg_fat"].value
    prior_sd = kb["rmr_individual_sd"].population_sd

    post = GaussianPosterior(mean=1.0, var=prior_sd ** 2)
    drift_var_per_day = (drift_sd_per_year ** 2) / 365.0

    ordered = sorted(logs, key=lambda w: w.day)
    for prev, cur in zip(ordered, ordered[1:]):
        dt = cur.day - prev.day
        if dt <= 0:
            continue
        mid_weight = 0.5 * (prev.weight_kg + cur.weight_kg)
        rmr_pred = _rmr_mifflin(mid_weight, height_cm, age, sex, kb)
        if rmr_pred <= 0:
            continue

        dw_per_day = (cur.weight_kg - prev.weight_kg) / dt
        m_obs = (cur.mean_daily_intake_kcal - dw_per_day * kcal) / (pal * rmr_pred)

        # var(m_obs) from weight noise on both weigh-ins, propagated through the map.
        gain = kcal / (pal * rmr_pred)
        obs_var = (gain ** 2) * (2.0 * weight_noise_kg ** 2) / (dt ** 2)

        post = post.step(m_obs, obs_var, process_var=drift_var_per_day * dt)

    return post


def personal_params_from_logs(logs: list[Weighin], height_cm: float, age: float,
                              sex: str, activity: str = "sedentary") -> MetabolicPersonalParams:
    """Convenience: posterior -> MetabolicPersonalParams the module consumes directly."""
    post = estimate_rmr_multiplier(logs, height_cm, age, sex, activity)
    return MetabolicPersonalParams(
        rmr_multiplier=post.mean,
        rmr_multiplier_sd=post.sd,
        n_observations=max(0, len(logs) - 1),
    )
