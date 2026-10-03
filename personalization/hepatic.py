"""Layer 3 personalization for the hepatic module.

The latent parameter we personalize is the user's alcohol elimination rate `beta`
(g/100mL per hour). Prior: N(value, population_sd^2) from Layer 1 (`elimination_rate_beta`).

Observation source: BAC readings during a session's falling limb (e.g. breathalyzer
spot checks, or a known time-to-sober). On the descending limb BAC declines linearly
at rate beta, so any two post-peak readings give a direct observation:

    beta_obs = (bac_earlier - bac_later) / (t_later - t_earlier)

Reading noise on each BAC measurement propagates into the observation variance, which
shrinks for widely spaced readings. A small process variance lets beta drift slowly
(it rises with sustained heavy drinking) so the filter keeps tracking.

Reuses the generic Kalman updater in gaussian_bayes.py — no new machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

from knowledge_base import load_system
from modules.hepatic import PersonalParams
from personalization.gaussian_bayes import GaussianPosterior


@dataclass(frozen=True)
class BacReading:
    """One measured BAC reading during a session."""

    hour: float          # hours since the session start
    bac: float           # measured g/100mL


def estimate_elimination_rate(
    readings: list[BacReading],
    bac_reading_noise: float = 0.005,
    drift_sd_per_year: float = 0.002,
) -> GaussianPosterior:
    """Update the posterior over beta from falling-limb BAC readings.

    Only consecutive pairs where BAC is decreasing are used (the falling limb). Pairs
    on the rising limb (absorption) are skipped — beta is not identifiable there.
    """
    kb = load_system("hepatic")
    beta_param = kb["elimination_rate_beta"]
    post = GaussianPosterior(mean=beta_param.value, var=beta_param.population_sd ** 2)
    drift_var_per_hour = (drift_sd_per_year ** 2) / (365.0 * 24.0)

    ordered = sorted(readings, key=lambda r: r.hour)
    for prev, cur in zip(ordered, ordered[1:]):
        dt = cur.hour - prev.hour
        if dt <= 0 or cur.bac >= prev.bac:
            continue  # rising/flat limb: beta not observable here
        beta_obs = (prev.bac - cur.bac) / dt
        # var of a difference of two noisy readings, divided by dt.
        obs_var = (2.0 * bac_reading_noise ** 2) / (dt ** 2)
        lo, hi = beta_param.plausible_range
        if not (lo <= beta_obs <= hi):
            beta_obs = min(max(beta_obs, lo), hi)
        post = post.step(beta_obs, obs_var, process_var=drift_var_per_hour * dt)

    return post


def personal_params_from_readings(readings: list[BacReading]) -> PersonalParams:
    """Convenience: posterior -> PersonalParams the hepatic module consumes directly."""
    post = estimate_elimination_rate(readings)
    n_usable = sum(
        1 for a, b in zip(sorted(readings, key=lambda r: r.hour),
                          sorted(readings, key=lambda r: r.hour)[1:])
        if b.hour > a.hour and b.bac < a.bac
    )
    return PersonalParams(
        elimination_beta=post.mean,
        elimination_beta_sd=post.sd,
        n_observations=n_usable,
    )
