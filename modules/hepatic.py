"""Hepatic module — alcohol pharmacokinetics (BAC over time).

Mechanistic model, NOT a neural network (brief 2, Layer 2). Pharmacokinetics:

  - Rising limb: first-order absorption from gut into the body water compartment
    (rate constant ka). Food slows ka.
  - Distribution volume: Widmark r * body_mass (the body-water compartment).
  - Falling limb: zero-order elimination at rate beta (g/100mL per hour) — this is
    exactly how clinical BAC calculators work, just integrated over time so it
    extends cleanly to longer horizons (brief 2).

All coefficients come from Layer 1 (knowledge_base/hepatic.json) with citations.
Confidence intervals come from Monte-Carlo sampling the population priors, so they
widen the further out the prediction runs (brief 2: uncertainty must compound with
horizon, never a fixed-width band).

Per-user parameter overrides (Layer 3 Bayesian posteriors) can be passed in via
`PersonalParams`; with none, this is the population-prior projection and the
confidence label reflects that.

This module computes BAC over time so other modules (e.g. sleep) can consume it as
an input — the cross-system edge is declared in the dependency graph, not here
(brief 2.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from knowledge_base import load_system

Sex = str  # "male" | "female"


class EvidenceLevel(str, Enum):
    """The three-outcome rule (brief 8)."""

    STRONG = "strong"      # real number, normal confidence, traceable to equation
    WEAK = "weak"          # answer given but low-confidence, reason stated
    NONE = "none"          # no evidence — say so, never fabricate


@dataclass(frozen=True)
class Drink:
    """One drink consumed at a point in time."""

    grams_ethanol: float
    hour: float = 0.0  # hours since t=0 (start of the scenario)

    @classmethod
    def standard(cls, n: float = 1.0, hour: float = 0.0) -> "Drink":
        """`n` US standard drinks (14 g ethanol each) at `hour`."""
        g = load_system("hepatic")["standard_drink_grams"].value
        return cls(grams_ethanol=n * g, hour=hour)


@dataclass(frozen=True)
class PersonalParams:
    """Optional per-user overrides from Layer 3. None => use population prior.

    `n_observations` drives the confidence label: more logged data => the Bayesian
    posterior is tighter => higher confidence (brief 3).
    """

    widmark_r: float | None = None
    elimination_beta: float | None = None
    elimination_beta_sd: float | None = None   # Layer 3 posterior SD (proper hookup)
    absorption_ka: float | None = None
    n_observations: int = 0


@dataclass
class BacResult:
    times_h: np.ndarray            # time grid (hours)
    bac_median: np.ndarray         # g/100mL, median across MC samples
    bac_lower: np.ndarray          # lower CI band
    bac_upper: np.ndarray          # upper CI band
    peak_bac: float                # median peak
    peak_bac_ci: tuple[float, float]
    time_to_sober_h: float         # median hours until BAC < legal-zero threshold
    time_to_sober_ci: tuple[float, float]
    drivers: list[tuple[str, float]]  # (parameter, relative sensitivity), ranked
    evidence: EvidenceLevel
    confidence_label: str
    citations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core deterministic simulation
# ---------------------------------------------------------------------------

def _simulate_bac(
    drinks: list[Drink],
    weight_kg: float,
    r: float,
    beta: float,
    ka: float,
    times_h: np.ndarray,
) -> np.ndarray:
    """Forward-simulate the BAC curve for one fixed parameter set.

    Units: BAC returned in g/100mL. Distribution volume in 100-mL units is
    r * weight_kg * 10 (since r*W liters = r*W*10 * 100 mL).
    """
    vd_100ml = r * weight_kg * 10.0
    dt = float(times_h[1] - times_h[0])

    gut = 0.0       # unabsorbed ethanol in gut (g)
    central = 0.0   # ethanol in body-water compartment (g)
    drink_idx = sorted(drinks, key=lambda d: d.hour)
    di = 0

    bac = np.empty_like(times_h)
    for i, t in enumerate(times_h):
        # Add any drinks consumed by this time into the gut.
        while di < len(drink_idx) and drink_idx[di].hour <= t + 1e-9:
            gut += drink_idx[di].grams_ethanol
            di += 1

        # First-order absorption gut -> central.
        absorbed = ka * gut * dt
        absorbed = min(absorbed, gut)
        gut -= absorbed
        central += absorbed

        # Zero-order elimination (only while ethanol present).
        if central > 0:
            elim = beta * vd_100ml * dt
            central = max(0.0, central - elim)

        bac[i] = central / vd_100ml

    return bac


def _time_to_sober(times_h: np.ndarray, bac: np.ndarray, threshold: float = 0.001) -> float:
    """Hours until BAC falls (and stays) below `threshold` g/100mL."""
    below = np.where(bac < threshold)[0]
    if below.size == 0:
        return float(times_h[-1])  # still not sober within horizon
    # First index after the peak where it stays under threshold.
    peak_i = int(np.argmax(bac))
    after = below[below >= peak_i]
    if after.size == 0:
        return float(times_h[-1])
    return float(times_h[after[0]])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_bac(
    drinks: list[Drink],
    weight_kg: float,
    sex: Sex,
    fed: bool = False,
    personal: PersonalParams | None = None,
    horizon_h: float = 24.0,
    dt_h: float = 0.05,
    n_samples: int = 400,
    ci: float = 0.90,
    seed: int = 0,
) -> BacResult:
    """Project blood alcohol concentration over time with a confidence interval.

    Strong-evidence outcome (brief 8): the Widmark/zero-order model is well
    validated, so this returns a real number at normal confidence.
    """
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")
    if not drinks:
        # No exposure => no projection to make. Honest "none" outcome.
        times = np.arange(0.0, horizon_h + dt_h, dt_h)
        zeros = np.zeros_like(times)
        return BacResult(
            times_h=times, bac_median=zeros, bac_lower=zeros, bac_upper=zeros,
            peak_bac=0.0, peak_bac_ci=(0.0, 0.0),
            time_to_sober_h=0.0, time_to_sober_ci=(0.0, 0.0),
            drivers=[], evidence=EvidenceLevel.NONE,
            confidence_label="no exposure provided",
            citations=[],
        )

    kb = load_system("hepatic")
    r_param = kb["widmark_r_male" if sex == "male" else "widmark_r_female"]
    beta_param = kb["elimination_rate_beta"]
    ka_param = kb["absorption_rate_ka"]

    personal = personal or PersonalParams()
    times = np.arange(0.0, horizon_h + dt_h, dt_h)

    # Prior means (or personal posterior means where available).
    r_mu = personal.widmark_r if personal.widmark_r is not None else r_param.value
    beta_mu = personal.elimination_beta if personal.elimination_beta is not None else beta_param.value
    ka_mu = personal.absorption_ka if personal.absorption_ka is not None else ka_param.value
    if fed:
        ka_mu *= 0.5  # food roughly halves absorption rate

    # Prior spread shrinks as the user logs data (Bayesian posterior narrowing).
    # Sqrt(n+1) is a stand-in until Layer 3 supplies real posteriors.
    shrink = 1.0 / np.sqrt(personal.n_observations + 1.0)
    rng = np.random.default_rng(seed)

    def sample(param, mu, sd=None):
        spread = sd if sd is not None else param.population_sd * shrink
        s = rng.normal(mu, spread, size=n_samples)
        lo, hi = param.plausible_range
        return np.clip(s, lo, hi)

    r_s = sample(r_param, r_mu)
    # Prefer a real Layer 3 posterior SD for the elimination rate when supplied.
    beta_s = sample(beta_param, beta_mu, sd=personal.elimination_beta_sd)
    ka_s = np.clip(
        rng.normal(ka_mu, ka_param.population_sd * shrink, size=n_samples),
        *ka_param.plausible_range,
    )

    curves = np.empty((n_samples, times.size))
    peaks = np.empty(n_samples)
    sobers = np.empty(n_samples)
    for k in range(n_samples):
        c = _simulate_bac(drinks, weight_kg, r_s[k], beta_s[k], ka_s[k], times)
        curves[k] = c
        peaks[k] = c.max()
        sobers[k] = _time_to_sober(times, c)

    lo_q = (1 - ci) / 2 * 100
    hi_q = (1 + ci) / 2 * 100

    drivers = _rank_drivers(drinks, weight_kg, r_param, beta_param, ka_param,
                            r_mu, beta_mu, ka_mu, times)
    confidence_label = _confidence_label(personal.n_observations, times, curves)

    return BacResult(
        times_h=times,
        bac_median=np.median(curves, axis=0),
        bac_lower=np.percentile(curves, lo_q, axis=0),
        bac_upper=np.percentile(curves, hi_q, axis=0),
        peak_bac=float(np.median(peaks)),
        peak_bac_ci=(float(np.percentile(peaks, lo_q)), float(np.percentile(peaks, hi_q))),
        time_to_sober_h=float(np.median(sobers)),
        time_to_sober_ci=(float(np.percentile(sobers, lo_q)), float(np.percentile(sobers, hi_q))),
        drivers=drivers,
        evidence=EvidenceLevel.STRONG,
        confidence_label=confidence_label,
        citations=[r_param.citation, beta_param.citation, ka_param.citation],
    )


def _rank_drivers(drinks, weight_kg, r_param, beta_param, ka_param,
                  r_mu, beta_mu, ka_mu, times) -> list[tuple[str, float]]:
    """One-at-a-time sensitivity: how much does peak BAC move per 1-SD parameter
    change. Gives the orchestration LLM an explanation of what drove the number."""
    base = _simulate_bac(drinks, weight_kg, r_mu, beta_mu, ka_mu, times).max()
    if base <= 0:
        return []
    out = []
    for label, mu, param in (
        ("body composition (Widmark r)", r_mu, r_param),
        ("elimination rate (beta)", beta_mu, beta_param),
        ("absorption rate (ka)", ka_mu, ka_param),
    ):
        bumped_mu = param.clamp(mu + param.population_sd)
        if label.startswith("body composition"):
            bumped = _simulate_bac(drinks, weight_kg, bumped_mu, beta_mu, ka_mu, times).max()
        elif label.startswith("elimination"):
            bumped = _simulate_bac(drinks, weight_kg, r_mu, bumped_mu, ka_mu, times).max()
        else:
            bumped = _simulate_bac(drinks, weight_kg, r_mu, beta_mu, bumped_mu, times).max()
        out.append((label, abs(bumped - base) / base))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _confidence_label(n_obs: int, times: np.ndarray, curves: np.ndarray) -> str:
    """Confidence reflects how much real data we have on THIS user (brief 1)."""
    if n_obs >= 20:
        data = "high (well-personalized)"
    elif n_obs >= 5:
        data = "moderate (some personal data)"
    else:
        data = "population-level only (no personal calibration yet)"
    return data
