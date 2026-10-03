"""Tests for hepatic Layer 3 (personal elimination rate)."""

from __future__ import annotations

import numpy as np
import pytest

from knowledge_base import load_system
from modules.hepatic import Drink, compute_bac
from personalization.hepatic import (
    BacReading,
    estimate_elimination_rate,
    personal_params_from_readings,
)


def _falling_limb(true_beta, peak=0.10, t_peak=1.0, n=8, spacing=0.5, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    readings = []
    for i in range(n):
        t = t_peak + i * spacing
        bac = max(0.0, peak - true_beta * (t - t_peak)) + rng.normal(0, noise)
        readings.append(BacReading(hour=t, bac=max(0.0, bac)))
    return readings


def test_recovers_known_beta():
    true_beta = 0.020
    readings = _falling_limb(true_beta, noise=0.001, seed=1)
    post = estimate_elimination_rate(readings, bac_reading_noise=0.001)
    assert post.mean == pytest.approx(true_beta, abs=0.003)


def test_more_readings_tighten_posterior():
    few = estimate_elimination_rate(_falling_limb(0.018, n=3, seed=2), bac_reading_noise=0.002)
    many = estimate_elimination_rate(_falling_limb(0.018, n=12, seed=2), bac_reading_noise=0.002)
    assert many.sd < few.sd


def test_rising_limb_readings_ignored():
    """Readings where BAC is increasing must not be used (beta unobservable)."""
    rising = [BacReading(0.0, 0.00), BacReading(0.3, 0.03), BacReading(0.6, 0.06)]
    post = estimate_elimination_rate(rising)
    prior = load_system("hepatic")["elimination_rate_beta"]
    # No usable pairs => posterior unchanged from prior.
    assert post.mean == pytest.approx(prior.value)
    assert post.sd == pytest.approx(prior.population_sd)


def test_personal_params_feed_module_and_narrow_ci():
    readings = _falling_limb(0.020, n=10, noise=0.001, seed=3)
    pp = personal_params_from_readings(readings)
    assert pp.elimination_beta_sd is not None and pp.elimination_beta_sd > 0

    prior = compute_bac([Drink.standard(3)], weight_kg=80, sex="male")
    personalized = compute_bac([Drink.standard(3)], weight_kg=80, sex="male", personal=pp)
    wp = float(np.max(prior.bac_upper - prior.bac_lower))
    wpp = float(np.max(personalized.bac_upper - personalized.bac_lower))
    assert wpp < wp
