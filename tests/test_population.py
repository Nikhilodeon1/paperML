"""Fast unit-level guards for the synthetic mock-user harness.

The exhaustive population validation (200 users + archetypes, 4 property families) lives
in `evaluation/backtest_population.py` and rolls into the accuracy report. These tests
keep a quick CI tripwire: the generator produces well-formed users, the simulator runs
on the whole population and the edge archetypes without crashing, and the headline
properties hold on a small sample.
"""

from __future__ import annotations

import math

from personalization.synthetic import make_population, archetypes, make_user
import numpy as np
from personalization.systems import compute_systems

_EXPECTED = {"metabolic", "cardiovascular", "sleep", "stress", "activity", "hepatic"}


def test_population_is_wellformed_and_simulates():
    users = make_population(n=25, seed=1)
    assert len(users) == 25
    for u in users:
        p = u["profile"]
        assert p["sex"] in ("male", "female")
        assert 18 <= p["age"] <= 90
        bmi = p["weight_kg"] / ((p["height_cm"] / 100) ** 2)
        assert 12 <= bmi <= 75
        tiles = compute_systems(u)
        assert _EXPECTED.issubset({t["key"] for t in tiles})
        for t in tiles:
            assert t["status"] in {"good", "watch", "none"}


def test_population_is_deterministic_under_seed():
    a = make_population(n=10, seed=42)
    b = make_population(n=10, seed=42)
    assert [u["profile"] for u in a] == [u["profile"] for u in b]


def test_archetypes_degrade_gracefully():
    for u in archetypes():
        tiles = compute_systems(u)  # must not raise on any extreme body
        for t in tiles:
            for m in t["metrics"]:
                for tok in str(m.get("value", "")).replace("%", " ").split():
                    try:
                        assert math.isfinite(float(tok))
                    except ValueError:
                        pass


def test_no_data_archetype_shows_none_tiles():
    nod = next(u for u in archetypes() if u["user_id"] == "arch_no_data")
    tiles = {t["key"]: t for t in compute_systems(nod)}
    assert tiles["cardiovascular"]["status"] == "none"
    assert tiles["stress"]["status"] == "none"


def test_weighins_carry_recoverable_metabolic_signal():
    """A slow, low true-RMR user should personalize below the population prior."""
    rng = np.random.default_rng(0)
    u = make_user("t", rng=rng, sex="male", age=35, height_cm=178, weight_kg=84,
                  true_rmr_mult=0.85, n_weeks=12)
    from personalization.user_store import metabolic_params
    mp = metabolic_params(u)
    assert mp is not None and mp.n_observations >= 2
    assert 0 < mp.rmr_multiplier_sd < 0.10  # tighter than the ~0.10 prior
