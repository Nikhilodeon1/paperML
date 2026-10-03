"""Per-user learning + Accuracy Journal tests (on synthetic wearable data).

The point of these is that the machinery RECOVERS a hidden truth and PREDICTS a specific
user better than the population average — validated on synthetic data with a known ground
truth, so it's proven before any real device is connected.
"""

from __future__ import annotations

import numpy as np

from personalization.synthetic import make_user
from personalization.wearable_learning import learn_wearable, learned_params
from personalization.accuracy_journal import accuracy_summary, build_journal


def _user(rhr=58, hrv=52, days=90, seed=3, **kw):
    rng = np.random.default_rng(seed)
    return make_user("t", rng=rng, sex="male", age=35, height_cm=178, weight_kg=80,
                     resting_hr=rhr, hrv_rmssd=hrv, n_wearable_days=days, **kw)


def test_learning_recovers_hidden_baseline():
    w = learn_wearable(_user(rhr=58, hrv=52, days=90))
    assert abs(w["resting_hr"]["mean"] - 58) < 3      # recovers true resting HR
    assert abs(w["hrv_rmssd"]["mean"] - 52) < 5       # recovers true HRV
    assert w["resting_hr"]["n"] == 90


def test_uncertainty_shrinks_with_more_data():
    few = learn_wearable(_user(days=10, seed=1))["resting_hr"]["sd"]
    many = learn_wearable(_user(days=90, seed=1))["resting_hr"]["sd"]
    assert many < few                                 # more days -> tighter posterior


def test_learned_params_feed_the_twin():
    import personalization.user_store as us
    u = _user(rhr=56, days=60)
    us.refresh_derived(u)
    over = learned_params(u)
    assert abs(over["hr_rest"] - 56) < 3
    from simulation import PhysioParams
    p = PhysioParams.from_profile(u["profile"], learned=over)
    assert abs(p.hr_rest - over["hr_rest"]) < 0.01    # engine uses the learned baseline


def test_baseline_journal_beats_population_constant():
    """Beating a population constant is TRUE BY CONSTRUCTION for anyone whose average differs
    from it — so this asserts the arithmetic works, and deliberately claims nothing about the
    simulation engine (which this module never calls). Engine validation lives in
    `evaluation/forward_validation.py`."""
    rng = np.random.default_rng(5)
    u = make_user("t", rng=rng, sex="female", age=32, height_cm=165, weight_kg=60,
                  resting_hr=52, hrv_rmssd=62, n_wearable_days=90)
    s = accuracy_summary(u)
    rhr = s["resting_hr"]
    assert "not the simulation engine" in rhr["predictor"].lower()
    assert rhr["mae"] < rhr["population_mae"]         # trivially true; the point of the test
    assert rhr["vs_population_pct"] > 30
    assert abs(rhr["bias"]) < 3                       # roughly unbiased
    assert 0.75 <= rhr["coverage_90"] <= 1.0          # calibrated-ish 90% band


def test_persistence_baseline_is_reported():
    """The honest bar must always be present, even when it is unflattering."""
    rng = np.random.default_rng(5)
    u = make_user("t", rng=rng, sex="female", age=32, height_cm=165, weight_kg=60,
                  resting_hr=52, hrv_rmssd=62, n_wearable_days=90)
    rhr = accuracy_summary(u)["resting_hr"]
    assert rhr["persistence_mae"] > 0
    assert "vs_persistence_pct" in rhr


def test_journal_entries_wellformed():
    j = build_journal(_user(days=40), "resting_hr")
    assert j and all({"predicted", "actual", "error", "in_90_band"} <= set(e) for e in j)
