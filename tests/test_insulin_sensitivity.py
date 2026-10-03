"""Per-user insulin sensitivity learner + the confidence gate + live-path wiring."""

from __future__ import annotations

import pytest

from orchestration import tools
from orchestration.context import current_user
from personalization.insulin_sensitivity import (learn_insulin_sensitivity,
                                                 make_meal_responses)

_PROF = {"weight_kg": 82, "height_cm": 178, "age": 45, "sex": "male"}


def _user(true_si, n, seed=1):
    return {"profile": _PROF,
            "meal_responses": make_meal_responses(true_si, n=n, profile=_PROF, seed=seed)}


def test_fitted_si_recovers_hidden_truth_with_enough_meals():
    res = learn_insulin_sensitivity(_user(0.55, n=8))
    assert res["source"] == "fitted"
    assert res["value"] == pytest.approx(0.55, abs=0.2)


def test_resistant_user_fits_low_si():
    res = learn_insulin_sensitivity(_user(0.4, n=10, seed=3))
    assert res["source"] == "fitted" and res["value"] < 0.7


def test_too_few_meals_falls_back_to_prior():
    res = learn_insulin_sensitivity(_user(0.55, n=2))
    assert res["source"] == "prior" and res["value"] == pytest.approx(1.0)   # population default


def test_no_data_is_prior_not_crash():
    assert learn_insulin_sensitivity({"profile": _PROF})["source"] == "prior"


def test_missing_profile_returns_none():
    assert learn_insulin_sensitivity({"meal_responses": []}) is None


def _peak(user):
    tok = current_user.set(user)
    try:
        r = tools.dispatch("simulate_scenario",
                           {"duration_min": 180, "meals": [{"t_min": 0, "carbs_g": 75}]})
    finally:
        current_user.reset(tok)
    return r["summary"]["glucose_mg_dl"]["max"]


def test_live_path_uses_a_fitted_si():
    """A trustworthy fit must actually change the simulated glucose response."""
    resistant = _peak({"profile": _PROF,
                       "derived": {"insulin_sensitivity": {"source": "fitted", "value": 0.4, "n": 8}}})
    population = _peak({"profile": _PROF, "derived": {}})
    assert resistant > population + 5          # resistant -> higher peak


def test_live_path_ignores_a_weak_fit():
    """THE gate: a low-confidence fit must NOT reach the engine — fall back to population."""
    weak = _peak({"profile": _PROF,
                  "derived": {"insulin_sensitivity": {"source": "weak", "value": 0.4, "n": 3}}})
    population = _peak({"profile": _PROF, "derived": {}})
    assert weak == pytest.approx(population, abs=1.0)
