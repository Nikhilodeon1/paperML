"""Optimization-engine tests — ranked, quantified interventions by counterfactual sim."""

from __future__ import annotations

import pytest

from personalization.optimize import (optimize, optimize_cardiovascular, optimize_glucose)


@pytest.fixture()
def user(tmp_path, monkeypatch):
    import personalization.user_store as us
    monkeypatch.setattr(us, "USER_DIR", tmp_path)
    return us.seed_demo("u_opt")


def test_cardiovascular_levers_reduce_risk(user):
    r = optimize_cardiovascular(user)
    assert r and r["levers"]
    for lv in r["levers"]:
        assert lv["to"] <= lv["from"]                 # every lever lowers (or holds) risk
        assert lv["improvement"] >= 0
    # ranked by improvement, descending
    imps = [lv["improvement"] for lv in r["levers"]]
    assert imps == sorted(imps, reverse=True)


def test_glucose_food_swap_is_top_lever(user):
    r = optimize_glucose(user)
    top = r["levers"][0]
    assert "lentils" in top["change"].lower()          # low-GI swap beats a post-meal walk
    assert top["improvement"] > 20                     # meaningful mg/dL reduction


def test_optimize_dispatch_by_goal(user):
    heart = optimize(user, "lower my heart risk")
    assert any(d["target"].startswith("10-year") for d in heart["domains"])
    sugar = optimize(user, "flatten my blood sugar")
    assert any("glucose" in d["target"] for d in sugar["domains"])
    assert heart["top_change"] is not None


def test_general_goal_surfaces_multiple_domains(user):
    r = optimize(user, "")
    assert len(r["domains"]) >= 2                       # tries all supported domains
    assert r["top_change"]["improvement"] >= 0
