"""Tests for the explainability layer (plain-language driver bullets + indicators)
and the fixed alcohol keyword parsing (vodka/shots/spirits)."""

from __future__ import annotations

from orchestration.router import UserProfile, route

PROFILE = UserProfile(weight_kg=80, height_cm=180, age=35, sex="male",
                      total_chol=230, hdl=42, sbp=138, smoker=True)


def test_vodka_shots_recognized_as_alcohol():
    a = route("if I drink 2 shots of vodka now, how will my sleep be affected?", PROFILE)
    assert a.tool == "alcohol_effect_on_sleep"
    assert a.indicators["standard_drinks"] == 2.0


def test_whiskey_recognized_for_bac():
    a = route("2 whiskeys, when am I sober?", PROFILE)
    assert a.tool == "estimate_bac"
    assert a.indicators["standard_drinks"] == 2.0


def test_bac_explanation_bullets_reference_real_drivers():
    a = route("4 beers tonight, when am I sober?", PROFILE)
    assert len(a.explanation) >= 1
    # Every bullet should name one of the module's actual ranked drivers.
    driver_names = {d for d, _ in a.raw["drivers"]}
    assert any(any(name in bullet for name in driver_names) for bullet in a.explanation)


def test_cvd_explanation_ties_to_indicators():
    a = route("what's my heart disease risk?", PROFILE)
    assert "risk_10yr_pct" in a.indicators
    assert a.indicators["risk_10yr_pct"] == round(a.raw["risk_10yr"] * 100, 1)
    assert a.explanation  # at least one modifiable-factor bullet


def test_alcohol_sleep_explanation_is_plain_language():
    a = route("3 shots of tequila before bed, how's my sleep?", PROFILE)
    assert a.tool == "alcohol_effect_on_sleep"
    assert "REM" in a.explanation[0] or "suppress" in a.explanation[0].lower()
    assert a.indicators["alcohol_g_per_kg_at_bedtime"] > 0


def test_unsupported_question_has_no_fabricated_explanation():
    a = route("will standing on my head improve my eyesight?", PROFILE)
    assert a.explanation == []
    assert a.indicators == {}
    assert a.citations == []
