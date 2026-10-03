"""The 5 rich demo users — data density + real per-user Si recovery."""

from __future__ import annotations

import pytest

from personalization.cohort import _PERSONAS, make_rich_user


@pytest.fixture(scope="module")
def maya(tmp_path_factory, ):
    # Maya: true Si 1.4, far from the population prior -> must fit. Short horizon for speed.
    import personalization.user_store as us
    d = tmp_path_factory.mktemp("cohort")
    _orig = us.USER_DIR
    us.USER_DIR = d
    try:
        u = make_rich_user(_PERSONAS[0], n_days=14, seed=0)
    finally:
        us.USER_DIR = _orig
    return u


def test_user_is_data_rich(maya):
    assert len(maya["wearable"]["daily"]) == 90          # months of daily wearable
    assert len(maya["wearable"]["cgm"]) > 500            # intraday CGM stream
    assert len(maya["logs"]["meals"]) == 14 * 3          # 3 logged meals/day
    assert len(maya["meal_responses"]) == 14 * 3         # each built into a fittable response
    assert maya["medical"]["labs"]["wbc"] and maya["medical"]["family_history"] is not None


def test_distinct_si_is_recovered_and_goes_live(maya):
    """Maya's hidden Si (1.4) is far from the population prior, so the fit must recover it and
    the gate must accept it -> personalization is real, not a stub."""
    isf = maya["derived"]["insulin_sensitivity"]
    assert isf["source"] == "fitted"
    assert isf["value"] == pytest.approx(1.4, abs=0.25)


def test_personas_span_the_clinical_range():
    a1c = {p["uid"]: p["labs"]["hba1c"] for p in _PERSONAS}
    assert a1c["u_maya"] < 5.7 and a1c["u_raj"] >= 5.7 and a1c["u_elena"] >= 6.5   # healthy/pre/diabetic
    assert any(p["labs"]["wbc"] < 4.0 for p in _PERSONAS)     # an immune-flag user (Grace)
