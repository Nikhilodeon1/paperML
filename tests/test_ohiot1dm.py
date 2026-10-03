"""OhioT1DM insulin-pathway validation — plumbing + the with/without-bolus direction."""

from __future__ import annotations

import pytest

from evaluation.ohiot1dm import _DIR, _pred_iauc, load_meals, validate


def test_modelling_a_bolus_lowers_predicted_iauc():
    """The mechanism the real-data test relies on: adding the insulin bolus must reduce the
    predicted glucose excursion for the same meal."""
    with_bolus = _pred_iauc(60, bolus_u=6)
    without = _pred_iauc(60, bolus_u=0)
    assert with_bolus < without


def test_loader_pairs_meals_with_nearby_boluses(tmp_path):
    # synthetic OhioT1DM-format CSV: a meal at interval 100 (=500 min) with a bolus, then a rise
    hdr = "5minute_intervals_timestamp,missing_cbg,cbg,finger,basal,hr,gsr,carbInput,bolus"
    lines = [hdr]
    for i in range(88, 145):                              # -60 .. +220 min around the meal
        g = 120 if i < 100 else 200 - (i - 100) * 2
        carb = "45" if i == 100 else ""
        bol = "5" if i == 100 else ""
        lines.append(f"{i}.0,0.0,{g},,,60,0.0002,{carb},{bol}")
    p = tmp_path / "999-ws-training_processed.csv"
    p.write_text("\n".join(lines))
    meals = load_meals(p)
    assert len(meals) == 1
    assert meals[0]["carbs_g"] == 45 and meals[0]["bolus_u"] == 5
    assert len(meals[0]["window"]) >= 20


@pytest.mark.skipif(not _DIR.exists(), reason="OhioT1DM dataset not present")
def test_real_dataset_bolus_helps():
    r = validate()
    assert r["n_meals"] > 100
    # modelling the bolus should fit real T1D glucose better than ignoring it
    assert r["with_bolus_mae"] < r["without_bolus_mae"]
