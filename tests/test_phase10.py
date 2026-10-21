"""Phase 10 (Amendment 6): the recalibration statistic and the heteroskedastic noise models."""
from __future__ import annotations

import numpy as np
import pytest

from evaluation import profile_lik, recalibration
from personalization import noise_model as nm
from personalization.objectives import ObjectiveSpec


def _entry(truth, minimum=1.0, curvature=4.0, n=15, lo=0.2, hi=5.0):
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    profile = 10.0 + curvature * (np.log(grid) - np.log(minimum)) ** 2
    return {"grid": grid.tolist(), "n_in_box": n, "profile": profile.tolist(), "reference": 10.0,
            "truth": truth}


def test_excess_at_truth_is_the_profile_rise():
    entry = _entry(truth=np.exp(0.5))
    assert recalibration.excess_at_truth(entry) == pytest.approx(4.0 * 0.25, rel=0.05)


def test_excess_is_none_off_the_grid():
    assert recalibration.excess_at_truth(_entry(truth=100.0)) is None
    assert recalibration.excess_at_truth({**_entry(truth=1.0), "truth": None}) is None


def test_calibrated_cutoff_is_the_empirical_quantile():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        # truths scattered around the minimum: the excess is curvature * offset^2
        offset = rng.normal(0.0, 0.4)
        rows.append({"subject_id": f"s{i % 50}", "profiles": {"p": _entry(truth=np.exp(offset))}})
    out = recalibration.calibrated_cutoff(rows, "p", resamples=200)
    excess = [recalibration.excess_at_truth(r["profiles"]["p"]) for r in rows]
    assert out["n"] == len(excess)
    assert out["cutoff"] == pytest.approx(np.quantile(excess, 0.95))
    assert out["ci95"][0] <= out["cutoff"] <= out["ci95"][1]
    # 95% of the subjects have the truth inside an interval at that cutoff, by construction
    cover = recalibration.coverage_at(rows, "p", out["cutoff"])
    assert cover["fraction"] >= 0.94


def test_bounded_fraction_falls_as_the_cutoff_rises():
    rows = [{"subject_id": "a", "distance_to_bound": {"p": 0.3},
             "profiles": {"p": _entry(truth=1.0, curvature=c)}} for c in (0.5, 1.0, 2.0, 8.0, 30.0)]
    low = recalibration.bounded_among_interior(rows, "p", 1.92)["fraction"]
    high = recalibration.bounded_among_interior(rows, "p", 12.0)["fraction"]
    assert high <= low
    assert recalibration.bounded_among_interior(rows, "p", 1.92)["n"] == 5


def test_estimates_on_a_bound_are_not_interior():
    rows = [{"subject_id": "a", "distance_to_bound": {"p": 0.0}, "profiles": {"p": _entry(1.0)}}]
    assert recalibration.bounded_among_interior(rows, "p", 1.92)["n"] == 0


def test_proportional_scale_has_a_floor_and_an_exponent():
    area = np.array([0.0, 100.0, 300.0])
    floor = 0.1 * area.mean()
    assert nm.proportional_scale(area, "proportional").tolist() == pytest.approx([floor, 100.0, 300.0])
    assert nm.proportional_scale(area, "root").tolist() == pytest.approx(np.sqrt([floor, 100.0, 300.0]))


def test_heteroskedastic_noise_estimates_a_relative_sigma():
    rng = np.random.default_rng(1)
    area = rng.uniform(50, 400, size=60)
    scale = nm.proportional_scale(area, "proportional")
    residual = 0.25 * scale * rng.normal(size=60)
    model = nm.build({"iauc": residual}, n_parameters=3, structure="proportional", meal_scale=scale)
    assert model.structure == "proportional"
    assert model.sigma["iauc"] == pytest.approx(0.25, rel=0.25)
    assert len(model.meal_scale) == 60
    assert nm.NoiseModel(**model.as_dict()).structure == "proportional"      # survives a round trip


def test_heteroskedastic_models_exist_for_iauc_only():
    with pytest.raises(ValueError):
        ObjectiveSpec(name="trace", noise_structure="root")
    with pytest.raises(ValueError):
        ObjectiveSpec(name="iauc", noise_structure="quadratic")
    with pytest.raises(ValueError):
        nm.build({"trace": np.zeros((3, 4))}, n_parameters=3, structure="root", meal_scale=np.ones(3))


def test_result_sets_for_different_noise_structures_do_not_mix():
    config = {**profile_lik.default_config(), "noise_structure": "root", "profile_only": ["insulin_sensitivity"]}
    payload = {"objective": "iauc", "bounds_scale": 1.0, "cohort": "cgmacros", "parameterization": "rates",
               "replica": None, "carb_scale": None, "polish": None}
    assert not profile_lik._matches(payload, config)
    assert profile_lik._matches({**payload, "noise_structure": "root",
                                 "profile_only": ["insulin_sensitivity"]}, config)
    assert profile_lik._matches(payload, profile_lik.default_config())


# --- Phase 11 (Amendment 7): the noise-scaled replica and the H24 rule --------------------------------------

def test_replica_noise_scale_shrinks_the_same_draws():
    import numpy as np
    from evaluation import subject_source as ss
    ss._replica_cached.cache_clear()
    args = ("cgmacros", 10, "CGMacros-001", 0, True, 0.0, None, "fit")
    clean = np.array([r["glucose"]["values"] for r in ss._replica_cached(*args[:4], False, *args[5:])[0]])
    full = np.array([r["glucose"]["values"] for r in ss._replica_cached(*args, 1.0)[0]])
    half = np.array([r["glucose"]["values"] for r in ss._replica_cached(*args, 0.5)[0]])
    assert np.allclose(half - clean, 0.5 * (full - clean), atol=1e-8)
    assert np.abs(full - clean).max() > 0
    ss._replica_cached.cache_clear()


def test_noise_scale_replica_keys_are_distinct_and_backward_compatible():
    from evaluation import noise_scale
    assert noise_scale.replica(0) == {"seed": 0, "cgm": True, "carb_cv": 0.25}
    assert noise_scale.replica(0, 0.5)["cgm_scale"] == 0.5
