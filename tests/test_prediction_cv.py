"""A9: the prediction cross-validation cells.

The two claims that matter are equivalence claims. A fold fitted as a mask over the full set of meals
must be the same fit as one built from the training meals alone, because the second is what the
submitted protocol did. And a grid cell that works from precomputed tables must pick the same point as
one that re-evaluates the loss from scratch. Everything else here is bookkeeping: shapes, the fold
partition, determinism, and that no cell silently drops meals.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.jax_config import configure

configure()

from evaluation import prediction_cv as pcv                            # noqa: E402
from evaluation.cohort_data import load_cohort                          # noqa: E402
from evaluation.cv_utils import make_folds                              # noqa: E402
from personalization.fit_general import fit                             # noqa: E402
from personalization.objectives import (                                # noqa: E402
    ObjectiveSpec, build_objective, observed_values,
)
from personalization.subject_loss import TARGETS, subject_arrays        # noqa: E402

SMALL = {"steps": 25}


@pytest.fixture(scope="module")
def config():
    cfg = pcv.default_config()
    cfg.update(SMALL)
    cfg["limit"] = 2
    return cfg


@pytest.fixture(scope="module")
def ctx(config):
    try:
        subjects = load_cohort(config["cohort"], min_meals=config["min_meals"], limit=2)
    except FileNotFoundError as exc:
        pytest.skip(f"CGMacros not available: {exc}")
    if not subjects:
        pytest.skip("no subjects")
    return pcv.SubjectContext(subjects[0], config)


def test_masked_fit_equals_fit_on_training_meals(ctx, config):
    folds = make_folds(ctx.subject.subject_id, ctx.n, k=5, seed=0)
    train = np.flatnonzero(folds != 0)
    records = [ctx.records[i] for i in train]
    spec = ObjectiveSpec(name="iauc", lam=config["lam"], beta=config["beta"], free=TARGETS)
    reference = fit(build_objective(spec, ctx.base, subject_arrays(records),
                                    observed_values(records, ctx.window)),
                    steps=config["steps"], learning_rate=config["learning_rate"])
    masked = ctx.fitter("grad3").fit(train)
    expected = np.array([reference.theta[name] for name in TARGETS])
    np.testing.assert_allclose(masked["theta_full"], expected, rtol=1e-6, atol=1e-8)
    assert masked["final_loss"] == pytest.approx(reference.final_loss, rel=1e-6)


def test_masked_trace_fit_equals_fit_on_training_meals(ctx, config):
    folds = make_folds(ctx.subject.subject_id, ctx.n, k=5, seed=1)
    train = np.flatnonzero(folds != 2)
    records = [ctx.records[i] for i in train]
    spec = ObjectiveSpec(name="trace", lam=config["lam"], beta=config["beta"],
                         free=("insulin_sensitivity",))
    reference = fit(build_objective(spec, ctx.base, subject_arrays(records),
                                    observed_values(records, ctx.window)),
                    steps=config["steps"], learning_rate=config["learning_rate"])
    masked = ctx.fitter("grad1_trace").fit(train)
    assert masked["theta_full"][0] == pytest.approx(reference.theta["insulin_sensitivity"],
                                                    rel=1e-6)


def test_grid_from_tables_matches_a_fresh_evaluation(ctx, config):
    """The table argmin must equal the argmin of a loss recomputed point by point."""
    folds = make_folds(ctx.subject.subject_id, ctx.n, k=5, seed=0)
    train = np.flatnonzero(folds != 3)
    table = ctx.grid1_table()
    chosen = pcv._grid_choice(ctx, table, train, config["lam"])
    spec = ObjectiveSpec(name="iauc", lam=config["lam"], beta=config["beta"], free=TARGETS)
    records = [ctx.records[i] for i in train]
    objective = build_objective(spec, ctx.base, subject_arrays(records),
                                observed_values(records, ctx.window))
    losses = [float(objective.loss(np.asarray(theta))) for theta in table["thetas"]]
    assert chosen == int(np.argmin(losses))


def test_grid_prediction_is_the_prediction_at_the_chosen_point(ctx, config):
    table = ctx.grid1_table()
    index = 7
    direct = ctx.predict(table["thetas"][index])
    for key in ("iauc", "peak", "trace_rmse"):
        np.testing.assert_allclose(table[key][index], direct[key], rtol=1e-9, atol=1e-9)


def test_unit_partitions_every_meal_exactly_once(config):
    cfg = dict(config)
    cfg["cells"] = ["grad1", "grid1", "grid1_legacy", "personal_mean", "population",
                    "persistence"]
    subject_id = load_cohort(cfg["cohort"], min_meals=cfg["min_meals"], limit=1)[0].subject_id
    result = pcv.run_unit(f"{subject_id}__r0", cfg)
    folds = np.asarray(result["folds"])
    assert set(folds.tolist()) == set(range(cfg["folds"]))
    for name, cell in result["cells"].items():
        assert cell["n_missing"] == 0, name
        assert len(cell["pred_iauc"]) == result["n_meals"], name
        assert np.isfinite(cell["pred_iauc"]).all(), name
        assert sum(f["n_test"] for f in cell["folds"]) == result["n_meals"], name


def test_unit_is_deterministic(config):
    cfg = dict(config)
    cfg["cells"] = ["grad3", "grid1", "personal_mean"]
    subject_id = load_cohort(cfg["cohort"], min_meals=cfg["min_meals"], limit=1)[0].subject_id
    a = pcv.run_unit(f"{subject_id}__r1", cfg)
    pcv._CONTEXTS.clear()
    b = pcv.run_unit(f"{subject_id}__r1", cfg)
    for name in cfg["cells"]:
        np.testing.assert_allclose(a["cells"][name]["pred_iauc"], b["cells"][name]["pred_iauc"],
                                   rtol=0, atol=1e-6)


def test_repeats_differ_and_fold_assignment_is_the_shared_one(config):
    cfg = dict(config)
    cfg["cells"] = ["personal_mean"]
    subject_id = load_cohort(cfg["cohort"], min_meals=cfg["min_meals"], limit=1)[0].subject_id
    a = pcv.run_unit(f"{subject_id}__r0", cfg)
    b = pcv.run_unit(f"{subject_id}__r1", cfg)
    assert a["folds"] != b["folds"]
    np.testing.assert_array_equal(a["folds"], make_folds(subject_id, a["n_meals"], 5, 0))


def test_unit_names_round_trip():
    assert pcv.split_unit("CGMacros-007__r3") == ("CGMacros-007", 3)
    assert pcv.stable_seed("a", 1) == pcv.stable_seed("a", 1)
    assert pcv.stable_seed("a", 1) != pcv.stable_seed("a", 2)
