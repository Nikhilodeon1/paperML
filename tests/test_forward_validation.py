"""Forward-validation harness — held-out discipline, parameter recovery, and TEETH.

A validator that cannot fail is worthless, so the load-bearing test here is
`test_harness_detects_a_wrong_model`: feed it a model that is wrong and the error must rise.
Without that, "the engine scores well" means nothing.
"""

from __future__ import annotations

import copy

import pytest

from evaluation.forward_validation import (_mae, _params, fit_insulin_sensitivity,
                                           postprandial_metrics, predict_glucose,
                                           run_forward_validation, run_postprandial_validation,
                                           schedule_from_events, summarize,
                                           summarize_postprandial, synth_days)


@pytest.fixture(scope="module")
def days():
    return synth_days(n_days=9, true_si=0.55, seed=0)


def test_fit_recovers_the_hidden_parameter(days):
    """Si is otherwise a hardcoded population guess. Fitted from glucose response it must
    land near the truth the synthetic person was generated with."""
    si = fit_insulin_sensitivity(days[:4])
    assert si == pytest.approx(0.55, abs=0.2), f"fitted Si {si} missed hidden truth 0.55"


def test_harness_detects_a_wrong_model(days):
    """THE test. A badly wrong insulin sensitivity must produce visibly worse predictions —
    otherwise the harness would rubber-stamp any model."""
    day = days[-1]
    actual = day["observations"]["glucose"]["values"]
    good = _mae(actual, predict_glucose(_params(0.55), day["events"]))   # the truth
    bad = _mae(actual, predict_glucose(_params(1.5), day["events"]))     # very wrong
    assert bad > good, "harness cannot distinguish a wrong model from the right one"
    assert bad - good > 1.0, f"wrong model only cost {bad - good:.2f} mg/dL — too blunt"


def test_no_leakage_from_the_held_out_day(days):
    """The final day is only ever the TEST day. Corrupting its observations must change its
    SCORE but never the fit or the prediction — that is what 'held out' means."""
    clean = run_forward_validation(days, min_history=5)
    dirty_days = copy.deepcopy(days)
    obs = dirty_days[-1]["observations"]["glucose"]
    obs["values"] = [999.0] * len(obs["values"])
    dirty = run_forward_validation(dirty_days, min_history=5)

    assert clean[-1].fitted_si == dirty[-1].fitted_si          # fit never saw the test day
    assert clean[-1].engine_fitted != dirty[-1].engine_fitted  # but the score reacted


def test_engine_beats_persistence_on_engine_generated_days(days):
    """Persistence ('tomorrow = yesterday') is the honest bar the old accuracy journal never
    tried. On engine-generated days the engine should clear it — if it cannot even do that,
    the harness or the fit is broken."""
    s = summarize(run_forward_validation(days, min_history=5))
    assert s["engine_fitted_mae"] < s["persistence_mae"]
    assert s["engine_fitted_mae"] < s["personal_mean_mae"]


def test_fitted_beats_uninformed_population_default(days):
    """The point of personalisation: a fitted Si must predict better than the population
    default it currently ships with."""
    s = summarize(run_forward_validation(days, min_history=5))
    assert s["engine_fitted_mae"] <= s["engine_population_mae"]


def test_events_map_to_engine_stimuli():
    """An event log is the engine's INPUT; without it there are outputs with no inputs."""
    sch = schedule_from_events({
        "meals": [{"t_min": 480, "carbs_g": 60}],
        "exercise": [{"start_min": 600, "end_min": 630, "intensity_mets": 8}],
        "drinks": [{"t_min": 1200, "standard_drinks": 2}],
        "caffeine": [{"t_min": 500, "mg": 95}],
        "sleep": [{"start_min": 0, "end_min": 420}],
    })
    assert len(sch.meals) == 1 and sch.meals[0].carbs_g == 60
    assert len(sch.exercise) == 1 and sch.exercise[0].intensity_mets == 8
    assert len(sch.drinks) == 1 and len(sch.caffeine) == 1 and len(sch.sleep) == 1


def test_synthetic_days_carry_inputs_and_outputs(days):
    d = days[0]
    assert d["events"]["meals"] and d["observations"]["glucose"]["values"]
    assert d["observations"]["glucose"]["step_min"] == 5.0


# --- PRIMARY metric: postprandial response ---------------------------------------------

def _curve(pre: list[float], window: list[float], step: float = 5.0, meal_t: float = 30.0):
    """Values on a 5-min grid from t=0: `pre` before the meal, `window` from the meal on."""
    return pre + window, 0.0, step, meal_t


def test_iauc_is_zero_for_a_flat_curve():
    vals, t0, step, meal = _curve([90] * 6, [90] * 37)
    m = postprandial_metrics(vals, t0, step, meal)
    assert m["iauc"] == pytest.approx(0.0, abs=1e-6)
    assert m["baseline"] == pytest.approx(90.0)


def test_iauc_matches_the_exact_trapezoid_for_a_known_rectangle():
    """+20 mg/dL held across the whole 0-3h window -> 20 * 180 = 3600 mg/dL*min exactly."""
    vals, t0, step, meal = _curve([90] * 6, [110] * 37)
    assert postprandial_metrics(vals, t0, step, meal)["iauc"] == pytest.approx(3600.0)


def test_iauc_ignores_area_below_baseline():
    """Convention: only area ABOVE the pre-meal baseline counts."""
    vals, t0, step, meal = _curve([90] * 6, [70] * 37)      # entirely below baseline
    assert postprandial_metrics(vals, t0, step, meal)["iauc"] == pytest.approx(0.0)


def test_baseline_is_the_premeal_mean_not_the_single_meal_time_sample():
    """A lone pre-meal reading carries CGM noise into EVERY point of the integral. Here the
    t=meal sample is a 120 spike; using it as baseline would wrongly yield iAUC=0."""
    vals, t0, step, meal = _curve([90] * 6, [120] + [110] * 36)
    m = postprandial_metrics(vals, t0, step, meal)
    assert m["baseline"] == pytest.approx(90.0)             # mean of the pre-meal window
    assert m["iauc"] > 3000.0                               # not swallowed by the spike


def test_peak_magnitude_and_timing_are_found():
    window = [90] * 12 + [150] + [90] * 24                  # peak 60 min after the meal
    vals, t0, step, meal = _curve([90] * 6, window)
    m = postprandial_metrics(vals, t0, step, meal)
    assert m["peak"] == pytest.approx(150.0)
    assert m["peak_time_min"] == pytest.approx(60.0)


def test_postprandial_metrics_returns_none_without_enough_samples():
    assert postprandial_metrics([90, 90], 0.0, 5.0, 500.0) is None


def test_engine_beats_persistence_and_personal_mean_on_iauc(days):
    """The PRIMARY claim: simulating the meal beats 'same meal yesterday' and 'that meal's
    average'. Whole-day MAE hides this — it buries the response in fasting baseline."""
    ps = summarize_postprandial(run_postprandial_validation(days, min_history=5))
    assert ps["n_meals"] > 0
    assert ps["fitted_iauc_mae"] < ps["persistence_iauc_mae"]
    assert ps["fitted_iauc_mae"] < ps["personal_mean_iauc_mae"]


def test_personalisation_matters_far_more_on_iauc_than_whole_day(days):
    """An UNpersonalised engine is worse than trivial baselines at meal response — the exact
    thing whole-day MAE concealed, and the reason iAUC is now primary."""
    ps = summarize_postprandial(run_postprandial_validation(days, min_history=5))
    assert ps["fitted_iauc_mae"] < ps["population_iauc_mae"]
    assert ps["population_iauc_mae"] > ps["persistence_iauc_mae"]
