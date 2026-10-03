"""Uncertainty-quantification tests — honest confidence bands.

Exhaustive validation is in evaluation/backtest_sim_uncertainty.py; these are fast guards
that the ensemble is well-formed, its median tracks the deterministic run, priors control
band width, and personalization sharpens a learned parameter's forecast.
"""

from __future__ import annotations

from simulation import Simulator, PhysioParams, Schedule, Meal, Exercise
from simulation.uncertainty import run_ensemble, ParamUncertainty, Spec


def _p():
    return PhysioParams(weight_kg=80, sex="male", age=35)


def test_median_tracks_deterministic_and_bands_ordered():
    sched = Schedule().add(Meal(0, 70))
    det = Simulator(_p()).run(sched, 180, outputs=["glucose_mg_dl"])
    ens = run_ensemble(_p(), sched, 180, n=80, outputs=["glucose_mg_dl"], seed=1)
    med = ens.bands["glucose_mg_dl"]["median"]
    lo, hi = ens.bands["glucose_mg_dl"]["lo"], ens.bands["glucose_mg_dl"]["hi"]
    assert abs(max(med) - max(det.series["glucose_mg_dl"])) / max(det.series["glucose_mg_dl"]) < 0.08
    assert all(lo[i] - 1e-6 <= med[i] <= hi[i] + 1e-6 for i in range(len(med)))


def test_wider_priors_widen_bands():
    sched = Schedule().add(Meal(0, 70))
    wide = ParamUncertainty({"carb_absorption": Spec(0.5), "insulin_secretion": Spec(0.5)})
    narrow = ParamUncertainty({"carb_absorption": Spec(0.05), "insulin_secretion": Spec(0.05)})
    w = run_ensemble(_p(), sched, 180, n=80, unc=wide, outputs=["glucose_mg_dl"], seed=2)
    n = run_ensemble(_p(), sched, 180, n=80, unc=narrow, outputs=["glucose_mg_dl"], seed=2)
    ws = w.summary(["glucose_mg_dl"])["glucose_mg_dl"]["peak"]
    ns = n.summary(["glucose_mg_dl"])["glucose_mg_dl"]["peak"]
    assert (ws["hi"] - ws["lo"]) > (ns["hi"] - ns["lo"])


def test_personalization_tightens_learned_forecast():
    sched = Schedule().add(Exercise(0, 60, 6))
    pop = run_ensemble(_p(), sched, 120, n=80, outputs=["energy_expended_kcal"], seed=3)
    learned = {"derived": {"metabolic": {"rmr_multiplier": 0.9, "rmr_multiplier_sd": 0.03,
                                         "n_observations": 12}}}
    per = run_ensemble(_p(), sched, 120, n=80, unc=ParamUncertainty.from_user(learned),
                       outputs=["energy_expended_kcal"], seed=3)
    pw = pop.summary(["energy_expended_kcal"])["energy_expended_kcal"]["end"]
    rw = per.summary(["energy_expended_kcal"])["energy_expended_kcal"]["end"]
    assert (rw["hi"] - rw["lo"]) < (pw["hi"] - pw["lo"])


def test_summary_rounds_bac_finely():
    from simulation import Drink
    ens = run_ensemble(_p(), Schedule().add(Drink(0, 3)), 240, n=60, outputs=["bac_g_dl"], seed=4)
    peak = ens.summary(["bac_g_dl"])["bac_g_dl"]["peak"]["median"]
    assert 0.03 <= peak <= 0.09          # not rounded to 0.1


def test_tool_uncertainty_mode():
    from orchestration.tools import dispatch
    r = dispatch("simulate_scenario", {"duration_min": 180, "meals": [{"t_min": 0, "carbs_g": 70}],
                                       "uncertainty": True, "n_samples": 60})
    assert "summary_ci" in r and "bands" in r
    g = r["summary_ci"]["glucose_mg_dl"]["peak"]
    assert g["lo"] <= g["median"] <= g["hi"]
    assert "median" in r["bands"]["glucose_mg_dl"] and "lo" in r["bands"]["glucose_mg_dl"]
