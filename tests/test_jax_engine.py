"""Phase-1 gate tests for the differentiable JAX engine + gradient fit."""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import equinox as eqx    # noqa: E402


# --- structural ---------------------------------------------------------------------------------
def test_jax_engine_imports_cleanly():
    from simulation import jax_engine as J
    assert J.N_STATE == 21
    assert J.IDX["glucose_mg_dl"] == 0


def test_population_defaults_finite():
    from simulation.jax_engine import JaxPhysioParams
    p = JaxPhysioParams.from_population_defaults()
    for f in ("insulin_sensitivity", "gastric_emptying", "carb_absorption"):
        assert np.isfinite(float(getattr(p, f)))


def test_softplus_relu_approximates_relu():
    from simulation.jax_engine import softplus_relu
    for x in (-5.0, -1.0, 1.0, 5.0):
        assert abs(float(softplus_relu(jnp.asarray(x))) - max(0.0, x)) < 0.11


def test_sigmoid_switch():
    from simulation.jax_engine import sigmoid_switch
    assert float(sigmoid_switch(jnp.asarray(15.6), 15.0, 10.0)) > 0.95
    assert float(sigmoid_switch(jnp.asarray(14.4), 15.0, 10.0)) < 0.05


def test_meal_gaussian_integrates_to_mass():
    from simulation.jax_engine import _meal_rate
    ts = jnp.arange(0.0, 60.0, 0.1)
    total = float(jnp.trapezoid(_meal_rate(ts, 30.0, 1000.0), ts))
    assert abs(total - 1000.0) / 1000.0 < 0.02


# --- correctness vs numpy -----------------------------------------------------------------------
def _numpy_glucose(carbs=75.0):
    from simulation import Simulator, PhysioParams, Schedule, Meal
    from simulation.observation import observe_series, spec_for
    s = Schedule(); s.add(Meal(30.0, carbs_g=carbs))
    traj = Simulator(PhysioParams()).run(s, 210, dt=1.0, record_every=5, outputs=["glucose_mg_dl"])
    obs = observe_series(traj, spec_for("cgm", "glucose"), step_min=5.0)
    return np.array(obs["values"])


def test_jax_numpy_glucose_agreement_within_5pct():
    from simulation.jax_engine import JaxPhysioParams, run_meal
    from simulation import PhysioParams
    g_np = _numpy_glucose(75.0)
    _, g_jx = run_meal(JaxPhysioParams.from_numpy(PhysioParams()), 75.0)
    g_jx = np.array(g_jx)
    n = min(len(g_np), len(g_jx))
    rel = np.abs(g_jx[:n] - g_np[:n]) / np.maximum(np.abs(g_np[:n]), 1e-6)
    assert rel.max() < 0.05, f"max rel err {rel.max():.3f}"


def test_gradient_Si_finite_nonzero_and_correct():
    from simulation.jax_engine import JaxPhysioParams, run_meal
    from simulation.jax_observation import iauc

    def sim_iauc(Si):
        p = eqx.tree_at(lambda m: m.insulin_sensitivity,
                        JaxPhysioParams.from_population_defaults(), Si)
        ts, g = run_meal(p, 75.0)
        return iauc(ts, g)

    Si = jnp.asarray(1.0, jnp.float32)
    gA = float(jax.grad(sim_iauc)(Si))
    eps = 1e-2
    gFD = float((sim_iauc(Si + eps) - sim_iauc(Si - eps)) / (2 * eps))
    assert np.isfinite(gA) and abs(gA) > 1e-3
    assert gA < 0                                   # more Si -> lower iAUC
    assert abs(gA - gFD) / max(abs(gFD), 1e-6) < 0.02   # autodiff matches finite diff


# --- inference (slow) ---------------------------------------------------------------------------
@pytest.mark.slow
def test_gradient_fit_recovers_synthetic_Si():
    from personalization.gradient_fit import fit_parameters
    from simulation.jax_engine import JaxPhysioParams, run_meal
    from simulation.jax_observation import iauc

    true_si = 0.5
    p = eqx.tree_at(lambda m: m.insulin_sensitivity,
                    JaxPhysioParams.from_population_defaults(), jnp.asarray(true_si, jnp.float32))
    specs = [(45., 0., 0.), (70., 10., 5.), (60., 5., 3.), (80., 15., 8.), (50., 0., 2.)]
    meals = []
    for c, f, fb in specs:
        ts, g = run_meal(p, c, f, fb)
        meals.append({"carbs_g": c, "fat_g": f, "fiber_g": fb, "observed_iAUC": float(iauc(ts, g))})
    res = fit_parameters(meals, n_steps=120, learning_rate=0.05)
    assert abs(res["Si"] - true_si) < 0.12
    assert res["loss_curve"][-1] < res["loss_curve"][0]
    # normalized gradient: Si (identifiable) should dominate gastric/carb (non-identifiable)
    gn = res["gradient_norms"]
    assert gn["insulin_sensitivity"] > gn["gastric_emptying"]
    assert gn["insulin_sensitivity"] > gn["carb_absorption"]
