"""Differentiable physiological engine in JAX (Phase 1 — the paper's core contribution).

A JAX/diffrax re-implementation of the numpy engine (`simulation/engine.py`) whose glucose
subsystem is differentiable end-to-end: `jax.grad(loss)(theta)` flows through the ODE solve.
The numpy engine stays the reference; this is built in parallel and validated against it.

SCOPE OF THIS MODULE (staged): the GLUCOSE subsystem — the metabolic (Bergman glucose-insulin +
2-compartment gut) and substrate (fasting/ketosis) modules, which are exactly what per-meal CGM
inference on CGMacros needs. Cortisol is held at its diurnal baseline (no acute stressor in
CGMacros meals, and the hepatic-glucose coupling only activates above cortisol 15), caffeine=0,
no exercise. The alcohol / autonomic / circadian / thermoregulation modules extend this same
`vector_field` and are added in a follow-up; pharmacology is deferred (open `substances` dict does
not map to a fixed JAX array). Every smoothing decision is recorded in `SMOOTHING_NOTES.md`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController

# --- state index map (full 21-var vector; this stage evolves the glucose subset, rest constant) -
STATE_VARS = [
    "glucose_mg_dl", "insulin_uU_ml", "x_insulin_action", "stomach_glucose_mg",
    "gut_glucose_mg", "glycogen_g", "ketones_mmol_l", "sc_insulin_u",
    "water_deficit_ml", "core_temp_c", "bac_g_dl", "gut_alcohol_g", "caffeine_mg",
    "cortisol_ug_dl", "heart_rate_bpm", "hrv_rmssd_ms", "sbp_mmhg", "dbp_mmhg",
    "energy_expended_kcal", "sleep_pressure", "alertness",
]
IDX = {name: i for i, name in enumerate(STATE_VARS)}
N_STATE = len(STATE_VARS)

_GB, _IB = 90.0, 10.0
_INSULIN_KA, _INSULIN_PLASMA = 0.025, 44.0


# --- smoothing primitives (Decisions 1-4; see SMOOTHING_NOTES.md) -------------------------------
def softplus_relu(x, beta: float = 10.0):
    """Smooth max(0, x). beta higher = sharper. Uses jax.nn.softplus for numerical stability."""
    return jax.nn.softplus(beta * x) / beta


def sigmoid_switch(x, thresh: float, sharp: float):
    """Smooth Heaviside: ~0 below `thresh`, ~1 above, transition width ~1/sharp."""
    return jax.nn.sigmoid(sharp * (x - thresh))


def smooth_clamp01(z, sharp: float = 6.0):
    """Smooth version of clamp(z, 0, 1): sigmoid mapping, 0.5 at z=0.5, ->0/1 outside [0,1]."""
    return jax.nn.sigmoid(sharp * (z - 0.5))


def soft_clamp(x, lo, hi):
    """Smooth saturating clamp (tanh). Near-identity in the interior; asymptotes to bounds.
    Applied only to TIGHT-range regulated variables (glucose, insulin), NOT to wide-range
    accumulator compartments (stomach/gut glucose 0..1e6), where tanh over a huge range would
    distort small physiological values — those stay >=0 by their first-order dynamics."""
    mid = (hi + lo) / 2.0
    half = (hi - lo) / 2.0
    return mid + half * jnp.tanh((x - mid) / half)


def diurnal_cortisol(clock_hour):
    return jnp.maximum(4.0, 9.5 + 5.0 * jnp.cos(2.0 * jnp.pi * (clock_hour - 8.0) / 24.0))


# --- parameters as a differentiable pytree ------------------------------------------------------
class JaxPhysioParams(eqx.Module):
    weight_kg: jax.Array
    insulin_sensitivity: jax.Array
    circadian_amp: jax.Array
    rmr_kcal_min: jax.Array
    glucose_effectiveness: jax.Array
    x_decay: jax.Array
    x_gain: jax.Array
    insulin_clearance: jax.Array
    insulin_secretion: jax.Array
    gastric_emptying: jax.Array
    carb_absorption: jax.Array

    @staticmethod
    def from_numpy(p) -> "JaxPhysioParams":
        """Copy the glucose-subsystem parameters out of a numpy `PhysioParams`."""
        f = lambda v: jnp.asarray(float(v), dtype=jnp.float32)
        return JaxPhysioParams(
            weight_kg=f(p.weight_kg), insulin_sensitivity=f(p.insulin_sensitivity),
            circadian_amp=f(p.circadian_amp), rmr_kcal_min=f(p.rmr_kcal_min),
            glucose_effectiveness=f(p.glucose_effectiveness), x_decay=f(p.x_decay),
            x_gain=f(p.x_gain), insulin_clearance=f(p.insulin_clearance),
            insulin_secretion=f(p.insulin_secretion), gastric_emptying=f(p.gastric_emptying),
            carb_absorption=f(p.carb_absorption))

    @staticmethod
    def from_population_defaults() -> "JaxPhysioParams":
        from .params import PhysioParams
        return JaxPhysioParams.from_numpy(PhysioParams())


def initial_state() -> jnp.ndarray:
    y = jnp.zeros(N_STATE, dtype=jnp.float32)
    y = y.at[IDX["glucose_mg_dl"]].set(90.0)
    y = y.at[IDX["insulin_uU_ml"]].set(10.0)
    y = y.at[IDX["glycogen_g"]].set(450.0)
    y = y.at[IDX["ketones_mmol_l"]].set(0.1)
    return y


def _meal_rate(t, meal_time, mass_mg, sigma: float = 2.0):
    """Gaussian rate pulse (Dirac approximation, width sigma min) delivering `mass_mg` of glucose
    to the stomach around `meal_time`. Integrates to mass_mg; differentiable everywhere."""
    amp = mass_mg / (sigma * jnp.sqrt(2.0 * jnp.pi))
    return amp * jnp.exp(-0.5 * ((t - meal_time) / sigma) ** 2)


def vector_field(t, y, args):
    """dstate/dt for the glucose subsystem. Pure, differentiable, no state-value Python branches."""
    p, meal_time, mass_mg, clock_hour = args

    # No state clamping in the RHS: the numpy per-step hard clamp was an explicit-Euler artifact
    # that never activates in normal operation (glucose stays 70-200 for a meal, well inside
    # 15-700). Boundedness at parameter extremes is enforced by projecting theta to plausible
    # ranges during optimization, NOT by distorting the vector field. (soft_clamp is kept as a
    # helper for that projection / for guarding pathological runs.) See SMOOTHING_NOTES.md.
    glucose = y[IDX["glucose_mg_dl"]]
    insulin = y[IDX["insulin_uU_ml"]]
    x_act = y[IDX["x_insulin_action"]]
    stomach = y[IDX["stomach_glucose_mg"]]
    gut = y[IDX["gut_glucose_mg"]]
    glycogen = y[IDX["glycogen_g"]]
    ketones = y[IDX["ketones_mmol_l"]]

    cortisol = diurnal_cortisol(clock_hour)  # constant baseline (no acute stressor)

    Vg = 1.6 * p.weight_kg
    emptying = p.gastric_emptying * stomach
    Ra = p.carb_absorption * gut
    # hepatic glucose output: only cortisol ABOVE 15 contributes (softplus of the old max(0,.))
    hepatic = 0.10 * softplus_relu(cortisol - 15.0)

    dG_met = (-(p.glucose_effectiveness + x_act) * glucose
              + p.glucose_effectiveness * _GB + Ra / Vg + hepatic)
    dI = (-p.insulin_clearance * (insulin - _IB)
          + p.insulin_secretion * softplus_relu(glucose - _GB))
    si_eff = p.insulin_sensitivity * (1.0 - p.circadian_amp)  # resistance(clock)=const here
    dX = -p.x_decay * x_act + p.x_gain * si_eff * (insulin - _IB)
    dStomach = -emptying + _meal_rate(t, meal_time, mass_mg)
    dGut = emptying - p.carb_absorption * gut
    # glycogen restoration switch (insulin>15) smoothed to a sigmoid dose-response
    dGly_met = 0.15 * sigmoid_switch(insulin, 15.0, 10.0)
    dE = p.rmr_kcal_min

    # substrate (fasting) contributions
    fasted = smooth_clamp01((_IB + 3.0 - insulin) / 6.0)
    glycogen_low = smooth_clamp01((350.0 - glycogen) / 150.0)
    keto_drive = fasted * glycogen_low
    d_ketones = 0.12 * keto_drive - 0.020 * (ketones - 0.1)
    drain = -0.6 * keto_drive
    gng = 0.08 * softplus_relu(72.0 - glucose) * fasted
    dGly_sub = -0.14 * fasted

    dy = jnp.zeros(N_STATE, dtype=jnp.float32)
    dy = dy.at[IDX["glucose_mg_dl"]].set(dG_met + drain + gng)
    dy = dy.at[IDX["insulin_uU_ml"]].set(dI)
    dy = dy.at[IDX["x_insulin_action"]].set(dX)
    dy = dy.at[IDX["stomach_glucose_mg"]].set(dStomach)
    dy = dy.at[IDX["gut_glucose_mg"]].set(dGut)
    dy = dy.at[IDX["glycogen_g"]].set(dGly_met + dGly_sub)
    dy = dy.at[IDX["ketones_mmol_l"]].set(d_ketones)
    dy = dy.at[IDX["energy_expended_kcal"]].set(dE)
    return dy


def run_meal(params: JaxPhysioParams, carbs_g: float, fat_g: float = 0.0, fiber_g: float = 0.0,
             meal_time: float = 30.0, duration_min: float = 210.0, clock_hour: float = 8.0,
             step_min: float = 5.0):
    """Simulate a single meal; returns (times, glucose_trajectory) at `step_min` resolution.

    Mirrors numpy `cgmacros.predict_meal_iauc`: meal at t=meal_time, carbs*1000*0.90*blunt mg of
    glucose, fat/fibre blunting factor identical to the numpy engine.
    """
    blunt = 1.0 / (1.0 + 0.08 * fiber_g + 0.005 * fat_g)
    mass_mg = carbs_g * 1000.0 * 0.90 * blunt
    ts = jnp.arange(0.0, duration_min + 1e-6, step_min)
    sol = diffeqsolve(
        ODETerm(vector_field), Tsit5(), t0=0.0, t1=float(duration_min), dt0=1.0,
        y0=initial_state(), args=(params, meal_time, mass_mg, clock_hour),
        saveat=SaveAt(ts=ts),
        stepsize_controller=PIDController(rtol=1e-4, atol=1e-4), max_steps=10000,
    )
    glucose = sol.ys[:, IDX["glucose_mg_dl"]]
    return ts, glucose
