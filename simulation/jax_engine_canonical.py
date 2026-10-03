"""The meal pathway in controllable canonical form, so the identifiable combination is a parameter.

The analysis needs to fit `tau1 = 1/k_e + 1/k_a` directly rather than fit `k_e` and `k_a` and report a
function of them, because the whole claim of H11 is that `tau1` is identifiable where the individual
rates are not. The naive route -- parameterize by `(tau1, p)` and recover the rates by solving the
quadratic -- is a trap:

    k_e, k_a = (sigma1 +/- sqrt(sigma1^2 - 4c)) / 2

That square root vanishes exactly at `k_e = k_a`, so its derivative is infinite there. `k_e = k_a` is
not an obscure corner: it is the tied-rate submodel of H12, the symmetric point where the swap
degeneracy lives, and the place the fits are most likely to land. Differentiating through the root map
would produce NaN gradients precisely where the interesting answer is.

The fix is to never form the roots. The stomach-to-gut cascade

    stomach' = -k_e * stomach + u(t)
    gut'     = k_e * stomach - k_a * gut
    Ra       = k_a * gut

is a second-order linear system whose transfer function from the meal input to the rate of appearance
is `f c / (s^2 + sigma1 s + c)` with `sigma1 = k_e + k_a` and `c = k_e k_a`. Any realization of that
transfer function produces identical `Ra` for identical input. Controllable canonical form is one:

    x1' = x2
    x2' = -c * x1 - sigma1 * x2 + u(t)
    Ra  = f * c * x1

It depends on `(sigma1, c)` -- symmetric functions of the rates -- and is a polynomial in them, so it
is analytic everywhere including at `k_e = k_a`, and `tau1 = sigma1 / c`, `p = 1 / c` are smooth
coordinates on it. Nothing is approximated: the two forms are the same linear system in different
bases.

Scope check before building this: inside the JAX engine the individual `stomach_glucose_mg` and
`gut_glucose_mg` states are read only by `vector_field` itself, so replacing the pair with `(x1, x2)`
changes no other consumer. The numpy reference engine (`simulation/modules/metabolic.py`) keeps the
cascade form and is untouched.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve

from simulation.jax_engine import (
    _GB, _IB, _meal_rate, diurnal_cortisol, sigmoid_switch, smooth_clamp01, softplus_relu,
)

__all__ = ["CanonicalParams", "STATE_VARS", "IDX", "N_STATE", "initial_state", "vector_field",
           "run_meal", "rates_to_canonical", "canonical_to_rates", "tau_p_to_canonical",
           "canonical_to_tau_p", "CARB_FRACTION"]

# The state of the glucose subsystem, with the two-compartment gut cascade replaced by the canonical
# pair (x1, x2). Everything else matches `simulation/jax_engine.STATE_VARS` one for one.
STATE_VARS = [
    "glucose_mg_dl", "insulin_uU_ml", "x_insulin_action", "meal_x1", "meal_x2",
    "glycogen_g", "ketones_mmol_l", "energy_expended_kcal",
]
IDX = {name: i for i, name in enumerate(STATE_VARS)}
N_STATE = len(STATE_VARS)

# Fraction of a meal's carbohydrate mass that reaches the gut as glucose. In the cascade form this is
# folded into the input mass; here it is the output gain `f`, which keeps the input a pure impulse.
CARB_FRACTION = 1.0


class CanonicalParams(eqx.Module):
    """The glucose subsystem parameterized by the symmetric functions of the gut rates.

    `sigma1 = k_e + k_a` and `c = k_e k_a`. Both are polynomials in the rates, so every derivative with
    respect to them is finite everywhere, including on the tied-rate boundary.
    """
    weight_kg: jax.Array
    insulin_sensitivity: jax.Array
    circadian_amp: jax.Array
    rmr_kcal_min: jax.Array
    glucose_effectiveness: jax.Array
    x_decay: jax.Array
    x_gain: jax.Array
    insulin_clearance: jax.Array
    insulin_secretion: jax.Array
    sigma1: jax.Array
    c: jax.Array

    @staticmethod
    def from_jax_params(p, sigma1=None, c=None) -> "CanonicalParams":
        """Copy a `JaxPhysioParams`, converting its gut rates to canonical coordinates."""
        s1, cc = rates_to_canonical(p.gastric_emptying, p.carb_absorption)
        return CanonicalParams(
            weight_kg=p.weight_kg, insulin_sensitivity=p.insulin_sensitivity,
            circadian_amp=p.circadian_amp, rmr_kcal_min=p.rmr_kcal_min,
            glucose_effectiveness=p.glucose_effectiveness, x_decay=p.x_decay, x_gain=p.x_gain,
            insulin_clearance=p.insulin_clearance, insulin_secretion=p.insulin_secretion,
            sigma1=jnp.asarray(s1 if sigma1 is None else sigma1),
            c=jnp.asarray(cc if c is None else c))


# --- coordinate maps ------------------------------------------------------------------------------

def rates_to_canonical(k_e, k_a):
    """`(k_e, k_a) -> (sigma1, c)`. Smooth, and the direction that is always safe."""
    return k_e + k_a, k_e * k_a


def canonical_to_rates(sigma1, c):
    """`(sigma1, c) -> (k_e, k_a)`, for REPORTING only.

    Singular at `k_e = k_a`, where the derivative of the square root is infinite. Never call this
    inside anything that will be differentiated: that is the entire reason this module exists. It is
    here so a fitted result can be quoted in the original units, and it returns NaN rather than a
    complex number when the poles are not real.
    """
    discriminant = sigma1 ** 2 - 4.0 * c
    root = jnp.sqrt(jnp.where(discriminant >= 0.0, discriminant, jnp.nan))
    return 0.5 * (sigma1 + root), 0.5 * (sigma1 - root)


def canonical_to_tau_p(sigma1, c):
    """`(sigma1, c) -> (tau1, p)` with `tau1 = 1/k_e + 1/k_a` and `p = 1/(k_e k_a)`."""
    return sigma1 / c, 1.0 / c


def tau_p_to_canonical(tau1, p):
    """`(tau1, p) -> (sigma1, c)`. Real poles require `p <= tau1^2 / 4`."""
    c = 1.0 / p
    return tau1 / p, c


# --- the engine -----------------------------------------------------------------------------------

def initial_state() -> jnp.ndarray:
    y = jnp.zeros(N_STATE, dtype=jnp.float64)
    y = y.at[IDX["glucose_mg_dl"]].set(90.0)
    y = y.at[IDX["insulin_uU_ml"]].set(10.0)
    y = y.at[IDX["glycogen_g"]].set(450.0)
    y = y.at[IDX["ketones_mmol_l"]].set(0.1)
    return y


def vector_field(t, y, args):
    """dstate/dt with the meal pathway in canonical form. Same physiology, different gut realization."""
    p, meal_time, mass_mg, clock_hour = args

    glucose = y[IDX["glucose_mg_dl"]]
    insulin = y[IDX["insulin_uU_ml"]]
    x_act = y[IDX["x_insulin_action"]]
    x1 = y[IDX["meal_x1"]]
    x2 = y[IDX["meal_x2"]]
    glycogen = y[IDX["glycogen_g"]]
    ketones = y[IDX["ketones_mmol_l"]]

    cortisol = diurnal_cortisol(clock_hour)
    Vg = 1.6 * p.weight_kg

    # Canonical realization of the stomach-to-gut cascade. Ra = c * x1 reproduces k_a * gut exactly.
    dx1 = x2
    dx2 = -p.c * x1 - p.sigma1 * x2 + _meal_rate(t, meal_time, mass_mg)
    Ra = CARB_FRACTION * p.c * x1

    hepatic = 0.10 * softplus_relu(cortisol - 15.0)
    dG_met = (-(p.glucose_effectiveness + x_act) * glucose
              + p.glucose_effectiveness * _GB + Ra / Vg + hepatic)
    dI = (-p.insulin_clearance * (insulin - _IB)
          + p.insulin_secretion * softplus_relu(glucose - _GB))
    si_eff = p.insulin_sensitivity * (1.0 - p.circadian_amp)
    dX = -p.x_decay * x_act + p.x_gain * si_eff * (insulin - _IB)
    dGly_met = 0.15 * sigmoid_switch(insulin, 15.0, 10.0)
    dE = p.rmr_kcal_min

    fasted = smooth_clamp01((_IB + 3.0 - insulin) / 6.0)
    glycogen_low = smooth_clamp01((350.0 - glycogen) / 150.0)
    keto_drive = fasted * glycogen_low
    d_ketones = 0.12 * keto_drive - 0.020 * (ketones - 0.1)
    drain = -0.6 * keto_drive
    gng = 0.08 * softplus_relu(72.0 - glucose) * fasted
    dGly_sub = -0.14 * fasted

    dy = jnp.zeros(N_STATE, dtype=y.dtype)
    dy = dy.at[IDX["glucose_mg_dl"]].set(dG_met + drain + gng)
    dy = dy.at[IDX["insulin_uU_ml"]].set(dI)
    dy = dy.at[IDX["x_insulin_action"]].set(dX)
    dy = dy.at[IDX["meal_x1"]].set(dx1)
    dy = dy.at[IDX["meal_x2"]].set(dx2)
    dy = dy.at[IDX["glycogen_g"]].set(dGly_met + dGly_sub)
    dy = dy.at[IDX["ketones_mmol_l"]].set(d_ketones)
    dy = dy.at[IDX["energy_expended_kcal"]].set(dE)
    return dy


def run_meal(params: CanonicalParams, carbs_g: float, fat_g: float = 0.0, fiber_g: float = 0.0,
             meal_time: float = 30.0, duration_min: float = 210.0, clock_hour: float = 8.0,
             step_min: float = 5.0, return_ra: bool = False):
    """Simulate one meal. Same signature, blunting and solver settings as the cascade engine."""
    blunt = 1.0 / (1.0 + 0.08 * fiber_g + 0.005 * fat_g)
    mass_mg = carbs_g * 1000.0 * 0.90 * blunt
    ts = jnp.arange(0.0, duration_min + 1e-6, step_min)
    solution = diffeqsolve(
        ODETerm(vector_field), Tsit5(), t0=0.0, t1=float(duration_min), dt0=1.0,
        y0=initial_state(), args=(params, meal_time, mass_mg, clock_hour),
        saveat=SaveAt(ts=ts),
        stepsize_controller=PIDController(rtol=1e-4, atol=1e-4), max_steps=10000,
    )
    glucose = solution.ys[:, IDX["glucose_mg_dl"]]
    if return_ra:
        return ts, glucose, CARB_FRACTION * params.c * solution.ys[:, IDX["meal_x1"]]
    return ts, glucose
