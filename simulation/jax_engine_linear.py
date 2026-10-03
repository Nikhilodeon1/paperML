"""The glucose subsystem linearized about its fasting operating point.

H2 is a statement about what an AREA can see. In the linearized model the claim is sharp: the
sensitivity of iAUC to either gut rate, relative to its sensitivity to insulin sensitivity, is at most
0.02 over a six-hour window. That is a statement about a linear time-invariant system, and it is much
stronger evidence than the same ratio measured on the nonlinear engine, because in the linear case the
area of the response is an exactly computable functional of the transfer function rather than a number
that came out of a solver.

The subsystem linearized here is the FAST one -- glucose, insulin, insulin action, and the two gut
compartments. Glycogen, ketones and expended energy are held at their fasting values. That is not a
convenience: those three have no fixed point at all in this model (glycogen drifts down at about
0.07 g/min while fasting), so "the basal state" only exists for the fast block, and including the slow
drift in a linearization would mean linearizing about a point the system is not at.

The operating point is found by integrating the fast block with no meal until it stops moving, then
polishing with Newton steps. The residual `||f(y*)||` is reported rather than assumed to be zero, so a
reader can see how good the approximation is.

WHICH BRANCH TO LINEARIZE ON, and why this is not a detail. The engine smooths two switches with a
softplus and a sigmoid. The insulin-secretion softplus, `beta * softplus_relu(G - 90)`, has a
transition half-width of 1/10 mg/dL. At the fasting point its gain is exactly half the above-threshold
value, and a 1 g meal -- 900 mg into a distribution volume of about 95 dL -- produces an excursion of
order 9 mg/dL, which is ninety transition widths. Every real meal therefore spends its entire
excursion on the ACTIVE branch, where the gain is the full `beta`.

Linearizing at the fasting point anyway gives an insulin-secretion gain three times too small, so the
model under-disposes glucose and the linear iAUC comes out about twice the nonlinear one -- measured at
67% too large even for a 1 g meal, which no amount of making the meal smaller fixes, because the
discrepancy is in the gain and not in the amplitude.

`regime="active"` is therefore the default and the only one used for H2. It replaces the secretion
softplus by its above-threshold branch `G - 90` and sets the fasting switch to zero (at post-meal
insulin levels `smooth_clamp01((13 - I) / 6)` is already 0.007 at 15 uU/mL and zero above it). This is
the classical linear minimal model, which is what the H2 proposition is a statement about.
`regime="basal"` is kept so the comparison can be shown rather than asserted.

Forward-mode autodiff is used for the Jacobian, which is allowed here and only here: `A` is the
derivative of a plain vector field with no solver inside it. Everything that differentiates THROUGH a
solve uses reverse mode, because diffrax does not support forward mode through `diffeqsolve`.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve

from simulation.jax_engine import (
    _GB, _IB, _meal_rate, diurnal_cortisol, smooth_clamp01, softplus_relu,
)

__all__ = ["FAST_VARS", "FAST_IDX", "N_FAST", "LinearModel", "fast_vector_field",
           "basal_point", "linearize", "run_meal_linear", "run_meal_fast", "iauc_sensitivity",
           "REGIMES"]

# The fast block. The slow states (glycogen, ketones, energy) are parameters of this field, not
# variables of it.
FAST_VARS = ["glucose_mg_dl", "insulin_uU_ml", "x_insulin_action",
             "stomach_glucose_mg", "gut_glucose_mg"]
FAST_IDX = {name: i for i, name in enumerate(FAST_VARS)}
N_FAST = len(FAST_VARS)

BASAL_GLYCOGEN = 450.0
BASAL_KETONES = 0.1


@dataclass
class LinearModel:
    """`dy/dt = A (y - y*) + B u(t)`, with `u` the meal rate into the stomach."""
    A: np.ndarray
    B: np.ndarray
    basal: np.ndarray
    residual: float          # ||f(y*)||, how far from a true fixed point the operating point is
    eigenvalues: np.ndarray
    clock_hour: float
    regime: str = "active"

    def as_dict(self) -> dict:
        return {"A": self.A.tolist(), "B": self.B.tolist(), "basal": self.basal.tolist(),
                "residual_at_basal": self.residual,
                "eigenvalues_real": np.real(self.eigenvalues).tolist(),
                "eigenvalues_imag": np.imag(self.eigenvalues).tolist(),
                "stable": bool(np.all(np.real(self.eigenvalues) < 0.0)),
                "clock_hour": self.clock_hour, "regime": self.regime}


REGIMES = ("active", "basal")


def fast_vector_field(y, p, clock_hour: float = 8.0, glycogen: float = BASAL_GLYCOGEN,
                      ketones: float = BASAL_KETONES, regime: str = "active"):
    """The fast block of `simulation/jax_engine.vector_field`, as a plain function of the state.

    Term for term identical to the full field restricted to these five states, with the slow states
    frozen. No solver, no meal: the meal enters through `B`.

    `regime="active"` takes the above-threshold branch of the two smoothed switches, which is where a
    real postprandial excursion lives; `regime="basal"` keeps the smoothed forms. See the module
    docstring: the difference is a factor of three in the insulin-secretion gain, not a refinement.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; known: {', '.join(REGIMES)}")
    glucose, insulin, x_act, stomach, gut = (y[0], y[1], y[2], y[3], y[4])
    cortisol = diurnal_cortisol(clock_hour)
    Vg = 1.6 * p.weight_kg

    emptying = p.gastric_emptying * stomach
    Ra = p.carb_absorption * gut
    hepatic = 0.10 * softplus_relu(cortisol - 15.0)

    if regime == "active":
        secretion = glucose - _GB                      # above-threshold branch of the softplus
        fasted = 0.0                                   # the fasting switch is shut after a meal
    else:
        secretion = softplus_relu(glucose - _GB)
        fasted = smooth_clamp01((_IB + 3.0 - insulin) / 6.0)

    dG_met = (-(p.glucose_effectiveness + x_act) * glucose
              + p.glucose_effectiveness * _GB + Ra / Vg + hepatic)
    dI = -p.insulin_clearance * (insulin - _IB) + p.insulin_secretion * secretion
    si_eff = p.insulin_sensitivity * (1.0 - p.circadian_amp)
    dX = -p.x_decay * x_act + p.x_gain * si_eff * (insulin - _IB)

    glycogen_low = smooth_clamp01((350.0 - glycogen) / 150.0)
    keto_drive = fasted * glycogen_low
    drain = -0.6 * keto_drive
    gng = 0.08 * softplus_relu(72.0 - glucose) * fasted

    return jnp.array([dG_met + drain + gng, dI, dX, -emptying, emptying - Ra])


def basal_point(p, clock_hour: float = 8.0, settle_min: float = 2000.0,
                newton_steps: int = 40, regime: str = "active") -> tuple[jnp.ndarray, float]:
    """The fasting operating point of the fast block, and the residual there.

    Integrated first, then Newton-polished. Integration alone leaves a residual of order the solver
    tolerance; Newton takes it to round-off, which matters because the Jacobian is evaluated there and a
    sloppy operating point would show up as a spurious drift in the linear response.
    """
    y0 = jnp.array([90.0, 10.0, 0.0, 0.0, 0.0])
    solution = diffeqsolve(
        ODETerm(lambda t, y, args: fast_vector_field(y, p, clock_hour, regime=regime)), Tsit5(),
        t0=0.0, t1=float(settle_min), dt0=1.0, y0=y0, saveat=SaveAt(t1=True),
        stepsize_controller=PIDController(rtol=1e-8, atol=1e-10), max_steps=200000)
    y = solution.ys[-1]

    def field(state):
        return fast_vector_field(state, p, clock_hour, regime=regime)

    for _ in range(newton_steps):
        residual = field(y)
        jacobian = jax.jacfwd(field)(y)
        try:
            step = jnp.linalg.solve(jacobian, residual)
        except Exception:
            break
        y = y - step
    return y, float(jnp.linalg.norm(field(y)))


def linearize(p, clock_hour: float = 8.0, regime: str = "active") -> LinearModel:
    """`A` and `B` at the fasting operating point.

    `A` is `jacfwd` of the plain vector field -- the one place forward mode is appropriate, because
    there is no solver in the function being differentiated. `B` is the meal direction: a meal enters
    the stomach compartment and nothing else.
    """
    basal, residual = basal_point(p, clock_hour, regime=regime)
    A = np.asarray(jax.jacfwd(lambda y: fast_vector_field(y, p, clock_hour, regime=regime))(basal),
                   dtype=float)
    B = np.zeros(N_FAST)
    B[FAST_IDX["stomach_glucose_mg"]] = 1.0
    return LinearModel(A=A, B=B, basal=np.asarray(basal, dtype=float), residual=residual,
                       eigenvalues=np.linalg.eigvals(A), clock_hour=clock_hour,
                       regime=regime)


def run_meal_linear(p, carbs_g: float, fat_g: float = 0.0, fiber_g: float = 0.0,
                    meal_time: float = 30.0, duration_min: float = 210.0,
                    clock_hour: float = 8.0, step_min: float = 5.0,
                    model: LinearModel | None = None, regime: str = "active"):
    """Response of the linearized system to one meal, returned on the same grid as the full engine.

    The returned glucose is the absolute level, `basal + deviation`, so every observable in
    `simulation/jax_observables` applies unchanged.
    """
    model = model or linearize(p, clock_hour, regime=regime)
    A = jnp.asarray(model.A)
    B = jnp.asarray(model.B)
    blunt = 1.0 / (1.0 + 0.08 * fiber_g + 0.005 * fat_g)
    mass_mg = carbs_g * 1000.0 * 0.90 * blunt

    def field(t, deviation, args):
        return A @ deviation + B * _meal_rate(t, meal_time, mass_mg)

    ts = jnp.arange(0.0, duration_min + 1e-6, step_min)
    solution = diffeqsolve(
        ODETerm(field), Tsit5(), t0=0.0, t1=float(duration_min), dt0=1.0,
        y0=jnp.zeros(N_FAST), args=None, saveat=SaveAt(ts=ts),
        stepsize_controller=PIDController(rtol=1e-6, atol=1e-8), max_steps=200000)
    glucose = model.basal[FAST_IDX["glucose_mg_dl"]] + solution.ys[:, FAST_IDX["glucose_mg_dl"]]
    return ts, glucose


def iauc_sensitivity(p, carbs_g: float, window_min: float = 360.0, clock_hour: float = 8.0,
                     meal_time: float = 30.0, step_min: float = 5.0,
                     regime: str = "active") -> dict:
    """`d iAUC / d log theta` in the LINEARIZED model, for insulin sensitivity and the two gut rates.

    This is the H2 quantity. The ratio of a timing derivative to the insulin-sensitivity derivative is
    predicted to be at most 0.02 over a six-hour window: an area is almost blind to when glucose
    arrives, and in a linear system that can be stated exactly rather than inferred.

    Differentiated with reverse mode through the linear solve, so the Jacobian of `A` with respect to
    the parameters is accounted for -- `A` itself depends on the gut rates, which is the entire point.
    """
    import equinox as eqx

    from simulation.jax_observables import Window, iauc

    window = Window(meal_time_min=meal_time, post_min=window_min, step_min=step_min)
    duration = meal_time + window_min
    names = ("insulin_sensitivity", "gastric_emptying", "carb_absorption")

    def area_at(log_scale, name):
        """iAUC when `name` is multiplied by exp(log_scale); the derivative at 0 is d iAUC / d log."""
        scaled = eqx.tree_at(lambda m: getattr(m, name), p,
                             getattr(p, name) * jnp.exp(log_scale))
        basal, _ = basal_point(scaled, clock_hour, regime=regime)
        A = jax.jacfwd(lambda y: fast_vector_field(y, scaled, clock_hour, regime=regime))(basal)
        B = jnp.zeros(N_FAST).at[FAST_IDX["stomach_glucose_mg"]].set(1.0)
        mass = carbs_g * 1000.0 * 0.90

        def field(t, deviation, args):
            return A @ deviation + B * _meal_rate(t, meal_time, mass)

        ts = jnp.arange(0.0, duration + 1e-6, step_min)
        solution = diffeqsolve(
            ODETerm(field), Tsit5(), t0=0.0, t1=float(duration), dt0=1.0,
            y0=jnp.zeros(N_FAST), args=None, saveat=SaveAt(ts=ts),
            stepsize_controller=PIDController(rtol=1e-6, atol=1e-8), max_steps=200000)
        glucose = basal[FAST_IDX["glucose_mg_dl"]] + solution.ys[:, FAST_IDX["glucose_mg_dl"]]
        return iauc(ts, glucose, window, beta=None)

    derivatives = {name: float(jax.grad(lambda s, n=name: area_at(s, n))(0.0)) for name in names}
    reference = abs(derivatives["insulin_sensitivity"])
    return {
        "window_min": window_min,
        "carbs_g": carbs_g,
        "regime": regime,
        "d_iauc_d_log": derivatives,
        "ratio_to_si": {name: (abs(value) / reference if reference > 0 else float("inf"))
                        for name, value in derivatives.items()},
    }


def run_meal_fast(p, carbs_g: float, fat_g: float = 0.0, fiber_g: float = 0.0,
                  meal_time: float = 30.0, duration_min: float = 210.0,
                  clock_hour: float = 8.0, step_min: float = 5.0, regime: str = "active"):
    """The fast block solved NONLINEARLY, as the matching reference for the linear model.

    This exists to separate two different questions that the comparison against the full engine mixes
    together:

    1. *Is the Jacobian right?* Answered by comparing the linear model against THIS, the same vector
       field without the linearization. The difference must go to zero as the meal shrinks, at the
       second order in the perturbation, and if it does not then `A` is wrong.
    2. *How far does linearity hold for a real meal?* Answered by comparing against the full engine,
       where the smoothed switches make the effective insulin-secretion gain depend on the size of the
       excursion. That error does NOT vanish for a small meal -- it grows, because a small excursion
       sits in the transition region of a softplus whose half-width is 0.1 mg/dL.

    Reporting only the second number would hide a Jacobian bug behind a modelling approximation.
    """
    blunt = 1.0 / (1.0 + 0.08 * fiber_g + 0.005 * fat_g)
    mass_mg = carbs_g * 1000.0 * 0.90 * blunt
    basal, _ = basal_point(p, clock_hour, regime=regime)

    def field(t, y, args):
        base = fast_vector_field(y, p, clock_hour, regime=regime)
        return base.at[FAST_IDX["stomach_glucose_mg"]].add(_meal_rate(t, meal_time, mass_mg))

    ts = jnp.arange(0.0, duration_min + 1e-6, step_min)
    solution = diffeqsolve(
        ODETerm(field), Tsit5(), t0=0.0, t1=float(duration_min), dt0=1.0, y0=basal, args=None,
        saveat=SaveAt(ts=ts), stepsize_controller=PIDController(rtol=1e-10, atol=1e-12),
        max_steps=1000000)
    return ts, solution.ys[:, FAST_IDX["glucose_mg_dl"]]
