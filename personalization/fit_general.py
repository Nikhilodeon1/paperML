"""One fit routine for every cell of every later phase.

`gradient_fit.fit_parameters` could do exactly one thing: Adam, 3 parameters, the iAUC objective,
starting from the population defaults. The revision needs the same model fitted with a chosen
objective, a chosen subset of free parameters, a chosen optimizer, from a chosen starting point, in a
box of chosen width -- because the reviewers asked whether the reported ceiling is a property of the
data or of those choices, and the only way to answer is to vary them and report what happens.

That module is left untouched so there is a fixed thing to reproduce. `test_fit_general.py` asserts
that this routine, pointed at the old objective definition, recovers the old results to 1e-6.

What a fit reports, beyond the parameters:

* **the projected gradient norm**, not the plain one. At a constrained optimum the plain gradient does
  not vanish -- it points into the bound -- so using it as a convergence test marks every
  bound-limited fit unconverged. The projected norm removes the components pushing through an active
  bound, which is the standard criterion and the one the interior-converged subject set is defined by.
* **which parameters ended on a bound.** Reviewer point 6 was that the headline S_I gradient might
  simply reflect an active bound. Every later analysis splits subjects on this flag, so it has to come
  out of the fit rather than be reconstructed afterwards.
* **the loss curve**, so a fit that was still moving when the step budget ran out is visible rather
  than indistinguishable from one that converged.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dataclasses import replace

from personalization import noise_model as nm
from personalization.objectives import Objective, ObjectiveSpec, build_objective, observed_values
from personalization.subject_loss import TARGETS, base_params, subject_arrays

__all__ = ["FitResult", "MLFit", "fit", "fit_subject", "fit_ml", "OPTIMIZERS", "BOUND_TOLERANCE",
           "CONVERGENCE_TOLERANCE", "PILOT_LAM"]

# The regularization the submitted prediction fit used. The pilot pass keeps it so that the noise
# model is estimated at a sane, non-degenerate parameter estimate even for a subject whose likelihood
# is nearly flat; the maximum-likelihood pass that follows is unregularized.
PILOT_LAM = 0.01

OPTIMIZERS = ("adam", "lbfgs")

# Within this fraction of the box width of a bound counts as being AT that bound. One percent of the
# range, which for insulin sensitivity is 0.013 -- far outside solver noise and far inside anything
# physiologically meaningful.
BOUND_TOLERANCE = 0.01

# A fit is called converged when the projected gradient norm falls below this fraction of the norm it
# started at. Relative rather than absolute because the three objectives differ by six orders of
# magnitude in scale (an iAUC loss is in mg/dL*min squared, a standardized one is order one), so no
# single absolute threshold could serve all of them.
CONVERGENCE_TOLERANCE = 1e-3


@dataclass
class FitResult:
    """Everything a later analysis needs from one fit, and nothing it has to recompute."""
    theta: dict                      # fitted value per free parameter
    theta_full: dict                 # every parameter, including the ones held fixed
    final_loss: float
    initial_loss: float
    projected_grad_norm: float
    initial_projected_grad_norm: float
    converged: bool
    at_bound: dict                   # parameter -> "lower" | "upper" | None
    n_at_bound: int
    interior: bool                   # no parameter at a bound AND converged
    loss_curve: list
    still_moving: bool               # the loss was improving when the budget ran out
    steps: int
    optimizer: str
    seed: int | None
    init: str
    spec: dict
    n_residuals: int
    wall_seconds: float
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _optimizer(kind: str, learning_rate: float):
    if kind == "adam":
        return optax.with_extra_args_support(optax.adam(learning_rate))
    if kind == "lbfgs":
        return optax.lbfgs()
    raise ValueError(f"unknown optimizer {kind!r}; use one of {OPTIMIZERS}")


def _project_gradient(grad, theta, lower, upper) -> np.ndarray:
    """Drop the gradient components that push through an active bound."""
    grad = np.asarray(grad, dtype=float).copy()
    theta = np.asarray(theta, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    tolerance = BOUND_TOLERANCE * (upper - lower)
    at_lower = theta <= lower + tolerance
    at_upper = theta >= upper - tolerance
    grad[at_lower & (grad > 0)] = 0.0
    grad[at_upper & (grad < 0)] = 0.0
    return grad


def _bound_flags(theta, lower, upper, names) -> dict:
    theta = np.asarray(theta, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    tolerance = BOUND_TOLERANCE * (upper - lower)
    out = {}
    for i, name in enumerate(names):
        if theta[i] <= lower[i] + tolerance[i]:
            out[name] = "lower"
        elif theta[i] >= upper[i] - tolerance[i]:
            out[name] = "upper"
        else:
            out[name] = None
    return out


def _initial_theta(objective: Objective, init: str, seed: int | None,
                   theta_start=None) -> jnp.ndarray:
    """Where the fit starts.

    `base` is the subject's population defaults, which is what the submitted fit used and what the
    regularization is centred on. `random` draws uniformly from the box, which is what the multistart
    and the inference-gap analyses need; the draw is seeded so a reported start can be reproduced.
    `warm` continues from a supplied estimate, which is how the maximum-likelihood pass starts from the
    pilot rather than walking back across the box.
    """
    if init == "warm":
        if theta_start is None:
            raise ValueError("a warm start needs theta_start")
        return jnp.asarray(theta_start)
    if init == "base":
        return objective.theta0
    if init == "random":
        if seed is None:
            raise ValueError("a random start needs a seed, so that the start can be reproduced")
        rng = np.random.default_rng(seed)
        lower = np.asarray(objective.lower, dtype=float)
        upper = np.asarray(objective.upper, dtype=float)
        return jnp.asarray(rng.uniform(lower, upper))
    raise ValueError(f"unknown init {init!r}; use 'base', 'random' or 'warm'")


def fit(objective: Objective, steps: int = 500, learning_rate: float = 0.02,
        optimizer: str = "adam", seed: int | None = None, init: str = "base",
        theta_start=None) -> FitResult:
    """Minimize `objective` inside its box and report the fit with its diagnostics."""
    started = time.perf_counter()
    loss_fn = objective.loss
    lower, upper = objective.lower, objective.upper
    # Projection rather than a plain clip: the reparameterized fits also have to stay inside the
    # real-pole region, and that constraint is not a box.
    theta_init = objective.project(_initial_theta(objective, init, seed, theta_start))
    opt = _optimizer(optimizer, learning_rate)

    @jax.jit
    def run(theta):
        def body(carry, _):
            current, state = carry
            value, grad = jax.value_and_grad(loss_fn)(current)
            updates, state = opt.update(grad, state, current, value=value, grad=grad,
                                        value_fn=loss_fn)
            # Projection after the update, not a penalty: the box is a hard constraint on where the
            # model is defined, and a soft version would let the solver wander outside it mid-run.
            current = objective.project(optax.apply_updates(current, updates))
            return (current, state), value

        (final, _), curve = jax.lax.scan(body, (theta, opt.init(theta)), None, length=steps)
        return final, curve

    theta_final, curve = run(theta_init)
    theta_final = objective.project(theta_final)
    curve = np.asarray(curve, dtype=float)

    grad_initial = _project_gradient(jax.grad(loss_fn)(theta_init), theta_init, lower, upper)
    grad_final = _project_gradient(jax.grad(loss_fn)(theta_final), theta_final, lower, upper)
    norm_initial = float(np.linalg.norm(grad_initial))
    norm_final = float(np.linalg.norm(grad_final))

    names = objective.spec.theta_names
    at_bound = _bound_flags(theta_final, lower, upper, names)
    n_at_bound = sum(1 for v in at_bound.values() if v is not None)
    converged = bool(norm_final <= CONVERGENCE_TOLERANCE * max(norm_initial, 1e-300))

    # "Still moving" compares the last tenth of the run against the preceding tenth. A fit whose loss
    # is visibly falling at the final step has not found its optimum, whatever its gradient norm says.
    tail = max(1, steps // 10)
    still_moving = bool(curve.size > 2 * tail
                        and (curve[-2 * tail:-tail].mean() - curve[-tail:].mean())
                        > 1e-6 * abs(curve[-tail:].mean()))

    return FitResult(
        theta={name: float(theta_final[i]) for i, name in enumerate(names)},
        theta_full=objective.to_natural(theta_final),
        final_loss=float(curve[-1]), initial_loss=float(curve[0]),
        projected_grad_norm=norm_final, initial_projected_grad_norm=norm_initial,
        converged=converged, at_bound=at_bound, n_at_bound=n_at_bound,
        interior=bool(converged and n_at_bound == 0),
        loss_curve=[float(v) for v in curve], still_moving=still_moving,
        steps=int(steps), optimizer=optimizer, seed=seed, init=init,
        spec=objective.spec.as_dict(), n_residuals=objective.n_residuals,
        wall_seconds=round(time.perf_counter() - started, 3),
        extras={"counts": objective.counts, "noise": objective.noise.as_dict(),
                "parameterization": objective.spec.parameterization})


def fit_subject(subject, spec: ObjectiveSpec | None = None, records=None, steps: int = 500,
                learning_rate: float = 0.02, optimizer: str = "adam", seed: int | None = None,
                init: str = "base", noise: nm.NoiseModel | None = None,
                theta_start=None) -> FitResult:
    """Fit one subject (or one training fold of one) end to end.

    `records` defaults to all of the subject's meals; pass a subset for a cross-validation fold. The
    padded width follows from the number of records, so a fold shares a compiled shape with every
    other fold of the same size.
    """
    spec = spec or ObjectiveSpec()
    records = list(subject.records if records is None else records)
    base = base_params(subject.profile)
    arrays = subject_arrays(records)
    observed = observed_values(records, spec.window)
    objective = build_objective(spec, base, arrays, observed, noise=noise)
    result = fit(objective, steps=steps, learning_rate=learning_rate, optimizer=optimizer,
                 seed=seed, init=init, theta_start=theta_start)
    result.extras["subject_id"] = subject.subject_id
    result.extras["n_meals"] = len(records)
    result.extras["padded_width"] = arrays["width"]
    return result


@dataclass
class MLFit:
    """A pilot fit, the noise model it implies, and the maximum-likelihood fit that follows.

    This is the three-step procedure the pre-registration amendment requires (B1, B2, B3), kept as one
    object so a result file records all of it together. Reporting the maximum-likelihood estimate
    without the noise model it was weighted by would be unreproducible.
    """
    pilot: FitResult
    noise: nm.NoiseModel
    ml: FitResult
    spec_pilot: dict
    spec_ml: dict

    def as_dict(self) -> dict:
        return {"pilot": self.pilot.as_dict(), "noise": self.noise.as_dict(),
                "ml": self.ml.as_dict(), "spec_pilot": self.spec_pilot, "spec_ml": self.spec_ml}


def _build_for(subject, spec: ObjectiveSpec, records, noise):
    base = base_params(subject.profile)
    arrays = subject_arrays(records)
    observed = observed_values(records, spec.window)
    return build_objective(spec, base, arrays, observed, noise=noise), arrays


def fit_ml(subject, spec: ObjectiveSpec | None = None, records=None, steps: int = 500,
           pilot_steps: int | None = None, learning_rate: float = 0.02,
           optimizer: str = "adam", pilot_lam: float = PILOT_LAM) -> MLFit:
    """Pilot fit, then noise model, then the unregularized maximum-likelihood fit.

    This is what every identifiability analysis uses. The three steps and the reasons, from
    `PREREG_AMENDMENT_1.md`:

    1. **Pilot** -- the regularized mean-squared-error fit, unweighted. Its only job is to provide a
       sensible point at which to measure how much each observable scatters.
    2. **Noise model** -- per-observable residual standard deviations, and for the trace an AR(1)
       coefficient, estimated at the pilot estimate and then FROZEN. Re-estimating them during the fit
       would let the optimizer lower its own loss by reclassifying its errors as noise.
    3. **Maximum likelihood** -- `lam = 0`, negative-log-likelihood scale, weighted by that frozen
       noise model, warm-started from the pilot. This is `theta_hat_ML`: the estimate the Fisher
       matrix, the profile likelihood and the interior-versus-bound classification all refer to.

    A regularized estimate would make every parameter look better identified than the likelihood
    supports, which is the first thing a reviewer would check.
    """
    spec = spec or ObjectiveSpec()
    records = list(subject.records if records is None else records)
    pilot_steps = steps if pilot_steps is None else pilot_steps

    spec_pilot = replace(spec, lam=pilot_lam, normalization="mean")
    objective_pilot, arrays = _build_for(subject, spec_pilot, records, None)
    pilot = fit(objective_pilot, steps=pilot_steps, learning_rate=learning_rate,
                optimizer=optimizer, init="base")

    theta_pilot = jnp.asarray([pilot.theta[name] for name in spec.theta_names])
    noise = nm.build(objective_pilot.residuals(theta_pilot), n_parameters=len(spec.free),
                     whiten_trace=spec.whiten_trace)

    spec_ml = replace(spec, lam=0.0, normalization="nll")
    objective_ml, _ = _build_for(subject, spec_ml, records, noise)
    ml = fit(objective_ml, steps=steps, learning_rate=learning_rate, optimizer=optimizer,
             init="warm", theta_start=theta_pilot)

    for result in (pilot, ml):
        result.extras["subject_id"] = subject.subject_id
        result.extras["n_meals"] = len(records)
        result.extras["padded_width"] = arrays["width"]
        result.extras["counts"] = objective_ml.counts
    ml.extras["noise"] = noise.as_dict()
    ml.extras["theta_pilot"] = dict(pilot.theta)

    return MLFit(pilot=pilot, noise=noise, ml=ml,
                 spec_pilot=spec_pilot.as_dict(), spec_ml=spec_ml.as_dict())
