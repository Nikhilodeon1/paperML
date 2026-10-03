"""Objectives for the observable ladder: area, area plus timing, and the full trace.

The ladder answers the reviewer who pointed out that fitting an area and then reporting that timing is
unidentifiable is close to circular. Each rung is the SAME model, the same bounds, the same optimizer,
fitted to strictly more of the observed curve:

* **iauc** -- one residual per meal. Reproduces the submitted results.
* **iauc_centroid** -- two residuals per meal, each weighted by the inverse of its own residual
  variance. Without that weighting the comparison would be meaningless: an iAUC residual is a few
  thousand mg/dL*min and a centroid residual a few tens of minutes, so an unweighted sum is an iAUC fit
  with rounding error attached.
* **trace** -- one residual per scored sample, 37 per meal on a 5 min grid, AR(1) whitened.

Four properties, each of which is easy and expensive to get wrong.

**One simulation per meal, not one per observable.** All three rungs read the same simulated
trajectory.

**The observed side uses the HARD operators.** Data need no gradient, so there is no reason to
approximate a positive part or an argmax when measuring what happened; the smooth forms are used only
where a derivative has to flow. The hard observed iAUC reproduces the loader's own value to 1e-11
across all 1640 CGMacros meals, so the bottom rung is the quantity the submitted results were fitted
to.

**Weights are inverse RESIDUAL variance, frozen from a pilot fit** (amendment B2). Weighting by the
spread of the observed values instead would make the H3 condition number an artifact of how variable
the cohort happens to be.

**The trace is whitened** (amendment B3). Thirty-seven nearly-random-walk samples per meal do not carry
thirty-seven independent observations.

`normalization` selects what the loss IS. `"mean"` is a mean squared error, which is what the submitted
prediction fit minimized and what A9 keeps using. `"nll"` is `0.5 * sum (residual / sigma)^2` -- a
negative log-likelihood up to constants, which is what makes the chi-square threshold of 1.92 in the
profile likelihood mean a 95% interval. Identifiability analyses must use `"nll"` with `lam = 0`, and
the class refuses any other combination.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from personalization import coords as co
from personalization import noise_model as nm
from personalization.subject_loss import LOWER, RANGE, TARGETS, UPPER, build_params, theta_of
from simulation.jax_engine import run_meal
from simulation.jax_engine_canonical import CanonicalParams
from simulation.jax_engine_canonical import run_meal as run_meal_canonical
from simulation.jax_observation import iauc as legacy_iauc
from simulation.jax_observables import Window, centroid, iauc, post_indices, trace

__all__ = ["ObjectiveSpec", "Objective", "observed_values", "simulate_meals", "build_objective",
           "LADDER", "NORMALIZATIONS", "PARAMETERIZATIONS"]

LADDER = ("iauc", "iauc_centroid", "trace")
NORMALIZATIONS = ("mean", "nll")

# How the free parameters are coordinatized.
#
#   rates  -- (S_I, k_e, k_a) in natural units, the cascade engine. What was submitted.
#   coords -- (log S_I, log tau1, log p), the canonical engine. Fits the combination H11 says is
#             identifiable instead of inferring it from a rate split the data cannot resolve.
#   tied   -- (log S_I, log tau1) with k_e = k_a = 2 / tau1, the two-parameter model of H12.
#
# `coords` and `tied` go through `simulation/jax_engine_canonical`, which realizes the same gut transfer
# function without ever forming the square root that is singular at k_e = k_a.
PARAMETERIZATIONS = ("rates", "coords", "tied")


@dataclass(frozen=True)
class ObjectiveSpec:
    """Everything that defines a fit except the data and the starting point."""
    name: str = "iauc"
    lam: float = 0.01
    beta: float = 10.0
    free: tuple[str, ...] = TARGETS
    window: Window = field(default_factory=Window)
    bounds_scale: float = 1.0
    normalization: str = "mean"
    whiten_trace: bool = True
    legacy_full_window: bool = False
    parameterization: str = "rates"

    def __post_init__(self):
        if self.parameterization not in PARAMETERIZATIONS:
            raise ValueError(f"unknown parameterization {self.parameterization!r}; "
                             f"known: {', '.join(PARAMETERIZATIONS)}")
        if self.parameterization != "rates" and self.legacy_full_window:
            raise ValueError("the legacy window exists only to reproduce the submitted rate-space fit")
        if self.name not in LADDER:
            raise ValueError(f"unknown objective {self.name!r}; known: {', '.join(LADDER)}")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"unknown normalization {self.normalization!r}; "
                             f"known: {', '.join(NORMALIZATIONS)}")
        if self.parameterization == "rates":
            unknown = [p for p in self.free if p not in TARGETS]
            if unknown:
                raise ValueError(f"not fittable parameters: {unknown}; known: {list(TARGETS)}")
            if not self.free:
                raise ValueError("at least one parameter must be free")
        if self.normalization == "nll" and self.lam != 0.0:
            raise ValueError(
                "a negative log-likelihood objective must be unregularized (lam = 0): the chi-square "
                "profile threshold is a statement about a likelihood, and a penalty toward the prior "
                "would narrow every interval it produces")

    @property
    def free_index(self) -> tuple[int, ...]:
        return tuple(TARGETS.index(name) for name in self.free)

    @property
    def theta_names(self) -> tuple[str, ...]:
        """What the fitted vector means, in order. The reparameterizations have a fixed free set."""
        if self.parameterization == "coords":
            return co.COORD_NAMES
        if self.parameterization == "tied":
            return co.TIED_COORD_NAMES
        return tuple(self.free)

    @property
    def duration_min(self) -> float:
        return self.window.meal_time_min + self.window.post_min

    @property
    def observables(self) -> tuple[str, ...]:
        return {"iauc": ("iauc",), "iauc_centroid": ("iauc", "centroid"),
                "trace": ("trace",)}[self.name]

    def as_dict(self) -> dict:
        return {"name": self.name, "lam": self.lam, "beta": self.beta, "free": list(self.free),
                "bounds_scale": self.bounds_scale, "normalization": self.normalization,
                "parameterization": self.parameterization,
                "theta_names": list(self.theta_names),
                "whiten_trace": self.whiten_trace,
                "legacy_full_window": self.legacy_full_window,
                "window": {"meal_time_min": self.window.meal_time_min,
                           "pre_min": self.window.pre_min, "post_min": self.window.post_min,
                           "step_min": self.window.step_min, "stride": self.window.stride}}


@dataclass
class Objective:
    """A built loss, with everything needed to optimize and interpret it."""
    loss: callable
    weighted_residuals: callable   # theta -> flat vector r with nll == 0.5 * sum(r^2)
    residuals: callable            # theta -> {observable: raw residual array on the real meals}
    predicted: callable            # theta -> {observable: prediction on the real meals}
    spec: ObjectiveSpec
    noise: nm.NoiseModel
    theta0: jnp.ndarray
    lower: jnp.ndarray
    upper: jnp.ndarray
    n_residuals: int
    counts: dict
    expand: callable
    project: callable              # theta -> theta, inside the box and the real-pole region
    to_natural: callable           # theta -> {parameter: value} in the original units, for reporting
    simulate: callable = None      # theta -> glucose, meals x samples, padded meals included
    penalty: callable = None       # theta -> the regularization term, before the factor lam

    def params_at(self, base, theta_free):
        return build_params(base, self.expand(theta_free))


def _scaled_bounds(scale: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The box, widened or narrowed about its own centre, scaled in LOG space.

    Scaled about the centre so that a wider box is not also a shifted one, and in log space because
    every one of these parameters is a positive rate or gain. Scaling linearly would push the insulin
    sensitivity lower bound to -0.35 at the 2x setting of the H10 sweep -- a negative insulin
    sensitivity, which is not a loose prior but a meaningless one, and which lets a fit report a
    sensitivity of 0.009 as though it were interior.
    """
    log_lower, log_upper = jnp.log(LOWER), jnp.log(UPPER)
    centre = 0.5 * (log_lower + log_upper)
    half = 0.5 * (log_upper - log_lower) * scale
    return jnp.exp(centre - half), jnp.exp(centre + half)


def observed_values(records, window: Window) -> dict:
    """What the CGM actually showed, per meal, using the hard (non-smoothed) operators."""
    rows_trace, rows_iauc, rows_centroid = [], [], []
    for record in records:
        glucose = jnp.asarray(record["glucose"]["values"], dtype=jnp.float64)
        window.require(len(glucose))
        rows_iauc.append(float(iauc(None, glucose, window, beta=None)))
        rows_centroid.append(float(centroid(None, glucose, window, beta=None)))
        rows_trace.append(np.asarray(trace(None, glucose, window), dtype=float))
    area = np.array(rows_iauc)
    # 82 of the 1640 CGMacros meals (5.0%) have an observed iAUC of exactly zero: glucose never rose
    # above the pre-meal baseline. For those the centroid is not a measurement but the floor of a
    # zero-over-zero ratio, reading as 0 min -- an impossibly early peak that would drag the timing fit
    # earlier for every meal the subject ate. The centroid residual is scored only where an excursion
    # exists. This excludes no meal from the iAUC rung, where "no rise" is a good observation, and it is
    # not a tunable threshold: it is exactly the set on which the statistic is undefined.
    return {"iauc": area, "centroid": np.array(rows_centroid),
            "centroid_defined": (area > 0.0).astype(float),
            "trace": np.array(rows_trace) if rows_trace else np.zeros((0, 0))}


def _one_meal(params, carbs, fat, fiber, duration_min: float, step_min: float):
    _, glucose = run_meal(params, carbs, fat, fiber, meal_time=30.0,
                          duration_min=duration_min, step_min=step_min)
    return glucose


def simulate_meals(params, carbs, fat, fiber, duration_min: float, step_min: float = 5.0):
    """Glucose trajectories for every meal in one batched solve. Meals x samples."""
    return jax.vmap(_one_meal, in_axes=(None, 0, 0, 0, None, None))(
        params, carbs, fat, fiber, duration_min, step_min)


def _whiten(rows, rho: float):
    """Prais-Winsten transform of each row, with a frozen `rho`.

    Row-wise on purpose: each meal is its own three-hour series, days apart from the next, so the last
    sample of one meal must never be paired with the first sample of another.
    """
    if rho <= 0.0:
        return rows
    head = jnp.sqrt(jnp.maximum(1.0 - rho ** 2, 0.0)) * rows[:, :1]
    tail = rows[:, 1:] - rho * rows[:, :-1]
    return jnp.concatenate([head, tail], axis=1)


def build_objective(spec: ObjectiveSpec, base, arrays: dict, observed: dict,
                    noise: nm.NoiseModel | None = None) -> Objective:
    """Assemble the loss for one subject and one rung of the ladder.

    `noise` carries the frozen per-observable sigmas and, for the trace, the AR(1) coefficient. Pass
    `None` for the pilot pass, which is unweighted by construction because no estimate exists yet.
    """
    window = spec.window
    noise = noise or nm.unit_noise()
    index = jnp.asarray(spec.free_index)
    theta_full0 = theta_of(base)
    lower_full, upper_full = _scaled_bounds(spec.bounds_scale)
    mask = arrays["mask"]
    n_real = int(arrays["n"])
    n_meals = max(n_real, 1)
    width = int(arrays["width"])

    def expand(theta_free):
        return theta_full0.at[index].set(theta_free)

    obs_iauc = jnp.asarray(_pad_to(observed["iauc"], width))
    obs_centroid = jnp.asarray(_pad_to(observed["centroid"], width))
    centroid_mask = mask * jnp.asarray(_pad_to(observed["centroid_defined"], width))
    n_centroid = max(float(np.sum(observed["centroid_defined"])), 1.0)

    sigma_iauc = noise.for_observable("iauc")
    sigma_centroid = noise.for_observable("centroid")
    sigma_trace = noise.for_observable("trace")
    rho = float(noise.rho) if (spec.whiten_trace and noise.whitened) else 0.0

    if spec.name == "trace":
        n_samples = observed["trace"].shape[1] if observed["trace"].size else 0
        obs_trace = jnp.asarray(_pad_rows(observed["trace"], width))
        sample_index = jnp.asarray(np.asarray(post_indices(window)))
    else:
        n_samples = 0
        obs_trace = jnp.zeros((width, 1))
        sample_index = jnp.zeros((1,), dtype=jnp.int32)

    if spec.legacy_full_window:
        # The submitted fit integrated the SIMULATED area over the whole 0-210 min trajectory, baseline
        # and tail included, while the OBSERVED area was integrated over 0-180 min after the meal. The
        # mismatch is real but small -- about 2.1 mg/dL*min, 0.04% of a 60 g meal, almost all of it the
        # smooth positive part accumulating its floor over the 30 min of pre-meal samples. This flag
        # reproduces the old quantity exactly so the refactor can be shown faithful; no reported result
        # uses it.
        simulated_times = jnp.arange(0.0, spec.duration_min + 1e-9, window.step_min)

        def _sim_iauc(g):
            return legacy_iauc(simulated_times, g)
    else:
        def _sim_iauc(g):
            return iauc(None, g, window, spec.beta)

    if spec.parameterization == "rates":
        def _simulate(theta_free):
            params = build_params(base, expand(theta_free))
            return simulate_meals(params, arrays["carbs"], arrays["fat"], arrays["fiber"],
                                  spec.duration_min, window.step_min)
    else:
        canonical_base = CanonicalParams.from_jax_params(base)
        tied = spec.parameterization == "tied"

        def _canonical_at(theta_log):
            """Build canonical parameters from log coordinates. No square roots, so no NaN at k_e = k_a."""
            si = jnp.exp(theta_log[0])
            tau1 = jnp.exp(theta_log[1])
            if tied:
                sigma1, c = co.tied_to_canonical(tau1)
            else:
                sigma1, c = co.tau_p_to_canonical(tau1, jnp.exp(theta_log[2]))
            return eqx.tree_at(
                lambda m: (m.insulin_sensitivity, m.sigma1, m.c), canonical_base, (si, sigma1, c))

        def _simulate(theta_free):
            params = _canonical_at(theta_free)
            return jax.vmap(
                lambda cg, ft, fb: run_meal_canonical(
                    params, cg, ft, fb, meal_time=30.0, duration_min=spec.duration_min,
                    step_min=window.step_min)[1],
                in_axes=(0, 0, 0))(arrays["carbs"], arrays["fat"], arrays["fiber"])

    def _predict(glucose):
        out = {}
        if "iauc" in spec.observables:
            out["iauc"] = jax.vmap(_sim_iauc)(glucose)
        if "centroid" in spec.observables:
            out["centroid"] = jax.vmap(lambda g: centroid(None, g, window, spec.beta))(glucose)
        if "trace" in spec.observables:
            baseline = jnp.mean(glucose[:, window.pre_slice[0]:window.pre_slice[1]], axis=1)
            out["trace"] = glucose[:, sample_index] - baseline[:, None]
        return out

    def penalty(theta_free):
        """Squared scaled distance from the starting point.

        In rate space the scale is the width of each parameter's interval, which is what the submitted
        fit used. In log coordinates the natural scale is the width of the log interval, so the penalty
        means the same thing -- a fraction of the available range -- rather than silently changing
        strength with the parameterization. Identifiability fits set lam = 0 and never touch this.
        """
        if spec.parameterization == "rates":
            return jnp.sum(((theta_free - theta_full0[index]) / RANGE[index]) ** 2)
        span = jnp.maximum(upper - lower, 1e-12)
        return jnp.sum(((theta_free - theta_start) / span) ** 2)

    def weighted_residuals(theta_free):
        """The flat vector `r` for which the likelihood is `0.5 * sum(r^2)`.

        This exists so the Fisher matrix cannot drift away from the loss. `F = J^T J` with
        `J = d r / d theta` IS the Gauss-Newton Fisher information of this likelihood by construction,
        rather than a separate derivation that has to be kept in step by hand. Masked meals contribute
        exact zeros, so they add nothing to either the loss or the information.

        Every weighting the objective applies is already inside `r`: the per-observable sigmas from
        amendment B2, and the Prais-Winsten whitening from B3. A test asserts
        `0.5 * sum(r^2) == loss` on the likelihood scale.
        """
        predicted = _predict(_simulate(theta_free))
        if spec.name == "iauc":
            return (mask * (predicted["iauc"] - obs_iauc) / sigma_iauc).ravel()
        if spec.name == "iauc_centroid":
            area = mask * (predicted["iauc"] - obs_iauc) / sigma_iauc
            time = centroid_mask * (predicted["centroid"] - obs_centroid) / sigma_centroid
            return jnp.concatenate([area.ravel(), time.ravel()])
        raw = mask[:, None] * (predicted["trace"] - obs_trace)
        return (_whiten(raw, rho) / sigma_trace).ravel()

    def loss(theta_free):
        residual = weighted_residuals(theta_free)
        if spec.normalization == "nll":
            fit = 0.5 * jnp.sum(residual ** 2)
        elif spec.name == "iauc":
            fit = jnp.sum(residual ** 2) / n_meals
        elif spec.name == "iauc_centroid":
            # The two residual families keep their own denominators on the mean scale, so neither is
            # diluted by how many meals the other happened to be scored on.
            half = residual.shape[0] // 2
            fit = (jnp.sum(residual[:half] ** 2) / n_meals
                   + jnp.sum(residual[half:] ** 2) / n_centroid)
        else:
            fit = jnp.sum(residual ** 2) / (n_meals * max(n_samples, 1))
        return fit + spec.lam * penalty(theta_free)

    def residuals(theta_free) -> dict:
        """Raw, unweighted, unwhitened residuals on the REAL meals only, for noise estimation."""
        predicted = _predict(_simulate(theta_free))
        out = {}
        if "iauc" in spec.observables:
            out["iauc"] = np.asarray(predicted["iauc"])[:n_real] - observed["iauc"][:n_real]
        if "centroid" in spec.observables:
            keep = observed["centroid_defined"][:n_real] > 0
            out["centroid"] = (np.asarray(predicted["centroid"])[:n_real][keep]
                               - observed["centroid"][:n_real][keep])
        if "trace" in spec.observables:
            out["trace"] = np.asarray(predicted["trace"])[:n_real] - observed["trace"][:n_real]
        return out

    def predicted_fn(theta_free) -> dict:
        predicted = _predict(_simulate(theta_free))
        return {k: np.asarray(v)[:n_real] for k, v in predicted.items()}

    if spec.parameterization == "rates":
        theta_start = theta_full0[index]
        lower, upper = lower_full[index], upper_full[index]

        def project(theta):
            return jnp.clip(theta, lower, upper)

        def to_natural(theta):
            full = np.asarray(expand(jnp.asarray(theta)), dtype=float)
            return {name: float(full[i]) for i, name in enumerate(TARGETS)}
    else:
        tied = spec.parameterization == "tied"
        raw_lower, raw_upper = co.log_bounds(tied=tied)
        if spec.bounds_scale != 1.0:
            centre = 0.5 * (raw_lower + raw_upper)
            half = 0.5 * (raw_upper - raw_lower) * spec.bounds_scale
            raw_lower, raw_upper = centre - half, centre + half
        lower, upper = jnp.asarray(raw_lower), jnp.asarray(raw_upper)

        # Start from the subject's population-default rates, expressed in these coordinates.
        tau0, p0 = co.rates_to_tau_p(float(theta_full0[1]), float(theta_full0[2]))
        start = ([math.log(float(theta_full0[0])), math.log(tau0)] if tied
                 else [math.log(float(theta_full0[0])), math.log(tau0), math.log(p0)])
        theta_start = jnp.clip(jnp.asarray(start), lower, upper)

        def project(theta):
            return co.project_log(theta, lower, upper, tied=tied)

        def to_natural(theta):
            values = np.asarray(theta, dtype=float)
            si, tau1 = float(np.exp(values[0])), float(np.exp(values[1]))
            if tied:
                k = 2.0 / tau1
                return {"insulin_sensitivity": si, "tau1": tau1, "p": tau1 ** 2 / 4.0,
                        "gastric_emptying": k, "carb_absorption": k, "tied": True,
                        "real_poles": True, "inside_original_box": bool(
                            co.tied_tau_range()[0] <= tau1 <= co.tied_tau_range()[1])}
            p_value = float(np.exp(values[2]))
            out = {"insulin_sensitivity": si, "tau1": tau1, "p": p_value, "tied": False,
                   "on_tied_boundary": co.on_tied_boundary(values)}
            out.update(co.rates_in_original_box(tau1, p_value))
            return out

    counts = {"n_meals": n_real, "n_centroid_scored": int(n_centroid),
              "n_centroid_undefined": n_real - int(n_centroid),
              "n_trace_samples": int(n_samples), "padded_width": width}
    n_residuals = {"iauc": n_real, "iauc_centroid": n_real + int(n_centroid),
                   "trace": n_real * max(n_samples, 1)}[spec.name]

    return Objective(
        loss=loss, weighted_residuals=weighted_residuals, residuals=residuals,
        predicted=predicted_fn, spec=spec, noise=noise,
        theta0=theta_start, lower=lower, upper=upper,
        n_residuals=n_residuals, counts=counts, expand=expand, project=project,
        to_natural=to_natural, simulate=_simulate, penalty=penalty)


def _pad_to(values: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros(width)
    out[:values.size] = values
    return out


def _pad_rows(rows: np.ndarray, width: int) -> np.ndarray:
    if rows.size == 0:
        return np.zeros((width, 1))
    out = np.zeros((width, rows.shape[1]))
    out[:rows.shape[0]] = rows
    return out
