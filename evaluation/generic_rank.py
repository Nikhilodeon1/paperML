"""A3: generic rank of the observable Jacobian.

Structural identifiability asks whether the parameters can be recovered from a noise-free observable at
generic parameter values. A necessary condition for LOCAL identifiability is that the Jacobian of the
observable with respect to the parameters has full column rank at generic points; a rank of two for a
three-parameter model proves a one-dimensional family of parameters is indistinguishable, at every
generic point, regardless of how much data is collected.

For each observable set, 200 parameter vectors are drawn uniformly in the box, the observables of ten
real meals are stacked, and the Jacobian with respect to the log parameters is decomposed. Each
observable's rows are divided by its own typical magnitude first: an area in thousands of mg/dL*min and
a peak time in tens of minutes would otherwise differ by orders of magnitude for reasons unrelated to
identifiability, and a relative-tolerance rank would read that as rank deficiency.

Observable sets: iAUC; iAUC + centroid; iAUC + peak time; the full trace.

Run:  python -m evaluation.runner evaluation.generic_rank
"""
from __future__ import annotations

import hashlib

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402

from evaluation.cohort_data import load_cohort                          # noqa: E402
from evaluation.identifiability_tools import numerical_rank            # noqa: E402
from personalization.objectives import _scaled_bounds, simulate_meals  # noqa: E402
from personalization.subject_loss import TARGETS, base_params, build_params  # noqa: E402
from simulation.jax_observables import (                               # noqa: E402
    Window, centroid, iauc, peak_time, post_indices, trace,
)

ANALYSIS_ID = "A3_generic_rank"
OBSERVABLE_SETS = {
    "iauc": ("iauc",),
    "iauc_centroid": ("iauc", "centroid"),
    "iauc_peak": ("iauc", "peak"),
    "trace": ("trace",),
}


def default_config() -> dict:
    return {
        "cohort": "cgmacros",
        "n_theta": 200,
        "n_meals": 10,
        "seed": 0,
        "rank_rtol": 1e-6,
        "bounds_scale": 1.0,
        "beta": 10.0,
        "min_meals": 10,
    }


def units(config: dict) -> list[str]:
    return list(OBSERVABLE_SETS)


def _rng(config: dict, tag: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{tag}|{config['seed']}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def sample_meals(config: dict) -> list[dict]:
    """Ten real meals, drawn without replacement from the pooled cohort, deterministically."""
    pool = []
    for subject in load_cohort(config["cohort"], min_meals=config["min_meals"]):
        for record in subject.records:
            pool.append((subject.subject_id, record))
    order = _rng(config, "meals").permutation(len(pool))[:config["n_meals"]]
    return [pool[i] for i in sorted(order)]


def run_unit(unit: str, config: dict) -> dict:
    names = OBSERVABLE_SETS[unit]
    meals = sample_meals(config)
    subject = load_cohort(config["cohort"], min_meals=config["min_meals"])[0]
    base = base_params(subject.profile)
    window, beta = Window(), config["beta"]
    carbs = jnp.asarray([m[1]["carbs_g"] for m in meals])
    fat = jnp.asarray([m[1]["fat_g"] for m in meals])
    fiber = jnp.asarray([m[1]["fiber_g"] for m in meals])
    duration = window.meal_time_min + window.post_min
    sample_index = post_indices(window)

    def observables(log_theta):
        params = build_params(base, jnp.exp(log_theta))
        glucose = simulate_meals(params, carbs, fat, fiber, duration, window.step_min)
        parts = []
        if "iauc" in names:
            parts.append(jax.vmap(lambda g: iauc(None, g, window, beta))(glucose)[:, None])
        if "centroid" in names:
            parts.append(jax.vmap(lambda g: centroid(None, g, window, beta))(glucose)[:, None])
        if "peak" in names:
            parts.append(jax.vmap(lambda g: peak_time(None, g, window, beta))(glucose)[:, None])
        if "trace" in names:
            baseline = jnp.mean(glucose[:, window.pre_slice[0]:window.pre_slice[1]], axis=1)
            parts.append(glucose[:, sample_index] - baseline[:, None])
        return parts

    def flat(log_theta):
        return jnp.concatenate([p.ravel() for p in observables(log_theta)])

    jacobian_fn = jax.jit(jax.jacrev(flat))
    value_fn = jax.jit(lambda t: [jnp.asarray(p) for p in observables(t)])

    lower, upper = _scaled_bounds(config["bounds_scale"])
    lower, upper = np.log(np.asarray(lower)), np.log(np.asarray(upper))
    rng = _rng(config, "theta")
    draws = rng.uniform(lower, upper, size=(config["n_theta"], len(TARGETS)))

    # Column counts per observable block, to rescale each block by its own typical magnitude.
    probe = [np.asarray(p) for p in value_fn(jnp.asarray(draws[0]))]
    block_sizes = [p.size for p in probe]

    jacobians, scales = [], []
    for log_theta in draws:
        jac = np.asarray(jacobian_fn(jnp.asarray(log_theta)), dtype=float)
        vals = [np.asarray(p) for p in value_fn(jnp.asarray(log_theta))]
        jacobians.append(jac)
        scales.append([float(np.median(np.abs(v)) + 1e-12) for v in vals])
    scale = np.median(np.asarray(scales), axis=0)         # one fixed scale per observable block

    spectra, ranks = [], []
    for jac in jacobians:
        rows, start = [], 0
        for size, s in zip(block_sizes, scale):
            rows.append(jac[start:start + size] / s)
            start += size
        scaled = np.concatenate(rows, axis=0)
        singular = np.linalg.svd(scaled, compute_uv=False)
        spectra.append(singular.tolist())
        ranks.append(numerical_rank(singular, rtol=config["rank_rtol"]))
    spectra = np.asarray(spectra)
    ranks = np.asarray(ranks)

    def q(values, p):
        return np.percentile(values, p, axis=0).tolist()

    return {
        "observables": list(names), "n_theta": int(config["n_theta"]),
        "n_meals": len(meals), "meal_sources": [m[0] for m in meals],
        "bounds_scale": config["bounds_scale"], "rank_rtol": config["rank_rtol"],
        "block_scale": dict(zip(names, scale.tolist())),
        "singular_values": {"median": q(spectra, 50), "q1": q(spectra, 25), "q3": q(spectra, 75),
                            "min": spectra.min(axis=0).tolist(),
                            "max": spectra.max(axis=0).tolist()},
        "rank_counts": {str(int(k)): int(v) for k, v in
                        zip(*np.unique(ranks, return_counts=True))},
        "rank_median": float(np.median(ranks)),
        "smallest_over_largest_median": float(np.median(spectra[:, -1] / spectra[:, 0])),
        "finite": bool(np.isfinite(spectra).all()),
        "macros": {},
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_unit("iauc", default_config()), indent=2)[:3000])
