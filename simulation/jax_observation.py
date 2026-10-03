"""Differentiable observation operator (JAX counterpart of `simulation/observation.py`).

Maps a simulated glucose trajectory to the summary statistics used for inference, with every
operation differentiable (soft-argmax replaces the hard argmax for peak/peak-time). Used by the
gradient-fit loss.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .jax_engine import softplus_relu


def _baseline(glucose, n_pre: int = 6):
    """Pre-meal fasting baseline: mean of the first `n_pre` samples (30 min at 5-min CGM)."""
    return jnp.mean(glucose[:n_pre])


def iauc(ts, glucose, n_pre: int = 6):
    """Incremental AUC above baseline (trapezoid), differentiable. Same metric as the numpy
    postprandial iAUC the CGMacros loader scores against."""
    base = _baseline(glucose, n_pre)
    above = softplus_relu(glucose - base)
    return jnp.trapezoid(above, ts)


def peak_glucose(glucose, temperature: float = 5.0):
    """Soft-max weighted peak (differentiable approx of max)."""
    w = jax.nn.softmax(glucose / temperature)
    return jnp.sum(w * glucose)


def peak_time(ts, glucose, temperature: float = 5.0):
    """Soft-argmax time of the peak (differentiable). Weights concentrate on the max as
    temperature -> 0; tune so it matches the hard argmax within ~5 min on real curves."""
    w = jax.nn.softmax(glucose / temperature)
    return jnp.sum(w * ts)


def early_slope(ts, glucose, cutoff_min: float = 45.0, n_pre: int = 6):
    """Rise rate over the first ~45 min post-meal (mg/dL per min), differentiable."""
    base = _baseline(glucose, n_pre)
    # smooth window weight for t in (baseline_end, cutoff]
    return (glucose[jnp.argmin(jnp.abs(ts - (ts[n_pre] + cutoff_min)))] - base) / cutoff_min


def smooth_observe(ts, glucose, metric: str):
    """Dispatch to a differentiable summary statistic."""
    if metric == "iAUC":
        return iauc(ts, glucose)
    if metric == "peak_glucose":
        return peak_glucose(glucose)
    if metric == "peak_time":
        return peak_time(ts, glucose)
    if metric == "fasting_baseline":
        return _baseline(glucose)
    if metric == "early_slope":
        return early_slope(ts, glucose)
    raise ValueError(f"unknown metric {metric!r}")
