"""Generic identifiability machinery: Fisher information, Schur complements, profile likelihood.

These are the tools the reviewers asked for. The rejected submission argued non-identifiability
from the magnitude of a gradient and from the diagonal of a Fisher matrix. Both are local
heuristics and neither can distinguish "this parameter has little effect" from "this parameter has
an effect that another parameter can cancel". The two things that can are here:

* The FULL Fisher matrix, with its eigenvectors. A near-zero eigenvalue with an eigenvector that
  mixes two parameters is a direction of the parameter space the data cannot see -- a statement
  about the pair, which a diagonal cannot make.
* The PROFILE likelihood. For each value of one parameter, the others are re-optimized; if the
  resulting curve stays within the chi-square threshold across the whole box, no amount of this
  data identifies that parameter, whatever any gradient says.

Everything here takes plain callables so it can be tested against toy problems with known answers
(`tests/test_identifiability_toys.py`) before being pointed at the ODE engine. That test is the
correctness gate for every later phase.

Conventions:

* Parameters are passed in NATURAL units; `log_coords=True` reports derivatives with respect to
  `log theta` instead, by scaling each column of the Jacobian by that parameter. Log coordinates
  are what makes an eigenvalue comparison across parameters with different units meaningful.
* `jax.jacrev` throughout, never forward mode: diffrax does not support forward-mode autodiff
  through a solve, and these tools must work on the engine as well as on toys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

__all__ = [
    "FisherResult", "fisher_matrix", "schur_complement", "timing_block",
    "profile_likelihood", "classify_profile", "numerical_rank", "CHI2_DELTA",
]

# Half of the 0.95 quantile of chi-square with one degree of freedom: the threshold for a
# 95% profile-likelihood interval on the NEGATIVE LOG-LIKELIHOOD scale.
CHI2_DELTA = 1.9207295


@dataclass
class FisherResult:
    """Fisher information at one point, with everything needed to interpret it."""
    jacobian: np.ndarray
    fisher: np.ndarray
    eigenvalues: np.ndarray          # ascending
    eigenvectors: np.ndarray         # columns, matching eigenvalues
    sigma: float
    log_coords: bool
    names: tuple[str, ...] = ()
    extras: dict = field(default_factory=dict)

    @property
    def condition_number(self) -> float:
        lo, hi = float(self.eigenvalues[0]), float(self.eigenvalues[-1])
        if lo <= 0.0:
            return float("inf")
        return hi / lo

    @property
    def eigenvalue_ratio(self) -> float:
        """Smallest over largest eigenvalue. Zero means a direction the data cannot see at all."""
        hi = float(self.eigenvalues[-1])
        return float(self.eigenvalues[0]) / hi if hi > 0.0 else float("nan")

    def weak_direction(self) -> np.ndarray:
        """Eigenvector of the smallest eigenvalue, sign-fixed so its largest entry is positive."""
        v = np.asarray(self.eigenvectors[:, 0], dtype=float)
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        return v

    def as_dict(self) -> dict:
        return {
            "jacobian": self.jacobian.tolist(),
            "fisher": self.fisher.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "eigenvectors": self.eigenvectors.tolist(),
            "condition_number": self.condition_number,
            "eigenvalue_ratio": self.eigenvalue_ratio,
            "weak_direction": self.weak_direction().tolist(),
            "sigma": self.sigma,
            "log_coords": self.log_coords,
            "names": list(self.names),
            **self.extras,
        }


def numerical_rank(singular_values, rtol: float = 1e-6) -> int:
    """Rank at a RELATIVE tolerance: how many singular values exceed `rtol * max`.

    Relative rather than absolute because these matrices carry physical units and their overall
    scale varies by orders of magnitude between subjects.
    """
    s = np.asarray(singular_values, dtype=float)
    if s.size == 0:
        return 0
    return int(np.sum(s > rtol * float(np.max(s))))


def fisher_matrix(obs_fn: Callable, theta, sigma: float = 1.0, log_coords: bool = True,
                  names: Sequence[str] = ()) -> FisherResult:
    """Fisher information for independent Gaussian observations of `obs_fn` at `theta`.

    `obs_fn(theta)` returns the predicted observation vector (any length, including 1). With
    independent noise of standard deviation `sigma`, the information is `J^T J / sigma^2`.

    `sigma` may be a scalar or one value per observation; a per-observation `sigma` is used when
    observables with different units are stacked (iAUC together with a centroid), where a single
    scalar would weight them arbitrarily.
    """
    theta_vec = jnp.asarray(theta, dtype=jnp.float64)
    jac = np.asarray(jax.jacrev(lambda t: jnp.atleast_1d(obs_fn(t)))(theta_vec), dtype=float)
    if jac.ndim == 1:
        jac = jac[None, :]

    if log_coords:
        jac = jac * np.asarray(theta_vec, dtype=float)[None, :]

    sig = np.asarray(sigma, dtype=float)
    if sig.ndim == 0:
        weighted = jac / float(sig)
    else:
        if sig.shape[0] != jac.shape[0]:
            raise ValueError(f"sigma has {sig.shape[0]} entries for {jac.shape[0]} observations")
        weighted = jac / sig[:, None]

    fisher = weighted.T @ weighted
    fisher = 0.5 * (fisher + fisher.T)          # symmetrize away round-off asymmetry
    evals, evecs = np.linalg.eigh(fisher)
    return FisherResult(jacobian=jac, fisher=fisher, eigenvalues=evals, eigenvectors=evecs,
                        sigma=float(sig) if sig.ndim == 0 else float(np.mean(sig)),
                        log_coords=bool(log_coords), names=tuple(names))


def schur_complement(fisher: np.ndarray, keep: Sequence[int],
                     eliminate: Sequence[int]) -> np.ndarray:
    """Information about `keep` after PROFILING OUT `eliminate`.

    `F_kk - F_ke F_ee^{-1} F_ek`. This, not the `keep` sub-block of the full matrix, is the
    information that survives re-optimizing the eliminated parameters -- the quantity a profile
    likelihood over `keep` would see. Taking the raw sub-block instead is the mistake that makes a
    compensating pair of parameters look informative.

    `F_ee` is inverted by least squares (`pinv`), so an eliminated block that is itself singular
    degrades gracefully rather than raising.
    """
    F = np.asarray(fisher, dtype=float)
    keep, eliminate = list(keep), list(eliminate)
    if not eliminate:
        return F[np.ix_(keep, keep)]
    Fkk = F[np.ix_(keep, keep)]
    Fke = F[np.ix_(keep, eliminate)]
    Fee = F[np.ix_(eliminate, eliminate)]
    out = Fkk - Fke @ np.linalg.pinv(Fee, rcond=1e-12) @ Fke.T
    return 0.5 * (out + out.T)


def timing_block(fisher: np.ndarray, names: Sequence[str],
                 timing: Sequence[str] = ("gastric_emptying", "carb_absorption"),
                 nuisance: Sequence[str] = ("insulin_sensitivity",)) -> dict:
    """The Schur-complement timing block and its summary statistics.

    This is the H3 metric: the condition number of the information about the two timing parameters
    once insulin sensitivity has been profiled out.
    """
    names = list(names)
    keep = [names.index(n) for n in timing if n in names]
    drop = [names.index(n) for n in nuisance if n in names]
    block = schur_complement(fisher, keep, drop)
    evals, evecs = np.linalg.eigh(block)
    lo, hi = float(evals[0]), float(evals[-1])
    weak = np.asarray(evecs[:, 0], dtype=float)
    if weak[np.argmax(np.abs(weak))] < 0:
        weak = -weak
    return {
        "block": block.tolist(),
        "names": [names[i] for i in keep],
        "eigenvalues": evals.tolist(),
        "condition_number": (hi / lo) if lo > 0 else float("inf"),
        "eigenvalue_ratio": (lo / hi) if hi > 0 else float("nan"),
        "weak_direction": weak.tolist(),
        "profiled_out": [names[i] for i in drop],
    }


# --- profile likelihood -------------------------------------------------------------------------

def _inner_optimizer(kind: str, learning_rate: float):
    """The inner optimizer, wrapped so both kinds take the same `update` call.

    L-BFGS needs the loss value and the loss function itself (its line search re-evaluates them);
    Adam does not. `with_extra_args_support` lets one call site pass both and have Adam ignore
    what it does not use, instead of branching on the optimizer at every step.
    """
    if kind == "adam":
        return optax.with_extra_args_support(optax.adam(learning_rate))
    if kind == "lbfgs":
        return optax.lbfgs()
    raise ValueError(f"unknown optimizer {kind!r}; use 'adam' or 'lbfgs'")


def profile_likelihood(loss_fn: Callable, theta_hat, index: int, grid,
                       lower=None, upper=None, steps: int = 80,
                       learning_rate: float = 0.02, optimizer: str = "adam",
                       vmap: bool = True) -> dict:
    """Profile the loss over parameter `index`, re-optimizing the others at each grid point.

    `loss_fn(theta)` is a scalar to be MINIMIZED, on the negative-log-likelihood scale (so a
    threshold of `CHI2_DELTA` above the minimum is a 95% interval). Regularization must be off
    (`lam = 0`) for the threshold to mean that: a penalty toward a prior narrows the profile and
    would turn a flat direction into an apparently bounded one.

    Each grid point starts from `theta_hat` (a warm start, which is both faster and keeps the inner
    optimum on the same branch) and takes `steps` projected updates with the profiled coordinate
    frozen. Returns the profile, the re-optimized parameters at each point, and a convergence flag
    per point.

    `vmap=True` evaluates every grid point in one batched call. The grid is short (15 points) and
    the solve is the expensive part, so this is the difference between seconds and minutes.
    """
    theta_hat = jnp.asarray(theta_hat, dtype=jnp.float64)
    grid = jnp.asarray(grid, dtype=jnp.float64)
    n = theta_hat.shape[0]
    lower = jnp.full(n, -jnp.inf) if lower is None else jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.full(n, jnp.inf) if upper is None else jnp.asarray(upper, dtype=jnp.float64)
    free = jnp.asarray([0.0 if i == index else 1.0 for i in range(n)], dtype=jnp.float64)
    opt = _inner_optimizer(optimizer, learning_rate)

    def one_point(fixed_value):
        start = theta_hat.at[index].set(fixed_value)

        def body(carry, _):
            theta, state = carry
            value, grad = jax.value_and_grad(loss_fn)(theta)
            grad = grad * free                      # the profiled coordinate never moves
            updates, state = opt.update(grad, state, theta, value=value,
                                       grad=grad, value_fn=loss_fn)
            theta = optax.apply_updates(theta, updates)
            theta = jnp.clip(theta, lower, upper)
            theta = theta.at[index].set(fixed_value)
            return (theta, state), value

        (theta_final, _), curve = jax.lax.scan(body, (start, opt.init(start)), None, length=steps)
        final_loss = loss_fn(theta_final)
        final_grad = jax.grad(loss_fn)(theta_final) * free
        # How much the inner loss was still falling over the last quarter of the run. A profile point
        # that is still descending overstates the profile there, which would make a flat parameter
        # look bounded, so the drift is reported for every point.
        tail = max(1, steps // 4)
        drift = curve[-tail] - final_loss
        return final_loss, theta_final, jnp.linalg.norm(final_grad), drift

    runner = jax.vmap(one_point) if vmap else (
        lambda g: jax.tree.map(lambda *xs: jnp.stack(xs), *[one_point(v) for v in g]))
    losses, thetas, grad_norms, drifts = runner(grid)

    losses = np.asarray(losses, dtype=float)
    base = float(np.min(losses))
    return {
        "index": int(index),
        "grid": np.asarray(grid, dtype=float).tolist(),
        "profile": losses.tolist(),
        "delta": (losses - base).tolist(),
        "theta": np.asarray(thetas, dtype=float).tolist(),
        "grad_norm": np.asarray(grad_norms, dtype=float).tolist(),
        "drift": np.asarray(drifts, dtype=float).tolist(),
        "min_loss": base,
        "argmin": float(np.asarray(grid, dtype=float)[int(np.argmin(losses))]),
        "optimizer": optimizer,
        "steps": int(steps),
    }


def classify_profile(grid, profile, delta: float = CHI2_DELTA,
                     reference: float | None = None) -> dict:
    """Classify one profile as identifiable, one-sided, or flat, and extract the interval.

    * **identifiable** -- the profile rises above `reference + delta` on BOTH sides of the minimum
      inside the box, so the 95% interval is bounded and lies inside the box.
    * **one-sided** -- it rises on one side only; the interval is bounded on that side and open on
      the other, which is still information but cannot be quoted as an interval.
    * **flat** -- it never rises above the threshold anywhere in the box. The data do not identify
      this parameter at all over this range.

    The interval endpoints are found by linear interpolation between adjacent grid points, so a
    15-point grid gives endpoints finer than its own spacing. `reference` defaults to the minimum
    over the grid; pass the full-fit loss instead when the profile minimum should be compared to
    the unconstrained optimum.
    """
    g = np.asarray(grid, dtype=float)
    p = np.asarray(profile, dtype=float)
    order = np.argsort(g)
    g, p = g[order], p[order]
    base = float(np.min(p)) if reference is None else float(reference)
    threshold = base + float(delta)
    i_min = int(np.argmin(p))

    def crossing(lo_idx: int, hi_idx: int, direction: int):
        """Walk outward from the minimum; interpolate where the profile crosses the threshold."""
        idx = i_min
        while 0 <= idx + direction < p.size:
            nxt = idx + direction
            if p[nxt] >= threshold:
                p0, p1 = p[idx], p[nxt]
                if p1 == p0:
                    return float(g[nxt])
                frac = (threshold - p0) / (p1 - p0)
                return float(g[idx] + frac * (g[nxt] - g[idx]))
            idx = nxt
        return None

    low = crossing(0, i_min, -1)
    high = crossing(i_min, p.size - 1, +1)
    if low is not None and high is not None:
        verdict = "identifiable"
    elif low is None and high is None:
        verdict = "flat"
    else:
        verdict = "one-sided"
    return {
        "verdict": verdict,
        "ci_low": low,
        "ci_high": high,
        "width": (high - low) if (low is not None and high is not None) else None,
        "bounded": verdict == "identifiable",
        "threshold": threshold,
        "reference": base,
        "argmin": float(g[i_min]),
        "max_rise": float(np.max(p) - base),
    }
