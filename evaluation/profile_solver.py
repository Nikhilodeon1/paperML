"""A converged inner solver for profile likelihoods (Amendment 12).

The profiles of the paper re-optimize the two free coordinates at each grid point with a fixed number of Adam steps
(`identifiability_tools.profile_likelihood`). A profile value computed that way is an upper bound on the true profile,
because the inner minimum is only approached. Phase 14 showed that more Adam steps change no profile class, which
does not show that Adam reaches the minimum. This module solves the same inner problem with L-BFGS-B until it stops,
so that "profile value" means the minimum found, and reports whether each point converged.

`fixed_coordinate_minimum` is one inner solve (also used by the profile-at-the-truth analysis, A19); `profile_lbfgsb`
walks the grid outward from the estimate and starts each point from its nearer neighbour, which keeps the inner
optimum on one branch and is much cheaper than a cold start at every point.
"""
from __future__ import annotations

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import jax                      # noqa: E402
import jax.numpy as jnp         # noqa: E402
from scipy.optimize import minimize  # noqa: E402


def _wrap_project(project, index: int):
    if project is None:
        return None
    import inspect
    _accepts_fixed = "fixed_index" in inspect.signature(project).parameters
    return (lambda t: project(t, fixed_index=index)) if _accepts_fixed else project


def reduced_value_and_grad(loss_fn, project, index: int, n: int):
    """`f(free, value)`: the loss with coordinate `index` held at `value` and the other coordinates `free`.

    The projection (the real-pole region of the reparameterized fits) is applied respecting `fixed_index`
    and the fixed coordinate is then restored, exactly as the Adam inner loop does, so both solvers
    minimize the same function.
    """
    free_index = jnp.asarray([i for i in range(n) if i != index])
    _proj = _wrap_project(project, index)

    def f(free, value):
        full = jnp.zeros(n, dtype=free.dtype).at[free_index].set(free).at[index].set(value)
        if _proj is not None:
            full = _proj(full).at[index].set(value)
        return loss_fn(full)

    return jax.jit(jax.value_and_grad(f)), free_index


def fixed_coordinate_minimum(value_and_grad, free_index, start, index: int, value: float, lower, upper,
                             maxiter: int = 200, project=None) -> dict:
    """L-BFGS-B over the free coordinates from `start` (a full vector), with coordinate `index` fixed at `value`."""
    start = np.asarray(start, dtype=float)
    free = np.asarray(free_index)
    lo, hi = np.asarray(lower, dtype=float)[free], np.asarray(upper, dtype=float)[free]
    x0 = np.clip(start[free], lo, hi)

    def fun(x):
        v, g = value_and_grad(jnp.asarray(x), jnp.asarray(value))
        return float(v), np.asarray(g, dtype=float)

    result = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)), options={"maxiter": maxiter})
    theta = start.copy()
    theta[free] = result.x
    theta[index] = value
    _proj = _wrap_project(project, index)
    if _proj is not None:
        theta = np.asarray(_proj(jnp.asarray(theta)).at[index].set(value), dtype=float)
    return {"loss": float(result.fun), "theta": theta, "converged": bool(result.success), "iterations": int(result.nit)}


def profile_lbfgsb(loss_fn, phi_hat, index: int, grid, lower, upper, project=None, maxiter: int = 200) -> dict:
    """Profile over `grid` (log coordinates) with a converged inner solve; the same keys as `profile_likelihood`."""
    phi_hat = np.asarray(phi_hat, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = phi_hat.shape[0]
    value_and_grad, free_index = reduced_value_and_grad(loss_fn, project, index, n)
    losses = np.zeros(len(grid))
    thetas = np.zeros((len(grid), n))
    converged = np.zeros(len(grid), dtype=bool)
    order = np.argsort(np.abs(grid - phi_hat[index]), kind="stable")
    previous = {-1: phi_hat, +1: phi_hat}                 # the nearest solved point on each side of the estimate
    for k in order:
        side = -1 if grid[k] < phi_hat[index] else +1
        out = fixed_coordinate_minimum(value_and_grad, free_index, previous[side], index, float(grid[k]), lower, upper,
                                       maxiter, project=project)
        losses[k], thetas[k], converged[k] = out["loss"], out["theta"], out["converged"]
        previous[side] = out["theta"]
    return {"index": int(index), "grid": grid.tolist(), "profile": losses.tolist(),
            "delta": (losses - losses.min()).tolist(), "theta": thetas.tolist(),
            "drift": [0.0] * len(grid), "converged": converged.tolist(), "min_loss": float(losses.min()),
            "argmin": float(grid[int(np.argmin(losses))]), "optimizer": "lbfgsb", "steps": int(maxiter)}
