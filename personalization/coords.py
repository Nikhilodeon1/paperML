"""Reparameterization: fit the combination the data can see, instead of the rates it cannot.

The central claim of the revision is that postprandial CGM identifies the mean transit time of the gut
cascade but not how that time is split between gastric emptying and carbohydrate absorption. Testing
that claim by fitting `(k_e, k_a)` and then reporting a function of them is weaker than fitting the
function directly, because the fit never has to commit to the split and so cannot be accused of having
found a split that the optimizer merely happened to land on.

Coordinates, all three fitted in logs:

    log S_I,   log tau1 = log(1/k_e + 1/k_a),   log p = log(1/(k_e k_a))

`tau1` is the mean transit time, in minutes, and it is the quantity H11 predicts is identifiable.
`p` is the remaining degree of freedom, and H11 predicts it is NOT identifiable under an area-plus-
timing observable. Logs because the parameters are positive and span a factor of three, so a relative
step is the natural one and the Fisher matrix in log coordinates is dimensionless.

**Real poles.** The gut cascade has two real rate constants only when the quadratic
`s^2 + sigma1 s + c` has real roots, which in these coordinates is exactly

    p <= tau1^2 / 4

The boundary `p = tau1^2 / 4` is `k_e = k_a`: the tied-rate model of H12. The constraint is enforced by
projection after each update, which lands a violating step on the tied-rate boundary rather than
rejecting it -- a fit that wants to be there should be allowed there, and how often that happens is
reported.

**The box is a superset of the exact image, deliberately.** The rectangle below is the hull of the image
of the original `(k_e, k_a)` box, not the image itself, which is a curvilinear region. Some corners of
the hull correspond to rate pairs outside the original box. The alternative -- a tighter box -- is the
dangerous direction: a bound that cuts into the likelihood manufactures apparent identifiability, which
is the reviewer objection this whole phase exists to answer. Every fit records whether its implied
rates fell inside the original box, so the looseness is visible rather than assumed harmless.

Nothing here is differentiated through a square root. `tau_p_to_canonical` maps straight to the
`(sigma1, c)` coordinates of `simulation/jax_engine_canonical`, which is analytic everywhere;
`tau_p_to_rates` exists only for reporting and says so.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from personalization.gradient_fit import PARAM_RANGES

__all__ = [
    "COORD_NAMES", "TIED_COORD_NAMES", "TAU_RANGE", "P_RANGE", "SI_RANGE",
    "rates_to_tau_p", "tau_p_to_rates", "tau_p_to_canonical", "tied_to_canonical",
    "log_bounds", "project_log", "on_tied_boundary", "rates_in_original_box",
    "tied_tau_range", "LOG_FOUR",
]

COORD_NAMES = ("log_insulin_sensitivity", "log_tau1", "log_p")
TIED_COORD_NAMES = ("log_insulin_sensitivity", "log_tau1")

LOG_FOUR = math.log(4.0)

_KE_LO, _KE_HI = PARAM_RANGES["gastric_emptying"]
_KA_LO, _KA_HI = PARAM_RANGES["carb_absorption"]
SI_RANGE = PARAM_RANGES["insulin_sensitivity"]

# Hull of the image of the original rate box. tau1 is largest when both rates are slowest; p is largest
# when their product is smallest, which is the same corner.
TAU_RANGE = (1.0 / _KE_HI + 1.0 / _KA_HI, 1.0 / _KE_LO + 1.0 / _KA_LO)
P_RANGE = (1.0 / (_KE_HI * _KA_HI), 1.0 / (_KE_LO * _KA_LO))


def rates_to_tau_p(k_e, k_a):
    """`(k_e, k_a) -> (tau1, p)`. Smooth, and the direction that is always safe."""
    return 1.0 / k_e + 1.0 / k_a, 1.0 / (k_e * k_a)


def tau_p_to_rates(tau1, p):
    """`(tau1, p) -> (k_e, k_a)`, for REPORTING only.

    Singular at `k_e = k_a`, where the square root has an infinite derivative. Never call this inside
    anything that will be differentiated. Returns NaN when the poles are not real.
    """
    sigma1 = tau1 / p
    c = 1.0 / p
    discriminant = sigma1 ** 2 - 4.0 * c
    root = np.sqrt(np.where(discriminant >= 0.0, discriminant, np.nan))
    return 0.5 * (sigma1 + root), 0.5 * (sigma1 - root)


def tau_p_to_canonical(tau1, p):
    """`(tau1, p) -> (sigma1, c)`, the coordinates the canonical engine integrates in."""
    return tau1 / p, 1.0 / p


def tied_to_canonical(tau1):
    """The tied-rate submodel `k_e = k_a = 2 / tau1`, so `sigma1 = 4 / tau1` and `c = 4 / tau1^2`."""
    return 4.0 / tau1, 4.0 / (tau1 ** 2)


def tied_tau_range() -> tuple[float, float]:
    """`tau1` range of the tied-rate model, from the original box on the shared rate.

    With `k_e = k_a = k`, `tau1 = 2 / k`. The shared rate has to lie in both original intervals, so it
    runs over their intersection; outside that there is no tied model inside the original box at all.
    """
    lo = max(_KE_LO, _KA_LO)
    hi = min(_KE_HI, _KA_HI)
    if lo >= hi:
        raise ValueError(
            f"the gastric and absorption intervals do not overlap ([{_KE_LO}, {_KE_HI}] and "
            f"[{_KA_LO}, {_KA_HI}]), so no tied-rate model lies inside the original box")
    return 2.0 / hi, 2.0 / lo


def log_bounds(tied: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds in log coordinates, in `COORD_NAMES` order."""
    if tied:
        tau_lo, tau_hi = tied_tau_range()
        return (np.array([math.log(SI_RANGE[0]), math.log(tau_lo)]),
                np.array([math.log(SI_RANGE[1]), math.log(tau_hi)]))
    return (np.array([math.log(SI_RANGE[0]), math.log(TAU_RANGE[0]), math.log(P_RANGE[0])]),
            np.array([math.log(SI_RANGE[1]), math.log(TAU_RANGE[1]), math.log(P_RANGE[1])]))


def project_log(theta_log, lower, upper, tied: bool = False):
    """Clip to the box, then enforce real poles by projecting onto the tied-rate boundary.

    In log coordinates the real-pole condition `p <= tau1^2 / 4` is linear:

        log p <= 2 log tau1 - log 4

    A violating step is projected onto that boundary rather than rejected, because the boundary is a
    legitimate model -- it is exactly `k_e = k_a` -- and because rejecting steps would stall the
    optimizer along the one direction the analysis cares most about.

    The tied parameterization has no `p`, so the constraint cannot be violated and only the box is
    applied.
    """
    theta_log = jnp.clip(jnp.asarray(theta_log), jnp.asarray(lower), jnp.asarray(upper))
    if tied:
        return theta_log
    ceiling = 2.0 * theta_log[1] - LOG_FOUR
    return theta_log.at[2].set(jnp.minimum(theta_log[2], ceiling))


def on_tied_boundary(theta_log, tolerance: float = 1e-6) -> bool:
    """Whether a fit sits on `p = tau1^2 / 4`, i.e. whether it chose equal rates."""
    theta_log = np.asarray(theta_log, dtype=float)
    if theta_log.size < 3:
        return True                     # the tied model is the boundary by construction
    return bool(theta_log[2] >= 2.0 * theta_log[1] - LOG_FOUR - tolerance)


def rates_in_original_box(tau1, p) -> dict:
    """Whether the rates implied by `(tau1, p)` lie inside the original `(k_e, k_a)` box.

    The fitted box is the hull of the image of that box, so some of it corresponds to rate pairs the
    original parameterization would have excluded. Reporting this per fit keeps the looseness visible.
    """
    k_e, k_a = tau_p_to_rates(np.asarray(tau1, dtype=float), np.asarray(p, dtype=float))
    real = bool(np.all(np.isfinite([k_e, k_a])))
    inside = bool(real
                  and _KE_LO <= float(k_e) <= _KE_HI
                  and _KA_LO <= float(k_a) <= _KA_HI)
    return {"k_e": float(k_e) if real else None, "k_a": float(k_a) if real else None,
            "real_poles": real, "inside_original_box": inside}
