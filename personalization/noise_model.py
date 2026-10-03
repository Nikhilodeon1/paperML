"""Per-subject noise model: how much each observable actually scatters, and how correlated it is.

Two corrections live here, both of which change results and both of which were pre-registered in
`PREREG_AMENDMENT_1.md` before any Phase 2 analysis ran.

**Inverse residual variance, not observed variance (B2).** When two observables with different units
are fitted together, something has to make their residuals comparable. Dividing by the spread of the
OBSERVED values -- what the first draft of the objective did -- makes the weighting depend on how
variable the cohort happens to be, which is not a property of the measurement. Dividing by the spread
of the RESIDUALS at a pilot fit weights each observable by how well the model can actually predict it,
which is the maximum-likelihood weighting. It matters because the H3 metric is the condition number of
a Schur-complement block, and a condition number is not invariant to relative weights: get the
weighting wrong and H3 measures the weighting.

**AR(1) whitening of the trace (B3).** A CGM trace is close to a random walk over three hours. Thirty-
seven samples of it do not carry thirty-seven independent observations, and a likelihood that assumes
they do overstates how much the full trace reveals -- biasing H3 and H5 in favour of the trace rung,
which is the direction that would most flatter the paper. Prais-Winsten whitening removes the lag-1
structure: the first residual of each meal is scaled by `sqrt(1 - rho^2)` and each later one becomes
`r_t - rho r_{t-1}`, after which the innovations are treated as independent.

Both `rho` and every `sigma` are estimated once from pilot residuals and then **frozen**. Re-estimating
them inside the optimization would let the fit lower its own loss by claiming its errors are noise,
which is the classic way to make a model look better than it is.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = ["NoiseModel", "estimate_sigma", "estimate_ar1", "whiten_rows", "unit_noise"]

# Floors, so a subject whose pilot fit happens to be near-perfect on one observable does not get an
# infinite weight on it. Each is well below the measurement resolution of its observable.
SIGMA_FLOOR = {"iauc": 1.0, "centroid": 0.1, "trace": 0.5}      # mg/dL*min, min, mg/dL
RHO_MAX = 0.99      # a unit root would make the whitened series a pure difference


@dataclass(frozen=True)
class NoiseModel:
    """Frozen noise parameters for one subject and one objective.

    `sigma` is the residual standard deviation per observable. `rho` and `innovation_sigma` apply to
    the trace only: after whitening, `innovation_sigma` is the standard deviation of the innovations,
    which is what the whitened residuals are divided by.
    """
    sigma: dict = field(default_factory=dict)
    rho: float = 0.0
    innovation_sigma: float = 1.0
    dof: dict = field(default_factory=dict)
    n_samples: dict = field(default_factory=dict)
    whitened: bool = False
    note: str = ""

    def for_observable(self, name: str) -> float:
        return float(self.sigma.get(name, 1.0))

    def as_dict(self) -> dict:
        return asdict(self)


def unit_noise() -> NoiseModel:
    """An unweighted model, for the pilot pass where no estimate exists yet."""
    return NoiseModel(sigma={}, rho=0.0, innovation_sigma=1.0, whitened=False,
                      note="pilot pass: unweighted")


def estimate_sigma(residuals, n_parameters: int, observable: str) -> tuple[float, int]:
    """Residual standard deviation and the degrees of freedom it was computed on.

    `n - p` in the denominator rather than `n`: with a median of 36 meals and three parameters the
    difference is a few percent of the variance, which is a few percent of every interval width.
    """
    values = np.asarray(residuals, dtype=float).ravel()
    values = values[np.isfinite(values)]
    floor = SIGMA_FLOOR.get(observable, 1e-9)
    if values.size == 0:
        return floor, 0
    dof = max(values.size - n_parameters, 1)
    sigma = float(np.sqrt(np.sum(values ** 2) / dof))
    return max(sigma, floor), int(dof)


def estimate_ar1(rows) -> tuple[float, float]:
    """Lag-1 coefficient and innovation standard deviation, pooled over a subject's meals.

    `rows` is meals x samples. Each meal is its own series -- the three-hour windows are days apart,
    so pooling the lag-1 products within meals while never pairing the last sample of one meal with
    the first of the next is the whole point of doing it this way rather than on a flattened vector.

    `rho` is clipped to be non-negative: a negative lag-1 coefficient on a CGM trace is a sign of
    noise dominating rather than of real anti-persistence, and whitening with it would amplify noise.
    """
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 3:
        return 0.0, 1.0

    numerator = float(np.sum(rows[:, 1:] * rows[:, :-1]))
    denominator = float(np.sum(rows[:, :-1] ** 2))
    rho = 0.0 if denominator <= 0.0 else numerator / denominator
    rho = float(min(max(rho, 0.0), RHO_MAX))

    whitened = whiten_rows(rows, rho)
    innovation = float(np.sqrt(np.mean(whitened ** 2))) if whitened.size else 1.0
    return rho, max(innovation, SIGMA_FLOOR["trace"])


def whiten_rows(rows, rho: float) -> np.ndarray:
    """Prais-Winsten transform of each row.

    First sample scaled by `sqrt(1 - rho^2)` so it carries the stationary variance rather than being
    discarded; later samples become `r_t - rho r_{t-1}`. Row count and width are preserved, so the
    whitened array can be used wherever the raw one was.
    """
    rows = np.asarray(rows, dtype=float)
    if rows.size == 0:
        return rows
    out = np.empty_like(rows)
    out[:, 0] = np.sqrt(max(1.0 - rho ** 2, 0.0)) * rows[:, 0]
    out[:, 1:] = rows[:, 1:] - rho * rows[:, :-1]
    return out


def build(residuals: dict, n_parameters: int, whiten_trace: bool = True) -> NoiseModel:
    """A frozen noise model from pilot residuals.

    `residuals` maps an observable name to its residual array: a vector for `iauc` and `centroid`, a
    meals x samples matrix for `trace`.
    """
    sigma, dof, counts = {}, {}, {}
    rho, innovation = 0.0, 1.0
    whitened = False

    for name, values in residuals.items():
        array = np.asarray(values, dtype=float)
        if name == "trace" and array.ndim == 2 and whiten_trace:
            rho, innovation = estimate_ar1(array)
            sigma[name] = innovation
            dof[name] = max(array.size - n_parameters, 1)
            counts[name] = int(array.size)
            whitened = True
            continue
        sigma[name], dof[name] = estimate_sigma(array, n_parameters, name)
        counts[name] = int(array.size)

    note = ("trace whitened with a frozen AR(1) coefficient" if whitened
            else "no autocorrelation correction (no trace residuals)")
    return NoiseModel(sigma=sigma, rho=rho, innovation_sigma=innovation, dof=dof,
                      n_samples=counts, whitened=whitened, note=note)
