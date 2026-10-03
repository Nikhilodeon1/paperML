"""Recursive Gaussian (1-D Kalman) updating for a single latent parameter.

Conjugate Gaussian update: prior N(mean, var) + observation `y` with noise variance
`obs_var` that is a linear readout of the latent (y ~ N(latent, obs_var)) gives a
closed-form posterior. Optional `process_var` lets the latent drift slowly between
updates (e.g. metabolism shifts over months), which keeps the filter from becoming
overconfident and unable to track real change.

This is the engine behind Layer 3. It is deliberately tiny and dependency-free so it
can run on-device.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GaussianPosterior:
    mean: float
    var: float

    @property
    def sd(self) -> float:
        return self.var ** 0.5

    def predict(self, process_var: float = 0.0) -> "GaussianPosterior":
        """Time update: let the latent drift, widening uncertainty."""
        return GaussianPosterior(self.mean, self.var + process_var)

    def update(self, y: float, obs_var: float) -> "GaussianPosterior":
        """Measurement update with a direct (unit-gain) observation of the latent.

        Kalman gain K = var / (var + obs_var); posterior var = (1-K)*var.
        """
        if obs_var <= 0:
            raise ValueError("obs_var must be positive")
        k = self.var / (self.var + obs_var)
        return GaussianPosterior(
            mean=self.mean + k * (y - self.mean),
            var=(1.0 - k) * self.var,
        )

    def step(self, y: float, obs_var: float, process_var: float = 0.0) -> "GaussianPosterior":
        """Predict (drift) then update — one full filter step."""
        return self.predict(process_var).update(y, obs_var)
