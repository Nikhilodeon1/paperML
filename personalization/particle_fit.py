"""Sequential Monte Carlo joint fit of engine parameters from a user's CGM.

The current personalization fits 3 parameters, ONE AT A TIME, with conjugate Gaussian scalar
updates. That cannot capture correlation between parameters (in real people insulin
sensitivity and gastric emptying co-vary) and assumes a linear observation model, which the
engine is not. This fits SEVERAL metabolic parameters JOINTLY from the glucose time-series,
with a full posterior (samples, not a point + symmetric band).

Method — particle filter / SMC (no torch, no sbi dependency):

  1. draw N particles from the prior over theta
  2. weight each by the likelihood of the observed CGM given the simulated CGM
        w ~ exp( -SSE / (2 * sigma^2) )      sigma = CGM measurement noise
  3. resample particles proportional to weight, jitter with a shrinking kernel
  4. repeat -> the particle cloud collapses onto the posterior

This is the lightweight, on-device-capable bridge to amortized SBI (a Neural Posterior
Estimator): the forward model + observation operator `h` here are exactly what an NPE trains
against. SMC does per-user inference correctly now; NPE later makes it instant across users.

VALIDATION NOTE: on synthetic data the days are engine-generated, so recovering the hidden
theta shows the METHOD works (joint, correlated, non-Gaussian), not that the engine matches a
real human. Real CGM + food log is the final step. See `test_particle_fit.py::
test_recovers_a_wrong_answer_is_impossible` for the teeth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation import Simulator, PhysioParams
from simulation.observation import observe_series, spec_for
from evaluation.forward_validation import DAY_MIN, _PROFILE, schedule_from_events

# parameter -> (low, high) prior support. These three shape the postprandial glucose curve:
# Si sets the depth of clearance, gastric_emptying + carb_absorption set the timing/shape.
PARAM_PRIORS: dict[str, tuple[float, float]] = {
    "insulin_sensitivity": (0.30, 1.40),
    "gastric_emptying": (0.015, 0.040),
    "carb_absorption": (0.012, 0.032),
}
_CGM_SIGMA = 10.0            # mg/dL measurement noise -> the likelihood scale


def _params_from(theta: dict) -> PhysioParams:
    p = PhysioParams.from_profile(_PROFILE)
    for k, v in theta.items():
        setattr(p, k, float(v))
    return p


def _sim_glucose(theta: dict, events: dict, dt: float, step_min: float) -> np.ndarray:
    traj = Simulator(_params_from(theta)).run(
        schedule_from_events(events), duration_min=DAY_MIN, dt=dt, record_every=5,
        outputs=["glucose_mg_dl"])
    return np.asarray(observe_series(traj, spec_for("cgm", "glucose"), step_min=step_min)["values"])


def _sse(theta: dict, days: list[dict], dt: float, step_min: float) -> float:
    total = 0.0
    for day in days:
        obs = np.asarray(day["observations"]["glucose"]["values"])
        pred = _sim_glucose(theta, day["events"], dt, step_min)
        n = min(len(obs), len(pred))
        total += float(np.sum((obs[:n] - pred[:n]) ** 2))
    return total


@dataclass
class Posterior:
    names: list[str]
    particles: np.ndarray          # (N, D)
    weights: np.ndarray            # (N,)

    def mean(self) -> dict:
        return dict(zip(self.names, np.average(self.particles, axis=0, weights=self.weights)))

    def sd(self) -> dict:
        mu = np.average(self.particles, axis=0, weights=self.weights)
        var = np.average((self.particles - mu) ** 2, axis=0, weights=self.weights)
        return dict(zip(self.names, np.sqrt(var)))

    def corr(self, a: str, b: str) -> float:
        """Posterior correlation between two params — the thing a scalar fit cannot see."""
        i, j = self.names.index(a), self.names.index(b)
        x, y = self.particles[:, i], self.particles[:, j]
        w = self.weights
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        cov = np.average((x - mx) * (y - my), weights=w)
        sx = np.sqrt(np.average((x - mx) ** 2, weights=w))
        sy = np.sqrt(np.average((y - my) ** 2, weights=w))
        return float(cov / (sx * sy)) if sx > 0 and sy > 0 else 0.0


def fit(days: list[dict], priors: dict[str, tuple[float, float]] | None = None,
        n_particles: int = 200, rounds: int = 4, dt: float = 1.0, step_min: float = 5.0,
        seed: int = 0) -> Posterior:
    """Joint posterior over the parameters in `priors` from the days' CGM."""
    priors = priors or PARAM_PRIORS
    names = list(priors)
    lo = np.array([priors[n][0] for n in names])
    hi = np.array([priors[n][1] for n in names])
    rng = np.random.default_rng(seed)

    particles = rng.uniform(lo, hi, size=(n_particles, len(names)))
    weights = np.full(n_particles, 1.0 / n_particles)

    for r in range(rounds):
        sse = np.array([_sse(dict(zip(names, p)), days, dt, step_min) for p in particles])
        # likelihood over all matched samples; subtract min for numerical stability
        n_pts = sum(len(d["observations"]["glucose"]["values"]) for d in days)
        loglik = -sse / (2.0 * _CGM_SIGMA ** 2)
        w = np.exp(loglik - loglik.max())
        weights = w / w.sum()

        if r == rounds - 1:
            break
        # resample + jitter with a kernel that shrinks each round (annealing)
        idx = rng.choice(n_particles, size=n_particles, p=weights)
        particles = particles[idx]
        band = (hi - lo) * 0.15 * (0.6 ** r)
        particles = np.clip(particles + rng.normal(0.0, band, particles.shape), lo, hi)
        weights = np.full(n_particles, 1.0 / n_particles)

    return Posterior(names=names, particles=particles, weights=weights)


# --- synthetic data with a MULTI-parameter hidden truth (for validating the method) ------

def synth_days_multi(n_days: int = 6, truth: dict | None = None, seed: int = 0) -> list[dict]:
    truth = truth or {"insulin_sensitivity": 0.55, "gastric_emptying": 0.030,
                      "carb_absorption": 0.020}
    rng = np.random.default_rng(seed)
    days = []
    for d in range(n_days):
        events = {
            "meals": [
                {"t_min": 8 * 60 + int(rng.integers(-30, 30)), "carbs_g": 45 + int(rng.integers(-10, 10))},
                {"t_min": 13 * 60 + int(rng.integers(-40, 40)), "carbs_g": 70 + int(rng.integers(-15, 15))},
                {"t_min": 19 * 60 + int(rng.integers(-40, 40)), "carbs_g": 60 + int(rng.integers(-15, 15))},
            ],
            "sleep": [{"start_min": 0, "end_min": 7 * 60}],
        }
        vals = _sim_glucose(truth, events, dt=1.0, step_min=5.0)
        vals = vals + rng.normal(0.0, _CGM_SIGMA, size=len(vals))     # CGM noise on the "real" data
        days.append({"date": f"day_{d:02d}", "events": events,
                     "observations": {"glucose": {"values": [round(float(v), 2) for v in vals],
                                                   "t0_min": 0.0, "step_min": 5.0}}})
    return days
