"""A12: the same identifiability questions on a second model class (Dalla Man 2007), iAUC observable.

The first draft claimed two model classes and the revision had no result for the second one. This runs the
two analyses that carry the claim, a full Fisher matrix and profile-likelihood intervals, on a gut model
that differs from the Bergman-style engine where it matters for the exchange symmetry:

* a three-compartment gut (solid stomach, liquid stomach, intestine) with a glucose-independent grinding
  rate and a nonlinear (tanh) gastric-emptying law that depends on the stomach content, so there is no
  exact two-compartment series chain and no exact exchange symmetry to inherit from;
* two glucose and two insulin compartments, delayed insulin action on endogenous glucose production, and
  a subcutaneous compartment, so the observation is the sensor lag a CGM has.

Fitted per subject (the Dalla Man subset used by `personalization/gradient_fit_dalla_man`): `Vmx` (insulin
sensitivity, the `S_I` analogue), `kabs` (intestinal absorption), `kmax` and `kmin` (gastric emptying),
`f` (fraction absorbed) and `Td` (sensor lag). The observable is the per-meal iAUC, exactly as for the
primary Bergman analysis. The model starts at the subject's own steady state (`Gb`, the mean pre-meal CGM
value), as the module documents.

Everything is evaluated in **float64** with tight solver tolerances. The shipped Dalla Man module builds
float32 parameters, which is adequate for a prediction fit and not for a Fisher matrix whose smallest
eigenvalue is the quantity of interest, so the parameters and the initial state are rebuilt in float64
here and nothing in `simulation/` is changed.

Noise model: independent Gaussian errors with one variance per subject, estimated from the residual of the
Adam fit and then held fixed (the same freezing as amendment B2), so the negative log-likelihood is
`0.5 * RSS / sigma^2` and a rise of 1.92 above its minimum is a 95% interval. This is simpler than the
Bergman analysis's per-observable weights, and the paper says so.

Run:  python -m evaluation.runner evaluation.dalla_man_identifiability --gate 2
"""
from __future__ import annotations

import time

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402
import optax                     # noqa: E402
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve  # noqa: E402

from evaluation.cohort_data import load_cohort                           # noqa: E402
from evaluation.identifiability_tools import (                           # noqa: E402
    CHI2_DELTA, classify_profile, fisher_matrix, profile_likelihood,
)
from personalization.gradient_fit_dalla_man import PARAM_RANGES, TARGETS  # noqa: E402
from simulation.dalla_man import DallaManParams, GSC, IDX, N_STATE, basal, rhs  # noqa: E402
from simulation.jax_observation import iauc                               # noqa: E402

ANALYSIS_ID = "A12_dalla_man"
BOUND_TOLERANCE = 0.01          # a fit within 1% of the (log) interval of a bound is "pinned", as elsewhere
MEAL_TIME_MIN = 30.0
DURATION_MIN = 210.0


def default_config() -> dict:
    return {
        "cohort": "cgmacros", "min_meals": 10, "limit": None,
        "fit_steps": 400, "fit_learning_rate": 0.03,
        "polish_starts": 5, "polish_maxiter": 300,
        "profile_parameters": ["Vmx", "kabs", "kmax", "kmin"],
        "grid_points": 11, "profile_steps": 50, "profile_learning_rate": 0.03,
        "rtol": 1e-6, "atol": 1e-6, "max_steps": 20000,
    }


def units(config: dict) -> list[str]:
    return [s.subject_id for s in load_cohort(config["cohort"], min_meals=config["min_meals"],
                                              limit=config["limit"])]


# --------------------------------------------------------------------------------------------------
# The float64 model
# --------------------------------------------------------------------------------------------------

def _params(theta: dict, weight_kg: float, gb: float) -> DallaManParams:
    f64 = lambda v: jnp.asarray(v, jnp.float64)                       # noqa: E731
    return DallaManParams(Vmx=f64(theta["Vmx"]), kabs=f64(theta["kabs"]), kmax=f64(theta["kmax"]),
                          kmin=f64(theta["kmin"]), f=f64(theta["f"]), Td=f64(theta["Td"]),
                          Gb=f64(gb), beta_sec=f64(0.14), weight_kg=f64(weight_kg))


def _initial_state(params: DallaManParams):
    Gpb, Gtb, _vm0, Ipb, Ilb, _sb, _kp1 = basal(params)
    from simulation.dalla_man import C
    y = jnp.zeros(N_STATE, dtype=jnp.float64)
    y = y.at[IDX["Gp"]].set(Gpb).at[IDX["Gt"]].set(Gtb).at[IDX["Ip"]].set(Ipb).at[IDX["Il"]].set(Ilb)
    y = y.at[IDX["I1"]].set(C["Ib"]).at[IDX["Id"]].set(C["Ib"]).at[IDX["Gsc"]].set(params.Gb)
    return y


def _meal_iauc(params: DallaManParams, carbs_g, rtol: float, atol: float, max_steps: int):
    ts = jnp.arange(0.0, DURATION_MIN + 1e-6, 5.0, dtype=jnp.float64)
    solution = diffeqsolve(
        ODETerm(rhs), Tsit5(), t0=0.0, t1=DURATION_MIN, dt0=1.0, y0=_initial_state(params),
        args=(params, MEAL_TIME_MIN, jnp.asarray(carbs_g, jnp.float64) * 1000.0, 2.0, 1.0),
        saveat=SaveAt(ts=ts), stepsize_controller=PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps)
    return iauc(ts, solution.ys[:, GSC])


def _subject_gb(records) -> float:
    return float(np.mean([np.mean(r["glucose"]["values"][:6]) for r in records]))


def _bounds_log() -> tuple[np.ndarray, np.ndarray]:
    lo = np.log([PARAM_RANGES[k][0] for k in TARGETS])
    hi = np.log([PARAM_RANGES[k][1] for k in TARGETS])
    return lo, hi


def _population_start() -> np.ndarray:
    base = DallaManParams.defaults(78.0, 100.0)
    return np.log([float(getattr(base, k)) for k in TARGETS])


# --------------------------------------------------------------------------------------------------
# One subject
# --------------------------------------------------------------------------------------------------

def run_unit(unit: str, config: dict) -> dict:
    started = time.perf_counter()
    subject = next(s for s in load_cohort(config["cohort"], min_meals=config["min_meals"],
                                          limit=config["limit"]) if s.subject_id == unit)
    records = list(subject.records)
    carbs = jnp.asarray([r["carbs_g"] for r in records], dtype=jnp.float64)
    observed = jnp.asarray([r["iauc"] for r in records], dtype=jnp.float64)
    gb, weight = _subject_gb(records), float(subject.profile["weight_kg"])
    rtol, atol, max_steps = config["rtol"], config["atol"], config["max_steps"]

    def predict(theta_vec):
        theta = {k: theta_vec[i] for i, k in enumerate(TARGETS)}
        params = _params(theta, weight, gb)
        return jax.vmap(lambda c: _meal_iauc(params, c, rtol, atol, max_steps))(carbs)

    def rss(phi):
        return jnp.sum((predict(jnp.exp(phi)) - observed) ** 2)

    lower, upper = _bounds_log()
    lower_j, upper_j = jnp.asarray(lower), jnp.asarray(upper)
    span = upper - lower
    start = np.clip(_population_start(), lower, upper)

    # Adam in log coordinates, projected to the box, to get a starting estimate and a noise variance.
    optimizer = optax.adam(config["fit_learning_rate"])
    value_and_grad = jax.jit(jax.value_and_grad(rss))

    @jax.jit
    def step(carry):
        phi, state = carry
        value, grad = value_and_grad(phi)
        updates, state = optimizer.update(grad, state)
        return jnp.clip(optax.apply_updates(phi, updates), lower_j, upper_j), state, value

    phi = jnp.asarray(start)
    state = optimizer.init(phi)
    for _ in range(config["fit_steps"]):
        phi, state, _ = step((phi, state))
    phi_adam = np.asarray(phi, dtype=float)
    n = len(records)
    rss_adam = float(rss(jnp.asarray(phi_adam)))
    sigma = float(np.sqrt(max(rss_adam, 1e-12) / n))

    def nll(phi_vec):
        return 0.5 * rss(phi_vec) / sigma ** 2

    from evaluation.profile_lik import polish_estimate
    polished = polish_estimate(nll, phi_adam, lower, upper, None, f"{unit}|dalla_man",
                               int(config["polish_starts"]), int(config["polish_maxiter"]))
    phi_hat = np.asarray(polished["phi"], dtype=float)
    theta_hat = np.exp(phi_hat)
    at_bound = {k: ("lower" if phi_hat[i] <= lower[i] + BOUND_TOLERANCE * span[i] else
                    "upper" if phi_hat[i] >= upper[i] - BOUND_TOLERANCE * span[i] else None)
                for i, k in enumerate(TARGETS)}

    fisher = fisher_matrix(lambda t: predict(t), theta_hat, sigma=sigma, log_coords=True, names=TARGETS)
    covariance = np.linalg.pinv(fisher.fisher, rcond=1e-12)
    information = (1.0 / np.maximum(np.diag(covariance), 1e-300)).tolist()

    loss_at_hat = float(nll(jnp.asarray(phi_hat)))
    profiles = {}
    for name in config["profile_parameters"]:
        index = TARGETS.index(name)
        phi_grid = np.linspace(lower[index], upper[index], config["grid_points"])
        raw = profile_likelihood(nll, phi_hat, index, phi_grid, lower=lower, upper=upper,
                                 steps=config["profile_steps"], learning_rate=config["profile_learning_rate"])
        profile = np.asarray(raw["profile"], dtype=float)
        drift = np.asarray(raw["drift"], dtype=float)
        reference = min(loss_at_hat, float(profile.min()))
        grid = np.exp(phi_grid)
        profiles[name] = {
            "grid": grid.tolist(), "profile": profile.tolist(), "reference": reference,
            "classification": classify_profile(grid, profile, delta=CHI2_DELTA, reference=reference),
            "unsettled": int(np.sum(np.abs(drift) > 0.05)),
            "better_than_theta_hat": bool(float(profile.min()) < loss_at_hat - 1e-6),
            "theta_hat": float(theta_hat[index]),
        }
    distance = {k: float(min(phi_hat[i] - lower[i], upper[i] - phi_hat[i]) / span[i])
                for i, k in enumerate(TARGETS)}
    return {
        "subject_id": unit, "n_meals": n, "Gb": gb, "weight_kg": weight, "sigma_iauc": sigma,
        "theta_hat": dict(zip(TARGETS, theta_hat.tolist())), "at_bound": at_bound,
        "distance_to_bound": distance, "polish": {k: v for k, v in polished.items() if k != "phi"},
        "fisher": fisher.as_dict(), "information_after_profiling_others": dict(zip(TARGETS, information)),
        "profiles": profiles, "delta": CHI2_DELTA, "wall_seconds": round(time.perf_counter() - started, 1),
        "macros": {},
    }


# --------------------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------------------

def summarize(config: dict | None = None) -> dict:
    from evaluation.results_io import largest_matching
    from evaluation.stats_utils import wilson_ci

    rows = list(largest_matching(ANALYSIS_ID, lambda p: True).values())
    if not rows:
        return {"n_subjects": 0}
    out = {"n_subjects": len(rows), "parameters": {}}
    for name in rows[0]["profiles"]:
        verdicts = [r["profiles"][name]["classification"]["verdict"] for r in rows]
        bounded = verdicts.count("identifiable")
        ci = wilson_ci(bounded, len(rows))
        inner = [r for r in rows if r["distance_to_bound"][name] > 0.01]
        inner_bounded = sum(r["profiles"][name]["classification"]["verdict"] == "identifiable" for r in inner)
        entry = {"bounded": bounded, "fraction": bounded / len(rows), "wilson": [ci.low, ci.high],
                 "one_sided": verdicts.count("one-sided"), "flat": verdicts.count("flat"),
                 "interior": {"n": len(inner), "bounded": inner_bounded,
                              "fraction": (inner_bounded / len(inner)) if inner else None},
                 "pinned": sum(r["at_bound"][name] is not None for r in rows),
                 "pinned_upper": sum(r["at_bound"][name] == "upper" for r in rows),
                 "pinned_lower": sum(r["at_bound"][name] == "lower" for r in rows)}
        out["parameters"][name] = entry
    eig = np.array([r["fisher"]["eigenvalues"] for r in rows])
    cond = np.array([r["fisher"]["condition_number"] for r in rows], dtype=float)
    out["fisher"] = {"eigenvalue_medians": np.median(eig, axis=0).tolist(),
                     "condition_number_median": float(np.median(cond[np.isfinite(cond)])) if np.isfinite(cond).any() else None,
                     "effective_rank_median": float(np.median([int(np.sum(np.array(r["fisher"]["eigenvalues"])
                                                                          > 1e-6 * max(r["fisher"]["eigenvalues"])))
                                                               for r in rows]))}
    gut = [k for k in ("kabs", "kmax", "kmin") if k in out["parameters"]]
    out["gut_max_bounded_fraction"] = max(out["parameters"][k]["fraction"] for k in gut) if gut else None
    out["polish_gap_median"] = float(np.median([r["polish"]["gap"] for r in rows]))
    out["wall_seconds_median"] = float(np.median([r["wall_seconds"] for r in rows]))
    return out


def main() -> None:
    import json
    print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
