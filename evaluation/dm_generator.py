"""The Dalla Man meal model as a replica generator (decision rule R11).

A subject's meal schedule and exact carbohydrate loads are simulated with the Dalla Man model, with the subject's body
weight and one parameter draw per subject (the same distribution and random stream as the earlier control, A27). The
sensor output is returned on the 5 minute grid of the other replicas, so the standard profile pipeline fits it.

The only truth carried over to the Bergman-type fit is the mean time of glucose appearance in plasma, `tau1_dm`,
computed from the intestinal content, which is proportional to the appearance rate.
"""
from __future__ import annotations

import numpy as np

from evaluation.cohort_data import Subject

MEAL_TIME_MIN = 30.0
SENSOR_MIN = 210.0                 # the replica window
MOMENT_MIN = 450.0                 # long enough for the appearance moment


def sample_parameters(subject_id: str) -> dict:
    from evaluation.a27_misspecified_dalla_man import _sample_dm_params
    return _sample_dm_params(subject_id)


def simulate(subject: Subject, parameters: dict) -> tuple[np.ndarray, float]:
    """Sensor glucose (meals x 43 samples, 0 to 210 min) and the mean appearance time after the meal, in minutes."""
    import jax
    import jax.numpy as jnp
    from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
    from simulation.dalla_man import C, GSC, IDX, N_STATE, DallaManParams, basal, rhs

    def f64(value):
        return jnp.asarray(value, dtype=jnp.float64)

    model = DallaManParams(
        Vmx=f64(parameters["Vmx"]), kabs=f64(parameters["kabs"]), kmax=f64(parameters["kmax"]),
        kmin=f64(parameters["kmin"]), f=f64(parameters["f"]), Td=f64(parameters["Td"]), Gb=f64(parameters["Gb"]),
        beta_sec=f64(0.14), weight_kg=f64(float(subject.profile["weight_kg"])))
    plasma, tissue, _, insulin_plasma, insulin_liver, _, _ = basal(model)
    start = jnp.zeros(N_STATE, dtype=jnp.float64)
    start = (start.at[IDX["Gp"]].set(plasma).at[IDX["Gt"]].set(tissue)
             .at[IDX["Ip"]].set(insulin_plasma).at[IDX["Il"]].set(insulin_liver)
             .at[IDX["I1"]].set(C["Ib"]).at[IDX["Id"]].set(C["Ib"]).at[IDX["Gsc"]].set(model.Gb))
    times = jnp.arange(0.0, MOMENT_MIN + 1e-6, 5.0, dtype=jnp.float64)

    def one_meal(carbs_g):
        solution = diffeqsolve(
            ODETerm(rhs), Tsit5(), t0=0.0, t1=MOMENT_MIN, dt0=1.0, y0=start,
            args=(model, MEAL_TIME_MIN, carbs_g * 1000.0, 2.0, 1.0), saveat=SaveAt(ts=times),
            stepsize_controller=PIDController(rtol=1e-5, atol=1e-5), max_steps=20000)
        return solution.ys[:, GSC], solution.ys[:, IDX["Qgut"]]

    carbs = jnp.asarray([r["carbs_g"] for r in subject.records], dtype=jnp.float64)
    sensor, gut = jax.jit(jax.vmap(one_meal))(carbs)
    sensor, gut = np.asarray(sensor, float), np.asarray(gut, float)
    first = int(round(MEAL_TIME_MIN / 5.0))
    after = np.maximum(gut[:, first:], 0.0)
    elapsed = 5.0 * np.arange(after.shape[1])
    moments = [np.trapezoid(elapsed * q, elapsed) / np.trapezoid(q, elapsed)
               for q in after if np.trapezoid(q, elapsed) > 1e-6]
    return sensor[:, : int(round(SENSOR_MIN / 5.0)) + 1], float(np.mean(moments))
