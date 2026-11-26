"""A28: the pseudo-true value of the Bergman-type fit to noise-free Dalla Man output (Amendment 17).

A27 generates traces from the Dalla Man model and fits the Bergman-type simulator. The fit does not target the Dalla Man mean
appearance time, because Bergman's tau1 absorbs whatever the Bergman structure lacks (sensor lag, a one-state insulin action
with fixed p2 and p3, a linear gut). Under misspecification the estimator converges to a pseudo-true value, the minimizer of the
fit to the noise-free output. This analysis computes that value per subject, so that two quantities can be reported
separately:

* the bias of the model: pseudo-true tau1 against the Dalla Man mean appearance time (and against that time plus the sensor
  lag), which is what no amount of data repairs;
* the variance of the estimator: coverage of the pseudo-true value by the A27 intervals, which is the question a profile
  interval answers.

The noise-free traces use the same Dalla Man parameters and carbohydrate schedules as A27 (`_sample_dm_params`). The Bergman
fit is a least-squares fit of the trace inside the original rate box (3-start L-BFGS-B, as in A27), with unit variance and no
whitening, since noise-free residuals have no variance to estimate.

Run:  python -m evaluation.a28_dm_pseudo_true            (writes results/phase20_amendment16/a28_dm_pseudo_true.json)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

from evaluation.a27_misspecified_dalla_man import KA_HI, KA_LO, KE_HI, KE_LO, SI_HI, SI_LO, _sample_dm_params  # noqa: E402
from evaluation.results_io import RESULTS  # noqa: E402
from personalization.coords import rates_to_tau_p  # noqa: E402

OUT_DIR = RESULTS / "phase20_amendment16"
UNIT_DIR = OUT_DIR / "a28_units"
UNIT_DIR.mkdir(parents=True, exist_ok=True)
A27_UNITS = OUT_DIR / "a27_units"


def solve_one(sid: str) -> dict:
    unit_file = UNIT_DIR / f"{sid}.json"
    if unit_file.exists():
        return json.loads(unit_file.read_text(encoding="utf-8"))

    from evaluation.jax_config import configure
    configure()
    import jax
    import jax.numpy as jnp
    from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
    from scipy.optimize import minimize

    from evaluation.a26_exact_domain_profiles import _get_kernels, _project_phi_to_rates_log
    from evaluation.cohort_data import load_cohort
    from evaluation.subject_source import window_for
    from personalization.objectives import ObjectiveSpec, build_objective, observed_values
    from personalization.subject_loss import base_params, subject_arrays
    from simulation.dalla_man import C as DM_C
    from simulation.dalla_man import DallaManParams, GSC, IDX, N_STATE, basal, rhs
    from simulation.jax_engine_canonical import CanonicalParams
    from simulation.jax_observables import post_indices

    subj = next(s for s in load_cohort("cgmacros", min_meals=10) if s.subject_id == sid)
    win = window_for(subj)
    dm = _sample_dm_params(sid)
    f64 = lambda v: jnp.asarray(v, dtype=jnp.float64)
    dm_p = DallaManParams(Vmx=f64(dm["Vmx"]), kabs=f64(dm["kabs"]), kmax=f64(dm["kmax"]), kmin=f64(dm["kmin"]), f=f64(dm["f"]),
                          Td=f64(dm["Td"]), Gb=f64(dm["Gb"]), beta_sec=f64(0.14), weight_kg=f64(float(subj.profile["weight_kg"])))
    Gpb, Gtb, _vm0, Ipb, Ilb, _sb, _kp1 = basal(dm_p)
    y0 = jnp.zeros(N_STATE, dtype=jnp.float64)
    y0 = (y0.at[IDX["Gp"]].set(Gpb).at[IDX["Gt"]].set(Gtb).at[IDX["Ip"]].set(Ipb).at[IDX["Il"]].set(Ilb)
          .at[IDX["I1"]].set(DM_C["Ib"]).at[IDX["Id"]].set(DM_C["Ib"]).at[IDX["Gsc"]].set(dm_p.Gb))
    ts_long = jnp.arange(0.0, 450.0 + 1e-6, 5.0, dtype=jnp.float64)

    def sim_meal(carbs_g):
        sol = diffeqsolve(ODETerm(rhs), Tsit5(), t0=0.0, t1=450.0, dt0=1.0, y0=y0,
                          args=(dm_p, 30.0, jnp.asarray(carbs_g, jnp.float64) * 1000.0, 2.0, 1.0), saveat=SaveAt(ts=ts_long),
                          stepsize_controller=PIDController(rtol=1e-5, atol=1e-5), max_steps=20000)
        return sol.ys[:, GSC], sol.ys[:, IDX["Qgut"]]

    carbs_arr = jnp.asarray([r["carbs_g"] for r in subj.records], dtype=jnp.float64)
    gsc_long, _ = jax.jit(jax.vmap(sim_meal))(carbs_arr)
    gsc_210 = np.asarray(gsc_long, float)[:, :43]
    records = [{"carbs_g": r["carbs_g"], "fat_g": 0.0, "fiber_g": 0.0, "protein_g": 0.0, "iauc": 0.0,
                "glucose": {"values": gsc_210[i].tolist(), "t0_min": -30.0, "step_min": 5.0, "meal_t_min": 0.0}}
               for i, r in enumerate(subj.records)]

    spec = ObjectiveSpec(name="trace", lam=0.0, normalization="nll", bounds_scale=1.0, window=win, parameterization="coords")
    arrays = subject_arrays(records)
    obs = observed_values(records, win)
    cbase = CanonicalParams.from_jax_params(base_params(subj.profile))
    width = int(arrays["width"])
    n_samples = obs["trace"].shape[1]
    padded = np.zeros((width, n_samples))
    padded[: obs["trace"].shape[0], :] = obs["trace"]
    sample_idx = jnp.asarray(np.asarray(post_indices(win)), dtype=jnp.int32)
    pre0, pre1 = int(win.pre_slice[0]), int(win.pre_slice[1])
    _nll, rate_vg, _prof = _get_kernels((width, n_samples, float(spec.duration_min), float(win.step_min), pre0, pre1))
    args = (cbase, jnp.asarray(arrays["carbs"], dtype=jnp.float64), jnp.asarray(arrays["fat"], dtype=jnp.float64),
            jnp.asarray(arrays["fiber"], dtype=jnp.float64), jnp.asarray(arrays["mask"], dtype=jnp.float64),
            jnp.asarray(padded, dtype=jnp.float64), sample_idx, jnp.asarray(1.0, dtype=jnp.float64), jnp.asarray(0.0, dtype=jnp.float64))
    bounds = [(math.log(SI_LO), math.log(SI_HI)), (math.log(KE_LO), math.log(KE_HI)), (math.log(KA_LO), math.log(KA_HI))]
    starts = [np.array([math.log(1.0), math.log(0.026), math.log(0.022)]),
              np.array([0.5 * (a + b) for a, b in bounds]),
              np.array([math.log(0.6), math.log(0.02), math.log(0.016)])]
    best, best_val = starts[0], float("inf")
    for s0 in starts:
        def fun(x):
            v, g = rate_vg(jnp.asarray(x, dtype=jnp.float64), *args)
            return float(v), np.asarray(g, dtype=float)
        res = minimize(fun, s0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400})
        if float(res.fun) < best_val:
            best_val, best = float(res.fun), np.asarray(res.x, float)
    tau1, p = rates_to_tau_p(math.exp(best[1]), math.exp(best[2]))
    out = {"subject_id": sid, "pseudo_tau1": tau1, "pseudo_p": p, "pseudo_si": math.exp(best[0]),
           "pseudo_ke": math.exp(best[1]), "pseudo_ka": math.exp(best[2]), "least_squares_value": best_val, "Td": dm["Td"]}
    unit_file.write_text(json.dumps(out), encoding="utf-8")
    return out


def main() -> int:
    from evaluation.cohort_data import load_cohort
    ids = [s.subject_id for s in load_cohort("cgmacros", min_meals=10)]
    rows = []
    for sid in ids:
        a27 = json.loads((A27_UNITS / f"{sid}.json").read_text(encoding="utf-8"))
        r = solve_one(sid)
        r["true_tau1_dm"], r["true_tau1_plus_td"] = a27["true_tau1_dm"], a27["true_tau1_plus_td"]
        r["tau1_ci"] = [a27["tau1"]["ci_low"], a27["tau1"]["ci_high"]]
        r["tau1_verdict"] = a27["tau1"]["verdict"]
        r["p_ci"] = [a27["p"]["ci_low"], a27["p"]["ci_high"]]
        r["hat_tau1"] = a27["hat_tau1"]
        rows.append(r)
    (OUT_DIR / "a28_dm_pseudo_true.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    print("wrote", OUT_DIR / "a28_dm_pseudo_true.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
