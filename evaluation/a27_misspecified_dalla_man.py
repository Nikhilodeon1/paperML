"""A27: Misspecified-generator control (Dalla Man 2007 -> Bergman-type simulator).

Generates postprandial CGM traces for the 45 CGMacros meal schedules using the nonlinear
three-compartment Dalla Man (2007) meal model with independently sampled physiological parameters,
exact carbohydrates (CV = 0), and CGMacros-matched subject-specific AR(1) CGM noise.

Fits the Bergman-type simulator strictly inside the exact feasible domain D and evaluates:
1. Exact-D boundedness of full-trace (tau1, p) profile intervals under structural misspecification.
2. Bias and 95% profile coverage of Bergman's tau1 interval against Dalla Man's exact
   model-independent mean plasma glucose appearance time tau1_DM (and lag-adjusted tau1_DM + Td).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

from evaluation.results_io import RESULTS  # noqa: E402
from personalization.coords import (  # noqa: E402
    P_RANGE,
    SI_RANGE,
    TAU_RANGE,
    feasible_p_bounds,
    feasible_tau1_bounds,
    in_feasible_domain,
    rates_to_tau_p,
)
from personalization.gradient_fit import PARAM_RANGES  # noqa: E402

OUT_DIR = RESULTS / "phase20_amendment16"
UNIT_DIR = OUT_DIR / "a27_units"
UNIT_DIR.mkdir(parents=True, exist_ok=True)

KE_LO, KE_HI = PARAM_RANGES["gastric_emptying"]
KA_LO, KA_HI = PARAM_RANGES["carb_absorption"]
SI_LO, SI_HI = SI_RANGE


def _seed_for(sid: str, salt: str = "a27_dm_v1") -> int:
    h = hashlib.sha256(f"{salt}:{sid}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**32 - 1)


def _sample_dm_params(sid: str) -> dict:
    rng = np.random.default_rng(_seed_for(sid, "a27_dm_params"))
    vmx = float(np.clip(0.047 * np.exp(rng.normal(0.0, 0.25)), 0.015, 0.120))
    kabs = float(np.clip(0.045 * np.exp(rng.normal(0.0, 0.25)), 0.022, 0.090))
    kmax = float(np.clip(0.0558 * np.exp(rng.normal(0.0, 0.20)), 0.032, 0.095))
    kmin = float(np.clip(0.028 * np.exp(rng.normal(0.0, 0.25)), 0.012, min(0.050, 0.85 * kmax)))
    td = float(rng.uniform(6.0, 16.0))
    return {"Vmx": vmx, "kabs": kabs, "kmax": kmax, "kmin": kmin, "f": 0.90, "Td": td, "Gb": 90.0}


def _ar1_noise(rng: np.random.Generator, n_meals: int, n_steps: int, sigma: float, rho: float) -> np.ndarray:
    rho = float(np.clip(rho, 0.0, 0.98))
    innov_sd = sigma * math.sqrt(max(1.0 - rho * rho, 1e-6))
    eps = np.zeros((n_meals, n_steps), dtype=float)
    eps[:, 0] = rng.normal(0.0, sigma, size=n_meals)
    for t in range(1, n_steps):
        eps[:, t] = rho * eps[:, t - 1] + rng.normal(0.0, innov_sd, size=n_meals)
    return eps


def _solve_one(args: tuple[str, dict]) -> dict:
    sid, stored_real = args
    unit_file = UNIT_DIR / f"{sid}.json"
    if unit_file.exists():
        return json.loads(unit_file.read_text(encoding="utf-8"))

    from evaluation.jax_config import configure
    configure()
    import jax
    import jax.numpy as jnp
    import optax
    from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
    from scipy.optimize import minimize

    from evaluation.a26_exact_domain_profiles import _get_kernels, _project_phi_to_rates_log
    from evaluation.cohort_data import load_cohort
    from evaluation.identifiability_tools import CHI2_DELTA, classify_profile
    from evaluation.subject_source import window_for
    from personalization import noise_model as nm
    from personalization.objectives import ObjectiveSpec, build_objective, observed_values
    from personalization.subject_loss import base_params, subject_arrays
    from simulation.dalla_man import DallaManParams, GSC, IDX, N_STATE, basal, rhs
    from simulation.jax_engine_canonical import CanonicalParams
    from simulation.jax_observables import post_indices

    subj = next(s for s in load_cohort("cgmacros", min_meals=10) if s.subject_id == sid)
    win = window_for(subj)
    dm_dict = _sample_dm_params(sid)
    weight_kg = float(subj.profile["weight_kg"])

    f64 = lambda v: jnp.asarray(v, dtype=jnp.float64)
    dm_p = DallaManParams(
        Vmx=f64(dm_dict["Vmx"]), kabs=f64(dm_dict["kabs"]), kmax=f64(dm_dict["kmax"]),
        kmin=f64(dm_dict["kmin"]), f=f64(dm_dict["f"]), Td=f64(dm_dict["Td"]),
        Gb=f64(dm_dict["Gb"]), beta_sec=f64(0.14), weight_kg=f64(weight_kg),
    )
    Gpb, Gtb, _vm0, Ipb, Ilb, _sb, _kp1 = basal(dm_p)
    from simulation.dalla_man import C as DM_C
    y0 = jnp.zeros(N_STATE, dtype=jnp.float64)
    y0 = (y0.at[IDX["Gp"]].set(Gpb).at[IDX["Gt"]].set(Gtb)
          .at[IDX["Ip"]].set(Ipb).at[IDX["Il"]].set(Ilb)
          .at[IDX["I1"]].set(DM_C["Ib"]).at[IDX["Id"]].set(DM_C["Ib"])
          .at[IDX["Gsc"]].set(dm_p.Gb))

    ts_long = jnp.arange(0.0, 450.0 + 1e-6, 5.0, dtype=jnp.float64)  # 0..450 min (91 pts); first 43 pts = 0..210 min

    def sim_dm_meal(carbs_g):
        sol = diffeqsolve(
            ODETerm(rhs), Tsit5(), t0=0.0, t1=450.0, dt0=1.0, y0=y0,
            args=(dm_p, 30.0, jnp.asarray(carbs_g, jnp.float64) * 1000.0, 2.0, 1.0),
            saveat=SaveAt(ts=ts_long),
            stepsize_controller=PIDController(rtol=1e-5, atol=1e-5),
            max_steps=20000,
        )
        return sol.ys[:, GSC], sol.ys[:, IDX["Qgut"]]

    carbs_arr = jnp.asarray([r["carbs_g"] for r in subj.records], dtype=jnp.float64)
    gsc_long, qgut_long = jax.jit(jax.vmap(sim_dm_meal))(carbs_arr)
    gsc_long = np.asarray(gsc_long, float)
    qgut_long = np.asarray(qgut_long, float)

    # True Dalla Man mean plasma glucose appearance time across subject's meals (t >= 30 min -> u = t - 30)
    u_post = np.arange(0.0, 420.0 + 1e-6, 5.0)
    q_post = np.maximum(qgut_long[:, 6:], 0.0)  # index 6 == t=30 min
    meal_tau1s = []
    for i in range(q_post.shape[0]):
        area = float(np.trapezoid(q_post[i], u_post))
        if area > 1e-6:
            meal_tau1s.append(float(np.trapezoid(u_post * q_post[i], u_post) / area))
    true_tau1_dm = float(np.mean(meal_tau1s))
    true_tau1_plus_td = true_tau1_dm + dm_dict["Td"]

    # Build noisy synthetic records using CGMacros-matched empirical AR(1) noise and EXACT carbohydrates
    noise_sigma = float(stored_real.get("noise", {}).get("trace", 10.0))
    noise_rho = float(stored_real.get("noise", {}).get("rho", 0.7))
    if not np.isfinite(noise_sigma) or noise_sigma <= 0.5 or noise_sigma > 60.0:
        noise_sigma, noise_rho = 10.0, 0.7
    rng = np.random.default_rng(_seed_for(sid, "a27_dm_noise"))
    gsc_210 = gsc_long[:, :43]
    noisy_g = gsc_210 + _ar1_noise(rng, gsc_210.shape[0], gsc_210.shape[1], noise_sigma, noise_rho)

    syn_records = []
    for i, r in enumerate(subj.records):
        vals = noisy_g[i].tolist()
        syn_records.append({
            "carbs_g": r["carbs_g"],
            "fat_g": 0.0,    # Dalla Man does not take fat/fiber blunting; pass exact glucose load
            "fiber_g": 0.0,
            "protein_g": 0.0,
            "iauc": 0.0,
            "glucose": {
                "values": vals,
                "t0_min": -30.0,
                "step_min": 5.0,
                "meal_t_min": 0.0,
            },
        })

    spec = ObjectiveSpec(
        name="trace", lam=0.0, normalization="nll", bounds_scale=1.0,
        window=win, parameterization="coords",
    )
    arrays = subject_arrays(syn_records)
    obs = observed_values(syn_records, win)
    base = base_params(subj.profile)
    cbase = CanonicalParams.from_jax_params(base)

    # Pilot fit to freeze noise model (identical protocol to real data)
    raw = build_objective(spec, base, arrays, obs, noise=None)
    pilot_opt = optax.adam(0.03)
    def _pbody(carry, _):
        th, st = carry
        v, g = jax.value_and_grad(raw.loss)(th)
        u, st = pilot_opt.update(g, st, th)
        th = raw.project(optax.apply_updates(th, u))
        return (th, st), v
    (phi_pilot, _), _ = jax.jit(
        lambda t0: jax.lax.scan(_pbody, (t0, pilot_opt.init(t0)), None, length=150)
    )(raw.theta0)
    noise = nm.build(raw.residuals(phi_pilot), n_parameters=len(spec.free), whiten_trace=spec.whiten_trace)

    width = int(arrays["width"])
    obs_tr_np = obs["trace"]
    n_samples = obs_tr_np.shape[1]
    padded_obs = np.zeros((width, n_samples), dtype=float)
    padded_obs[: obs_tr_np.shape[0], :] = obs_tr_np
    sample_idx = jnp.asarray(np.asarray(post_indices(win)), dtype=jnp.int32)
    pre0, pre1 = int(win.pre_slice[0]), int(win.pre_slice[1])
    key = (width, n_samples, float(spec.duration_min), float(win.step_min), pre0, pre1)
    nll_jit, rate_vg_jit, prof_jit = _get_kernels(key)

    carbs = jnp.asarray(arrays["carbs"], dtype=jnp.float64)
    fat = jnp.asarray(arrays["fat"], dtype=jnp.float64)
    fiber = jnp.asarray(arrays["fiber"], dtype=jnp.float64)
    mask = jnp.asarray(arrays["mask"], dtype=jnp.float64)
    obs_trace = jnp.asarray(padded_obs, dtype=jnp.float64)
    sigma_trace = jnp.asarray(float(noise.for_observable("trace")), dtype=jnp.float64)
    rho = jnp.asarray(float(noise.rho) if (spec.whiten_trace and noise.whitened) else 0.0, dtype=jnp.float64)

    # Exact MLE inside original rate box D
    rate_bounds = [
        (math.log(SI_LO), math.log(SI_HI)),
        (math.log(KE_LO), math.log(KE_HI)),
        (math.log(KA_LO), math.log(KA_HI)),
    ]
    starts = [
        _project_phi_to_rates_log(np.asarray(phi_pilot, float)),
        np.array([math.log(1.0), math.log(0.026), math.log(0.022)], dtype=float),
        np.array([
            0.5 * (math.log(SI_LO) + math.log(SI_HI)),
            0.5 * (math.log(KE_LO) + math.log(KE_HI)),
            0.5 * (math.log(KA_LO) + math.log(KA_HI)),
        ], dtype=float),
    ]
    best_u = starts[0]
    best_val = float("inf")
    for s0 in starts:
        def _fun(x):
            v, g = rate_vg_jit(
                jnp.asarray(x, dtype=jnp.float64),
                cbase, carbs, fat, fiber, mask, obs_trace, sample_idx, sigma_trace, rho,
            )
            return float(v), np.asarray(g, dtype=float)
        res = minimize(_fun, s0, jac=True, method="L-BFGS-B", bounds=rate_bounds, options={"maxiter": 120})
        if float(res.fun) < best_val:
            best_val = float(res.fun)
            best_u = np.asarray(res.x, dtype=float)

    hat_si = math.exp(best_u[0])
    hat_ke = math.exp(best_u[1])
    hat_ka = math.exp(best_u[2])
    hat_tau1, hat_p = rates_to_tau_p(hat_ke, hat_ka)
    phi_hat = jnp.asarray([math.log(hat_si), math.log(hat_tau1), math.log(hat_p)], dtype=jnp.float64)
    loss_hat = float(nll_jit(phi_hat, cbase, carbs, fat, fiber, mask, obs_trace, sample_idx, sigma_trace, rho))

    lo_tau, hi_tau = math.log(TAU_RANGE[0]), math.log(TAU_RANGE[1])
    lo_p, hi_p = math.log(P_RANGE[0]), math.log(P_RANGE[1])
    int_tau = bool(min(math.log(hat_tau1) - lo_tau, hi_tau - math.log(hat_tau1)) / (hi_tau - lo_tau) > 0.01)
    int_p = bool(min(math.log(hat_p) - lo_p, hi_p - math.log(hat_p)) / (hi_p - lo_p) > 0.01)

    tau_grid_log = np.linspace(lo_tau, hi_tau, 15)
    tau_grid_nat = np.exp(tau_grid_log)
    p_bounds_for_tau = np.array([feasible_p_bounds(t) for t in tau_grid_nat], dtype=float)
    tau_losses, _ = prof_jit(
        phi_hat,
        jnp.asarray(tau_grid_log, dtype=jnp.float64),
        jnp.asarray(np.log(p_bounds_for_tau[:, 0]), dtype=jnp.float64),
        jnp.asarray(np.log(p_bounds_for_tau[:, 1]), dtype=jnp.float64),
        jnp.asarray(False),
        cbase, carbs, fat, fiber, mask, obs_trace, sample_idx, sigma_trace, rho,
    )
    tau_losses = np.asarray(tau_losses, float)

    p_grid_log = np.linspace(lo_p, hi_p, 15)
    p_grid_nat = np.exp(p_grid_log)
    tau_bounds_for_p = np.array([feasible_tau1_bounds(pv) for pv in p_grid_nat], dtype=float)
    p_losses, _ = prof_jit(
        phi_hat,
        jnp.asarray(p_grid_log, dtype=jnp.float64),
        jnp.asarray(np.log(tau_bounds_for_p[:, 0]), dtype=jnp.float64),
        jnp.asarray(np.log(tau_bounds_for_p[:, 1]), dtype=jnp.float64),
        jnp.asarray(True),
        cbase, carbs, fat, fiber, mask, obs_trace, sample_idx, sigma_trace, rho,
    )
    p_losses = np.asarray(p_losses, float)

    ref_tau = min(loss_hat, float(tau_losses.min()))
    ref_p = min(loss_hat, float(p_losses.min()))
    cl_tau = classify_profile(tau_grid_nat, tau_losses, delta=CHI2_DELTA, reference=ref_tau)
    cl_p = classify_profile(p_grid_nat, p_losses, delta=CHI2_DELTA, reference=ref_p)

    def _cov(target_val):
        in_range = TAU_RANGE[0] <= target_val <= TAU_RANGE[1]
        cov_nom = bool(in_range and (float(np.interp(target_val, tau_grid_nat, tau_losses)) - ref_tau <= CHI2_DELTA))
        cov_bd = None
        if cl_tau["verdict"] == "identifiable" and cl_tau.get("ci_low") and cl_tau.get("ci_high"):
            cov_bd = bool(cl_tau["ci_low"] <= target_val <= cl_tau["ci_high"])
        return cov_nom, cov_bd

    cov_nom_dm, cov_bd_dm = _cov(true_tau1_dm)
    cov_nom_td, cov_bd_td = _cov(true_tau1_plus_td)

    out = {
        "subject_id": sid,
        "dm_params": dm_dict,
        "true_tau1_dm": true_tau1_dm,
        "true_tau1_plus_td": true_tau1_plus_td,
        "hat_tau1": hat_tau1,
        "hat_p": hat_p,
        "hat_si": hat_si,
        "tau1_bias_min": hat_tau1 - true_tau1_dm,
        "tau1_plus_td_bias_min": hat_tau1 - true_tau1_plus_td,
        "tau1": {
            "interior": int_tau,
            "verdict": cl_tau["verdict"],
            "ci_low": cl_tau.get("ci_low"),
            "ci_high": cl_tau.get("ci_high"),
            "cov_nom_dm": cov_nom_dm,
            "cov_bd_dm": cov_bd_dm,
            "cov_nom_td": cov_nom_td,
            "cov_bd_td": cov_bd_td,
        },
        "p": {
            "interior": int_p,
            "verdict": cl_p["verdict"],
            "ci_low": cl_p.get("ci_low"),
            "ci_high": cl_p.get("ci_high"),
        },
    }
    unit_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    from evaluation.phase13_summary import _coords
    from evaluation.recalibration import _rows

    stored_real = {r["subject_id"]: r for r in _rows(_coords({}))}
    tasks = [(sid, r) for sid, r in sorted(stored_real.items())]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_solve_one, t) for t in tasks]
        for fut in as_completed(futs):
            rows.append(fut.result())
    rows.sort(key=lambda x: x["subject_id"])

    n = len(rows)
    n_int_tau = sum(r["tau1"]["interior"] for r in rows)
    bd_tau = [r for r in rows if r["tau1"]["verdict"] == "identifiable"]
    n_bd_tau_int = sum(r["tau1"]["interior"] for r in bd_tau)

    n_int_p = sum(r["p"]["interior"] for r in rows)
    bd_p = [r for r in rows if r["p"]["verdict"] == "identifiable"]
    n_bd_p_int = sum(r["p"]["interior"] for r in bd_p)

    summary = {
        "n": n,
        "true_tau1_dm_median": float(np.median([r["true_tau1_dm"] for r in rows])),
        "true_tau1_dm_iqr": [float(np.percentile([r["true_tau1_dm"] for r in rows], 25)),
                             float(np.percentile([r["true_tau1_dm"] for r in rows], 75))],
        "hat_tau1_median": float(np.median([r["hat_tau1"] for r in rows])),
        "median_bias_dm_min": float(np.median([r["tau1_bias_min"] for r in rows])),
        "median_abs_err_dm_min": float(np.median([abs(r["tau1_bias_min"]) for r in rows])),
        "median_bias_td_min": float(np.median([r["tau1_plus_td_bias_min"] for r in rows])),
        "tau1_n_interior": n_int_tau,
        "tau1_bounded_all": len(bd_tau),
        "tau1_bounded_int": n_bd_tau_int,
        "tau1_bounded_all_pct": 100.0 * len(bd_tau) / n,
        "tau1_bounded_int_pct": 100.0 * n_bd_tau_int / max(n_int_tau, 1),
        "tau1_cov_nom_dm_pct": 100.0 * sum(r["tau1"]["cov_nom_dm"] for r in rows) / n,
        "tau1_cov_bd_dm_pct": 100.0 * sum(bool(r["tau1"]["cov_bd_dm"]) for r in bd_tau) / max(len(bd_tau), 1),
        "tau1_cov_nom_td_pct": 100.0 * sum(r["tau1"]["cov_nom_td"] for r in rows) / n,
        "tau1_cov_bd_td_pct": 100.0 * sum(bool(r["tau1"]["cov_bd_td"]) for r in bd_tau) / max(len(bd_tau), 1),
        "p_n_interior": n_int_p,
        "p_bounded_all": len(bd_p),
        "p_bounded_int": n_bd_p_int,
        "p_bounded_all_pct": 100.0 * len(bd_p) / n,
        "p_bounded_int_pct": 100.0 * n_bd_p_int / max(n_int_p, 1),
        "elapsed_s": time.time() - t0,
    }
    out_path = OUT_DIR / "a27_misspecified_dalla_man.json"
    out_path.write_text(json.dumps({"summary": summary, "subjects": rows}, indent=2), encoding="utf-8")
    print(f"[a27 ({summary['elapsed_s']:.1f}s)] {json.dumps(summary, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
