"""A26: Strictly feasible original-box D full-trace optimization and profile likelihoods.

Computes exact constrained MLEs and profile likelihoods for (log tau1, log p) strictly inside the
curved feasible domain:

    D = { (tau1, p) = (1/k_e + 1/k_a, 1/(k_e k_a)) : k_e in [0.015, 0.040], k_a in [0.012, 0.032] }

using exact closed-form slices [p_min(tau1), p_max(tau1)] and [tau1_min(p), tau1_max(p)] so that
100% of MLEs, inner grid optima, and profile CI endpoints correspond to valid rate pairs inside the
original parameter box.

Uses a module-level XLA-compiled kernel cache per worker (keyed by padded width & window shape) and
a single persistent 8-worker pool across all settings for ~5x faster execution.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
UNIT_DIR = OUT_DIR / "a26_units"
UNIT_DIR.mkdir(parents=True, exist_ok=True)

KE_LO, KE_HI = PARAM_RANGES["gastric_emptying"]
KA_LO, KA_HI = PARAM_RANGES["carb_absorption"]
SI_LO, SI_HI = SI_RANGE

# Per-worker compiled kernel cache keyed by (width, n_samples, duration_min, step_min, pre0, pre1)
_KERNEL_CACHE: dict[tuple, tuple] = {}


def _get_kernels(key: tuple):
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    from evaluation.jax_config import configure
    configure()
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    from personalization import coords as co
    from simulation.jax_engine_canonical import run_meal as run_meal_canonical

    _width, _nsamp, duration_min, step_min, pre0, pre1 = key
    lo_si, hi_si = math.log(SI_LO), math.log(SI_HI)
    opt = optax.adam(0.03)

    def trace_nll(phi_log, cbase, carbs, fat, fiber, mask, obs_trace, sample_idx, sigma_trace, rho):
        si = jnp.exp(phi_log[0])
        tau1 = jnp.exp(phi_log[1])
        p = jnp.exp(phi_log[2])
        sigma1, c = co.tau_p_to_canonical(tau1, p)
        params = eqx.tree_at(lambda m: (m.insulin_sensitivity, m.sigma1, m.c), cbase, (si, sigma1, c))
        glucose = jax.vmap(
            lambda cg, ft, fb: run_meal_canonical(
                params, cg, ft, fb, meal_time=30.0, duration_min=duration_min, step_min=step_min
            )[1],
            in_axes=(0, 0, 0),
        )(carbs, fat, fiber)
        baseline = jnp.mean(glucose[:, pre0:pre1], axis=1)
        pred = glucose[:, sample_idx] - baseline[:, None]
        raw = (pred - obs_trace) * mask[:, None]
        # Prais-Winsten whitening
        head = jnp.sqrt(jnp.maximum(1.0 - rho * rho, 0.0)) * raw[:, :1]
        tail = raw[:, 1:] - rho * raw[:, :-1]
        whitened = jnp.where(rho > 0.0, jnp.concatenate([head, tail], axis=1), raw)
        r = whitened / sigma_trace
        return 0.5 * jnp.sum(r * r)

    def rate_loss_and_grad(u_log, cbase, carbs, fat, fiber, mask, obs_trace, sample_idx, sigma_trace, rho):
        def _f(u):
            ke, ka = jnp.exp(u[1]), jnp.exp(u[2])
            phi = jnp.stack([u[0], jnp.log(1.0 / ke + 1.0 / ka), jnp.log(1.0 / (ke * ka))])
            return trace_nll(phi, cbase, carbs, fat, fiber, mask, obs_trace, sample_idx, sigma_trace, rho)
        return jax.value_and_grad(_f)(u_log)

    def profile_1d(phi_hat, grid_vals, free_lo, free_hi, fixed_idx_is_p,
                   cbase, carbs, fat, fiber, mask, obs_trace, sample_idx, sigma_trace, rho):
        # fixed_idx_is_p: bool scalar (False => profiling log_tau1, True => profiling log_p)
        def one_pt(fval, flo, fhi):
            th0 = jnp.where(
                fixed_idx_is_p,
                jnp.array([jnp.clip(phi_hat[0], lo_si, hi_si), jnp.clip(phi_hat[1], flo, fhi), fval]),
                jnp.array([jnp.clip(phi_hat[0], lo_si, hi_si), fval, jnp.clip(phi_hat[2], flo, fhi)]),
            )
            gmask = jnp.where(fixed_idx_is_p, jnp.array([1.0, 1.0, 0.0]), jnp.array([1.0, 0.0, 1.0]))

            def body(carry, _):
                th, st = carry
                val, g = jax.value_and_grad(trace_nll)(
                    th, cbase, carbs, fat, fiber, mask, obs_trace, sample_idx, sigma_trace, rho
                )
                g = g * gmask
                u, st = opt.update(g, st, th)
                th = optax.apply_updates(th, u)
                th = jnp.where(
                    fixed_idx_is_p,
                    jnp.array([jnp.clip(th[0], lo_si, hi_si), jnp.clip(th[1], flo, fhi), fval]),
                    jnp.array([jnp.clip(th[0], lo_si, hi_si), fval, jnp.clip(th[2], flo, fhi)]),
                )
                return (th, st), val

            (th_end, _), _ = jax.lax.scan(body, (th0, opt.init(th0)), None, length=100)
            end_loss = trace_nll(th_end, cbase, carbs, fat, fiber, mask, obs_trace, sample_idx, sigma_trace, rho)
            return end_loss, th_end

        return jax.vmap(one_pt)(grid_vals, free_lo, free_hi)

    kernels = (
        jax.jit(trace_nll),
        jax.jit(rate_loss_and_grad),
        jax.jit(profile_1d),
    )
    _KERNEL_CACHE[key] = kernels
    return kernels


def _project_phi_to_rates_log(phi_log: np.ndarray) -> np.ndarray:
    lsi = float(np.clip(phi_log[0], math.log(SI_LO), math.log(SI_HI)))
    tau1 = float(np.clip(math.exp(phi_log[1]), TAU_RANGE[0], TAU_RANGE[1]))
    plo, phi = feasible_p_bounds(tau1)
    p = float(np.clip(math.exp(phi_log[2]), plo, phi))
    disc = max(tau1 * tau1 - 4.0 * p, 0.0)
    r1 = (tau1 + math.sqrt(disc)) / (2.0 * p)
    r2 = (tau1 - math.sqrt(disc)) / (2.0 * p)
    ke1, ka1 = float(np.clip(r1, KE_LO, KE_HI)), float(np.clip(r2, KA_LO, KA_HI))
    ke2, ka2 = float(np.clip(r2, KE_LO, KE_HI)), float(np.clip(r1, KA_LO, KA_HI))
    if abs(ke1 - r1) + abs(ka1 - r2) <= abs(ke2 - r2) + abs(ka2 - r1):
        ke, ka = ke1, ka1
    else:
        ke, ka = ke2, ka2
    return np.array([lsi, math.log(ke), math.log(ka)], dtype=float)


def _solve_one(args: tuple[str, dict, str, dict | None]) -> dict:
    setting, cfg, sid, stored = args
    unit_file = UNIT_DIR / f"{setting}__{sid}.json"
    if unit_file.exists():
        return json.loads(unit_file.read_text(encoding="utf-8"))

    from evaluation.jax_config import configure
    configure()
    import jax
    import jax.numpy as jnp
    import optax
    from scipy.optimize import minimize
    from evaluation.identifiability_tools import CHI2_DELTA, classify_profile
    from evaluation.subject_source import get_subject, window_for
    from personalization import noise_model as nm
    from personalization.objectives import ObjectiveSpec, build_objective, observed_values
    from personalization.subject_loss import base_params, subject_arrays
    from simulation.jax_engine_canonical import CanonicalParams
    from simulation.jax_observables import post_indices

    subj = get_subject(cfg, sid)
    win = window_for(subj)
    spec = ObjectiveSpec(
        name="trace", lam=0.0, normalization="nll", bounds_scale=1.0,
        window=win, parameterization="coords",
    )
    records = list(subj.records)
    arrays = subject_arrays(records)
    obs = observed_values(records, win)
    base = base_params(subj.profile)
    cbase = CanonicalParams.from_jax_params(base)
    names = list(spec.theta_names)

    # 1. Exact frozen noise model (identical to stored phase13 / a23)
    raw = build_objective(spec, base, arrays, obs, noise=None)
    if stored is not None:
        phi_pilot = jnp.asarray([stored["theta_ml"][n] for n in names], dtype=jnp.float64)
    else:
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

    # 2. Exact MLE strictly inside original rate box ([SI_LO, SI_HI] x D) via 3-start L-BFGS-B
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
    assert in_feasible_domain(hat_tau1, hat_p), f"MLE outside D: {hat_tau1}, {hat_p}"

    lo_tau, hi_tau = math.log(TAU_RANGE[0]), math.log(TAU_RANGE[1])
    lo_p, hi_p = math.log(P_RANGE[0]), math.log(P_RANGE[1])
    dist_tau = min(math.log(hat_tau1) - lo_tau, hi_tau - math.log(hat_tau1)) / (hi_tau - lo_tau)
    dist_p = min(math.log(hat_p) - lo_p, hi_p - math.log(hat_p)) / (hi_p - lo_p)
    int_tau = bool(dist_tau > 0.01)
    int_p = bool(dist_p > 0.01)

    # 3. Exact D-constrained profile for log_tau1
    tau_grid_log = np.linspace(lo_tau, hi_tau, 15)
    tau_grid_nat = np.exp(tau_grid_log)
    p_bounds_for_tau = np.array([feasible_p_bounds(t) for t in tau_grid_nat], dtype=float)
    tau_losses, tau_thetas = prof_jit(
        phi_hat,
        jnp.asarray(tau_grid_log, dtype=jnp.float64),
        jnp.asarray(np.log(p_bounds_for_tau[:, 0]), dtype=jnp.float64),
        jnp.asarray(np.log(p_bounds_for_tau[:, 1]), dtype=jnp.float64),
        jnp.asarray(False),
        cbase, carbs, fat, fiber, mask, obs_trace, sample_idx, sigma_trace, rho,
    )
    tau_losses = np.asarray(tau_losses, float)
    tau_thetas = np.asarray(tau_thetas, float)

    # 4. Exact D-constrained profile for log_p
    p_grid_log = np.linspace(lo_p, hi_p, 15)
    p_grid_nat = np.exp(p_grid_log)
    tau_bounds_for_p = np.array([feasible_tau1_bounds(pv) for pv in p_grid_nat], dtype=float)
    p_losses, p_thetas = prof_jit(
        phi_hat,
        jnp.asarray(p_grid_log, dtype=jnp.float64),
        jnp.asarray(np.log(tau_bounds_for_p[:, 0]), dtype=jnp.float64),
        jnp.asarray(np.log(tau_bounds_for_p[:, 1]), dtype=jnp.float64),
        jnp.asarray(True),
        cbase, carbs, fat, fiber, mask, obs_trace, sample_idx, sigma_trace, rho,
    )
    p_losses = np.asarray(p_losses, float)
    p_thetas = np.asarray(p_thetas, float)

    n_infeas = sum(
        0 if in_feasible_domain(math.exp(th[1]), math.exp(th[2])) else 1
        for th in np.vstack([tau_thetas, p_thetas])
    )

    ref_tau = min(loss_hat, float(tau_losses.min()))
    ref_p = min(loss_hat, float(p_losses.min()))
    cl_tau = classify_profile(tau_grid_nat, tau_losses, delta=CHI2_DELTA, reference=ref_tau)
    cl_p = classify_profile(p_grid_nat, p_losses, delta=CHI2_DELTA, reference=ref_p)

    def _eval_truth(pinfo_key, grid_nat, losses, ref, cl):
        if stored is None:
            return None, None, None
        truth = stored["profiles"][pinfo_key].get("truth")
        if truth is None:
            return None, None, None
        exc = float(np.interp(truth, grid_nat, losses)) - ref
        cov_nom = bool(exc <= CHI2_DELTA)
        cov_bd = None
        if cl["verdict"] == "identifiable" and cl.get("ci_low") and cl.get("ci_high"):
            cov_bd = bool(cl["ci_low"] <= truth <= cl["ci_high"])
        return float(truth), cov_nom, cov_bd

    t_tau, cov_nom_tau, cov_bd_tau = _eval_truth("log_tau1", tau_grid_nat, tau_losses, ref_tau, cl_tau)
    t_p, cov_nom_p, cov_bd_p = _eval_truth("log_p", p_grid_nat, p_losses, ref_p, cl_p)

    def _check_ci_in_D(cl, grid_log, thetas, fixed_idx):
        if cl["verdict"] != "identifiable" or not cl.get("ci_low") or not cl.get("ci_high"):
            return None
        ok = []
        for ep in (cl["ci_low"], cl["ci_high"]):
            if fixed_idx == 1:
                p_ep = math.exp(float(np.interp(math.log(ep), grid_log, thetas[:, 2])))
                ok.append(in_feasible_domain(ep, p_ep))
            else:
                t_ep = math.exp(float(np.interp(math.log(ep), grid_log, thetas[:, 1])))
                ok.append(in_feasible_domain(t_ep, ep))
        return all(ok)

    res = {
        "setting": setting,
        "subject_id": sid,
        "loss_hat": loss_hat,
        "hat_tau1": hat_tau1,
        "hat_p": hat_p,
        "hat_si": hat_si,
        "n_infeasible_grid_points": int(n_infeas),
        "tau1": {
            "interior": int_tau,
            "verdict": cl_tau["verdict"],
            "ci_low": cl_tau.get("ci_low"),
            "ci_high": cl_tau.get("ci_high"),
            "truth": t_tau,
            "cov_nom": cov_nom_tau,
            "cov_bd": cov_bd_tau,
            "both_ep_in_D": _check_ci_in_D(cl_tau, tau_grid_log, tau_thetas, 1),
            "log_width": (math.log(cl_tau["ci_high"] / cl_tau["ci_low"])
                          if cl_tau["verdict"] == "identifiable" and cl_tau.get("ci_low") and cl_tau.get("ci_high")
                          else None),
        },
        "p": {
            "interior": int_p,
            "verdict": cl_p["verdict"],
            "ci_low": cl_p.get("ci_low"),
            "ci_high": cl_p.get("ci_high"),
            "truth": t_p,
            "cov_nom": cov_nom_p,
            "cov_bd": cov_bd_p,
            "both_ep_in_D": _check_ci_in_D(cl_p, p_grid_log, p_thetas, 2),
            "log_width": (math.log(cl_p["ci_high"] / cl_p["ci_low"])
                          if cl_p["verdict"] == "identifiable" and cl_p.get("ci_low") and cl_p.get("ci_high")
                          else None),
        },
    }
    unit_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def _summarize(rows: list[dict], coord: str) -> dict:
    n = len(rows)
    n_int = sum(r[coord]["interior"] for r in rows)
    bd_rows = [r for r in rows if r[coord]["verdict"] == "identifiable"]
    n_bd = len(bd_rows)
    n_bd_int = sum(r[coord]["interior"] for r in bd_rows)
    n_os = sum(r[coord]["verdict"] == "one-sided" for r in rows)
    n_flat = sum(r[coord]["verdict"] == "flat" for r in rows)
    both_in_d = sum(bool(r[coord]["both_ep_in_D"]) for r in bd_rows)
    cov_nom_vals = [r[coord]["cov_nom"] for r in rows if r[coord]["cov_nom"] is not None]
    cov_bd_vals = [r[coord]["cov_bd"] for r in bd_rows if r[coord]["cov_bd"] is not None]
    widths = [r[coord]["log_width"] for r in bd_rows if r[coord]["log_width"] is not None]
    return {
        "n": n,
        "n_interior": n_int,
        "bounded_all": n_bd,
        "bounded_int": n_bd_int,
        "bounded_all_pct": 100.0 * n_bd / max(n, 1),
        "bounded_int_pct": 100.0 * n_bd_int / max(n_int, 1),
        "one_sided": n_os,
        "flat": n_flat,
        "both_ep_in_D": both_in_d,
        "both_ep_in_D_pct": 100.0 * both_in_d / max(n_bd, 1),
        "cov_nom_pct": (100.0 * sum(cov_nom_vals) / len(cov_nom_vals)) if cov_nom_vals else None,
        "cov_bd_pct": (100.0 * sum(cov_bd_vals) / len(cov_bd_vals)) if cov_bd_vals else None,
        "median_log_width": float(np.median(widths)) if widths else None,
        "total_infeasible_grid_points": int(sum(r["n_infeasible_grid_points"] for r in rows)),
    }


def main() -> int:
    from evaluation.cohort_data import load_cohort
    from evaluation.phase13_summary import _coords
    from evaluation.recalibration import _rows

    settings = [
        ("cgmacros_real", {"cohort": "cgmacros", "min_meals": 10, "limit": None, "replica": None}, {}),
        ("hall_real", {"cohort": "hall", "min_meals": 5, "limit": None, "replica": None}, None),
        ("replica_cv0", {"cohort": "cgmacros", "min_meals": 10, "limit": None, "replica": {"seed": 0, "cgm": True, "carb_cv": 0.0}}, {"replica": {"seed": 0, "cgm": True, "carb_cv": 0.0}}),
        ("replica_cv025", {"cohort": "cgmacros", "min_meals": 10, "limit": None, "replica": {"seed": 0, "cgm": True, "carb_cv": 0.25}}, {"replica": {"seed": 0, "cgm": True, "carb_cv": 0.25}}),
        ("replica_cv075", {"cohort": "cgmacros", "min_meals": 10, "limit": None, "replica": {"seed": 0, "cgm": True, "carb_cv": 0.75}}, {"replica": {"seed": 0, "cgm": True, "carb_cv": 0.75}}),
    ]

    all_tasks = []
    setting_order = [s[0] for s in settings]
    for name, cfg, extra in settings:
        if extra is not None:
            stored_rows = _rows(_coords(extra))
            for r in stored_rows:
                all_tasks.append((name, cfg, r["subject_id"], r))
        else:
            subjs = load_cohort(cfg["cohort"], min_meals=cfg["min_meals"])
            for s in subjs:
                all_tasks.append((name, cfg, s.subject_id, None))

    t0 = time.time()
    by_setting: dict[str, list[dict]] = {k: [] for k in setting_order}
    done = 0
    total = len(all_tasks)
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_solve_one, t) for t in all_tasks]
        for fut in as_completed(futs):
            res = fut.result()
            by_setting[res["setting"]].append(res)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"[a26] progress {done}/{total} ({time.time() - t0:.1f}s)", flush=True)

    out = {}
    for name in setting_order:
        rows = sorted(by_setting[name], key=lambda x: x["subject_id"])
        s_tau = _summarize(rows, "tau1")
        s_p = _summarize(rows, "p")
        out[name] = {"tau1": s_tau, "p": s_p, "subjects": rows}
        print(
            f"[a26 {name}] "
            f"tau1: {s_tau['bounded_int']}/{s_tau['n_interior']} ({s_tau['bounded_int_pct']:.0f}% int, {s_tau['bounded_all']}/{s_tau['n']} all, cov_nom={s_tau['cov_nom_pct']}, cov_bd={s_tau['cov_bd_pct']}) | "
            f"p: {s_p['bounded_int']}/{s_p['n_interior']} ({s_p['bounded_int_pct']:.0f}% int, {s_p['bounded_all']}/{s_p['n']} all, cov_nom={s_p['cov_nom_pct']}, cov_bd={s_p['cov_bd_pct']}) | "
            f"infeas={s_tau['total_infeasible_grid_points']}",
            flush=True,
        )

    out_path = OUT_DIR / "a26_exact_domain.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[a26] wrote {out_path} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
