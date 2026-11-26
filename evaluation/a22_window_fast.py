"""Fast, resumable, multi-worker runner for A22 (H40): CGMacros iAUC profile likelihood across post-meal windows (90, 120, 60 min).

Mathematically identical to `profile_lik` (unregularized MLE, frozen pilot sigma, 15 log-spaced grid points,
100 Adam steps at lr=0.03 in log coordinates, CHI2_DELTA=1.92072941). Checkpoints every subject to
`results/phase20_amendment16/a22_units/{w_min}/{subject_id}.json` so interrupted runs resume instantly.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np

# Restrict per-worker thread oversubscription when running multi-process
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

from evaluation.results_io import RESULTS  # noqa: E402

OUT_DIR = RESULTS / "phase20_amendment16"
UNITS_DIR = OUT_DIR / "a22_units"

_KERNELS = {}


def _get_kernels(post_min: float):
    key = int(round(post_min))
    if key in _KERNELS:
        return _KERNELS[key]

    from evaluation.jax_config import configure
    configure()
    import jax
    import jax.numpy as jnp
    import optax
    from personalization.subject_loss import LOWER, UPPER, build_params
    from simulation.jax_engine import run_meal
    from simulation.jax_observables import Window, iauc

    win = Window(post_min=float(post_min), stride=1)
    dur = float(win.meal_time_min + win.post_min)
    step = float(win.step_min)
    beta = 10.0

    def _sim_iaucs(base, theta, carbs, fat, fiber):
        params = build_params(base, theta)
        g = jax.vmap(lambda c, f, b: run_meal(params, c, f, b, meal_time=30.0, duration_min=dur, step_min=step)[1])(carbs, fat, fiber)
        return jax.vmap(lambda gi: iauc(None, gi, win, beta))(g)

    def _mse_loss(base, theta, carbs, fat, fiber, mask, obs, n_meals):
        pred = _sim_iaucs(base, theta, carbs, fat, fiber)
        return jnp.sum((mask * (pred - obs)) ** 2) / n_meals

    def _nll_loss_log(base, phi, carbs, fat, fiber, mask, obs, sigma):
        pred = _sim_iaucs(base, jnp.exp(phi), carbs, fat, fiber)
        r = mask * (pred - obs) / sigma
        return 0.5 * jnp.sum(r ** 2)

    @jax.jit
    def fit_adam_rate(base, theta0, carbs, fat, fiber, mask, obs, n_meals, n_steps):
        opt = optax.adam(0.02)
        state = opt.init(theta0)

        def body(_, carry):
            th, st = carry
            _, grad = jax.value_and_grad(lambda t: _mse_loss(base, t, carbs, fat, fiber, mask, obs, n_meals))(th)
            updates, st = opt.update(grad, st, th)
            th = jnp.clip(optax.apply_updates(th, updates), LOWER, UPPER)
            return th, st

        th_final, _ = jax.lax.fori_loop(0, n_steps, body, (theta0, state))
        pred = _sim_iaucs(base, th_final, carbs, fat, fiber)
        return th_final, pred

    @jax.jit
    def eval_nll(base, phi, carbs, fat, fiber, mask, obs, sigma):
        return _nll_loss_log(base, phi, carbs, fat, fiber, mask, obs, sigma)

    @lru_cache(maxsize=4)
    def make_profile_kernel(idx: int):
        phi_lo = jnp.log(LOWER)
        phi_hi = jnp.log(UPPER)
        opt = optax.adam(0.03)

        def solve_one(base, phi_hat, grid_val, carbs, fat, fiber, mask, obs, sigma):
            phi0 = phi_hat.at[idx].set(grid_val)
            state = opt.init(phi0)

            def body(_, carry):
                p, st = carry
                _, grad = jax.value_and_grad(lambda x: _nll_loss_log(base, x, carbs, fat, fiber, mask, obs, sigma))(p)
                grad = grad.at[idx].set(0.0)
                updates, st = opt.update(grad, st, p)
                p = jnp.clip(optax.apply_updates(p, updates), phi_lo, phi_hi).at[idx].set(grid_val)
                return p, st

            p_end, _ = jax.lax.fori_loop(0, 100, body, (phi0, state))
            return _nll_loss_log(base, p_end, carbs, fat, fiber, mask, obs, sigma)

        return jax.jit(jax.vmap(solve_one, in_axes=(None, None, 0, None, None, None, None, None, None)))

    _KERNELS[key] = (win, fit_adam_rate, eval_nll, make_profile_kernel)
    return _KERNELS[key]


def _run_subject_worker(args: tuple[float, str]) -> dict:
    w_min, sid = args
    unit_dir = UNITS_DIR / str(int(round(w_min)))
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / f"{sid}.json"
    if unit_path.exists():
        return json.loads(unit_path.read_text(encoding="utf-8"))

    win, fit_adam_rate, eval_nll, make_profile_kernel = _get_kernels(w_min)
    import jax.numpy as jnp
    from evaluation.cohort_data import load_cohort
    from evaluation.identifiability_tools import CHI2_DELTA, classify_profile
    from evaluation.profile_lik import log_grid
    from personalization.objectives import observed_values
    from personalization.subject_loss import LOWER, TARGETS, UPPER, base_params, subject_arrays, theta_of

    subjects = load_cohort("cgmacros", min_meals=10)
    subj = next(s for s in subjects if s.subject_id == sid)
    lo_np = np.asarray(LOWER, dtype=float)
    hi_np = np.asarray(UPPER, dtype=float)
    phi_lo_np, phi_hi_np = np.log(lo_np), np.log(hi_np)

    records = list(subj.records)
    base = base_params(subj.profile)
    arrays = subject_arrays(records)
    obs_dict = observed_values(records, win)
    n_real = int(arrays["n"])
    width = int(arrays["width"])
    obs_pad = np.zeros(width, dtype=float)
    obs_pad[:n_real] = obs_dict["iauc"][:n_real]

    carbs, fat, fiber, mask = arrays["carbs"], arrays["fat"], arrays["fiber"], arrays["mask"]
    obs_j = jnp.asarray(obs_pad)
    theta0 = theta_of(base)

    _, pred_pilot = fit_adam_rate(base, theta0, carbs, fat, fiber, mask, obs_j, float(n_real), 300)
    resid_pilot = np.asarray(pred_pilot)[:n_real] - obs_dict["iauc"][:n_real]
    sigma = max(float(np.sqrt(np.sum(resid_pilot ** 2) / max(n_real - 3, 1))), 1e-6)

    th_hat, _ = fit_adam_rate(base, theta0, carbs, fat, fiber, mask, obs_j, float(n_real), 600)
    theta_hat = np.asarray(th_hat, dtype=float)
    phi_hat = np.log(theta_hat)
    phi_hat_j = jnp.asarray(phi_hat)
    loss_at_hat = float(eval_nll(base, phi_hat_j, carbs, fat, fiber, mask, obs_j, sigma))

    profs, dists = {}, {}
    for idx, pname in enumerate(TARGETS):
        dists[pname] = float(min(phi_hat[idx] - phi_lo_np[idx], phi_hi_np[idx] - phi_hat[idx]) / (phi_hi_np[idx] - phi_lo_np[idx]))
        grid_nat, _ = log_grid(lo_np[idx], hi_np[idx], 15, extension=0)
        kern = make_profile_kernel(idx)
        prof = np.asarray(kern(base, phi_hat_j, jnp.asarray(np.log(grid_nat)), carbs, fat, fiber, mask, obs_j, sigma), dtype=float)
        ref = min(loss_at_hat, float(prof.min()))
        cl = classify_profile(grid_nat, prof, delta=CHI2_DELTA, reference=ref)
        profs[pname] = {
            "verdict": cl["verdict"],
            "theta_hat": float(theta_hat[idx]),
            "max_rise": cl.get("max_rise"),
            "interior": bool(dists[pname] > 0.01),
        }

    row = {"subject_id": sid, "sigma": sigma, "profiles": profs, "distance_to_bound": dists}
    unit_path.write_text(json.dumps(row, indent=2, default=float), encoding="utf-8")
    return row


def run_window(w_min: float, workers: int = 8) -> dict:
    from evaluation.cohort_data import load_cohort

    t0 = time.time()
    subjects = load_cohort("cgmacros", min_meals=10)
    sids = [s.subject_id for s in subjects]
    rows_by_id = {}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_subject_worker, (w_min, sid)): sid for sid in sids}
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            rows_by_id[row["subject_id"]] = row
            done += 1
            if done % 9 == 0 or done == 1 or done == len(sids):
                print(f"[a22 T={int(w_min)}m] {done}/{len(sids)} done ({time.time()-t0:.1f}s)", flush=True)

    rows = [rows_by_id[sid] for sid in sids]
    summ = {"window_min": int(w_min), "n": len(rows), "elapsed_s": time.time() - t0}
    for pname, short in (("insulin_sensitivity", "S_I"), ("gastric_emptying", "k_e"), ("carb_absorption", "k_a")):
        n_int = sum(r["profiles"][pname]["interior"] for r in rows)
        n_bd = sum(r["profiles"][pname]["verdict"] == "identifiable" for r in rows)
        n_os = sum(r["profiles"][pname]["verdict"] == "one-sided" for r in rows)
        n_fl = sum(r["profiles"][pname]["verdict"] == "flat" for r in rows)
        summ[short] = {
            "interior": int(n_int),
            "bounded": int(n_bd),
            "bounded_all_pct": 100.0 * n_bd / len(rows),
            "bounded_int_pct": (100.0 * n_bd / n_int) if n_int else 0.0,
            "one_sided": int(n_os),
            "flat": int(n_fl),
        }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"a22_window_{int(w_min)}.json"
    p.write_text(json.dumps({"summary": summ, "subjects": rows}, indent=2, default=float), encoding="utf-8")
    print(f"[a22 T={int(w_min)}m DONE in {time.time()-t0:.1f}s] "
          f"SI={summ['S_I']['bounded']}/{summ['S_I']['interior']} int ({summ['S_I']['bounded_all_pct']:.0f}% all), "
          f"ke={summ['k_e']['bounded']}/{summ['k_e']['interior']} int, "
          f"ka={summ['k_a']['bounded']}/{summ['k_a']['interior']} int", flush=True)
    return summ


if __name__ == "__main__":
    wins = [float(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [90.0, 120.0, 60.0]
    for w in wins:
        run_window(w, workers=8)
