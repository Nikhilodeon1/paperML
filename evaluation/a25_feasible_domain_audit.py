"""A25: Real-pole-enforced log_p profiling and exact feasible-domain D endpoint audit.

Audits the transformed-coordinate trace profiles (log S_I, log tau1, log p) for:
1. Exact original-box feasible-domain D = {(tau1, p) : {k_e, k_a} in [0.015, 0.040] x [0.012, 0.032]}
   endpoint validity for stored log_tau1 profiles (which already satisfy p <= tau1^2/4 at 100% of points).
2. Recomputing log_p profiles with proper real-pole projection enforced during inner optimization
   (log_tau1 >= 0.5 * (fixed_log_p + log 4)), fixing the projection overwrite bug where clamping log_p
   was undone by resetting the fixed grid coordinate.
"""
from __future__ import annotations

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

OUT_DIR = RESULTS / "phase20_amendment16"
LOG_FOUR = math.log(4.0)
KE_LO, KE_HI = 0.015, 0.040
KA_LO, KA_HI = 0.012, 0.032


def _in_feasible_D(tau1: float, p: float, tol: float = 1e-5) -> bool:
    disc = tau1 * tau1 - 4.0 * p
    if disc < -tol * tau1 * tau1:
        return False
    sq = math.sqrt(max(disc, 0.0))
    r1 = (tau1 + sq) / (2.0 * p)
    r2 = (tau1 - sq) / (2.0 * p)
    c1 = (KE_LO - tol <= r1 <= KE_HI + tol) and (KA_LO - tol <= r2 <= KA_HI + tol)
    c2 = (KE_LO - tol <= r2 <= KE_HI + tol) and (KA_LO - tol <= r1 <= KA_HI + tol)
    return bool(c1 or c2)


def _audit_tau1_rows(rows: list[dict]) -> dict:
    int_n = 0
    bd = 0
    hat_in_d = 0
    both_ep_in_d = 0
    ep_out = 0
    ep_tot = 0
    for r in rows:
        if r["distance_to_bound"]["log_tau1"] > 0.01:
            int_n += 1
        pinfo = r["profiles"]["log_tau1"]
        cl = pinfo["classification"]
        if cl["verdict"] != "identifiable":
            continue
        bd += 1
        names = list(r["theta_ml"].keys())
        i_p = names.index("log_p")
        tau_hat = float(np.exp(r["theta_ml"]["log_tau1"]))
        p_hat = float(np.exp(r["theta_ml"]["log_p"]))
        if _in_feasible_D(tau_hat, p_hat):
            hat_in_d += 1
        grid = np.asarray(pinfo["grid"], float)[: pinfo["n_in_box"]]
        inner = np.asarray(pinfo["inner_theta"], float)[: pinfo["n_in_box"]]
        log_g = np.log(grid)
        ok = []
        for ep in (cl["ci_low"], cl["ci_high"]):
            ep_tot += 1
            p_ep = float(np.exp(np.interp(np.log(ep), log_g, inner[:, i_p])))
            f = _in_feasible_D(ep, p_ep)
            ok.append(f)
            if not f:
                ep_out += 1
        if all(ok):
            both_ep_in_d += 1
    return {
        "n": len(rows),
        "n_interior": int_n,
        "rect_bounded": bd,
        "rect_bounded_int_pct": 100.0 * bd / max(int_n, 1),
        "hat_in_D": hat_in_d,
        "both_ep_in_D": both_ep_in_d,
        "both_ep_in_D_of_bd_pct": 100.0 * both_ep_in_d / max(bd, 1),
        "both_ep_in_D_of_int_pct": 100.0 * both_ep_in_d / max(int_n, 1),
        "ep_outside_D": ep_out,
        "ep_total": ep_tot,
        "ep_outside_D_pct": 100.0 * ep_out / max(ep_tot, 1),
    }


def _solve_one_p(args: tuple[str, dict, dict]) -> dict:
    setting, cfg, stored = args
    from evaluation.jax_config import configure
    configure()
    import jax
    import jax.numpy as jnp
    import optax
    from evaluation.identifiability_tools import CHI2_DELTA, classify_profile
    from evaluation.subject_source import get_subject, window_for
    from personalization import noise_model as nm
    from personalization.objectives import ObjectiveSpec, build_objective, observed_values
    from personalization.subject_loss import base_params, subject_arrays

    sid = stored["subject_id"]
    subj = get_subject(cfg, sid)
    win = window_for(subj)
    spec = ObjectiveSpec(name="trace", lam=0.0, normalization="nll", bounds_scale=1.0, window=win, parameterization="coords")
    records = list(subj.records)
    arrays = subject_arrays(records)
    obs = observed_values(records, win)
    names = list(spec.theta_names)
    phi_hat = jnp.asarray([stored["theta_ml"][n] for n in names], dtype=float)
    raw = build_objective(spec, base_params(subj.profile), arrays, obs, noise=None)
    noise = nm.build(raw.residuals(phi_hat), n_parameters=len(spec.free), whiten_trace=spec.whiten_trace)
    obj = build_objective(spec, base_params(subj.profile), arrays, obs, noise=noise)
    lo, hi = obj.lower, obj.upper
    pinfo = stored["profiles"]["log_p"]
    grid_nat = np.asarray(pinfo["grid"], float)[: pinfo["n_in_box"]]
    grid_log = jnp.asarray(np.log(grid_nat))
    opt = optax.adam(0.03)

    def solve_pt(fixed_lp):
        min_ltau = 0.5 * (fixed_lp + LOG_FOUR)
        th0 = phi_hat.at[2].set(fixed_lp).at[1].set(jnp.maximum(phi_hat[1], min_ltau))
        th0 = jnp.clip(th0, lo, hi).at[2].set(fixed_lp).at[1].set(jnp.maximum(th0[1], min_ltau))

        def body(carry, _):
            th, st = carry
            val, g = jax.value_and_grad(obj.loss)(th)
            g = g.at[2].set(0.0)
            upd, st = opt.update(g, st, th)
            th = jnp.clip(optax.apply_updates(th, upd), lo, hi)
            th = th.at[2].set(fixed_lp).at[1].set(jnp.maximum(th[1], min_ltau))
            return (th, st), val

        (th_end, _), _ = jax.lax.scan(body, (th0, opt.init(th0)), None, length=100)
        return obj.loss(th_end), th_end

    losses, thetas = jax.jit(jax.vmap(solve_pt))(grid_log)
    losses = np.asarray(losses, float)
    thetas = np.asarray(thetas, float)
    loss_at_hat = float(obj.loss(phi_hat))
    ref = min(loss_at_hat, float(losses.min()))
    cl = classify_profile(grid_nat, losses, delta=CHI2_DELTA, reference=ref)

    truth = pinfo.get("truth")
    cov_nom = None
    cov_bd = None
    if truth is not None:
        exc = float(np.interp(truth, grid_nat, losses)) - ref
        cov_nom = bool(exc <= CHI2_DELTA)
        if cl["verdict"] == "identifiable" and cl.get("ci_low") and cl.get("ci_high"):
            cov_bd = bool(cl["ci_low"] <= truth <= cl["ci_high"])

    both_ep_in_d = False
    ep_out = 0
    if cl["verdict"] == "identifiable" and cl.get("ci_low") and cl.get("ci_high"):
        log_g = np.log(grid_nat)
        ok = []
        for ep in (cl["ci_low"], cl["ci_high"]):
            t_ep = float(np.exp(np.interp(np.log(ep), log_g, thetas[:, 1])))
            f = _in_feasible_D(t_ep, ep)
            ok.append(f)
            if not f:
                ep_out += 1
        both_ep_in_d = all(ok)

    max_viol = float(np.max(np.exp(thetas[:, 2]) - (np.exp(thetas[:, 1]) ** 2) / 4.0))
    return {
        "subject_id": sid,
        "interior": bool(stored["distance_to_bound"]["log_p"] > 0.01),
        "old_verdict": pinfo["classification"]["verdict"],
        "new_verdict": cl["verdict"],
        "ci_low": cl.get("ci_low"),
        "ci_high": cl.get("ci_high"),
        "cov_nom": cov_nom,
        "cov_bd": cov_bd,
        "both_ep_in_D": both_ep_in_d,
        "ep_out_D": ep_out,
        "max_real_pole_viol": max_viol,
    }


def main() -> int:
    from evaluation.phase13_summary import _coords
    from evaluation.recalibration import _rows

    settings = [
        ("cgmacros_real", {}, None),
        ("replica_cv0", {"replica": {"seed": 0, "cgm": True, "carb_cv": 0.0}}, {"seed": 0, "cgm": True, "carb_cv": 0.0}),
        ("replica_cv025", {"replica": {"seed": 0, "cgm": True, "carb_cv": 0.25}}, {"seed": 0, "cgm": True, "carb_cv": 0.25}),
        ("replica_cv075", {"replica": {"seed": 0, "cgm": True, "carb_cv": 0.75}}, {"seed": 0, "cgm": True, "carb_cv": 0.75}),
    ]
    out = {}
    for name, extra, rep in settings:
        t0 = time.time()
        rows = _rows(_coords(extra))
        tau_audit = _audit_tau1_rows(rows)
        cfg = {"cohort": "cgmacros", "min_meals": 10, "limit": None, "replica": rep}
        p_rows = []
        with ProcessPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(_solve_one_p, (name, cfg, r)) for r in rows]
            for fut in as_completed(futs):
                p_rows.append(fut.result())
        n_int = sum(x["interior"] for x in p_rows)
        old_bd = sum(x["old_verdict"] == "identifiable" for x in p_rows)
        new_bd = sum(x["new_verdict"] == "identifiable" for x in p_rows)
        both_d = sum(x["both_ep_in_D"] for x in p_rows)
        ep_out = sum(x["ep_out_D"] for x in p_rows)
        cov_nom = [x["cov_nom"] for x in p_rows if x["cov_nom"] is not None]
        cov_bd = [x["cov_bd"] for x in p_rows if x["cov_bd"] is not None]
        p_audit = {
            "n": len(p_rows),
            "n_interior": n_int,
            "old_bounded": old_bd,
            "old_bounded_int_pct": 100.0 * old_bd / max(n_int, 1),
            "real_pole_bounded": new_bd,
            "real_pole_bounded_all_pct": 100.0 * new_bd / len(p_rows),
            "real_pole_bounded_int_pct": 100.0 * new_bd / max(n_int, 1),
            "both_ep_in_D": both_d,
            "both_ep_in_D_of_bd_pct": 100.0 * both_d / max(new_bd, 1),
            "ep_outside_D": ep_out,
            "ep_outside_D_pct": 100.0 * ep_out / max(2 * new_bd, 1),
            "cov_nom_pct": 100.0 * float(np.mean(cov_nom)) if cov_nom else None,
            "cov_bd_pct": 100.0 * float(np.mean(cov_bd)) if cov_bd else None,
            "max_real_pole_viol": float(max(x["max_real_pole_viol"] for x in p_rows)),
            "elapsed_s": time.time() - t0,
        }
        out[name] = {"tau1": tau_audit, "p_real_pole": p_audit, "subjects": p_rows}
        print(f"[a25 {name} ({time.time()-t0:.1f}s)] "
              f"tau1 both_ep_in_D={tau_audit['both_ep_in_D']}/{tau_audit['rect_bounded']} ({tau_audit['both_ep_in_D_of_bd_pct']:.0f}%) | "
              f"p old_bd={old_bd} -> real_pole_bd={new_bd}/{n_int} ({p_audit['real_pole_bounded_int_pct']:.0f}% int), "
              f"both_ep_in_D={both_d}/{new_bd} ({p_audit['both_ep_in_D_of_bd_pct']:.0f}%), "
              f"cov_nom={p_audit['cov_nom_pct']}, cov_bd={p_audit['cov_bd_pct']}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "a25_feasible_domain.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"[a25] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
