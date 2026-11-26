"""Phase 20 (Amendment 16): R25 + A21..A24 analyses for TMLR revision.

Subcommands:
  python -m evaluation.phase20_amendment16 r25   # Stored-profile 6->28 diagnostic (fixed subjects + shift)
  python -m evaluation.phase20_amendment16 a21   # H39: Negative control for iAUC prediction power
  python -m evaluation.phase20_amendment16 a22   # H40: CGMacros iAUC window sweep (60, 90, 120, 180 min)
  python -m evaluation.phase20_amendment16 a23   # H41: Hall (n=15) exploratory full-trace coordinate profiles
  python -m evaluation.phase20_amendment16 a24   # H42: Meal-clustered sandwich variance remedy (replica seed 0)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from evaluation.results_io import RESULTS

OUT_DIR = RESULTS / "phase20_amendment16"


def _save(name: str, payload: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"[{name}] wrote {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# R25: Zero-compute stored-profile diagnostic (6/45 -> 28/45 mechanism)
# ---------------------------------------------------------------------------
def run_r25() -> dict:
    from evaluation.phase13_summary import _coords
    from evaluation.recalibration import _rows

    cvs = (0.0, 0.1, 0.25, 0.5, 0.75)
    by_cv = {
        cv: {r["subject_id"]: r for r in _rows(_coords({"replica": {"seed": 0, "cgm": True, "carb_cv": cv}}))}
        for cv in cvs
    }
    out = {"cvs": list(cvs)}
    for key, label in (("log_tau1", "tau1"), ("log_p", "p")):
        per_cv = {}
        common_int = set.intersection(*[
            {sid for sid, r in by_cv[cv].items() if r["distance_to_bound"][key] > 0.01}
            for cv in cvs
        ])
        for cv in cvs:
            rows = by_cv[cv]
            n = len(rows)
            n_int = sum(r["distance_to_bound"][key] > 0.01 for r in rows.values())
            n_bd = sum(r["profiles"][key]["classification"]["verdict"] == "identifiable" for r in rows.values())
            n_os = sum(r["profiles"][key]["classification"]["verdict"] == "one-sided" for r in rows.values())
            n_fl = sum(r["profiles"][key]["classification"]["verdict"] == "flat" for r in rows.values())
            n_fixed_bd = sum(rows[sid]["profiles"][key]["classification"]["verdict"] == "identifiable" for sid in common_int)
            signed_err = [
                float(np.log(r["profiles"][key]["theta_hat"] / r["profiles"][key]["truth"]))
                for r in rows.values()
                if r["profiles"][key].get("theta_hat") and r["profiles"][key].get("truth")
            ]
            per_cv[str(cv)] = {
                "n": n,
                "n_interior": int(n_int),
                "bounded_all": int(n_bd),
                "bounded_all_pct": 100.0 * n_bd / max(n, 1),
                "bounded_int_pct": 100.0 * n_bd / max(n_int, 1),
                "one_sided": int(n_os),
                "flat": int(n_fl),
                "fixed_common_n": len(common_int),
                "fixed_common_bounded": int(n_fixed_bd),
                "fixed_common_pct": 100.0 * n_fixed_bd / max(len(common_int), 1),
                "median_signed_log_bias": float(np.median(signed_err)),
                "median_dist_to_bound": float(np.median([r["distance_to_bound"][key] for r in rows.values()])),
            }
        # Transitions 0 -> 0.75
        os_to_bd = 0
        bd_to_bd = 0
        bd_to_os = 0
        os_to_os = 0
        shifts = []
        dist_delta_newly_bd = []
        for sid in sorted(by_cv[0.0]):
            r0, r75 = by_cv[0.0][sid], by_cv[0.75][sid]
            v0 = r0["profiles"][key]["classification"]["verdict"]
            v75 = r75["profiles"][key]["classification"]["verdict"]
            h0, h75 = r0["profiles"][key]["theta_hat"], r75["profiles"][key]["theta_hat"]
            if h0 and h75:
                shifts.append(float(np.log(h75 / h0)))
            if v0 == "one-sided" and v75 == "identifiable":
                os_to_bd += 1
                dist_delta_newly_bd.append(float(r75["distance_to_bound"][key] - r0["distance_to_bound"][key]))
            elif v0 == "identifiable" and v75 == "identifiable":
                bd_to_bd += 1
            elif v0 == "identifiable" and v75 == "one-sided":
                bd_to_os += 1
            elif v0 == "one-sided" and v75 == "one-sided":
                os_to_os += 1
        out[label] = {
            "per_cv": per_cv,
            "transitions_0_to_075": {
                "one_sided_to_bounded": os_to_bd,
                "bounded_to_bounded": bd_to_bd,
                "bounded_to_one_sided": bd_to_os,
                "one_sided_to_one_sided": os_to_os,
            },
            "median_log_shift_0_to_075": float(np.median(shifts)),
            "frac_upward_shift_0_to_075": float(np.mean(np.array(shifts) > 0)),
            "median_dist_gain_newly_bounded": float(np.median(dist_delta_newly_bd)) if dist_delta_newly_bd else None,
            "frac_dist_gain_positive": float(np.mean(np.array(dist_delta_newly_bd) > 0)) if dist_delta_newly_bd else None,
        }
    _save("r25_stored_mechanism", out)
    return out


# ---------------------------------------------------------------------------
# A21 (H39): Negative control for iAUC prediction power
# ---------------------------------------------------------------------------
def run_a21() -> dict:
    from evaluation.jax_config import configure
    configure()
    from evaluation import prediction_cv
    from evaluation.cohort_data import load_cohort
    from evaluation.subject_source import get_subject, replica_truth
    from personalization.subject_loss import LOWER, UPPER, TARGETS

    # Slow corner: (k_e=0.015, k_a=0.012); Fast corner: (k_e=0.040, k_a=0.032)
    corners = {
        "slow_corner": (float(LOWER[1]), float(LOWER[2])),
        "fast_corner": (float(UPPER[1]), float(UPPER[2])),
    }
    si_grid = np.linspace(float(LOWER[0]), float(UPPER[0]), 41)
    base_cfg = prediction_cv.default_config()
    subjects = load_cohort("cgmacros", min_meals=10)
    out = {"seeds": {}}

    for seed in (0, 1):
        t0 = time.time()
        cfg = {**base_cfg, "replica": {"seed": seed, "cgm": True, "carb_cv": 0.25, "truth": "random"}}
        per_subj = []
        for s_raw in subjects:
            sid = s_raw.subject_id
            truth = replica_truth(cfg, sid)
            theta_true = np.asarray([truth["theta_true"][n] for n in TARGETS], dtype=float)
            subj = get_subject(cfg, sid)
            ctx = prediction_cv.SubjectContext(subj, cfg)
            obs_iauc = np.asarray(ctx.obs_iauc, dtype=float)
            obs_peak = np.asarray(ctx.obs_peak, dtype=float)
            floor_out = ctx.predict(theta_true)
            floor_iauc = float(np.mean(np.abs(np.asarray(floor_out["iauc"]) - obs_iauc)))
            floor_peak = float(np.mean(np.abs(np.asarray(floor_out["peak"]) - obs_peak)))
            floor_trace = float(np.mean(np.asarray(floor_out["trace_rmse"])))

            row = {"subject_id": sid, "floor_iauc": floor_iauc, "floor_peak": floor_peak, "floor_trace": floor_trace}
            # 5-fold split (repeat 0, deterministic SHA-256)
            rng = np.random.default_rng(prediction_cv.stable_seed("a21", sid, seed))
            perm = rng.permutation(ctx.n)
            folds = np.array_split(perm, 5)

            for cname, (ke_bad, ka_bad) in corners.items():
                # (a) Fixed true S_I + wrong corner rates
                th_fixed = np.array([theta_true[0], ke_bad, ka_bad], dtype=float)
                o_fix = ctx.predict(th_fixed)
                iauc_fix = float(np.mean(np.abs(np.asarray(o_fix["iauc"]) - obs_iauc)))
                peak_fix = float(np.mean(np.abs(np.asarray(o_fix["peak"]) - obs_peak)))
                trace_fix = float(np.mean(np.asarray(o_fix["trace_rmse"])))

                # (b) S_I re-fitted on training folds via 41-pt grid (exact same loss as grid1) with wrong corner rates
                # Precompute predictions across 41 S_I values once for this corner
                grid_iaucs = np.zeros((len(si_grid), ctx.n), dtype=float)
                grid_peaks = np.zeros((len(si_grid), ctx.n), dtype=float)
                grid_traces = np.zeros((len(si_grid), ctx.n), dtype=float)
                for gi, si_val in enumerate(si_grid):
                    og = ctx.predict(np.array([si_val, ke_bad, ka_bad], dtype=float))
                    grid_iaucs[gi] = np.asarray(og["iauc"], dtype=float)
                    grid_peaks[gi] = np.asarray(og["peak"], dtype=float)
                    grid_traces[gi] = np.asarray(og["trace_rmse"], dtype=float)

                pred_iauc_cv = np.zeros(ctx.n, dtype=float)
                pred_peak_cv = np.zeros(ctx.n, dtype=float)
                pred_trace_cv = np.zeros(ctx.n, dtype=float)
                for fidx in range(5):
                    te = folds[fidx]
                    tr = np.concatenate([folds[j] for j in range(5) if j != fidx])
                    mse = np.mean((grid_iaucs[:, tr] - obs_iauc[None, tr]) ** 2, axis=1)
                    pen = 0.01 * ((si_grid - 1.0) / (float(UPPER[0]) - float(LOWER[0]))) ** 2
                    best_g = int(np.argmin(mse + pen))
                    pred_iauc_cv[te] = grid_iaucs[best_g, te]
                    pred_peak_cv[te] = grid_peaks[best_g, te]
                    pred_trace_cv[te] = grid_traces[best_g, te]

                iauc_refit = float(np.mean(np.abs(pred_iauc_cv - obs_iauc)))
                peak_refit = float(np.mean(np.abs(pred_peak_cv - obs_peak)))
                trace_refit = float(np.mean(pred_trace_cv))

                row[f"{cname}_fixed_si"] = {
                    "iauc_ratio": iauc_fix / max(floor_iauc, 1e-9),
                    "peak_ratio": peak_fix / max(floor_peak, 1e-9),
                    "trace_ratio": trace_fix / max(floor_trace, 1e-9),
                }
                row[f"{cname}_refit_si"] = {
                    "iauc_ratio": iauc_refit / max(floor_iauc, 1e-9),
                    "peak_ratio": peak_refit / max(floor_peak, 1e-9),
                    "trace_ratio": trace_refit / max(floor_trace, 1e-9),
                }
            per_subj.append(row)

        summary = {"n": len(per_subj), "elapsed_s": time.time() - t0}
        for cname in corners:
            for mode in ("fixed_si", "refit_si"):
                key = f"{cname}_{mode}"
                summary[key] = {
                    m: {
                        "median": float(np.median([r[key][m] for r in per_subj])),
                        "q25": float(np.percentile([r[key][m] for r in per_subj], 25)),
                        "q75": float(np.percentile([r[key][m] for r in per_subj], 75)),
                    }
                    for m in ("iauc_ratio", "peak_ratio", "trace_ratio")
                }
        out["seeds"][str(seed)] = {"summary": summary, "subjects": per_subj}
        print(f"[a21] seed {seed} done in {time.time() - t0:.1f}s: "
              f"slow_refit_iauc={summary['slow_corner_refit_si']['iauc_ratio']['median']:.3f}, "
              f"fast_refit_iauc={summary['fast_corner_refit_si']['iauc_ratio']['median']:.3f}, "
              f"slow_fixed_iauc={summary['slow_corner_fixed_si']['iauc_ratio']['median']:.3f}", flush=True)

    s0 = out["seeds"]["0"]["summary"]
    h39_met = (
        s0["slow_corner_refit_si"]["iauc_ratio"]["median"] <= 1.10
        and s0["fast_corner_refit_si"]["iauc_ratio"]["median"] <= 1.10
        and s0["slow_corner_refit_si"]["peak_ratio"]["median"] > s0["slow_corner_refit_si"]["iauc_ratio"]["median"]
        and s0["slow_corner_refit_si"]["trace_ratio"]["median"] > s0["slow_corner_refit_si"]["iauc_ratio"]["median"]
    )
    out["h39_status"] = "met" if h39_met else "not met"
    _save("a21_neg_control", out)
    return out


# ---------------------------------------------------------------------------
# A22 (H40): CGMacros iAUC window sweep (60, 90, 120, 180 min)
# ---------------------------------------------------------------------------
def run_a22() -> dict:
    from evaluation.jax_config import configure
    configure()
    import jax.numpy as jnp
    from evaluation.cohort_data import load_cohort
    from evaluation.identifiability_tools import CHI2_DELTA, classify_profile, profile_likelihood
    from evaluation.profile_lik import log_grid
    from personalization.fit_general import fit_ml
    from personalization.objectives import ObjectiveSpec, build_objective, observed_values
    from personalization.subject_loss import TARGETS, base_params, subject_arrays
    from simulation.jax_observables import Window

    windows = (60.0, 90.0, 120.0, 180.0)
    subjects = load_cohort("cgmacros", min_meals=10)
    out = {"windows": {}}
    all_pass = True

    for w_min in windows:
        t0 = time.time()
        win = Window(post_min=w_min, stride=1)
        spec_fit = ObjectiveSpec(name="iauc", bounds_scale=1.0, window=win, parameterization="rates")
        spec_nll = ObjectiveSpec(name="iauc", lam=0.0, normalization="nll", bounds_scale=1.0, window=win, parameterization="rates")
        rows = []
        for subj in subjects:
            ml = fit_ml(subj, spec_fit, steps=600, pilot_steps=300, learning_rate=0.02)
            theta_hat = np.array([ml.ml.theta[n] for n in TARGETS], dtype=float)
            records = list(subj.records)
            obj = build_objective(spec_nll, base_params(subj.profile), subject_arrays(records),
                                  observed_values(records, win), noise=ml.noise)
            lower = np.asarray(obj.lower, dtype=float)
            upper = np.asarray(obj.upper, dtype=float)
            phi_hat = np.log(theta_hat)
            phi_lo, phi_hi = np.log(lower), np.log(upper)

            def loss_fn(phi, _obj=obj):
                return _obj.loss(jnp.exp(phi))

            loss_at_hat = float(loss_fn(jnp.asarray(phi_hat)))
            profs = {}
            dists = {}
            for idx, pname in enumerate(TARGETS):
                dists[pname] = float(min(phi_hat[idx] - phi_lo[idx], phi_hi[idx] - phi_hat[idx]) / (phi_hi[idx] - phi_lo[idx]))
                grid_nat, n_in = log_grid(lower[idx], upper[idx], 15, extension=0)
                res = profile_likelihood(loss_fn, phi_hat, idx, np.log(grid_nat), phi_lo, phi_hi,
                                         steps=100, learning_rate=0.03)
                prof = np.asarray(res["profile"], dtype=float)
                ref = min(loss_at_hat, float(prof.min()))
                cl = classify_profile(grid_nat, prof, delta=CHI2_DELTA, reference=ref)
                profs[pname] = {"verdict": cl["verdict"], "theta_hat": float(theta_hat[idx]),
                                "max_rise": cl.get("max_rise"), "interior": dists[pname] > 0.01}
            rows.append({"subject_id": subj.subject_id, "profiles": profs, "distance_to_bound": dists})

        summ = {"n": len(rows), "elapsed_s": time.time() - t0}
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
            if short in ("k_e", "k_a") and (n_bd / len(rows)) > 0.25:
                all_pass = False
        out["windows"][str(int(w_min))] = {"summary": summ, "subjects": rows}
        print(f"[a22] T={int(w_min)}m ({time.time()-t0:.1f}s): "
              f"SI={summ['S_I']['bounded']}/{summ['S_I']['interior']} int ({summ['S_I']['bounded_all_pct']:.0f}% all), "
              f"ke={summ['k_e']['bounded']}/{summ['k_e']['interior']} int, "
              f"ka={summ['k_a']['bounded']}/{summ['k_a']['interior']} int", flush=True)

    out["h40_status"] = "met" if all_pass else "not met"
    _save("a22_window_profiles", out)
    return out


# ---------------------------------------------------------------------------
# A23 (H41): Hall standardized-meal (n=15) exploratory trace coordinate profiles
# ---------------------------------------------------------------------------
def run_a23() -> dict:
    from evaluation.jax_config import configure
    configure()
    from evaluation import profile_lik
    from evaluation.cohort_data import load_cohort

    t0 = time.time()
    cfg = {
        **profile_lik.default_config(),
        "cohort": "hall",
        "min_meals": 5,
        "objective": "trace",
        "parameterization": "coords",
        "bounds_scale": 1.0,
        "polish": True,
        "polish_starts": 5,
        "polish_maxiter": 300,
    }
    subjects = load_cohort("hall", min_meals=5)
    rows = []
    for s in subjects:
        r = profile_lik.run_unit(s.subject_id, cfg)
        rows.append(r)

    summ = {"n": len(rows), "elapsed_s": time.time() - t0}
    for key, short in (("log_insulin_sensitivity", "S_I"), ("log_tau1", "tau1"), ("log_p", "p")):
        n_int = sum(r["distance_to_bound"][key] > 0.01 for r in rows)
        n_bd = sum(r["profiles"][key]["classification"]["verdict"] == "identifiable" for r in rows)
        n_os = sum(r["profiles"][key]["classification"]["verdict"] == "one-sided" for r in rows)
        n_fl = sum(r["profiles"][key]["classification"]["verdict"] == "flat" for r in rows)
        summ[short] = {
            "interior": int(n_int),
            "bounded": int(n_bd),
            "bounded_all_pct": 100.0 * n_bd / len(rows),
            "bounded_int_pct": (100.0 * n_bd / n_int) if n_int else 0.0,
            "one_sided": int(n_os),
            "flat": int(n_fl),
        }
    out = {"summary": summ, "h41_status": "reported (exploratory)", "subjects": rows}
    print(f"[a23] Hall trace done in {time.time()-t0:.1f}s: "
          f"SI={summ['S_I']['bounded']}/{summ['S_I']['interior']}, "
          f"tau1={summ['tau1']['bounded']}/{summ['tau1']['interior']}, "
          f"p={summ['p']['bounded']}/{summ['p']['interior']}", flush=True)
    _save("a23_hall_trace", out)
    return out


# ---------------------------------------------------------------------------
# A24 (H42): Meal-clustered Huber-White sandwich variance remedy (replica seed 0)
# ---------------------------------------------------------------------------
def run_a24() -> dict:
    from evaluation.jax_config import configure
    configure()
    import jax
    import jax.numpy as jnp
    from evaluation.identifiability_tools import CHI2_DELTA, classify_profile
    from evaluation.phase13_summary import _coords
    from evaluation.recalibration import _rows
    from evaluation.subject_source import get_subject, window_for
    from personalization import noise_model as nm
    from personalization.objectives import ObjectiveSpec, build_objective, observed_values
    from personalization.subject_loss import base_params, subject_arrays

    cvs = (0.0, 0.25, 0.75)
    out = {"cvs": {}}

    for cv in cvs:
        t0 = time.time()
        cfg = {"cohort": "cgmacros", "min_meals": 10, "limit": None,
               "replica": {"seed": 0, "cgm": True, "carb_cv": cv},
               "bounds_scale": 1.0, "objective": "trace", "parameterization": "coords"}
        stored_list = _rows(_coords({"replica": {"seed": 0, "cgm": True, "carb_cv": cv}}))
        per_subj = []
        for stored in stored_list:
            sid = stored["subject_id"]
            subj = get_subject(cfg, sid)
            win = window_for(subj)
            spec = ObjectiveSpec(name="trace", lam=0.0, normalization="nll", bounds_scale=1.0,
                                 window=win, parameterization="coords")
            records = list(subj.records)
            arrays = subject_arrays(records)
            obs = observed_values(records, win)
            names = list(spec.theta_names)
            phi_hat = jnp.asarray([stored["theta_ml"][n] for n in names], dtype=float)
            raw = build_objective(spec, base_params(subj.profile), arrays, obs, noise=None)
            noise = nm.build(raw.residuals(phi_hat), n_parameters=len(spec.free), whiten_trace=spec.whiten_trace)
            obj = build_objective(spec, base_params(subj.profile), arrays, obs, noise=noise)

            r_flat = np.asarray(obj.weighted_residuals(phi_hat), dtype=float)
            J_flat = np.asarray(jax.jacrev(obj.weighted_residuals)(phi_hat), dtype=float)
            width = int(arrays["width"])
            n_real = int(arrays["n"])
            n_samples = len(r_flat) // width
            R = r_flat.reshape(width, n_samples)[:n_real]        # (M, S)
            J = J_flat.reshape(width, n_samples, len(names))[:n_real]  # (M, S, 3)

            # Meal-level scores g_m = J_m^T r_m, meat B = sum_m g_m g_m^T, bread H = sum_m J_m^T J_m
            G = np.einsum("msp,ms->mp", J, R)                    # (M, 3)
            B = G.T @ G
            H = np.einsum("msp,msq->pq", J, J)
            H = 0.5 * (H + H.T)
            # Small ridge for numerical stability on boundary/near-singular Hessians
            eig_max = float(np.max(np.linalg.eigvalsh(H)))
            H_reg = H + max(1e-10 * eig_max, 1e-12) * np.eye(len(names))
            H_inv = np.linalg.inv(H_reg)
            dof_scale = n_real / max(n_real - len(names), 1)
            V_naive = H_inv
            V_sand = dof_scale * (H_inv @ B @ H_inv)

            entry = {"subject_id": sid, "n_meals": n_real, "coords": {}}
            for j, cname in enumerate(names):
                v_n = max(float(V_naive[j, j]), 1e-12)
                v_s = max(float(V_sand[j, j]), 1e-12)
                infl = float(np.clip(v_s / v_n, 1.0, 100.0))  # floor at 1.0 (never narrow below model-based)
                pinfo = stored["profiles"][cname]
                grid = np.asarray(pinfo["grid"], dtype=float)
                losses = np.asarray(pinfo["profile"], dtype=float)
                n_in = int(pinfo.get("n_in_box", len(grid)))
                grid_in, losses_in = grid[:n_in], losses[:n_in]
                ref_loss = float(pinfo["reference"])
                truth = pinfo.get("truth")
                theta_hat = pinfo.get("theta_hat")

                # 1. Naive profile (delta = 1.92)
                cl_naive = classify_profile(grid_in, losses_in, delta=CHI2_DELTA, reference=ref_loss)
                # 2. Sandwich-rescaled profile (Kent/Chandler-Bate: threshold = infl * delta)
                cl_sand = classify_profile(grid_in, losses_in, delta=CHI2_DELTA * infl, reference=ref_loss)

                # Coverage on stored profile by linear interpolation at truth
                cov_naive = None
                cov_sand_prof = None
                cov_sand_wald = None
                if truth is not None and theta_hat is not None:
                    log_err = abs(float(np.log(theta_hat / truth)))
                    se_sand = float(np.sqrt(v_s))
                    cov_sand_wald = bool(log_err <= 1.959964 * se_sand)
                    excess_at_truth = float(np.interp(truth, grid_in, losses_in)) - ref_loss
                    cov_naive = bool(excess_at_truth <= CHI2_DELTA)
                    cov_sand_prof = bool(excess_at_truth <= CHI2_DELTA * infl)

                w_naive = (float(np.log(cl_naive["ci_high"] / cl_naive["ci_low"]))
                           if cl_naive["verdict"] == "identifiable" and cl_naive.get("ci_low") and cl_naive.get("ci_high") else None)
                w_sand = (float(np.log(cl_sand["ci_high"] / cl_sand["ci_low"]))
                          if cl_sand["verdict"] == "identifiable" and cl_sand.get("ci_low") and cl_sand.get("ci_high") else None)

                entry["coords"][cname] = {
                    "inflation_factor": infl,
                    "se_naive": float(np.sqrt(v_n)),
                    "se_sand": float(np.sqrt(v_s)),
                    "interior": bool(stored["distance_to_bound"][cname] > 0.01),
                    "naive_verdict": cl_naive["verdict"],
                    "sand_verdict": cl_sand["verdict"],
                    "cov_naive_prof": cov_naive,
                    "cov_sand_prof": cov_sand_prof,
                    "cov_sand_wald": cov_sand_wald,
                    "width_naive": w_naive,
                    "width_sand": w_sand,
                }
            per_subj.append(entry)

        summ = {"n": len(per_subj), "elapsed_s": time.time() - t0}
        for cname, short in (("log_tau1", "tau1"), ("log_p", "p"), ("log_insulin_sensitivity", "S_I")):
            cs = [r["coords"][cname] for r in per_subj]
            n_int = sum(c["interior"] for c in cs)
            bd_naive = sum(c["naive_verdict"] == "identifiable" for c in cs)
            bd_sand = sum(c["sand_verdict"] == "identifiable" for c in cs)
            cov_n = [c["cov_naive_prof"] for c in cs if c["cov_naive_prof"] is not None]
            cov_sp = [c["cov_sand_prof"] for c in cs if c["cov_sand_prof"] is not None]
            cov_sw = [c["cov_sand_wald"] for c in cs if c["cov_sand_wald"] is not None]
            cov_sp_bd = [c["cov_sand_prof"] for c in cs if c["sand_verdict"] == "identifiable" and c["cov_sand_prof"] is not None]
            ws = [c["width_sand"] for c in cs if c["width_sand"] is not None]
            wn = [c["width_naive"] for c in cs if c["width_naive"] is not None]
            summ[short] = {
                "median_inflation": float(np.median([c["inflation_factor"] for c in cs])),
                "naive_bounded_all": int(bd_naive),
                "naive_bounded_int_pct": 100.0 * bd_naive / max(n_int, 1),
                "sand_bounded_all": int(bd_sand),
                "sand_bounded_int_pct": 100.0 * bd_sand / max(n_int, 1),
                "naive_cov_pct": 100.0 * float(np.mean(cov_n)) if cov_n else None,
                "sand_prof_cov_pct": 100.0 * float(np.mean(cov_sp)) if cov_sp else None,
                "sand_prof_cov_of_bounded_pct": 100.0 * float(np.mean(cov_sp_bd)) if cov_sp_bd else None,
                "sand_wald_cov_pct": 100.0 * float(np.mean(cov_sw)) if cov_sw else None,
                "median_width_naive": float(np.median(wn)) if wn else None,
                "median_width_sand": float(np.median(ws)) if ws else None,
            }
        out["cvs"][str(cv)] = {"summary": summ, "subjects": per_subj}
        print(f"[a24] CV={cv} ({time.time()-t0:.1f}s): "
              f"tau1 infl={summ['tau1']['median_inflation']:.2f}x cov {summ['tau1']['naive_cov_pct']:.0f}%->{summ['tau1']['sand_prof_cov_pct']:.0f}% (wald {summ['tau1']['sand_wald_cov_pct']:.0f}%) bd {summ['tau1']['naive_bounded_all']}->{summ['tau1']['sand_bounded_all']} | "
              f"p infl={summ['p']['median_inflation']:.2f}x cov {summ['p']['naive_cov_pct']:.0f}%->{summ['p']['sand_prof_cov_pct']:.0f}% (wald {summ['p']['sand_wald_cov_pct']:.0f}%) bd {summ['p']['naive_bounded_all']}->{summ['p']['sand_bounded_all']}", flush=True)

    c25 = out["cvs"]["0.25"]["summary"]
    h42_met = (c25["tau1"]["sand_prof_cov_pct"] >= 85.0 and c25["p"]["sand_prof_cov_pct"] >= 85.0)
    out["h42_status"] = "met" if h42_met else "not met"
    _save("a24_sandwich_coverage", out)
    return out


def main() -> int:
    cmds = {"r25": run_r25, "a21": run_a21, "a22": run_a22, "a23": run_a23, "a24": run_a24}
    targets = sys.argv[1:] or list(cmds.keys())
    for t in targets:
        if t not in cmds:
            raise SystemExit(f"unknown subcommand {t}; choose from {list(cmds)}")
        cmds[t]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
