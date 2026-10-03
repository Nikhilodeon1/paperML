"""Phase 2 — gradient-based inference on real CGMacros (Table 1/2 gradient column).

For each subject: fit (Si, gastric, carb) by gradient descent through the differentiable engine
(`personalization/gradient_fit.py`), using the subject's REAL demographics as the base params.
Produces the same metrics as `clinical_recovery.py` / `snpe_kfold.py` for a direct comparison:
in-sample Si vs HbA1c/HOMA-IR (Table 1) and held-out iAUC MAE (Table 2). Also records the per-step
gradient-norm trajectory per parameter (the identifiability diagnostic / paper's best figure).

Run:  python -m evaluation.gradient_inference_results --subjects 6   # smoke
      python -m evaluation.gradient_inference_results                # full 45
"""
from __future__ import annotations

import statistics
import time

import numpy as np

import paper_config as cfg
from simulation import PhysioParams
from simulation.jax_engine import JaxPhysioParams
from evaluation.cgmacros import _profile, load_bio, subjects
from personalization import gradient_fit

_TARGETS = ("insulin_sensitivity", "gastric_emptying", "carb_absorption")


def _subject_meals(meals) -> list[dict]:
    out = []
    for m in meals:
        r = m.real_iauc()
        if r is None:
            continue
        out.append({"carbs_g": float(m.carbs_g), "fat_g": float(m.fat_g),
                    "fiber_g": float(m.fiber_g), "observed_iAUC": float(r)})
    return out


def _base_for(prof: dict) -> JaxPhysioParams:
    return JaxPhysioParams.from_numpy(PhysioParams.from_profile(prof))


def fit_subject(meals: list[dict], prof: dict, n_steps: int = 250) -> dict:
    return gradient_fit.fit_parameters(meals, targets=_TARGETS, n_steps=n_steps,
                                       learning_rate=0.02, base=_base_for(prof))


def collect(limit: int | None = None, n_steps: int = 250, min_meals: int = 4) -> dict:
    cfg.set_all_seeds()
    bio = load_bio()
    rows, t0 = [], time.time()
    grad_hist_accum = {k: [] for k in _TARGETS}   # per-subject step trajectories (for the figure)
    for sid, meals in subjects(limit=limit):
        b = bio.get(sid)
        prof = _profile(b)
        ms = _subject_meals(meals)
        if not (b and prof and len(ms) >= min_meals):
            continue
        res = fit_subject(ms, prof, n_steps=n_steps)
        rows.append({"subject": sid, "si": res["Si"], "hba1c": b["hba1c"],
                     "homa_ir": b["homa_ir"], "status": b["status"], "n": len(ms),
                     "grad_norms": res["gradient_norms"]})
        for k in _TARGETS:
            grad_hist_accum[k].append(res["grad_norm_history"][k])
    ms_per = 1000 * (time.time() - t0) / max(len(rows), 1)
    return {"rows": rows, "ms_per_subject": ms_per, "grad_history": grad_hist_accum}


def kfold_subject(meals: list[dict], prof: dict, n_steps: int = 250) -> dict | None:
    """Chronological split: fit on first half, predict second-half iAUC. MAE vs personal-mean."""
    from simulation.jax_engine import run_meal
    from simulation.jax_observation import iauc
    if len(meals) < 8:
        return None
    k = len(meals) // 2
    train, test = meals[:k], meals[k:]
    res = fit_subject(train, prof, n_steps=n_steps)
    p = res["params"]
    eng, pm = [], []
    personal_mean = statistics.fmean(m["observed_iAUC"] for m in train)
    for m in test:
        ts, g = run_meal(p, m["carbs_g"], m["fat_g"], m["fiber_g"])
        pred = float(iauc(ts, g))
        eng.append(abs(pred - m["observed_iAUC"]))
        pm.append(abs(personal_mean - m["observed_iAUC"]))
    return {"engine_mae": statistics.fmean(eng), "personal_mean_mae": statistics.fmean(pm),
            "beats": statistics.fmean(eng) < statistics.fmean(pm)}


def run_kfold(limit: int | None = None, n_steps: int = 250) -> dict:
    bio = load_bio()
    eng, pm, wins, n = [], [], 0, 0
    for sid, meals in subjects(limit=limit):
        prof = _profile(bio.get(sid))
        ms = _subject_meals(meals)
        if not prof:
            continue
        r = kfold_subject(ms, prof, n_steps=n_steps)
        if r:
            eng.append(r["engine_mae"]); pm.append(r["personal_mean_mae"])
            wins += r["beats"]; n += 1
    if not n:
        return {}
    return {"n": n, "engine_mae": statistics.fmean(eng), "personal_mean_mae": statistics.fmean(pm),
            "gap": statistics.fmean(pm) - statistics.fmean(eng), "winrate": 100 * wins / n}


def main() -> None:
    import argparse
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=None)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--kfold", action="store_true")
    args = ap.parse_args()

    from evaluation.clinical_recovery import bootstrap_corr
    c = collect(limit=args.subjects, n_steps=args.steps)
    rows = c["rows"]
    si = [r["si"] for r in rows]
    print("=" * 76)
    print(f"GRADIENT INFERENCE — {len(rows)} subjects, {c['ms_per_subject']:.0f} ms/subj")
    sp = bootstrap_corr(si, [r["hba1c"] for r in rows], "spearman")
    pe = bootstrap_corr(si, [r["hba1c"] for r in rows], "pearson")
    print(f"  Spearman(Si,HbA1c) = {sp['r']:+.3f} [{sp['lo']:+.2f},{sp['hi']:+.2f}]")
    print(f"  Pearson(Si,HbA1c)  = {pe['r']:+.3f} [{pe['lo']:+.2f},{pe['hi']:+.2f}]")
    gn = {k: statistics.fmean(r["grad_norms"][k] for r in rows) for k in _TARGETS}
    print("  mean |gradient| per param:  " + "  ".join(f"{k.split('_')[0]} {gn[k]:.3g}" for k in _TARGETS))
    if args.kfold:
        kf = run_kfold(limit=args.subjects, n_steps=args.steps)
        print(f"  HELD-OUT: engine MAE {kf['engine_mae']:.0f} | personal-mean {kf['personal_mean_mae']:.0f}"
              f" | gap {kf['gap']:+.0f} | beats {kf['winrate']:.0f}%")
    print("=" * 76)


if __name__ == "__main__":
    main()
