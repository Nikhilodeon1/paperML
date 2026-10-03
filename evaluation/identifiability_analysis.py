"""Identifiability characterization (paper §4.3 / Table 3, Figures 2-3) — the novel finding.

The claim: insulin sensitivity Si is identifiable from a meal-glucose curve, while gastric
emptying and carb absorption are structurally non-identifiable from that observation. A point
estimator (RF) hides this by shrinking un-identifiable parameters toward the prior mean; a
calibrated posterior (SNPE) reveals it as posterior WIDTH.

Three analyses:
  * ``posterior_width_analysis``  — per real CGMacros subject: marginal posterior std of each
                                    parameter (Si narrow, gastric/carb wide).
  * ``rf_shrinkage_analysis``     — synthetic subjects at known true Si across the range: RF
                                    point vs SNPE posterior mean+std. RF shrinks at the extremes;
                                    SNPE reports wide posteriors there instead (Figure 3).
  * ``multi_meal_convergence``    — synthetic subject, vary n_meals: Si posterior std shrinks with
                                    data (identifiable), gastric std stays ~flat (structural).

Run:  python -m evaluation.identifiability_analysis
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sstats

import paper_config as cfg
from personalization import npe, snpe_infer
from simulation import Simulator, PhysioParams, Schedule, Meal
from simulation.observation import observe_series, spec_for

_PROFILE = dict(npe._PROFILE)
_TRUE_GASTRIC, _TRUE_CARB = 0.030, 0.020
_MEAL_T = 30.0


def _synth_meals(true_si: float, n_meals: int, seed: int, profile: dict | None = None,
                 gastric: float = _TRUE_GASTRIC, carb: float = _TRUE_CARB,
                 noise_sd: float = 10.0) -> list[dict]:
    """Simulate a subject's meals at a KNOWN (Si, gastric, carb) -> snpe meal dicts."""
    profile = profile or _PROFILE
    rng = np.random.default_rng(seed)
    meals = []
    for _ in range(n_meals):
        carbs = float(rng.uniform(40.0, 80.0))
        p = PhysioParams.from_profile(profile)
        p.insulin_sensitivity, p.gastric_emptying, p.carb_absorption = true_si, gastric, carb
        s = Schedule()
        s.add(Meal(_MEAL_T, carbs_g=carbs))
        traj = Simulator(p).run(s, duration_min=_MEAL_T + 180.0, dt=1.0, record_every=5,
                                outputs=["glucose_mg_dl"])
        obs = observe_series(traj, spec_for("cgm", "glucose"), step_min=5.0)
        vals = list(np.asarray(obs["values"]) + rng.normal(0, noise_sd, len(obs["values"])))
        meals.append({"carbs_g": carbs,
                      "glucose": {"values": vals, "t0_min": obs["t0_min"], "step_min": 5.0,
                                  "meal_t_min": _MEAL_T}})
    return meals


# --- 1. per real subject -----------------------------------------------------------------------

def posterior_width_analysis(posterior, limit: int | None = None) -> list[dict]:
    """Per CGMacros subject: marginal posterior std of each parameter + labs + n_meals."""
    from evaluation.cgmacros import _profile, load_bio, subjects
    from evaluation.clinical_recovery import _subject_meals_snpe

    cfg.set_all_seeds()
    bio = load_bio()
    out = []
    subs = subjects(limit=limit)
    for k, (sid, meals) in enumerate(subs):
        b = bio.get(sid)
        prof = _profile(b)
        good = [m for m in meals if m.real_iauc() is not None]
        if not (b and prof and len(good) >= 4):
            continue
        post = snpe_infer.infer_posterior(_subject_meals_snpe(good), posterior, profile=prof,
                                          combine="per_meal", max_meals=12)
        if not post:
            continue
        out.append({"subject": sid, "n_meals": len(good), "hba1c": b["hba1c"],
                    "homa_ir": b["homa_ir"], "status": b["status"],
                    "Si_std": post["Si_std"], "gastric_std": post["gastric_std"],
                    "carb_std": post["carb_std"], "Si_mean": post["Si_mean"]})
    return out


def width_summary(rows: list[dict]) -> dict:
    """Aggregate: mean posterior std per parameter, normalised by prior width."""
    lo, hi = cfg.prior_bounds()
    widths = {"insulin_sensitivity": hi[0] - lo[0], "gastric_emptying": hi[1] - lo[1],
              "carb_absorption": hi[2] - lo[2]}
    keys = [("Si_std", "insulin_sensitivity"), ("gastric_std", "gastric_emptying"),
            ("carb_std", "carb_absorption")]
    summ = {}
    for k, name in keys:
        vals = np.array([r[k] for r in rows])
        summ[name] = {"mean_std": float(vals.mean()),
                      "frac_of_prior": float(vals.mean() / widths[name])}
    return {"n": len(rows), "per_param": summ}


# --- 2. RF shrinkage vs SNPE posteriors --------------------------------------------------------

def rf_shrinkage_analysis(posterior, rf=None, si_grid=None, n_subjects: int = 6,
                          n_meals: int = 10) -> list[dict]:
    """At each true Si: RF point estimate vs SNPE posterior mean+std (averaged over subjects)."""
    if rf is None:
        from evaluation.clinical_recovery import build_rf
        rf = build_rf()
    si_grid = si_grid if si_grid is not None else [0.3, 0.5, 0.7, 0.95, 1.1, 1.3, 1.6]
    si_idx = npe.PARAM_NAMES.index("insulin_sensitivity")

    rows = []
    for true_si in si_grid:
        rf_preds, snpe_means, snpe_stds = [], [], []
        for k in range(n_subjects):
            meals = _synth_meals(true_si, n_meals, seed=1000 + k)
            # RF point (median over meals, mirrors npe_cgmacros)
            feats = []
            for m in meals:
                g = m["glucose"]
                s = npe.summary_stats(g["values"], g["t0_min"], g["step_min"], g["meal_t_min"])
                if not np.isnan(s).any():
                    feats.append(npe.features_for(s, m["carbs_g"], _PROFILE))
            if feats:
                rf_preds.append(float(np.median(rf.predict(np.array(feats))[:, si_idx])))
            post = snpe_infer.infer_posterior(meals, posterior, profile=_PROFILE, combine="per_meal")
            if post:
                snpe_means.append(post["Si_mean"])
                snpe_stds.append(post["Si_std"])
        rows.append({
            "true_si": true_si,
            "rf_mean": float(np.mean(rf_preds)) if rf_preds else float("nan"),
            "rf_bias": float(np.mean(rf_preds) - true_si) if rf_preds else float("nan"),
            "snpe_mean": float(np.mean(snpe_means)) if snpe_means else float("nan"),
            "snpe_bias": float(np.mean(snpe_means) - true_si) if snpe_means else float("nan"),
            "snpe_std": float(np.mean(snpe_stds)) if snpe_stds else float("nan"),
        })
    return rows


# --- 3. convergence: Si narrows with data, gastric stays wide ----------------------------------

def multi_meal_convergence(posterior, true_si: float = 0.7,
                           n_meals_grid=None, combine: str = "product") -> dict:
    """Vary n_meals for a synthetic subject; track posterior std of each parameter vs n."""
    n_meals_grid = n_meals_grid or [1, 2, 4, 8, 16, 32]
    curves = {name: [] for name in cfg.PARAM_NAMES}
    for n in n_meals_grid:
        meals = _synth_meals(true_si, n, seed=7)
        post = snpe_infer.infer_posterior(meals, posterior, profile=_PROFILE, combine=combine)
        curves["insulin_sensitivity"].append(post.get("Si_std", np.nan))
        curves["gastric_emptying"].append(post.get("gastric_std", np.nan))
        curves["carb_absorption"].append(post.get("carb_std", np.nan))

    def r2_vs_n(std_list):
        x = np.array(n_meals_grid, float)
        y = np.array(std_list, float)
        ok = np.isfinite(y)
        if ok.sum() < 3:
            return float("nan")
        return float(sstats.pearsonr(x[ok], y[ok]).statistic ** 2)

    return {"n_meals_grid": n_meals_grid, "curves": curves, "combine": combine,
            "r2_n_vs_std": {name: r2_vs_n(curves[name]) for name in cfg.PARAM_NAMES}}


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    from personalization.snpe_trainer import SNPETrainer

    cfg.set_all_seeds()
    post = SNPETrainer.load().posterior

    print("=" * 78)
    print("IDENTIFIABILITY — Si should be narrow, gastric/carb wide (real CGMacros subjects)")
    print("=" * 78)
    rows = posterior_width_analysis(post)
    summ = width_summary(rows)
    print(f"  n subjects: {summ['n']}")
    for name, d in summ["per_param"].items():
        print(f"    {name:20}  mean posterior std {d['mean_std']:.4f}  "
              f"= {d['frac_of_prior']*100:.0f}% of prior width")

    print("\n  RF shrinkage vs SNPE posterior (synthetic, known true Si):")
    print(f"    {'true Si':>8}{'RF mean':>9}{'RF bias':>9}{'SNPE mean':>11}{'SNPE bias':>11}{'SNPE std':>10}")
    for r in rf_shrinkage_analysis(post):
        print(f"    {r['true_si']:>8.2f}{r['rf_mean']:>9.2f}{r['rf_bias']:>+9.2f}"
              f"{r['snpe_mean']:>11.2f}{r['snpe_bias']:>+11.2f}{r['snpe_std']:>10.3f}")

    print("\n  Convergence (product-combine, true Si=0.7): posterior std vs n_meals")
    conv = multi_meal_convergence(post)
    for name in cfg.PARAM_NAMES:
        vals = "  ".join(f"{v:.4f}" for v in conv["curves"][name])
        print(f"    {name:20} [{vals}]  R^2(n,std)={conv['r2_n_vs_std'][name]:.2f}")
    print("  (Si R^2 high + std falling = identifiable; gastric R^2 low + std flat = structural)")
    print("=" * 78)


if __name__ == "__main__":
    main()
