"""Publication figures (Block 6) — one function per figure, PDF + PNG into evaluation/figures/.

Consistent styling across the paper: one colour per estimator (SMC/grid=blue, RF=orange,
SNPE=green, baselines=grey), all axis labels >= 10pt (NeurIPS/AAAI requirement). Each figure is
guarded so one failure does not abort the rest; ``generate_all`` returns the list of files written.

Run:  python -m evaluation.figures
"""
from __future__ import annotations

import warnings

import numpy as np

import paper_config as cfg

warnings.filterwarnings("ignore")

COLORS = {"grid": "#3B6BB0", "smc": "#3B6BB0", "rf": "#E08A2B", "snpe": "#4C9F70",
          "baseline": "#9AA0A6", "normal": "#4C9F70", "prediabetic": "#E0B62B",
          "diabetic": "#C0504D"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 120})
    return plt


def _save(fig, name: str):
    cfg.ensure_dirs()
    pdf = cfg.FIGURES / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(cfg.FIGURES / f"{name}.png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return pdf


def fig_clinical_recovery(posterior=None, rows=None):
    """Figure 4: Si vs HbA1c scatter (colour by dx group) + Si boxplot by group (SNPE)."""
    plt = _mpl()
    from evaluation import clinical_recovery as cr
    if rows is None:
        rows, _ = cr.collect("snpe", posterior=posterior)
    res = cr.analyze(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    for g in ["normal", "prediabetic", "diabetic"]:
        gr = [r for r in rows if r["status"] == g]
        ax1.scatter([r["hba1c"] for r in gr], [r["si"] for r in gr],
                    c=COLORS[g], label=g, s=32, edgecolor="white", linewidth=0.5)
    xs = np.array([r["hba1c"] for r in rows]); ys = np.array([r["si"] for r in rows])
    if len(xs) > 2:
        b, a = np.polyfit(xs, ys, 1)
        xx = np.linspace(xs.min(), xs.max(), 50)
        ax1.plot(xx, a + b * xx, color="grey", ls="--", lw=1)
    ph = res["pearson_hba1c"]
    ax1.set_xlabel("HbA1c (%)"); ax1.set_ylabel("SNPE insulin sensitivity Si")
    ax1.set_title(f"r = {ph['r']:+.2f}  [{ph['lo']:+.2f}, {ph['hi']:+.2f}]")
    ax1.legend(frameon=False, fontsize=8)

    groups = ["normal", "prediabetic", "diabetic"]
    data = [[r["si"] for r in rows if r["status"] == g] for g in groups]
    bp = ax2.boxplot(data, tick_labels=[g[:4] for g in groups], patch_artist=True)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(COLORS[g]); patch.set_alpha(0.7)
    ax2.set_ylabel("Si"); ax2.set_title(f"ANOVA p = {res['anova_p']:.3g}")
    fig.tight_layout()
    return _save(fig, "fig4_clinical_recovery")


def fig_rf_vs_snpe(posterior=None, shrink_rows=None):
    """Figure 3: RF shrinks true Si toward the mean; SNPE reports wide posteriors instead."""
    plt = _mpl()
    from evaluation import identifiability_analysis as ida
    if shrink_rows is None:
        from personalization.snpe_trainer import SNPETrainer
        posterior = posterior or SNPETrainer.load().posterior
        shrink_rows = ida.rf_shrinkage_analysis(posterior)
    true = np.array([r["true_si"] for r in shrink_rows])
    rf = np.array([r["rf_mean"] for r in shrink_rows])
    sn = np.array([r["snpe_mean"] for r in shrink_rows])
    sd = np.array([r["snpe_std"] for r in shrink_rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    lim = [true.min() - 0.1, true.max() + 0.1]
    ax1.plot(lim, lim, color="grey", ls="--", lw=1, label="perfect")
    ax1.plot(true, rf, "o-", color=COLORS["rf"], label="RF point")
    ax1.errorbar(true, sn, yerr=sd, fmt="s-", color=COLORS["snpe"], capsize=3, label="SNPE mean±std")
    ax1.set_xlabel("true Si"); ax1.set_ylabel("estimated Si")
    ax1.set_title("RF shrinks at extremes; SNPE stays honest"); ax1.legend(frameon=False, fontsize=8)

    ax2.plot(true, sd, "s-", color=COLORS["snpe"])
    ax2.set_xlabel("true Si"); ax2.set_ylabel("SNPE posterior std")
    ax2.set_title("posterior width grows where Si is unresolved")
    fig.tight_layout()
    return _save(fig, "fig3_rf_vs_snpe")


def fig_identifiability(posterior=None):
    """Figure 2: convergence — Si posterior std falls with n_meals, gastric/carb stay wide."""
    plt = _mpl()
    from evaluation import identifiability_analysis as ida
    from personalization.snpe_trainer import SNPETrainer
    posterior = posterior or SNPETrainer.load().posterior
    conv = ida.multi_meal_convergence(posterior)
    n = conv["n_meals_grid"]
    lo, hi = cfg.prior_bounds()
    widths = dict(zip(cfg.PARAM_NAMES, np.array(hi) - np.array(lo)))

    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {"insulin_sensitivity": ("Si", COLORS["snpe"]),
              "gastric_emptying": ("gastric", COLORS["rf"]),
              "carb_absorption": ("carb", COLORS["grid"])}
    for name, (lab, col) in styles.items():
        frac = np.array(conv["curves"][name]) / widths[name]
        ax.plot(n, frac, "o-", color=col, label=f"{lab}  R²(n,std)={conv['r2_n_vs_std'][name]:.2f}")
    ax.set_xlabel("number of meals"); ax.set_ylabel("posterior std / prior width")
    ax.set_title("Si narrows with data; gastric/carb structurally wide")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, "fig2_identifiability")


def fig_kfold_utility(kfold=None):
    """Figure 5: held-out iAUC MAE by estimator + baselines (per-fold points)."""
    plt = _mpl()
    from evaluation import snpe_kfold
    r = kfold or snpe_kfold.run()
    methods, vals, cols = [], [], []
    for e, d in r["estimators"].items():
        methods.append(e); vals.append(d["mae"]["folds"]); cols.append(COLORS.get(e, "#666"))
    for b, d in r["baselines"].items():
        methods.append(b); vals.append(d["folds"]); cols.append(COLORS["baseline"])

    fig, ax = plt.subplots(figsize=(8, 4))
    means = [np.mean(v) for v in vals]
    ax.bar(range(len(methods)), means, color=cols, alpha=0.75)
    for i, v in enumerate(vals):
        ax.scatter([i] * len(v), v, color="black", s=12, zorder=3)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("held-out iAUC MAE (mg/dL·min)")
    ax.set_title("Personalized Si beats baselines on held-out meals")
    fig.tight_layout()
    return _save(fig, "fig5_kfold_utility")


def fig_ood_gap(gap=None):
    """New Figure: real meal summary stats fall in the tails of the SNPE training distribution.

    The mechanistic explanation for SNPE's compressed absolute Si (Table 2 utility gap): amortized
    inference is queried off its training support. Per-panel title shows % of real meals outside the
    simulated central-90% range."""
    plt = _mpl()
    from evaluation import ood_analysis as ood
    if gap is None:
        gap = ood.simulator_reality_gap()
    sim, real = gap["sim"], gap["real"]
    names = ood.SUMMARY_NAMES
    fig, axes = plt.subplots(1, len(names), figsize=(15, 3))
    for i, (ax, name) in enumerate(zip(axes, names)):
        s, r = sim[:, i], real[:, i]
        lo = min(np.percentile(s, 1), np.percentile(r, 1))
        hi = max(np.percentile(s, 99), np.percentile(r, 99))
        bins = np.linspace(lo, hi, 40)
        ax.hist(s, bins=bins, density=True, color=COLORS["baseline"], alpha=0.6, label="simulated")
        ax.hist(r, bins=bins, density=True, color=COLORS["snpe"], alpha=0.6, label="real")
        d = gap["per_stat"][name]
        ax.set_title(f"{name}\n{d['frac_real_outside_central90']*100:.0f}% real OOD "
                     f"(shift {d['median_shift_z']:+.1f}z)", fontsize=9)
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Simulator-reality gap: real meal summary statistics fall in the tails of the "
                 "training distribution", fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig6_ood_gap")


def fig_gradient_identifiability(grad_history):
    """Gradient-norm vs optimization step (mean over subjects, IQR band), one line per parameter.
    Si stays informative; gastric/carb sit near zero from step 1 — the signal was never there
    (structural non-identifiability), the two-independent-methods sibling of the SNPE posterior
    width. `grad_history` = {param: [per-subject [per-step normalized |grad|]]}."""
    plt = _mpl()
    styles = {"insulin_sensitivity": ("Si", COLORS["snpe"]),
              "gastric_emptying": ("gastric", COLORS["rf"]),
              "carb_absorption": ("carb", COLORS["grid"])}
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (lab, col) in styles.items():
        arr = np.array(grad_history[name])            # (n_subjects, n_steps)
        steps = np.arange(arr.shape[1])
        ax.plot(steps, arr.mean(axis=0), color=col, label=lab, lw=2)
        ax.fill_between(steps, np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0),
                        color=col, alpha=0.15)
    ax.set_yscale("log")                               # Si ~20-60x the others
    ax.set_xlabel("optimization step")
    ax.set_ylabel("normalized |gradient|")
    ax.set_title("Gradient identifiability: Si informative, gastric/carb near-zero from step 1")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig7_gradient_identifiability")


def fig_dalla_man_identifiability(n_subjects: int = 10, n_steps: int = 120):
    """Fig 8 — gradient-norm vs optimization step for the Dalla Man parameters (Fig 7 analog).
    Runs the Dalla Man gradient fit on a representative CGMacros subsample (full 45 is too slow for
    a figure) and plots the per-step normalized gradient norm for all 6 fitted params. Expectation
    (report honestly if it differs): Vmx dominant/identifiable; kabs/kmax/kmin the non-identifiable
    ridge; f/Td intermediate."""
    plt = _mpl()
    import paper_config as cfg
    from simulation import PhysioParams
    from simulation.dalla_man import DallaManParams
    from personalization.gradient_fit_dalla_man import fit_parameters, TARGETS
    from evaluation.cgmacros import _profile, load_bio, subjects
    from evaluation.snpe_kfold import _meal_records
    from evaluation.dalla_man_comparison import _subject_Gb, _blunt

    cfg.set_all_seeds()
    bio = load_bio()
    hist = {k: [] for k in TARGETS}
    n = 0
    for sid, meals in subjects():
        prof = _profile(bio.get(sid))
        recs = _meal_records(meals)
        if not prof or len(recs) < 10:
            continue
        Gb = _subject_Gb(recs)
        gmeals = [{"carbs_g": r["carbs_g"] * _blunt(r), "observed_iAUC": r["iauc"]} for r in recs]
        res = fit_parameters(gmeals, n_steps=n_steps, base=DallaManParams.defaults(prof["weight_kg"], Gb),
                             weight_kg=prof["weight_kg"], Gb=Gb)
        for k in TARGETS:
            hist[k].append(res["grad_norm_history"][k])
        n += 1
        if n >= n_subjects:
            break

    palette = ["#4C9F70", "#E08A2B", "#3B6BB0", "#9AA0A6", "#C0504D", "#7E57C2"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, k in enumerate(TARGETS):
        arr = np.array(hist[k])                    # (n_subjects, n_steps)
        ax.plot(np.arange(arr.shape[1]), arr.mean(axis=0), color=palette[i % len(palette)],
                label=k, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("normalized |gradient|")
    ax.set_title(f"Dalla Man identifiability (n={n}): Vmx identifiable, gut/timing not")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, "fig8_dalla_man_identifiability")


def generate_all(kfold=None, ood=None) -> list:
    """Generate every figure. Pass precomputed ``kfold``/``ood`` results to avoid recomputation
    (reproduce_all threads these through so the 20-min k-fold and the OOD sweep run only once)."""
    jobs = [
        (fig_clinical_recovery, {}),
        (fig_rf_vs_snpe, {}),
        (fig_identifiability, {}),
        (fig_ood_gap, {"gap": ood}),
        (fig_kfold_utility, {"kfold": kfold}),
    ]
    written = []
    for fn, kw in jobs:
        try:
            written.append(str(fn(**kw)))
            print(f"  wrote {written[-1]}")
        except Exception as e:
            import traceback
            print(f"  FAILED {fn.__name__}: {e}")
            traceback.print_exc()
    return written


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    print("Generating figures ->", cfg.FIGURES)
    generate_all()


if __name__ == "__main__":
    main()
