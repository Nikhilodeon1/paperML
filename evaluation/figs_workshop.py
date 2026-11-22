"""Figures for the CALE-AI workshop paper (AAAI two-column format).

Rules of the AAAI author kit that these figures follow: PDF output, fonts embedded as Type 42 (no Type 3), text at least nine
point at the size the figure is included, lines at least half a point, no clipping in LaTeX (figures are cropped here),
and a design that reads in grayscale (marker shape, line style and fill carry the distinction, colour only repeats it).
Colours are dark enough for a contrast of at least 4.5:1 on white.

Every number a figure shows is computed here from the stored results; the numbers that the text quotes are written to
`workshop_caleai/figure_numbers.json` and enter the paper as macros.

Run:  python -m evaluation.figs_workshop
"""
from __future__ import annotations

import json

import numpy as np

from evaluation.results_io import RESULTS, ROOT

OUT = ROOT / "workshop_caleai" / "figures"
NUMBERS = ROOT / "workshop_caleai" / "figure_numbers.json"
BLUE, ORANGE, GREEN, PURPLE, GREY, BLACK = "#00589B", "#A8420A", "#00704A", "#6B2D7A", "#595959", "#111111"
COLUMN, FULL = 3.0, 6.2          # inches; included at 0.9 of the column and at 0.9 of the text width
NUM: dict = {}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 9, "figure.dpi": 130, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8, "lines.linewidth": 1.1, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False})
    return plt


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return OUT / f"{name}.pdf"


def _panel(ax, letter: str, dx: float = -0.2) -> None:
    ax.text(dx, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def _load(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ---------------------------------------------------------------------------------------------------------------
# Figure 1: what an area can and cannot see
# ---------------------------------------------------------------------------------------------------------------

WINDOW = 180.0          # minutes after the meal that the illustration integrates over
MEAL_G = 60.0           # grams of carbohydrate of the illustrated meal
PAIRS = {"A": (1 / 26.5, 1 / 73.5), "B": (0.0250, 1 / 60.0), "C": (0.0200, 1 / 50.0)}      # tau1 = 100 min in every pair


def _simulate(ke: float, ka: float, duration: float = 240.0, meal_time: float = 30.0):
    import equinox as eqx
    import jax.numpy as jnp
    from simulation import jax_engine as engine
    base = engine.JaxPhysioParams.from_population_defaults()
    p = eqx.tree_at(lambda x: (x.gastric_emptying, x.carb_absorption), base, (jnp.float32(ke), jnp.float32(ka)))
    ts, g = engine.run_meal(p, MEAL_G, 0.0, 0.0, meal_time=meal_time, duration_min=duration + meal_time, step_min=1.0)
    return np.asarray(ts, dtype=float) - meal_time, np.asarray(g, dtype=float)


def _iauc(t, g, window: float = WINDOW) -> float:
    mask = (t >= 0) & (t <= window)
    return float(np.trapezoid(np.maximum(g[mask] - g[t == 0][0], 0.0), t[mask]))


def _noise_scales() -> dict:
    from evaluation.ladder import a4_results
    out = {}
    for objective in ("iauc", "trace"):
        rows = a4_results(objective, 1.0)
        values = [float(list(r["noise"]["sigma"].values())[0]) for r in rows.values()]
        out[objective] = float(np.median(values)) if values else None
    return out


def figure_what_the_area_sees():
    from evaluation.ladder import a4_results
    plt = _plt()
    ke_grid = np.linspace(0.015, 0.040, 15)
    ka_grid = np.linspace(0.012, 0.032, 15)
    surface = np.zeros((len(ka_grid), len(ke_grid)))
    for i, ka in enumerate(ka_grid):
        for j, ke in enumerate(ke_grid):
            t, g = _simulate(ke, ka)
            surface[i, j] = _iauc(t, g)
    sigma = _noise_scales()
    traces = {k: _simulate(*v) for k, v in PAIRS.items()}
    swap = _simulate(PAIRS["A"][1], PAIRS["A"][0])
    iaucs = {k: _iauc(*traces[k]) for k in traces}
    peaks = {k: float(traces[k][0][(traces[k][0] >= 0) & (traces[k][0] <= WINDOW)][np.argmax(traces[k][1][(traces[k][0] >= 0) & (traces[k][0] <= WINDOW)])]) for k in traces}
    peak_g = {k: float(traces[k][1][(traces[k][0] >= 0) & (traces[k][0] <= WINDOW)].max()) for k in traces}
    t_ref = traces["B"][0]
    diff = {k: traces[k][1] - traces["B"][1] for k in ("A", "C")}
    mask = (t_ref >= 0) & (t_ref <= WINDOW)
    NUM.update({
        "WindowMin": WINDOW, "MealGrams": MEAL_G, "PairTauOne": 100.0,
        "IaucSurfaceMin": float(surface.min()), "IaucSurfaceMax": float(surface.max()),
        "IaucSurfaceRange": float(surface.max() - surface.min()), "IaucNoiseSd": sigma["iauc"],
        "IaucRangeOverNoise": float((surface.max() - surface.min()) / sigma["iauc"]),
        "TraceNoiseSd": sigma["trace"],
        "PairIaucSpread": float(max(iaucs.values()) - min(iaucs.values())),
        "PairIaucSpreadPct": float(100 * (max(iaucs.values()) - min(iaucs.values())) / np.mean(list(iaucs.values()))),
        "PairIaucOverNoise": float((max(iaucs.values()) - min(iaucs.values())) / sigma["iauc"]),
        "PairPeakTimeSpread": float(max(peaks.values()) - min(peaks.values())),
        "PairPeakGlucoseSpread": float(max(peak_g.values()) - min(peak_g.values())),
        "PairTraceMaxDiff": float(max(np.abs(diff["A"][mask]).max(), np.abs(diff["C"][mask]).max())),
        "SwapMaxDiff": float(np.abs(swap[1] - traces["A"][1]).max()),
        "PairPA": 1 / (PAIRS["A"][0] * PAIRS["A"][1]), "PairPB": 1 / (PAIRS["B"][0] * PAIRS["B"][1]),
        "PairPC": 1 / (PAIRS["C"][0] * PAIRS["C"][1])})

    fig = plt.figure(figsize=(FULL, 2.75))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.05, 0.95], height_ratios=[1.5, 1.0], hspace=0.14, wspace=0.62,
                          left=0.085, right=0.985, top=0.88, bottom=0.16)
    # (a) iAUC over the physiological box
    ax = fig.add_subplot(gs[:, 0])
    mesh = ax.contourf(ke_grid * 100, ka_grid * 100, surface, levels=9, cmap="cividis")
    for tau, style in ((80.0, ":"), (100.0, "--"), (120.0, "-.")):
        ke_line = np.linspace(0.015, 0.040, 200)
        with np.errstate(divide="ignore", invalid="ignore"):
            ka_line = 1.0 / (tau - 1.0 / ke_line)
        ok = (ka_line >= 0.012) & (ka_line <= 0.032) & (ka_line > 0)
        ax.plot(ke_line[ok] * 100, ka_line[ok] * 100, color="white", lw=1.6, ls=style)
        ax.plot(ke_line[ok] * 100, ka_line[ok] * 100, color=BLACK, lw=0.8, ls=style)
        if ok.any():
            k = np.where(ok)[0][len(np.where(ok)[0]) // 2]
            ax.annotate(rf"$\tau_1={tau:.0f}$", (ke_line[k] * 100, ka_line[k] * 100), xytext=(2, 4), textcoords="offset points",
                        fontsize=9, color="white", fontweight="bold")
    for name, (ke, ka) in PAIRS.items():
        ax.plot(ke * 100, ka * 100, marker="o", ms=6, mfc="white", mec=BLACK, mew=1.0, ls="none")
        ax.annotate(name, (ke * 100, ka * 100), xytext=(3, 3), textcoords="offset points", fontsize=9, fontweight="bold", color="white")
    ax.set_xlabel(r"gastric emptying $k_e$ ($10^{-2}$ min$^{-1}$)")
    ax.set_ylabel(r"absorption $k_a$ ($10^{-2}$ min$^{-1}$)")
    cb = fig.colorbar(mesh, ax=ax, location="top", pad=0.03, fraction=0.075, aspect=18, ticks=[4100, 4500, 4900, 5200])
    cb.set_label("iAUC (mg min dL$^{-1}$)")
    cb.ax.tick_params(labelsize=9, width=0.8)
    cb.outline.set_linewidth(0.8)
    _panel(ax, "a", dx=-0.27)
    # (b) traces and differences for three pairs with the same tau_1
    t = traces["A"][0]
    ax = fig.add_subplot(gs[0, 1])
    sty = {"A": ("-", BLACK, 1.5), "B": ("--", BLUE, 1.2), "C": ("-.", ORANGE, 1.2)}
    for name, (ls, color, lw) in sty.items():
        ax.plot(t, traces[name][1], ls=ls, color=color, lw=lw, label=f"pair {name}")
    ax.plot(swap[0], swap[1], ls=":", color=GREY, lw=1.4, label="A swapped")
    ax.set_xlim(0, WINDOW)
    ax.set_xticks([0, 60, 120, 180])
    ax.tick_params(labelbottom=False)
    ax.set_ylim(85, 190)
    ax.set_ylabel("glucose (mg dL$^{-1}$)")
    ax.legend(loc="upper center", ncol=2, fontsize=9, handlelength=1.6, columnspacing=0.8, labelspacing=0.15,
              borderaxespad=0.1, handletextpad=0.4)
    _panel(ax, "b", dx=-0.3)
    ax = fig.add_subplot(gs[1, 1], sharex=ax)
    ax.tick_params(labelbottom=True)
    ax.axhspan(-sigma["trace"], sigma["trace"], color="#D9D9D9", lw=0)
    ax.plot(t, diff["A"], ls="-", color=BLACK, lw=1.2)
    ax.plot(t, diff["C"], ls="-.", color=ORANGE, lw=1.2)
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.set_xlim(0, WINDOW)
    ax.set_xlabel("minutes after the meal")
    ax.set_ylabel("vs. B (mg dL$^{-1}$)")
    ax.text(0.97, 0.06, r"grey band: $\pm$1 noise sd", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=GREY)
    # (c) Fisher eigenvalues under the iAUC, three cohorts
    ax = fig.add_subplot(gs[:, 2])
    fills = {"cgmacros": ("#FFFFFF", ""), "shanghai": ("#BDBDBD", ""), "hall": ("#7A7A7A", "")}
    labels = {"cgmacros": "CGMacros", "shanghai": "Shanghai", "hall": "Hall"}
    width = 0.2
    drawn = False
    sets = [(cohort, "iauc", fills[cohort][0], "") for cohort in fills] + [("cgmacros", "trace", "#FFFFFF", "////")]
    for j, (cohort, objective, face, hatch) in enumerate(sets):
        rows = a4_results(objective, 1.0, cohort=cohort)
        if not rows:
            continue
        ev = np.array([r["fisher"]["eigenvalues"] for r in rows.values()])
        for i in range(3):
            bp = ax.boxplot(np.log10(np.maximum(ev[:, i], 1e-16)), positions=[i + (j - 1.5) * width * 1.15], widths=width,
                            patch_artist=True, showfliers=False, medianprops={"color": BLACK, "lw": 1.2},
                            whiskerprops={"lw": 0.8}, capprops={"lw": 0.8}, boxprops={"lw": 0.8})
            bp["boxes"][0].set(facecolor=face, edgecolor=BLACK, hatch=hatch)
        if objective == "trace":
            med = np.median(ev, axis=0)
            for i, word_ in enumerate(("One", "Two", "Three")):
                NUM[f"EigTraceMedian{word_}"] = float(med[i])
        drawn = True
    ax.set_xticks(range(3))
    ax.set_xticklabels([r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"])
    ax.set_xlabel("Fisher eigenvalue")
    ax.set_ylabel(r"$\log_{10}$ eigenvalue")
    _panel(ax, "c", dx=-0.3)
    if not drawn:
        return None
    return _save(fig, "fig_area")


# ---------------------------------------------------------------------------------------------------------------
# Figure 2: held-out error against parameter error
# ---------------------------------------------------------------------------------------------------------------

def figure_prediction_vs_parameters():
    from scipy.stats import spearmanr
    plt = _plt()
    r1 = _load("phase13_summary.json").get("r1", {})
    r9 = _load("phase14_summary.json").get("r9", {})
    seeds = (("random_truth_seed0", "o", "seed 0"), ("random_truth_seed1", "s", "seed 1"))
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.75), sharex=True,
                             gridspec_kw={"wspace": 0.38, "left": 0.085, "right": 0.99, "bottom": 0.33, "top": 0.89})
    titles = ["iAUC (trained on)", "peak time", "glucose trace"]
    ylabels = ["held-out error /\ntrue-model error", "", ""]
    pooled = {k: [] for k in ("iauc", "peak", "trace")}
    for key, marker, label in seeds:
        rows = r9.get(key, {}).get("subjects") or []
        by_id = {r["subject_id"]: r for r in rows}
        iauc = r1.get(key, {}).get("subjects") or []
        timing_a = np.array([r["timing_rms"] for r in iauc])
        ratio_a = np.array([r["grad3_mae"] / r["floor"] for r in iauc])
        axes[0].scatter(timing_a, ratio_a, marker=marker, s=22, facecolor=BLUE, edgecolor=BLUE, alpha=0.85, lw=0.6)
        pooled["iauc"].append((timing_a, ratio_a))
        for ax, kind, name in ((axes[1], "peak", "peak"), (axes[2], "trace", "trace")):
            timing = np.array([r["timing_rms"] for r in rows])
            free = np.array([r[f"grad3_{kind}"] / r[f"floor_{kind}"] for r in rows])
            fixed = np.array([r[f"grad1_{kind}"] / r[f"floor_{kind}"] for r in rows])
            ax.scatter(timing, free, marker=marker, s=22, facecolor=BLUE, edgecolor=BLUE, alpha=0.85, lw=0.6)
            ax.scatter(timing, fixed, marker=marker, s=22, facecolor="white", edgecolor=ORANGE, alpha=0.95, lw=0.9)
            pooled[kind].append((timing, free, fixed))
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.axhline(1.0, color=BLACK, lw=0.9, ls="--")
        ax.set_xlabel("rate error (RMS log ratio)")
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_ylabel(ylabel)
    rho = {}
    for key in ("iauc", "peak", "trace"):
        timing = np.concatenate([p[0] for p in pooled[key]])
        value = np.concatenate([p[1] for p in pooled[key]])
        rho[key] = float(spearmanr(timing, value).statistic)
    for ax, key in zip(axes, ("iauc", "peak", "trace")):
        ax.text(0.04, 0.95, rf"$\rho_s$ = {rho[key]:.2f}", transform=ax.transAxes, va="top", fontsize=9)
    for ax, letter in zip(axes, "abc"):
        _panel(ax, letter, dx=-0.3 if letter == "a" else -0.22)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="none", mfc=GREY, mec=GREY, ms=5, label="seed 0"),
               Line2D([], [], marker="s", ls="none", mfc=GREY, mec=GREY, ms=5, label="seed 1"),
               Line2D([], [], marker="o", ls="none", mfc=BLUE, mec=BLUE, ms=5, label=r"fit of $S_I$, $k_e$, $k_a$"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec=ORANGE, ms=5, mew=1.0, label=r"fit of $S_I$ only"),
               Line2D([], [], color=BLACK, ls="--", lw=0.9, label="true model")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, handletextpad=0.2, columnspacing=1.0,
               bbox_to_anchor=(0.5, 0.0))
    NUM.update({"PvpRhoIauc": rho["iauc"], "PvpRhoPeak": rho["peak"], "PvpRhoTrace": rho["trace"]})
    return _save(fig, "fig_prediction")


# ---------------------------------------------------------------------------------------------------------------
# Figure 3: the carbohydrate-error sweep
# ---------------------------------------------------------------------------------------------------------------

def figure_carbohydrate_sweep():
    plt = _plt()
    r6 = _load("phase13_summary.json").get("r6", {})
    cvs = [cv for cv in ("0", "0.1", "0.25", "0.5", "0.75") if r6.get(cv, {}).get("n")]
    if not cvs:
        return None
    x = np.array([float(cv) for cv in cvs])
    fig, ax = plt.subplots(figsize=(COLUMN, 3.0))
    for name, marker, color, label in (("tau1", "o", BLUE, r"$\tau_1$"), ("p", "s", ORANGE, r"$q$")):
        frac = np.array([100 * r6[cv][name]["bounded_interior"]["fraction"] for cv in cvs])
        low = np.array([100 * r6[cv][name]["bounded_interior"]["wilson"][0] for cv in cvs])
        high = np.array([100 * r6[cv][name]["bounded_interior"]["wilson"][1] for cv in cvs])
        ax.errorbar(x + (0.006 if name == "p" else -0.006), frac, yerr=[frac - low, high - frac], marker=marker, ms=5, color=color,
                    ls="-", lw=1.2, capsize=2, elinewidth=0.8, mfc=color, label=f"{label} bounded")
        from evaluation.stats_utils import wilson_ci
        cov = [(r6[cv][name]["nominal_coverage"] or {}) for cv in cvs]
        cover = np.array([100 * c.get("fraction", np.nan) for c in cov])
        ci = [wilson_ci(int(c["covered"]), int(c["n"])) for c in cov]
        lo, hi = np.array([100 * c.low for c in ci]), np.array([100 * c.high for c in ci])
        ax.errorbar(x + (0.012 if name == "p" else -0.012), cover, yerr=[cover - lo, hi - cover], marker=marker, ms=5, color=color,
                    ls="--", lw=1.2, mfc="white", mew=1.2, capsize=2, elinewidth=0.8, label=f"{label} covers truth")
        real = r6.get("real_polished", {}).get(name, {}).get("fraction")
        if real is not None:
            ax.axhline(100 * real, color=color, lw=0.9, ls=":")
            NUM[f"SweepReal{'Tau' if name == 'tau1' else 'P'}"] = 100 * real
    ax.plot([], [], color=GREY, lw=0.9, ls=":", label="real data, bounded")
    ax.set_xlabel("carbohydrate-log error (CV; 0 is exact)")
    ax.set_ylabel("% of subjects")
    ax.set_ylim(0, 102)
    ax.set_xticks(x)
    ax.set_xticklabels(cvs, fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, handlelength=2.0, columnspacing=0.9,
               labelspacing=0.2, handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.34)
    return _save(fig, "fig_sweep")


# ---------------------------------------------------------------------------------------------------------------
# Figure 4: a wrong fixed parameter is absorbed by the insulin-action scale
# ---------------------------------------------------------------------------------------------------------------

def figure_absorption():
    from evaluation.phase13_summary import _iauc_config
    from evaluation.phase15_summary import STANDARD_REPLICA
    from evaluation.recalibration import _rows
    plt = _plt()
    sets = [("0.7", {**STANDARD_REPLICA, "perturb": {"insulin_secretion": 0.7}}), ("1", STANDARD_REPLICA),
            ("1.3", {**STANDARD_REPLICA, "perturb": {"insulin_secretion": 1.3}})]
    fig, ax = plt.subplots(figsize=(COLUMN, 3.0))
    rng = np.random.default_rng(0)
    expected = {"0.7": np.log(0.7), "1": 0.0, "1.3": np.log(1.3)}
    for i, (label, replica) in enumerate(sets):
        rows = _rows(_iauc_config(replica))
        err, pinned = [], []
        for r in rows:
            truth = (r.get("replica_truth") or {}).get("theta_true")
            if not truth:
                continue
            err.append(np.log(r["theta_ml"]["insulin_sensitivity"]) - np.log(truth["insulin_sensitivity"]))
            pinned.append(r["distance_to_bound"]["insulin_sensitivity"] <= 0.01)
        err, pinned = np.array(err), np.array(pinned)
        jitter = rng.uniform(-0.17, 0.17, len(err))
        ax.scatter(i + jitter[~pinned], err[~pinned], s=18, marker="o", facecolor=BLUE, edgecolor=BLUE, alpha=0.85, lw=0.6,
                   label="interior estimate" if i == 0 else None)
        ax.scatter(i + jitter[pinned], err[pinned], s=22, marker="s", facecolor="white", edgecolor=ORANGE, lw=1.0,
                   label="on a bound of its box" if i == 0 else None)
        ax.plot([i - 0.3, i + 0.3], [np.median(err)] * 2, color=BLACK, lw=1.8)
        ax.plot([i - 0.3, i + 0.3], [expected[label]] * 2, color=GREY, lw=1.0, ls="--")
        NUM[f"AbsorbMedian{label.replace('.', '')}"] = float(np.median(err))
        NUM[f"AbsorbPinned{label.replace('.', '')}"] = int(pinned.sum())
        NUM[f"AbsorbN{label.replace('.', '')}"] = int(len(err))
    ax.axhline(0, color=BLACK, lw=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([r"$\times 0.7$", "unperturbed", r"$\times 1.3$"])
    ax.set_xlabel("insulin action of the truth vs. the fit")
    ax.set_ylabel(r"$\log(\hat S_I\,/\,S_I)$")
    ax.plot([], [], color=BLACK, lw=1.8, label="median")
    ax.plot([], [], color=GREY, lw=1.0, ls="--", label="log of the factor")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, handletextpad=0.3, columnspacing=0.9,
               labelspacing=0.2, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.19, right=0.98, top=0.98, bottom=0.33)
    return _save(fig, "fig_absorption")


def extra_numbers() -> None:
    """Numbers that answer the review: all-subject fractions, the cutoff the truth would need, and the bias without input error."""
    from evaluation import profile_lik
    from evaluation.identifiability_tools import CHI2_DELTA
    from evaluation.phase13_summary import _coords, _iauc_config
    from evaluation.recalibration import _rows, bounded_among_interior, calibrated_cutoff
    for cv, tag in (("0", "Exact"), ("0.25", "Quarter"), ("0.75", "ThreeQuarter")):
        rows = _rows(_coords({"replica": {"seed": 0, "cgm": True, "carb_cv": float(cv)}}))
        for key, name in (("log_tau1", "Tau"), ("log_p", "Q")):
            NUM[f"SweepAll{name}Bounded{tag}"] = int(sum(r["profiles"][key]["classification"]["verdict"] == "identifiable" for r in rows))
            cutoff = calibrated_cutoff(rows, key)["cutoff"]
            NUM[f"SweepSigma{name}{tag}"] = float(np.sqrt(cutoff / CHI2_DELTA))
            NUM[f"SweepCalBounded{name}{tag}"] = 100 * float(bounded_among_interior(rows, key, cutoff)["fraction"])
        NUM[f"SweepAllN{tag}"] = len(rows)
    rows = _rows(_iauc_config({"seed": 0, "cgm": True, "carb_cv": 0.0}))
    err = [np.log(r["theta_ml"]["insulin_sensitivity"]) - np.log(r["replica_truth"]["theta_true"]["insulin_sensitivity"]) for r in rows]
    NUM["AbsorbMedianExact"] = float(np.median(err))
    NUM["AbsorbPinnedExact"] = int(sum(r["distance_to_bound"]["insulin_sensitivity"] <= 0.01 for r in rows))
    NUM["AbsorbNExact"] = len(rows)
    real = _rows({**profile_lik.default_config(), "objective": "iauc", "bounds_scale": 1.0})
    for key, name in (("insulin_sensitivity", "SI"), ("gastric_emptying", "Ke"), ("carb_absorption", "Ka")):
        NUM[f"RealIauc{name}BoundedAll"] = int(sum(r["profiles"][key]["classification"]["verdict"] == "identifiable" for r in real))
    NUM["RealIaucN"] = len(real)

    def median_sigma(cv):
        found = _rows(_coords({"replica": {"seed": 0, "cgm": True, "carb_cv": cv}}))
        values = [float(list(r["noise"]["sigma"].values())[0]) for r in found
                  if r.get("noise") and isinstance(r["noise"].get("sigma"), dict)]
        return float(np.median(values)) if values else None
    a, b = median_sigma(0.0), median_sigma(0.1)
    if a and b:
        NUM["SweepSigmaChangePct"] = 100 * (b / a - 1)
    NUM["RealIaucSIInterior"] = len(bounded_rows(real))
    phase18_numbers()


def phase18_numbers() -> None:
    """Numbers of Amendment 14 for the text: re-estimated noise, extra seeds, a systematic error, chronological folds."""
    path = RESULTS / "phase18_summary.json"
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    for cv, tag in (("0.25", "Quarter"), ("0.75", "ThreeQuarter")):
        block = d["noise_final"][cv]
        NUM[f"FinalCoverTau{tag}"] = 100 * block["tau1"]["nominal_coverage"]["fraction"]
        NUM[f"FinalCoverQ{tag}"] = 100 * block["p"]["nominal_coverage"]["fraction"]
    sysb = d["r17"]["trace"]
    NUM["SysBoundedTau"] = 100 * sysb["tau1"]["bounded_interior"]["fraction"]
    NUM["SysBoundedQ"] = 100 * sysb["p"]["bounded_interior"]["fraction"]
    NUM["SysCoverTau"] = 100 * sysb["tau1"]["nominal_coverage"]["fraction"]
    NUM["SysCoverQ"] = 100 * sysb["p"]["nominal_coverage"]["fraction"]
    rise_q, drop_q = [], []
    for seed, word_ in (("seed1", "One"), ("seed2", "Two")):
        blocks = d["seeds"][seed]
        NUM[f"Seed{word_}TauBoundedZero"] = 100 * blocks["0"]["tau1"]["bounded_interior"]["fraction"]
        NUM[f"Seed{word_}TauBoundedHigh"] = 100 * blocks["0.75"]["tau1"]["bounded_interior"]["fraction"]
        NUM[f"Seed{word_}TauCoverZero"] = 100 * blocks["0"]["tau1"]["nominal_coverage"]["fraction"]
        NUM[f"Seed{word_}TauCoverHigh"] = 100 * blocks["0.75"]["tau1"]["nominal_coverage"]["fraction"]
        detail = d["h36"]["results"][seed]["detail"]["p"]
        rise_q.append(detail["bounded_rise_points"])
        drop_q.append(detail["coverage_drop_points"])
    NUM["SeedMinRiseQ"] = min(rise_q)
    NUM["SeedMinDropQ"] = min(drop_q)
    for scheme, tag in (("blocked", "Blocked"), ("forward", "Forward")):
        block = d["r19"][scheme]
        NUM[f"Chrono{tag}PersonalMean"] = block["grad3_minus_personal_mean"]["mean"]
        NUM[f"Chrono{tag}PersonalMeanLow"] = block["grad3_minus_personal_mean"]["ci95"][0]
        NUM[f"Chrono{tag}PersonalMeanHigh"] = block["grad3_minus_personal_mean"]["ci95"][1]


def bounded_rows(rows):
    return [r for r in rows if r["distance_to_bound"]["insulin_sensitivity"] > 0.01]


ALL = {"area": figure_what_the_area_sees, "prediction": figure_prediction_vs_parameters,
       "sweep": figure_carbohydrate_sweep, "absorption": figure_absorption}


def main() -> int:
    made = []
    extra_numbers()
    for name, fn in ALL.items():
        try:
            out = fn()
        except Exception as exc:
            import traceback
            print(f"  {name}: FAILED {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        print(f"  {name}: {'drawn' if out else 'skipped (inputs missing)'}")
        made.append(bool(out))
    NUMBERS.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS.write_text(json.dumps(NUM, indent=2, default=float), encoding="utf-8")
    print("wrote", NUMBERS)
    return 0 if all(made) else 1


if __name__ == "__main__":
    raise SystemExit(main())
