"""Figures for the TMLR manuscript, drawn from stored results only (PDF and PNG).

Replaces the AISTATS figure set (`figs_aistats.py`), whose legends, axis labels and log-axis ticks were
not good enough for a journal: a legend that showed one colour for three series, an axis label clipped at
the top, overlapping log ticks. Every figure here is built on a shared style, uses a colour-blind-safe
palette, labels its panels, and states in its caption (in the manuscript) what each panel shows.

F1  The exchange symmetry (analytic panel) and the weak Fisher direction against it, per cohort.
F2  Fisher information: eigenvalue spectra and the timing-block condition number by observable.
F3  Profile likelihood: two exemplar subjects, and the bounded-interval fractions by parameter, observable
    and box.
F4  The true-model replica: bounded S_I fractions (real data, seeds, random truth by tercile) and the
    what-if noise budgets.
F5  The identifiable combinations (tau1, p) in coordinates, real data against replica.
F6  Replication on Shanghai and Hall.
F7  Prediction: paired differences, CGMacros and Shanghai.
F8  The window dependence of the area's sensitivity to timing (H2).
F9  Leakage: correlation of S_I with HbA1c, raw and partial.
A1  Residual gut content breaking the symmetry; box sweep.   A2  Generic-rank spectra.
A3  Second model class (Dalla Man).   A4  Optimizer-robustness of the diagnostic (H10).

A figure whose inputs are missing is skipped and reported, never drawn from placeholder data.

Run:  python -m evaluation.figs_tmlr
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation.results_io import RESULTS, ROOT, largest_matching

OUT = ROOT / "paper" / "figures" / "tmlr"
# Okabe-Ito: distinguishable under the common forms of colour-vision deficiency.
C = {"iauc": "#0072B2", "iauc_centroid": "#E69F00", "trace": "#009E73", "grey": "#7F7F7F",
     "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "black": "#222222"}
LABEL = {"iauc": "iAUC", "iauc_centroid": "iAUC + centroid", "trace": "full trace"}
PARAM = {"insulin_sensitivity": r"$S_I$", "gastric_emptying": r"$k_e$", "carb_absorption": r"$k_a$"}
DELTA = 1.9207295


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 130, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8, "lines.linewidth": 1.4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
    return plt


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return [OUT / f"{name}.pdf", OUT / f"{name}.png"]


def _panel(ax, letter: str) -> None:
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def _load(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _p2() -> dict:
    return _load("phase2_summary.json")


def _p8() -> dict:
    return _load("phase8_summary.json")


def _p9() -> dict:
    return _load("phase9_summary.json")


def _wilson_bars(ax, x, fraction, wilson, color, width=0.18, alpha=1.0, label=None):
    ax.bar(x, 100 * fraction, width, color=color, alpha=alpha, label=label, linewidth=0)
    ax.plot([x, x], [100 * wilson[0], 100 * wilson[1]], color=C["black"], lw=0.8)


# --------------------------------------------------------------------------------------------------
# F1
# --------------------------------------------------------------------------------------------------

def fig1_symmetry():
    from evaluation.ladder import a4_results, trade_off_cosine
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    t = np.linspace(0, 360, 721)

    def appearance(ke, ka, leftover=0.0):
        """Rate of appearance for a unit meal; `leftover` is residual intestinal content (fraction)."""
        main = ke * ka / (ka - ke) * (np.exp(-ke * t) - np.exp(-ka * t))
        extra = leftover * ka * np.exp(-ka * t)
        return main + extra
    ke, ka = 0.030, 0.016
    ax.plot(t, appearance(ke, ka), color=C["iauc"], lw=2.2, label=r"$(k_e,k_a)=(0.030,\,0.016)$")
    ax.plot(t, appearance(ka, ke), color=C["red"], lw=1.2, ls="--", label=r"swapped $(0.016,\,0.030)$")
    ax.plot(t, appearance(ke, ka, 0.05), color=C["grey"], lw=1.0, ls=":", label="5% residual gut, first pair")
    ax.plot(t, appearance(ka, ke, 0.05), color=C["purple"], lw=1.0, ls=":", label="5% residual gut, swapped")
    ax.set_xlabel("minutes after the meal")
    ax.set_ylabel("rate of appearance\n(fraction of meal per min)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _panel(ax, "a")

    ax = axes[1]
    drawn = False
    for key, box, cohort, label, color in (("cgmacros", 1.0, "cgmacros", "CGMacros", C["iauc"]),
                                           ("shanghai", 1.0, "shanghai", "Shanghai", C["trace"]),
                                           ("hall", 1.0, "hall", "Hall", C["iauc_centroid"])):
        rows = a4_results("iauc", box, cohort=cohort)
        if not rows:
            continue
        cos = np.array([trade_off_cosine(r["timing_block"]["weak_direction"], r["theta_ml"]["gastric_emptying"],
                                         r["theta_ml"]["carb_absorption"]) for r in rows.values()])
        miss = np.sort(np.maximum(1.0 - np.abs(cos), 1e-9))
        ax.step(miss, np.arange(1, len(miss) + 1) / len(miss), where="post", color=color,
                label=f"{label} (n = {len(miss)})")
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.axvline(0.2, color=C["grey"], lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(1e-9, 1.0)
    ax.set_xlabel("1 - |cosine| of the weak Fisher direction\nwith the exchange direction")
    ax.set_ylabel("cumulative share\nof subjects")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    _panel(ax, "b")
    fig.tight_layout()
    return _save(fig, "f1_symmetry")


# --------------------------------------------------------------------------------------------------
# F2
# --------------------------------------------------------------------------------------------------

def fig2_fisher():
    from evaluation.ladder import a4_results
    plt = _plt()
    data = {o: a4_results(o, 1.0) for o in ("iauc", "iauc_centroid", "trace")}
    if not all(data.values()):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    width = 0.25
    for j, objective in enumerate(data):
        ev = np.array([r["fisher"]["eigenvalues"] for r in data[objective].values()])
        for i in range(3):
            bp = ax.boxplot(np.log10(np.maximum(ev[:, i], 1e-16)), positions=[i + (j - 1) * width],
                            widths=width * 0.85, patch_artist=True, showfliers=False,
                            medianprops={"color": "black", "lw": 1.0}, whiskerprops={"lw": 0.8},
                            capprops={"lw": 0.8}, boxprops={"lw": 0.8})
            bp["boxes"][0].set(facecolor=C[objective], alpha=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([r"$\lambda_1$ (weakest)", r"$\lambda_2$", r"$\lambda_3$ (strongest)"])
    ax.set_ylabel(r"$\log_{10}$ Fisher eigenvalue")
    for objective in data:
        ax.plot([], [], "s", color=C[objective], label=LABEL[objective])
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    _panel(ax, "a")

    ax = axes[1]
    ladder = _p2().get("ladder", {})
    boxes = [b for b in ("0.5x", "1x", "2x") if b in ladder]
    for j, objective in enumerate(("iauc", "iauc_centroid", "trace")):
        values = [ladder[b].get("h3_common_reference", {}).get("median_condition_number", {}).get(objective)
                  for b in boxes]
        if any(v is None for v in values):
            continue
        ax.bar(np.arange(len(boxes)) + (j - 1) * 0.26, values, 0.24, color=C[objective], label=LABEL[objective])
    ax.set_yscale("log")
    ax.set_xticks(range(len(boxes)))
    ax.set_xticklabels(boxes)
    ax.set_xlabel("parameter box")
    ax.set_ylabel("timing-block condition number")
    _panel(ax, "b")
    fig.tight_layout()
    return _save(fig, "f2_fisher")


# --------------------------------------------------------------------------------------------------
# F3
# --------------------------------------------------------------------------------------------------

def fig3_profiles():
    plt = _plt()
    rows = largest_matching("A5_profile", lambda p: p["objective"] == "iauc" and abs(p["bounds_scale"] - 1.0) < 1e-12
                            and p.get("cohort", "cgmacros") == "cgmacros" and p.get("replica") is None
                            and p.get("carb_scale") in (None, 1, 1.0) and p.get("polish") is None
                            and p.get("parameterization", "rates") == "rates")
    if not rows:
        return None
    ids = sorted(rows)
    steep = max(ids, key=lambda s: rows[s]["profiles"]["insulin_sensitivity"]["classification"]["max_rise"])
    flat = min(ids, key=lambda s: rows[s]["profiles"]["gastric_emptying"]["classification"]["max_rise"])
    fig = plt.figure(figsize=(6.8, 4.9))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.55, wspace=0.28)
    for r_i, sid in enumerate((steep, flat)):
        for c_i, parameter in enumerate(PARAM):
            ax = fig.add_subplot(grid[r_i, c_i])
            entry = rows[sid]["profiles"][parameter]
            g, d, n_in = np.array(entry["grid"]), np.array(entry["profile"]) - entry["reference"], entry["n_in_box"]
            ax.plot(g[:n_in], d[:n_in], "-o", ms=2.8, color=C["iauc"])
            if len(g) > n_in:
                ax.plot(g[n_in - 1:], d[n_in - 1:], "--o", ms=2.8, color=C["grey"])
            ax.axhline(DELTA, color=C["red"], lw=0.9, ls=":")
            ax.set_xscale("log")
            ax.set_xlabel(PARAM[parameter])
            ax.set_ylim(-0.2, min(10.0, max(4.0, float(np.nanmax(d[:n_in])) * 1.05)))
            if c_i == 0:
                ax.set_ylabel("profile NLL above minimum")
            ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
            if r_i == 0 and c_i == 0:
                _panel(ax, "a")
            if r_i == 1 and c_i == 0:
                _panel(ax, "b")
    return _save(fig, "f3_profile_curves"), fig3_fractions()


def fig3_fractions():
    plt = _plt()
    p2 = _p2()
    series = [("profile_likelihood", "iauc"), ("profile_likelihood_iauc_centroid", "iauc_centroid"),
              ("profile_likelihood_trace", "trace")]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), sharey=True)
    drawn = False
    boxes = ["0.5x", "1x", "2x"]
    for ax, parameter in zip(axes, PARAM):
        for s_i, (key, objective) in enumerate(series):
            for b_i, box in enumerate(boxes):
                block = p2.get(key, {}).get(box)
                if not block or not block.get("n_subjects"):
                    continue
                d = block["parameters"][parameter]
                x = b_i + (s_i - 1) * 0.27
                _wilson_bars(ax, x, d["fraction"], d["wilson"], C[objective], width=0.25,
                             label=LABEL[objective] if (b_i == 0 and parameter == "insulin_sensitivity") else None)
                drawn = True
        ax.set_xticks(range(3))
        ax.set_xticklabels(boxes)
        ax.set_title(PARAM[parameter])
        ax.set_xlabel("parameter box")
    if not drawn:
        plt.close(fig)
        return None
    axes[0].set_ylabel("bounded 95% interval\n(% of subjects)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.5)
    for ax, letter in zip(axes, "abc"):
        _panel(ax, letter)
    fig.tight_layout()
    return _save(fig, "f3_bounded_fractions")


# --------------------------------------------------------------------------------------------------
# F4
# --------------------------------------------------------------------------------------------------

def fig4_replica():
    p8, p9 = _p8(), _p9()
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.7), gridspec_kw={"width_ratios": [1.15, 1.0, 1.05]})
    # (a) S_I bounded among interior: real, replica seeds, pooled
    ax = axes[0]
    p2 = _p2().get("profile_likelihood", {}).get("1x", {}).get("S_I_bounded_among_interior_for_S_I")
    bars = []
    if p2:
        bars.append(("real data", p2["fraction"], p2["wilson"], C["grey"]))
    seeds = p9.get("h17", {}).get("per_seed", {})
    for seed, d in sorted(seeds.items()):
        si = d.get("S_I_interior_bounded") if d.get("n") else None
        if si:
            bars.append((f"seed {seed}", si["fraction"], si["wilson"], C["iauc"]))
    pooled = p9.get("h17", {}).get("pooled_S_I_interior_bounded")
    if pooled:
        bars.append(("pooled", pooled["fraction"], pooled["wilson"], C["black"]))
    if not seeds:
        si = p8.get("h13", {}).get("S_I_interior_bounded")
        if si:
            bars.append(("replica", si["fraction"], si["wilson"], C["iauc"]))
    for i, (label, f, w, color) in enumerate(bars):
        _wilson_bars(ax, i, f, w, color, width=0.65)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], rotation=40, ha="right", fontsize=7.5)
    ax.axhline(50, color=C["red"], lw=0.9, ls=":")
    ax.set_ylabel(r"$S_I$ interval bounded" + "\n(% of interior subjects)")
    ax.set_ylim(0, 100)
    _panel(ax, "a")

    # (b) random-truth recovery by tercile of the true value
    ax = axes[1]
    recovery = p9.get("h6_synthetic", {}).get("recovery", {})
    drawn = False
    for j, (parameter, color) in enumerate((("insulin_sensitivity", C["iauc"]), ("gastric_emptying", C["iauc_centroid"]),
                                             ("carb_absorption", C["trace"]))):
        r = recovery.get(parameter)
        if not r:
            continue
        for t, tercile in enumerate(("low", "middle", "high")):
            d = r["terciles"][tercile]
            if d["fraction"] is None:
                continue
            x = t + (j - 1) * 0.27
            _wilson_bars(ax, x, d["fraction"], d["wilson"], color, width=0.25,
                         label=PARAM[parameter] if t == 0 else None)
            drawn = True
    ax.set_xticks(range(3))
    ax.set_xticklabels(["low", "middle", "high"])
    ax.set_xlabel("true value, tercile of the box")
    ax.set_ylabel("interval bounded\n(% of interior subjects)")
    ax.set_ylim(0, 100)
    if drawn:
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    else:
        ax.text(0.5, 0.5, "synthetic study\nnot available", transform=ax.transAxes, ha="center", va="center",
                color=C["grey"])
    _panel(ax, "b")

    # (c) the what-if noise budgets
    ax = axes[2]
    settings = p8.get("h14", {}).get("settings", {})
    names = [("cgm_off_carb_0", "none"), ("cgm_off_carb_025", "carbs\n25%"), ("cgm_on_carb_0", "CGM"),
             ("cgm_on_carb_025", "CGM +\ncarbs 25%"), ("cgm_on_carb_05", "CGM +\ncarbs 50%")]
    k = 0
    for key, label in names:
        d = settings.get(key, {})
        if not d.get("n"):
            continue
        ax.errorbar(k, d["ratio"], yerr=[[d["ratio"] - d["ratio_ci95"][0]], [d["ratio_ci95"][1] - d["ratio"]]],
                    fmt="o", color=C["iauc"], ms=4, capsize=2, lw=1.0)
        ax.text(k + 0.05, -0.04, label.replace("\n", " "), ha="right", va="top", fontsize=7, rotation=40,
                rotation_mode="anchor", transform=ax.get_xaxis_transform())
        k += 1
    ax.axhline(0.7, color=C["grey"], lw=0.8, ls="--")
    ax.axhline(0.9, color=C["grey"], lw=0.8, ls="--")
    ax.set_xticks([])
    ax.set_xlim(-0.6, k - 0.4)
    ax.set_ylabel("replica / real\nheld-out iAUC error")
    ax.set_ylim(0, 1.15)
    _panel(ax, "c")
    fig.tight_layout()
    return _save(fig, "f4_replica")


# --------------------------------------------------------------------------------------------------
# F5
# --------------------------------------------------------------------------------------------------

def fig5_coordinates():
    p8, p9 = _p8(), _p9()
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    series = [("real, Adam estimate", lambda o, k: (p8.get("h11", {}).get(o, {}).get(k) or {}).get("interior"),
               C["grey"]),
              ("real, polished", lambda o, k: ((p9.get("h21", {}).get("h11_on_polished_estimates", {}).get(o, {}) or {})
                                               .get(k) or {}).get("interior"), C["iauc"]),
              ("true-model replica, polished", lambda o, k: (p9.get("h18", {}).get(o, {}) or {}).get(
                  "tau1" if k == "tau1" else "p"), C["trace"])]
    for ax, (key, title) in zip(axes, (("tau1", r"$\tau_1=1/k_e+1/k_a$"), ("p", r"$p=1/(k_e k_a)$"))):
        for j, (label, getter, color) in enumerate(series):
            for i, objective in enumerate(("iauc", "iauc_centroid", "trace")):
                d = getter(objective, key)
                if not d or d.get("fraction") is None:
                    continue
                x = i + (j - 1) * 0.27
                _wilson_bars(ax, x, d["fraction"], d["wilson"], color, width=0.25,
                             label=label if i == 0 else None)
        ax.set_xticks(range(3))
        ax.set_xticklabels([LABEL[o] for o in ("iauc", "iauc_centroid", "trace")], fontsize=7.5)
        ax.set_title(title)
    axes[0].set_ylabel("interval bounded\n(% of interior subjects)")
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    for ax, letter in zip(axes, "ab"):
        _panel(ax, letter)
    fig.tight_layout()
    return _save(fig, "f5_coordinates")


# --------------------------------------------------------------------------------------------------
# F6
# --------------------------------------------------------------------------------------------------

def fig6_cohorts():
    from evaluation import ladder, profile_lik
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    cohorts = (("CGMacros", "cgmacros", C["iauc"]), ("Shanghai", "shanghai", C["trace"]), ("Hall", "hall", C["iauc_centroid"]))
    ax = axes[0]
    for j, (label, cohort, color) in enumerate(cohorts):
        rows = ladder.a4_results("iauc", 1.0, cohort=cohort)
        if not rows:
            continue
        for i, parameter in enumerate(PARAM):
            f = np.mean([r["fit"]["at_bound"].get(parameter) is not None for r in rows.values()])
            ax.bar(i + (j - 1) * 0.27, 100 * f, 0.25, color=color, label=label if i == 0 else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(PARAM.values()))
    ax.set_ylabel("estimate on a bound\n(% of subjects)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _panel(ax, "a")
    ax = axes[1]
    for j, (label, cohort, color) in enumerate(cohorts):
        block = profile_lik.summarize({**profile_lik.default_config(), "objective": "iauc", "bounds_scale": 1.0,
                                       "cohort": cohort})
        if not block.get("n_subjects"):
            continue
        for i, parameter in enumerate(PARAM):
            d = block["parameters"][parameter]
            _wilson_bars(ax, i + (j - 1) * 0.27, d["fraction"], d["wilson"], color, width=0.25)
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(PARAM.values()))
    ax.set_ylabel("bounded 95% interval\n(% of subjects)")
    ax.set_ylim(0, 40)
    _panel(ax, "b")
    fig.tight_layout()
    return _save(fig, "f6_cohorts")


# --------------------------------------------------------------------------------------------------
# F7
# --------------------------------------------------------------------------------------------------

PAIR_LABEL = {"grad3-grid3": "gradient vs grid (3 parameters)", "grad1-grid1": "gradient vs grid ($S_I$ only)",
              "grad3-grad1": "3 parameters vs $S_I$ only", "grad3_trace-grad3": "trace vs iAUC objective",
              "grad3-personal_mean": "gradient vs personal mean", "grad3-rf": "gradient vs random forest",
              "grad3-snpe": "gradient vs SNPE", "grad3-grid1": "gradient (3) vs grid ($S_I$ only)",
              "grad1_trace-grad1": "trace vs iAUC ($S_I$ only)"}


def fig7_prediction():
    p2, p9 = _p2(), _p9()
    plt = _plt()
    primary = [(f"{c['first']}-{c['second']}", c) for c in p2.get("prediction", {}).get("comparisons", [])
               if "ci95" in c]
    order = ["grad3-grid3", "grad1-grid1", "grad3-grad1", "grad3_trace-grad3", "grad1_trace-grad1",
             "grad3-personal_mean", "grad3-rf", "grad3-snpe"]
    primary = {k: c for k, c in primary}
    if not primary:
        return None
    shanghai = (p9.get("h20", {}).get("metrics", {}).get("iauc", {}).get("comparisons") or {})
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    keys = [k for k in order if k in primary]
    ypos = {k: len(keys) - 1 - i for i, k in enumerate(keys)}
    for k in keys:
        c = primary[k]
        ax.plot([c["ci95"]["low"], c["ci95"]["high"]], [ypos[k] + 0.12] * 2, color=C["iauc"], lw=2.0)
        ax.plot(c["mean_difference"], ypos[k] + 0.12, "o", color=C["iauc"], ms=4)
        s = shanghai.get(k)
        if s and s.get("ci95"):
            ax.plot([s["ci95"]["low"], s["ci95"]["high"]], [ypos[k] - 0.12] * 2, color=C["trace"], lw=2.0)
            ax.plot(s["mean_difference"], ypos[k] - 0.12, "s", color=C["trace"], ms=4)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvspan(-150, 150, color=C["grey"], alpha=0.14, lw=0)
    ax.set_yticks([ypos[k] for k in keys])
    ax.set_yticklabels([PAIR_LABEL.get(k, k) for k in keys])
    ax.set_xlabel(r"held-out iAUC MAE difference, mg/dL$\cdot$min (negative: first is better)")
    ax.plot([], [], "o-", color=C["iauc"], label="CGMacros")
    if shanghai:
        ax.plot([], [], "s-", color=C["trace"], label="Shanghai")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, "f7_prediction")


# --------------------------------------------------------------------------------------------------
# F8, F9
# --------------------------------------------------------------------------------------------------

def fig8_window():
    summary = _p2().get("moment_checks")
    if not summary or not summary.get("n_subjects"):
        return None
    plt = _plt()
    windows = summary["windows"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, key, title, threshold in ((axes[0], "linear", "linearized engine", 0.02),
                                      (axes[1], "nonlinear_default", "nonlinear engine", 0.25)):
        have = [w for w in windows if str(w) in summary[key]]
        for parameter, label, color in (("ke", r"$k_e$", C["iauc"]), ("ka", r"$k_a$", C["trace"])):
            med = [summary[key][str(w)][parameter]["median"] for w in have]
            lo = [summary[key][str(w)][parameter]["q1"] for w in have]
            hi = [summary[key][str(w)][parameter]["q3"] for w in have]
            ax.plot(have, med, "-o", ms=3.5, color=color, label=label)
            ax.fill_between(have, lo, hi, color=color, alpha=0.18, lw=0)
        ax.axhline(threshold, color=C["grey"], ls="--", lw=0.9)
        ax.axhline(1.0, color=C["red"], ls=":", lw=0.9)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("window after the meal (min)")
    axes[0].set_ylabel(r"$|\partial\mathrm{iAUC}/\partial\log k|$" + "\n" + r"$/\,|\partial\mathrm{iAUC}/\partial\log S_I|$")
    axes[0].legend(frameon=False, fontsize=7.5)
    for ax, letter in zip(axes, "ab"):
        _panel(ax, letter)
    fig.tight_layout()
    return _save(fig, "f8_window")


def fig9_leakage():
    stats = _p8().get("leakage", {}).get("statistics")
    if not stats:
        return None
    plt = _plt()
    rows = [("raw_snpe", "SNPE, raw", C["iauc_centroid"]), ("partial_snpe", "SNPE, partial", C["iauc_centroid"]),
            ("raw_gradient", "gradient, raw", C["iauc"]), ("partial_gradient", "gradient, partial", C["iauc"]),
            ("snpe_vs_own_baseline_feature", "SNPE vs its baseline feature", C["red"]),
            ("raw_difference", "gradient $-$ SNPE, raw", C["black"]), ("partial_difference", "gradient $-$ SNPE, partial", C["black"])]
    rows = [r for r in rows if r[0] in stats]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for i, (key, label, color) in enumerate(rows[::-1]):
        d = stats[key]
        ax.plot([d["low"], d["high"]], [i, i], color=color, lw=2.0)
        ax.plot(d["estimate"], i, "o", color=color, ms=4.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows[::-1]])
    ax.set_xlabel("Spearman correlation (95% bootstrap interval)")
    fig.tight_layout()
    return _save(fig, "f9_leakage")


# --------------------------------------------------------------------------------------------------
# Appendix
# --------------------------------------------------------------------------------------------------

def figA1_gut_and_boxes():
    p2 = _p2()
    rows = p2.get("gut_sweep", {}).get("sweep", {}).get("rows")
    sweep = p2.get("bounds_sweep_fisher_fit")
    if not rows and not sweep:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax = axes[0]
    if rows:
        x = [r["gut_content_percent_of_meal"] for r in rows]
        ax.plot(x, [r["max_abs_glucose_difference_mg_dl"] for r in rows], "-o", ms=3.5, color=C["iauc"], label="largest")
        ax.plot(x, [r["median_abs_glucose_difference_mg_dl"] for r in rows], "-o", ms=3.5, color=C["trace"], label="median")
        ax.axhspan(5, 10, color=C["grey"], alpha=0.18, lw=0)
        ax.set_xlabel("residual gut content (% of the meal)")
        ax.set_ylabel(r"glucose change under" + "\n" + r"$k_e\leftrightarrow k_a$ (mg/dL)")
        ax.legend(frameon=False, fontsize=7.5)
    _panel(ax, "a")
    ax = axes[1]
    if sweep:
        boxes = ["0.5x", "1x", "2x"]
        for parameter, color in zip(PARAM, (C["iauc"], C["iauc_centroid"], C["trace"])):
            ax.plot(range(3), [100 * sweep[b]["pinned"][parameter]["fraction"] for b in boxes], "-o", ms=3.5,
                    color=color, label=PARAM[parameter])
        ax.set_xticks(range(3))
        ax.set_xticklabels(boxes)
        ax.set_xlabel("box width relative to the primary box")
        ax.set_ylabel("estimate on a bound\n(% of subjects)")
        ax.legend(frameon=False, fontsize=7.5)
    _panel(ax, "b")
    fig.tight_layout()
    return _save(fig, "a1_gut_and_boxes")


def figA2_generic_rank():
    ranks = _p2().get("generic_rank")
    if not ranks:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for (name, d), color in zip(ranks.items(), (C["iauc"], C["iauc_centroid"], C["trace"], C["purple"])):
        ax.semilogy(range(1, 4), d["singular_values_median"][::-1], "-o", ms=3.5, color=color,
                    label=name.replace("_", " "))
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("singular value index (smallest first)")
    ax.set_ylabel("median singular value (scaled)")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    return _save(fig, "a2_generic_rank")


def figA3_dalla_man():
    from evaluation import dalla_man_identifiability as dm
    block = dm.summarize()
    rows = list(largest_matching(dm.ANALYSIS_ID, lambda p: True).values())
    if not block.get("n_subjects"):
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax = axes[0]
    ev = np.array([r["fisher"]["eigenvalues"] for r in rows])
    ax.boxplot(np.log10(np.maximum(ev, 1e-16)), showfliers=False, patch_artist=True,
               boxprops={"facecolor": C["sky"], "lw": 0.8}, medianprops={"color": "black"})
    ax.set_xticklabels([f"$\\lambda_{i}$" for i in range(1, ev.shape[1] + 1)])
    ax.set_ylabel(r"$\log_{10}$ Fisher eigenvalue")
    _panel(ax, "a")
    ax = axes[1]
    names = [n for n in ("Vmx", "kabs", "kmax", "kmin") if n in block["parameters"]]
    for i, n in enumerate(names):
        d = block["parameters"][n]
        _wilson_bars(ax, i, d["fraction"], d["wilson"], C["iauc"], width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([r"$V_{mx}$", r"$k_{abs}$", r"$k_{max}$", r"$k_{min}$"][:len(names)])
    ax.set_ylabel("bounded 95% interval\n(% of subjects)")
    ax.set_ylim(0, 60)
    _panel(ax, "b")
    fig.tight_layout()
    return _save(fig, "a3_dalla_man")


def figA4_robustness():
    h10 = _p9().get("h10", {})
    if not h10.get("factors"):
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    labels, values = [], []
    for factor, label in (("initialization", "initialization"), ("bounds", "box"), ("parameterization", "log scale"),
                          ("optimizer", "L-BFGS")):
        v = h10["factors"].get(factor, {}).get("median_of_mean_tau")
        if v is not None:
            labels.append(label)
            values.append(v)
    ax.bar(range(len(values)), values, 0.55, color=C["iauc"])
    ax.axhline(0.8, color=C["red"], ls=":", lw=0.9)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.2, 1.05)
    ax.set_ylabel("median Kendall tau with\nthe reference ordering")
    fig.tight_layout()
    return _save(fig, "a4_robustness")


ALL = {"f1": fig1_symmetry, "f2": fig2_fisher, "f3": fig3_profiles, "f4": fig4_replica, "f5": fig5_coordinates,
       "f6": fig6_cohorts, "f7": fig7_prediction, "f8": fig8_window, "f9": fig9_leakage,
       "a1": figA1_gut_and_boxes, "a2": figA2_generic_rank, "a3": figA3_dalla_man, "a4": figA4_robustness}


def main() -> int:
    made, skipped = [], []
    for name, fn in ALL.items():
        try:
            out = fn()
        except Exception as exc:                     # a broken figure must not hide the others
            import traceback
            print(f"  {name}: FAILED {type(exc).__name__}: {exc}")
            traceback.print_exc()
            skipped.append(name)
            continue
        (made if out else skipped).append(name)
        print(f"  {name}: {'drawn' if out else 'skipped (inputs missing)'}")
    print(f"{len(made)} drawn, {len(skipped)} skipped")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
