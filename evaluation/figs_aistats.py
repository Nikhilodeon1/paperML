"""Figures for the revision, drawn from stored results only (PDF and PNG, 9 pt minimum text).

F1  Fisher eigenvalue spectra of the 3x3 information matrix, one box per eigenvalue, per objective.
F2  Timing-to-S_I sensitivity ratio against window length, linearized and nonlinear engine (H2).
F3  Profile-likelihood curves for two representative subjects, and the fraction of subjects with a
    bounded interval per parameter, objective and box.
F4  Prediction comparison: paired differences with bootstrap intervals for the pre-specified pairs.
A1  Generic-rank singular value spectra.   A2  Exchange-symmetry break against residual gut content.
A3  Fraction of subjects on a bound against box width.

A figure whose inputs are missing is skipped and reported, never drawn from placeholder data. Figure 5
of the plan (diagnostic against profile likelihood) needs the Phase 5 analyses, which have not been
run, and is therefore absent.

Run:  python -m evaluation.figs_aistats
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation.results_io import RESULTS, ROOT, largest_matching

OUT = ROOT / "paper" / "figures" / "aistats"
PALETTE = {"iauc": "#1f6f8b", "iauc_centroid": "#c97b1a", "trace": "#4c9a5b", "grey": "#8a8f98",
           "bad": "#b04a4a"}
LABEL = {"iauc": "iAUC", "iauc_centroid": "iAUC + centroid", "trace": "full trace"}
PARAM_LABEL = {"insulin_sensitivity": r"$S_I$", "gastric_emptying": r"$k_e$",
               "carb_absorption": r"$k_a$"}
SIGNIFICANCE_DELTA = 1.9207295


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                         "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save(fig, name: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        paths.append(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def _summary() -> dict:
    path = RESULTS / "phase2_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fig1_spectra(box: float = 1.0):
    from evaluation.ladder import a4_results
    plt = _plt()
    data = {o: a4_results(o, box) for o in ("iauc", "iauc_centroid", "trace")}
    if not all(data.values()):
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.24
    for j, objective in enumerate(data):
        ev = np.array([r["fisher"]["eigenvalues"] for r in data[objective].values()])
        for i in range(3):
            pos = i + (j - 1) * width
            values = np.maximum(ev[:, i], 1e-16)
            bp = ax.boxplot(np.log10(values), positions=[pos], widths=width * 0.85,
                            patch_artist=True, showfliers=False, medianprops={"color": "black"})
            bp["boxes"][0].set(facecolor=PALETTE[objective], alpha=0.85)
    ax.set_xticks(range(3))
    ax.set_xticklabels([r"$\lambda_1$ (weakest)", r"$\lambda_2$", r"$\lambda_3$ (strongest)"])
    ax.set_ylabel(r"$\log_{10}$ eigenvalue of the Fisher information")
    for objective in data:
        ax.plot([], [], "s", color=PALETTE[objective], label=LABEL[objective])
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, "f1_fisher_spectra")


def fig2_window():
    summary = _summary().get("moment_checks")
    if not summary or not summary.get("n_subjects"):
        return None
    plt = _plt()
    windows = summary["windows"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, key, title in ((axes[0], "linear", "linearized engine"),
                           (axes[1], "nonlinear_default", "nonlinear engine")):
        have = [w for w in windows if str(w) in summary[key]]
        for parameter, label, color in (("ke", r"$k_e$", PALETTE["iauc"]),
                                        ("ka", r"$k_a$", PALETTE["trace"])):
            med = [summary[key][str(w)][parameter]["median"] for w in have]
            lo = [summary[key][str(w)][parameter]["q1"] for w in have]
            hi = [summary[key][str(w)][parameter]["q3"] for w in have]
            ax.plot(have, med, "-o", color=color, label=label)
            ax.fill_between(have, lo, hi, color=color, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("window after the meal (min)")
    axes[0].set_ylabel(r"$|\partial\,\mathrm{iAUC}/\partial\log k|\,/\,"
                       r"|\partial\,\mathrm{iAUC}/\partial\log S_I|$")
    axes[0].axhline(0.02, color=PALETTE["grey"], ls="--", lw=1)
    axes[1].axhline(0.25, color=PALETTE["grey"], ls="--", lw=1)
    axes[0].legend(frameon=False)
    return _save(fig, "f2_window_ratio")


def fig3_profiles(box: float = 1.0):
    plt = _plt()
    rows = largest_matching("A5_profile", lambda p: p["objective"] == "iauc"
                            and abs(p["bounds_scale"] - box) < 1e-12)
    if not rows:
        return None
    ids = sorted(rows)
    # One subject whose S_I profile rises steeply, one whose timing profiles are flattest.
    steep = max(ids, key=lambda s: rows[s]["profiles"]["insulin_sensitivity"]["classification"]["max_rise"])
    flat = min(ids, key=lambda s: rows[s]["profiles"]["gastric_emptying"]["classification"]["max_rise"])
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharey="row")
    for r_i, sid in enumerate((steep, flat)):
        for c_i, parameter in enumerate(PARAM_LABEL):
            entry = rows[sid]["profiles"][parameter]
            grid = np.array(entry["grid"])
            delta = np.array(entry["profile"]) - entry["reference"]
            n_in = entry["n_in_box"]
            ax = axes[r_i, c_i]
            ax.plot(grid[:n_in], delta[:n_in], "-o", ms=3, color=PALETTE["iauc"])
            if len(grid) > n_in:
                ax.plot(grid[n_in - 1:], delta[n_in - 1:], "--o", ms=3, color=PALETTE["grey"])
            ax.axhline(SIGNIFICANCE_DELTA, color=PALETTE["bad"], lw=1, ls=":")
            ax.set_xscale("log")
            ax.set_xlabel(PARAM_LABEL[parameter])
            if c_i == 0:
                ax.set_ylabel("profile NLL above minimum")
            ax.set_ylim(-0.2, min(12.0, max(4.0, float(np.nanmax(delta[:n_in])) * 1.05)))
    return _save(fig, "f3a_profile_curves")


def fig3_fractions():
    summary = _summary()
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    boxes = ["0.5x", "1x", "2x"]
    width = 0.1
    series = [("profile_likelihood", "iauc"), ("profile_likelihood_iauc_centroid", "iauc_centroid"),
              ("profile_likelihood_trace", "trace")]
    drawn = False
    for s_i, (key, objective) in enumerate(series):
        for p_i, parameter in enumerate(PARAM_LABEL):
            for b_i, box in enumerate(boxes):
                block = summary.get(key, {}).get(box)
                if not block or not block.get("n_subjects"):
                    continue
                d = block["parameters"][parameter]
                x = p_i + (s_i * 3 + b_i - 4) * width
                ax.bar(x, 100 * d["fraction"], width * 0.9, color=PALETTE[objective],
                       alpha=0.45 + 0.25 * b_i)
                ax.plot([x, x], [100 * d["wilson"][0], 100 * d["wilson"][1]], color="black", lw=0.8)
                drawn = True
    if not drawn:
        return None
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(PARAM_LABEL.values()))
    ax.set_ylabel("subjects with a bounded 95% interval (%)")
    for objective in ("iauc", "iauc_centroid", "trace"):
        ax.bar([], [], color=PALETTE[objective], label=LABEL[objective])
    ax.legend(frameon=False, title="darker = wider box (0.5x, 1x, 2x)", title_fontsize=9)
    return _save(fig, "f3b_bounded_fractions")


def fig4_prediction():
    prediction = _summary().get("prediction")
    if not prediction:
        return None
    plt = _plt()
    wanted = (("grad3", "grid1"), ("grad3", "rf"), ("grad3", "snpe"), ("grad3", "personal_mean"))
    pairs = [c for c in prediction["comparisons"] if "ci95" in c and (
        c["group"] in ("equivalence", "superiority") or (c["first"], c["second"]) in wanted)]
    fig, ax = plt.subplots(figsize=(5.4, 0.45 * len(pairs) + 1.0))
    for i, c in enumerate(pairs[::-1]):
        ci = c["ci95"]
        ax.plot([ci["low"], ci["high"]], [i, i], color=PALETTE["iauc"], lw=2)
        ax.plot(c["mean_difference"], i, "o", color="black", ms=4)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvspan(-150, 150, color=PALETTE["grey"], alpha=0.15, lw=0)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{c['first']} - {c['second']}".replace("_", " ") for c in pairs[::-1]])
    ax.set_xlabel(r"held-out iAUC MAE difference, mg/dL$\cdot$min (negative: first is better)")
    return _save(fig, "f4_prediction_pairs")


def figA_generic_rank():
    ranks = _summary().get("generic_rank")
    if not ranks:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for name, d in ranks.items():
        ax.semilogy(range(1, 4), d["singular_values_median"][::-1], "-o", label=name.replace("_", " "))
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("singular value index (smallest first)")
    ax.set_ylabel("median singular value (scaled)")
    ax.legend(frameon=False)
    return _save(fig, "a1_generic_rank")


def figA_gut_sweep():
    rows = _summary().get("gut_sweep", {}).get("sweep", {}).get("rows")
    if not rows:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    x = [r["gut_content_percent_of_meal"] for r in rows]
    ax.plot(x, [r["max_abs_glucose_difference_mg_dl"] for r in rows], "-o", label="maximum",
            color=PALETTE["iauc"])
    ax.plot(x, [r["median_abs_glucose_difference_mg_dl"] for r in rows], "-o", label="median",
            color=PALETTE["trace"])
    ax.axhspan(5, 10, color=PALETTE["grey"], alpha=0.2, lw=0)
    ax.set_xlabel("residual gut content (% of meal)")
    ax.set_ylabel(r"glucose change under $k_e\leftrightarrow k_a$ (mg/dL)")
    ax.legend(frameon=False)
    return _save(fig, "a2_gut_sweep")


def figA_box_sweep():
    sweep = _summary().get("bounds_sweep_fisher_fit")
    if not sweep:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    boxes = ["0.5x", "1x", "2x"]
    for parameter, label in PARAM_LABEL.items():
        ax.plot(range(3), [100 * sweep[b]["pinned"][parameter]["fraction"] for b in boxes], "-o",
                label=label)
    ax.set_xticks(range(3))
    ax.set_xticklabels(boxes)
    ax.set_xlabel("box width relative to the primary box")
    ax.set_ylabel("subjects on a bound (%)")
    ax.legend(frameon=False)
    return _save(fig, "a3_box_sweep")


ALL = {"f1": fig1_spectra, "f2": fig2_window, "f3a": fig3_profiles, "f3b": fig3_fractions,
       "f4": fig4_prediction, "a1": figA_generic_rank, "a2": figA_gut_sweep, "a3": figA_box_sweep}


def main() -> int:
    made, skipped = [], []
    for name, fn in ALL.items():
        try:
            out = fn()
        except Exception as exc:                     # a broken figure must not hide the others
            print(f"  {name}: FAILED {type(exc).__name__}: {exc}")
            skipped.append(name)
            continue
        (made if out else skipped).append(name)
        print(f"  {name}: {'drawn' if out else 'skipped (inputs missing)'}")
    print(f"{len(made)} drawn, {len(skipped)} skipped")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
