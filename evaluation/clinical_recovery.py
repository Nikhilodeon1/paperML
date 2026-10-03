"""Meaning result (paper §4.1 / Table 1): unsupervised recovery of clinical insulin resistance.

For each CGMacros subject, fit insulin sensitivity Si from their real meal-glucose curves with
one of three estimators, using ONLY the CGM + carbs + demographics (never a diagnosis or lab),
then correlate the recovered Si against independently-measured HbA1c and HOMA-IR. A Si that
tracks these labs is physiologically real, not a curve-fit knob.

Three estimators, one interface (``estimator=``):
  * ``grid``  — the classical per-subject 1-D grid fit on iAUC (``cgmacros.fit_subject_si``); this
                is the established r=-0.59 baseline. (The brief calls this "SMC"; the reproducible
                codebase method is the grid fit. ``particle_fit`` SMC is validated on full-day
                synthetic CGM, a different data shape than CGMacros' per-meal windows.)
  * ``rf``    — amortized RandomForest point estimator (``personalization/npe.py``), r=-0.66.
  * ``snpe``  — amortized calibrated SNPE-C posterior mean (``personalization/snpe_infer.py``).

Adds the statistics a reviewer will ask for and the prior scripts omitted: bootstrap 95% CI on
each Pearson r, Spearman rho as an outlier-robust check, one-way ANOVA across diagnosis groups
with Tukey HSD post-hoc, and per-group n.

Run:  python -m evaluation.clinical_recovery --estimator snpe
"""
from __future__ import annotations

import statistics
import time

import numpy as np
from scipy import stats as sstats

import paper_config as cfg
from evaluation.cgmacros import _profile, fit_subject_si, load_bio, subjects
from personalization import npe, snpe_infer

_GROUPS = ["normal", "prediabetic", "diabetic"]


def _subject_meals_snpe(meals) -> list[dict]:
    """CGMacros meal objects -> snpe_infer meal schema (meal at grid t=0)."""
    out = []
    for m in meals:
        vals, t0 = m._grid()
        out.append({"carbs_g": m.carbs_g,
                    "glucose": {"values": vals, "t0_min": t0, "step_min": 5.0, "meal_t_min": 0.0}})
    return out


def build_rf(n_train: int = 20_000, seed: int = cfg.SEED):
    theta, x = npe.generate_training_set(n_train, seed=seed)
    return npe.AmortizedEstimator().fit(theta, x)


def load_snpe(path=None):
    from personalization.snpe_trainer import SNPETrainer
    return SNPETrainer.load(path).posterior


def collect(estimator: str = "snpe", limit: int | None = None, *, rf=None, posterior=None,
            snpe_kwargs: dict | None = None) -> tuple[list[dict], float]:
    """Fit Si for every eligible subject with the chosen estimator. Returns (rows, ms/subject)."""
    cfg.set_all_seeds()
    if estimator == "rf" and rf is None:
        rf = build_rf()
    if estimator == "snpe" and posterior is None:
        posterior = load_snpe()
    snpe_kwargs = snpe_kwargs or {}

    bio = load_bio()
    rows, infer_t = [], 0.0
    for sid, meals in subjects(limit=limit):
        b = bio.get(sid)
        prof = _profile(b)
        good = [m for m in meals if m.real_iauc() is not None]
        if not (b and prof and len(good) >= 4):
            continue
        t = time.time()
        if estimator == "grid":
            si = fit_subject_si(meals, profile=prof)
        elif estimator == "rf":
            feats = []
            for m in good:
                vals, t0 = m._grid()
                s = npe.summary_stats(vals, t0, 5.0, meal_t_min=0.0)
                if not np.isnan(s).any():
                    feats.append(npe.features_for(s, m.carbs_g, prof))
            if not feats:
                continue
            si = float(np.median(rf.predict(np.array(feats))[:, npe.PARAM_NAMES.index("insulin_sensitivity")]))
        elif estimator == "snpe":
            pe = snpe_infer.point_estimate(_subject_meals_snpe(good), posterior, profile=prof,
                                           **snpe_kwargs)
            if not pe:
                continue
            si = pe["insulin_sensitivity"]
        else:
            raise ValueError(f"unknown estimator {estimator!r}")
        infer_t += time.time() - t
        if not np.isfinite(si):
            continue
        rows.append({"subject": sid, "si": float(si), "n": len(good), "hba1c": b["hba1c"],
                     "homa_ir": b["homa_ir"], "status": b["status"], "bmi": b["bmi"]})
    return rows, 1000 * infer_t / max(len(rows), 1)


def _clean(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def _corr(a, b, method):
    if method == "spearman":
        return float(sstats.spearmanr(a, b).statistic)
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_corr(xs, ys, method: str = "pearson", n_boot: int = 1000, seed: int = cfg.SEED):
    """Correlation (pearson|spearman) with a percentile bootstrap 95% CI."""
    a, b = _clean(xs, ys)
    if len(a) < 3:
        return {"r": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": len(a)}
    r = _corr(a, b, method)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    boots = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if a[s].std() > 0 and b[s].std() > 0:
            boots.append(_corr(a[s], b[s], method))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"r": r, "lo": float(lo), "hi": float(hi), "n": len(a)}


def bootstrap_pearson(xs, ys, n_boot: int = 1000, seed: int = cfg.SEED):
    """Back-compat alias for the Pearson bootstrap."""
    return bootstrap_corr(xs, ys, "pearson", n_boot=n_boot, seed=seed)


def analyze(rows: list[dict]) -> dict:
    """Full statistical summary of one estimator's Si-vs-labs recovery."""
    si = [r["si"] for r in rows]
    a1c = [r["hba1c"] for r in rows]
    homa = [r["homa_ir"] for r in rows]

    # Spearman is the PRIMARY metric: Si is not linearly scaled to HbA1c, and clinical status is a
    # rank question ("does higher Si go with lower HbA1c across groups"). Pearson is kept as a
    # secondary/point-estimator-favouring comparison.
    res = {
        "n": len(rows),
        "spearman_hba1c_ci": bootstrap_corr(si, a1c, "spearman"),
        "spearman_homa_ci": bootstrap_corr(si, homa, "spearman"),
        "pearson_hba1c": bootstrap_corr(si, a1c, "pearson"),
        "pearson_homa": bootstrap_corr(si, homa, "pearson"),
    }
    res["spearman_hba1c"] = res["spearman_hba1c_ci"]["r"]   # float back-compat
    res["spearman_homa"] = res["spearman_homa_ci"]["r"]

    # group means + one-way ANOVA + Tukey HSD
    groups = {g: [r["si"] for r in rows if r["status"] == g] for g in _GROUPS}
    res["group_means"] = {g: (round(statistics.fmean(v), 3), len(v)) if v else (None, 0)
                          for g, v in groups.items()}
    present = [v for v in groups.values() if len(v) >= 2]
    if len(present) >= 2:
        res["anova_p"] = float(sstats.f_oneway(*present).pvalue)
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            labels = [r["status"] for r in rows if r["status"] in _GROUPS]
            vals = [r["si"] for r in rows if r["status"] in _GROUPS]
            tuk = pairwise_tukeyhsd(vals, labels)
            res["tukey"] = str(tuk)
        except Exception as e:  # statsmodels optional at runtime
            res["tukey"] = f"(tukey unavailable: {e})"
    else:
        res["anova_p"] = float("nan")
        res["tukey"] = "(insufficient groups)"
    return res


def main() -> None:
    import argparse
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", choices=["grid", "rf", "snpe"], default="snpe")
    ap.add_argument("--subjects", type=int, default=None)
    args = ap.parse_args()

    rows, ms = collect(args.estimator, limit=args.subjects)
    res = analyze(rows)
    print("=" * 84)
    print(f"CLINICAL RECOVERY — estimator={args.estimator}  (Si vs real HbA1c / HOMA-IR, in-sample)")
    print("=" * 84)
    sh, sho = res["spearman_hba1c_ci"], res["spearman_homa_ci"]
    ph, pho = res["pearson_hba1c"], res["pearson_homa"]
    print(f"  n subjects: {res['n']}   inference: {ms:.1f} ms/subject")
    print("  PRIMARY (rank):")
    print(f"    Spearman(Si, HbA1c)   = {sh['r']:+.3f}  95% CI [{sh['lo']:+.3f}, {sh['hi']:+.3f}]")
    print(f"    Spearman(Si, HOMA-IR) = {sho['r']:+.3f}  95% CI [{sho['lo']:+.3f}, {sho['hi']:+.3f}]")
    print("  secondary (linear):")
    print(f"    Pearson(Si, HbA1c)    = {ph['r']:+.3f}  95% CI [{ph['lo']:+.3f}, {ph['hi']:+.3f}]")
    print(f"    Pearson(Si, HOMA-IR)  = {pho['r']:+.3f}  95% CI [{pho['lo']:+.3f}, {pho['hi']:+.3f}]")
    gm = res["group_means"]
    print("  group mean Si: " + "  ".join(f"{g} {gm[g][0]} (n={gm[g][1]})" for g in _GROUPS))
    print(f"  one-way ANOVA across groups: p = {res['anova_p']:.4g}")
    print("  " + res["tukey"].replace("\n", "\n  "))
    print("=" * 84)


if __name__ == "__main__":
    main()
