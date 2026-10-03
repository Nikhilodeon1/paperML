"""Statistics used by every AISTATS-revision analysis, in one place.

The rejected submission reported point estimates and a single fold assignment. Every comparison
here instead carries an interval, every family of comparisons is multiplicity-corrected, and every
equivalence claim is tested against a pre-registered margin rather than asserted from a
non-significant difference. Confidence intervals over subjects are bootstrap intervals because the
per-subject scores are neither normal nor independent of meal count.

Conventions:

* The resampling unit is the SUBJECT, never the meal. A subject contributes one number.
* Equivalence uses a 90% interval against a two-sided margin, which is the interval-based form of
  a two-one-sided-test procedure at alpha 0.05 (`tost`).
* Functions return plain dictionaries of floats so results serialize straight to JSON.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

__all__ = [
    "CI", "bootstrap_ci", "paired_bootstrap_ci", "wilcoxon", "tost", "holm",
    "wilson_ci", "fisher_z_difference", "kendall_tau", "spearman", "binomial_ci",
]

N_RESAMPLES = 2000


@dataclass(frozen=True)
class CI:
    """A point estimate with an interval. `level` is the nominal coverage, e.g. 0.95."""
    estimate: float
    low: float
    high: float
    level: float
    n: int
    method: str

    def as_dict(self) -> dict:
        return asdict(self)

    def excludes_zero(self) -> bool:
        return (self.low > 0.0) or (self.high < 0.0)

    def within(self, margin: float) -> bool:
        """Whether the whole interval lies inside (-margin, +margin) -- the equivalence verdict."""
        return (self.low > -margin) and (self.high < margin)


def _clean(values) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    return a[np.isfinite(a)]


def bootstrap_ci(values, statistic=np.mean, level: float = 0.95,
                 n_resamples: int = N_RESAMPLES, seed: int = 0, method: str = "BCa") -> CI:
    """Bootstrap CI for a one-sample statistic, resampling subjects.

    Falls back from BCa to the percentile interval when the acceleration constant is undefined
    (degenerate or near-constant samples); the returned `method` says which was used, so a
    fallback is visible in the result file rather than hidden.
    """
    a = _clean(values)
    if a.size < 2:
        point = float(statistic(a)) if a.size else float("nan")
        return CI(point, float("nan"), float("nan"), level, int(a.size), "insufficient-n")
    point = float(statistic(a))
    rng = np.random.default_rng(seed)
    for attempt in (method, "percentile"):
        try:
            res = stats.bootstrap((a,), statistic, n_resamples=n_resamples,
                                  confidence_level=level, method=attempt,
                                  random_state=rng, vectorized=False)
            low = float(res.confidence_interval.low)
            high = float(res.confidence_interval.high)
            if math.isfinite(low) and math.isfinite(high):
                return CI(point, low, high, level, int(a.size), attempt)
        except Exception:
            continue
    return CI(point, float("nan"), float("nan"), level, int(a.size), "failed")


def paired_bootstrap_ci(a, b, level: float = 0.95, n_resamples: int = N_RESAMPLES,
                        seed: int = 0) -> CI:
    """Bootstrap CI for the mean paired difference `a - b` over subjects.

    Pairs are resampled together, which is what makes this a paired comparison; resampling the two
    arms independently would discard the pairing that gives the comparison its power.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"paired arrays must have the same shape, got {x.shape} and {y.shape}")
    keep = np.isfinite(x) & np.isfinite(y)
    return bootstrap_ci(x[keep] - y[keep], np.mean, level=level,
                        n_resamples=n_resamples, seed=seed)


def wilcoxon(a, b=None) -> dict:
    """Wilcoxon signed-rank test on paired values (or on one sample against zero).

    `zero_method="wilcox"` drops exact ties, matching the usual convention. Returns the statistic,
    the p-value, the number of pairs used, and the median difference, which is the effect size the
    test is actually about.
    """
    x = np.asarray(a, dtype=float)
    if b is None:
        d = x[np.isfinite(x)]
    else:
        y = np.asarray(b, dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        d = x[keep] - y[keep]
    nonzero = d[d != 0.0]
    if nonzero.size < 1:
        return {"statistic": float("nan"), "p": 1.0, "n": int(d.size),
                "median_difference": 0.0, "note": "all differences zero"}
    res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(res.statistic), "p": float(res.pvalue), "n": int(d.size),
            "median_difference": float(np.median(d))}


def tost(a, b, margin: float, level: float = 0.90, n_resamples: int = N_RESAMPLES,
         seed: int = 0) -> dict:
    """Equivalence of two paired arms within +/- `margin`, by bootstrap interval.

    Equivalent in intent to two one-sided tests at alpha 0.05: the arms are declared equivalent
    only when the whole 90% interval for the mean difference lies inside the margin. A
    non-significant difference is NOT equivalence and is never reported as one.
    """
    ci = paired_bootstrap_ci(a, b, level=level, n_resamples=n_resamples, seed=seed)
    return {"margin": float(margin), "equivalent": bool(ci.within(margin)), **ci.as_dict()}


def holm(pvalues, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni correction over a family of p-values.

    Holm rather than Benjamini-Hochberg: these families are small and the pre-registration states
    familywise control, not a false-discovery rate.
    """
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(list(pvalues), dtype=float)
    if p.size == 0:
        return {"p_adjusted": [], "reject": [], "alpha": float(alpha), "n": 0}
    finite = np.isfinite(p)
    adjusted = np.ones_like(p)
    reject = np.zeros(p.shape, dtype=bool)
    if finite.any():
        rej, adj, _, _ = multipletests(p[finite], alpha=alpha, method="holm")
        adjusted[finite] = adj
        reject[finite] = rej
    return {"p_adjusted": [float(v) for v in adjusted],
            "reject": [bool(v) for v in reject], "alpha": float(alpha), "n": int(p.size)}


def wilson_ci(successes: int, n: int, level: float = 0.95) -> CI:
    """Wilson score interval for a proportion.

    Used for every "fraction of subjects with a bounded CI" in H5. The Wald interval is wrong at
    the proportions these analyses produce (near 0 and near 1) and can leave the unit interval.
    """
    if n <= 0:
        return CI(float("nan"), float("nan"), float("nan"), level, 0, "wilson")
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denom
    return CI(float(phat), max(0.0, centre - half), min(1.0, centre + half), level, int(n),
              "wilson")


def binomial_ci(successes: int, n: int, level: float = 0.95) -> CI:
    """Clopper-Pearson (exact) interval, for the win counts in the prediction comparison."""
    if n <= 0:
        return CI(float("nan"), float("nan"), float("nan"), level, 0, "clopper-pearson")
    res = stats.binomtest(int(successes), int(n)).proportion_ci(
        confidence_level=level, method="exact")
    return CI(successes / n, float(res.low), float(res.high), level, int(n), "clopper-pearson")


def _fisher_z(r: float) -> float:
    r = min(max(float(r), -0.999999), 0.999999)
    return 0.5 * math.log((1.0 + r) / (1.0 - r))


def fisher_z_difference(r1: float, n1: int, r2: float, n2: int) -> dict:
    """Test that two INDEPENDENT correlations differ (Fisher z).

    Used for the heterogeneity comparisons across cohorts, which are separate samples. It is not
    valid for two correlations measured on the same subjects (raw versus partial on one cohort);
    those are compared by the paired bootstrap of their difference instead.
    """
    if n1 < 4 or n2 < 4:
        return {"z": float("nan"), "p": float("nan"), "r1": float(r1), "r2": float(r2),
                "n1": int(n1), "n2": int(n2), "note": "n too small for Fisher z"}
    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z = (_fisher_z(r1) - _fisher_z(r2)) / se
    return {"z": float(z), "p": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
            "r1": float(r1), "r2": float(r2), "n1": int(n1), "n2": int(n2)}


def kendall_tau(a, b) -> dict:
    """Kendall tau-b between two orderings. Tau rather than Spearman for the robustness sweeps,
    which compare short rankings (3 parameters) where tie handling matters."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2:
        return {"tau": float("nan"), "p": float("nan"), "n": int(keep.sum())}
    res = stats.kendalltau(x[keep], y[keep])
    return {"tau": float(res.statistic), "p": float(res.pvalue), "n": int(keep.sum())}


def spearman(a, b) -> dict:
    """Spearman rank correlation, with the pair count actually used."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": int(keep.sum())}
    res = stats.spearmanr(x[keep], y[keep])
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n": int(keep.sum())}
