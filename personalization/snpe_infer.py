"""SNPE inference wrapper: a user's meals -> a calibrated joint posterior over physiology.

Given the trained SNPE posterior (``personalization/snpe_trainer.py``) and a user's logged meals
(same schema the CGMacros loader and observation model emit), return posterior mean **and std**
for all three parameters. ``Si_std`` is the headline: narrow => identifiable, wide =>
non-identifiable.

Moments are computed by rejection-free importance sampling on the flow's ``log_prob`` rather than
``posterior.sample`` — on real, out-of-distribution CGM the sbi rejection sampler loops on
leakage and is ~10x slower, while ``log_prob`` is a fixed forward pass. A two-stage proposal
(uniform prior -> Gaussian refit around the mode) keeps the effective sample size high even when
the Si posterior is sharp.

Combination modes for a subject with several meals:
  * ``per_meal`` (default): per-meal moments; subject Si = median of per-meal means (mirrors the
    RF baseline's median-over-meals), reported std = mean per-meal marginal std.
  * ``product``: multi-observation joint posterior as a product of per-meal posteriors (uniform
    prior cancels), via importance sampling. Correct Bayesian narrowing with more meals.
"""
from __future__ import annotations

import numpy as np

import paper_config as cfg
from personalization import npe

_SHORT = {"insulin_sensitivity": "Si", "gastric_emptying": "gastric", "carb_absorption": "carb"}
_LO, _HI = (np.array(x) for x in cfg.prior_bounds())


def _feature_rows(user_meals: list[dict], profile: dict | None,
                  max_meals: int | None = None) -> np.ndarray:
    """One feature row (npe.FEATURE_NAMES, 10-dim) per usable-glucose-window meal (optionally capped)."""
    profile = profile or npe._PROFILE
    rows = []
    for m in user_meals:
        g = m.get("glucose") or {}
        vals = g.get("values", m.get("values"))
        if vals is None:
            continue
        s = npe.summary_stats(vals, g.get("t0_min", 0.0), g.get("step_min", 5.0),
                              g.get("meal_t_min", npe._PRE))
        if np.isnan(s).any():
            continue
        rows.append(npe.features_for(s, m.get("carbs_g", 60.0), profile))
        if max_meals and len(rows) >= max_meals:
            break
    return np.asarray(rows, dtype=np.float32)


def _log_prob(posterior, theta_np, x_row):
    """Unnormalised log p(theta | x_row) for a batch of thetas (norm cancels in weights)."""
    import torch

    tt = torch.as_tensor(np.asarray(theta_np), dtype=torch.float32)
    xt = torch.as_tensor(np.asarray(x_row), dtype=torch.float32)
    try:
        return posterior.log_prob(tt, x=xt, norm_posterior=False).detach().cpu().numpy()
    except TypeError:
        posterior.set_default_x(xt)
        return posterior.log_prob(tt).detach().cpu().numpy()


def _norm_w(logw):
    logw = np.asarray(logw, dtype=np.float64)
    logw = logw - np.nanmax(logw)
    w = np.exp(logw)
    w = np.where(np.isfinite(w), w, 0.0)
    tot = w.sum()
    return (w / tot) if tot > 0 else None


def is_moments(posterior, x_row, seed: int = cfg.SEED, n1: int = 6000, n2: int = 10000):
    """Two-stage importance-sampling posterior moments for one observation.

    Stage 1 samples the uniform prior to locate the mode; stage 2 samples a Gaussian refit around
    it (inflated) so a sharp posterior still gets high ESS. Returns (mean[3], std[3], samples, ess)
    or None if the weights collapse.
    """
    rng = np.random.default_rng(seed)
    p1 = rng.uniform(_LO, _HI, size=(n1, 3))
    w1 = _norm_w(_log_prob(posterior, p1, x_row))
    if w1 is None:
        return None
    mean1 = np.average(p1, axis=0, weights=w1)
    cov = np.cov(p1.T, aweights=w1) + 1e-12 * np.eye(3)

    S = cov * 6.25  # inflate (2.5 sigma) for a heavy-enough proposal
    try:
        L = np.linalg.cholesky(S)
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        std1 = np.sqrt(np.average((p1 - mean1) ** 2, axis=0, weights=w1))
        idx = rng.choice(n1, size=2000, p=w1)
        return mean1, std1, p1[idx], float(1.0 / np.sum(w1 ** 2))

    z = rng.standard_normal((n2, 3))
    p2 = mean1 + z @ L.T
    inside = np.all((p2 >= _LO) & (p2 <= _HI), axis=1)
    p2 = p2[inside]
    if len(p2) < 200:
        std1 = np.sqrt(np.average((p1 - mean1) ** 2, axis=0, weights=w1))
        idx = rng.choice(n1, size=2000, p=w1)
        return mean1, std1, p1[idx], float(1.0 / np.sum(w1 ** 2))

    lp2 = _log_prob(posterior, p2, x_row)
    diff = p2 - mean1
    quad = np.einsum("ij,jk,ik->i", diff, Sinv, diff)   # (p2-m) Sinv (p2-m)
    w2 = _norm_w(lp2 + 0.5 * quad)                        # w ∝ post / gaussian_proposal
    if w2 is None:
        return None
    mean2 = np.average(p2, axis=0, weights=w2)
    std2 = np.sqrt(np.average((p2 - mean2) ** 2, axis=0, weights=w2))
    idx = rng.choice(len(p2), size=2000, p=w2)
    return mean2, std2, p2[idx], float(1.0 / np.sum(w2 ** 2))


def _pack(mean, std, n_meals, samples, extra=None):
    out = {"n_meals": int(n_meals), "source": "snpe",
           "joint_samples": np.asarray(samples), "Si_samples": np.asarray(samples)[:, 0]}
    for i, name in enumerate(cfg.PARAM_NAMES):
        out[f"{_SHORT[name]}_mean"] = float(mean[i])
        out[f"{_SHORT[name]}_std"] = float(std[i])
    if extra:
        out.update(extra)
    return out


def _product_posterior(posterior, feats, seed, n_proposal=20_000):
    """Joint multi-observation posterior via importance sampling (product of experts)."""
    rng = np.random.default_rng(seed)
    prop = rng.uniform(_LO, _HI, size=(n_proposal, 3))
    total = np.zeros(n_proposal, dtype=np.float64)
    for row in feats:
        total += _log_prob(posterior, prop, row)
    w = _norm_w(total)
    if w is None:
        return {}
    mean = np.average(prop, axis=0, weights=w)
    std = np.sqrt(np.average((prop - mean) ** 2, axis=0, weights=w))
    ess = float(1.0 / np.sum(w ** 2))
    idx = rng.choice(n_proposal, size=2000, p=w)
    return _pack(mean, std, len(feats), prop[idx], extra={"ess": ess, "combine": "product"})


def infer_posterior(user_meals: list[dict], trained_posterior, profile: dict | None = None,
                    combine: str = "per_meal", max_meals: int | None = None,
                    seed: int = cfg.SEED) -> dict:
    """Joint posterior over (Si, gastric, carb). Empty dict if no usable meal."""
    feats = _feature_rows(user_meals, profile, max_meals=max_meals)
    if len(feats) == 0:
        return {}

    if combine == "product":
        return _product_posterior(trained_posterior, feats, seed)

    means, stds, sample_sets = [], [], []
    for j, row in enumerate(feats):
        mom = is_moments(trained_posterior, row, seed=seed + j)
        if mom is None:
            continue
        m, s, samp, _ess = mom
        means.append(m)
        stds.append(s)
        sample_sets.append(samp)
    if not means:
        return {}
    subj_mean = np.median(np.array(means), axis=0)   # median over meals (mirrors RF)
    subj_std = np.array(stds).mean(axis=0)            # mean per-meal marginal std
    pooled = np.concatenate(sample_sets, axis=0)
    return _pack(subj_mean, subj_std, len(means), pooled, extra={"combine": "per_meal"})


def point_estimate(user_meals: list[dict], trained_posterior, profile: dict | None = None,
                   **kw) -> dict:
    """Posterior-mean point estimate as ``{param_name: value}`` (for the correlation tables)."""
    post = infer_posterior(user_meals, trained_posterior, profile=profile, **kw)
    if not post:
        return {}
    return {"insulin_sensitivity": post["Si_mean"], "gastric_emptying": post["gastric_mean"],
            "carb_absorption": post["carb_mean"]}
