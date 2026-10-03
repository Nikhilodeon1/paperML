"""SBC calibration diagnostic (paper appendix, Block 1e) — the gate before trusting posteriors.

Neural posterior estimators can be over/under-confident. Simulation-Based Calibration (Talts et
al. 2018) is the standard check: draw theta ~ prior, simulate x, infer the posterior, and record
the RANK of the true theta among posterior samples. If the posterior is calibrated, those ranks
are uniform, and central credible intervals have their nominal coverage (a 90% interval contains
the truth 90% of the time).

Run this and confirm coverage is near-nominal BEFORE using SNPE posterior widths in any paper
figure. If Si is badly miscalibrated the identifiability story is unsafe.

Run:  python -m evaluation.snpe_calibration
"""
from __future__ import annotations

import numpy as np

import paper_config as cfg
from personalization import npe

_LEVELS = (0.5, 0.9, 0.95)


def run_sbc_check(posterior, n_trials: int = 500, n_post: int = 500, seed: int = 123) -> dict:
    """SBC over fresh prior draws. Returns rank array + empirical coverage at 50/90/95%.

    Uses a DIFFERENT seed (123) from training (42) so the calibration set is genuinely held out.
    """
    import torch
    from sbi.diagnostics import run_sbc

    theta, x = npe.generate_training_set(n_trials, seed=seed)   # fresh (theta, x) from the prior
    th = torch.as_tensor(theta, dtype=torch.float32)
    xt = torch.as_tensor(x, dtype=torch.float32)
    ranks, _dap = run_sbc(th, xt, posterior, num_posterior_samples=n_post,
                          num_workers=1, show_progress_bar=False)
    ranks = ranks.detach().cpu().numpy() if hasattr(ranks, "detach") else np.asarray(ranks)

    u = ranks / float(n_post)                                  # normalised rank in [0, 1]
    coverage = {}
    for c in _LEVELS:
        coverage[c] = {name: float(np.mean(np.abs(u[:, i] - 0.5) <= c / 2.0))
                       for i, name in enumerate(cfg.PARAM_NAMES)}

    # uniformity KS test per parameter (p>0.05 => cannot reject uniform => calibrated)
    from scipy import stats as sstats
    ks = {name: float(sstats.kstest(u[:, i], "uniform").pvalue)
          for i, name in enumerate(cfg.PARAM_NAMES)}

    return {"ranks": ranks, "n_trials": int(len(ranks)), "n_post": n_post,
            "coverage": coverage, "ks_uniform_p": ks}


def save_rank_histogram(ranks, n_post, path=None):
    """Rank histogram per parameter — flat = calibrated. Saved to evaluation/figures/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg.ensure_dirs()
    path = path or (cfg.FIGURES / "sbc_rank_histogram.pdf")
    fig, axes = plt.subplots(1, len(cfg.PARAM_NAMES), figsize=(11, 3))
    for i, (ax, name) in enumerate(zip(axes, cfg.PARAM_NAMES)):
        ax.hist(ranks[:, i], bins=20, color="#4C9F70", edgecolor="white")
        ax.axhline(len(ranks) / 20.0, color="grey", ls="--", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("rank", fontsize=10)
    axes[0].set_ylabel("count", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    from personalization.snpe_trainer import SNPETrainer

    cfg.set_all_seeds()
    post = SNPETrainer.load().posterior
    res = run_sbc_check(post)
    print("=" * 74)
    print("SBC CALIBRATION — ranks should be uniform; coverage should match nominal")
    print("=" * 74)
    print(f"  trials: {res['n_trials']}  posterior samples/trial: {res['n_post']}")
    for c in _LEVELS:
        cov = res["coverage"][c]
        print(f"  {int(c*100)}% CI coverage:  " +
              "  ".join(f"{n.split('_')[0]} {cov[n]*100:.0f}%" for n in cfg.PARAM_NAMES))
    print("  KS uniformity p-values (>0.05 = calibrated): " +
          "  ".join(f"{n.split('_')[0]} {res['ks_uniform_p'][n]:.3f}" for n in cfg.PARAM_NAMES))
    path = save_rank_histogram(res["ranks"], res["n_post"])
    print(f"  rank histogram -> {path}")
    print("=" * 74)


if __name__ == "__main__":
    main()
