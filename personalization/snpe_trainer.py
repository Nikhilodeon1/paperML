"""SNPE-C training: calibrated posteriors over (Si, gastric_emptying, carb_absorption).

Upgrades the RF point estimator (``personalization/npe.py``) to a full Neural Posterior Estimator
(``sbi`` SNPE-C, a masked-autoregressive-flow density estimator). It uses the **same** forward
simulator, the **same** five summary statistics, and the **same** 10-dim feature vector as
``npe.py`` — the only thing that changes is the estimator, so SNPE-vs-RF is a clean
apples-to-apples comparison (paper Table 1).

The payoff is a posterior with *width*: insulin sensitivity (Si) comes out narrow (identifiable),
while gastric emptying and carb absorption come out wide (structurally non-identifiable from a
meal-glucose curve) — the paper's core finding, which a point estimator cannot express.

Train + save artifact::

    python -m personalization.snpe_trainer --n 50000

then load with :meth:`SNPETrainer.load` or via ``paper_config.SNPE_POSTERIOR``.
"""
from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import paper_config as cfg
from personalization import npe  # canonical simulator + summary stats + feature assembly


def build_prior():
    """A ``BoxUniform`` prior over the 3 parameters, in :data:`paper_config.PARAM_NAMES` order."""
    import torch
    from sbi.utils import BoxUniform

    lo, hi = cfg.prior_bounds()
    return BoxUniform(low=torch.tensor(lo, dtype=torch.float32),
                      high=torch.tensor(hi, dtype=torch.float32))


def generate_training_data(n: int, seed: int = cfg.SEED):
    """Sample theta ~ prior, simulate a meal response, summarise -> (theta[N,3], x[N,8]).

    Delegates to :func:`personalization.npe.generate_training_set` so the SNPE training set is
    byte-for-byte the same distribution the RF baseline is trained on (same PRIOR, same feature
    vector, same CGM-noise model). NaN rows (degenerate windows) are dropped inside npe.
    """
    return npe.generate_training_set(n, seed=seed)


@dataclass
class SNPETrainer:
    """End-to-end SNPE-C pipeline: generate -> train -> save/load a calibrated posterior."""

    n_simulations: int = 50_000
    seed: int = cfg.SEED
    density_estimator: str = "maf"
    theta: np.ndarray | None = None
    x: np.ndarray | None = None
    posterior: object = None
    meta: dict = field(default_factory=dict)

    # -- data ---------------------------------------------------------------------------------
    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.time()
        self.theta, self.x = generate_training_data(self.n_simulations, seed=self.seed)
        self.meta["n_kept"] = int(len(self.theta))
        self.meta["gen_seconds"] = round(time.time() - t0, 1)
        return self.theta, self.x

    # -- train --------------------------------------------------------------------------------
    def train(self, theta: np.ndarray | None = None, x: np.ndarray | None = None):
        """Fit the SNPE-C density estimator. Returns the sbi posterior object."""
        import torch
        from sbi.inference import NPE

        cfg.set_all_seeds(self.seed)
        if theta is None or x is None:
            if self.theta is None:
                self.generate()
            theta, x = self.theta, self.x

        theta_t = torch.as_tensor(np.asarray(theta), dtype=torch.float32)
        x_t = torch.as_tensor(np.asarray(x), dtype=torch.float32)

        t0 = time.time()
        inference = NPE(prior=build_prior(), density_estimator=self.density_estimator)
        inference.append_simulations(theta_t, x_t)
        estimator = inference.train()
        self.posterior = inference.build_posterior(estimator)
        self.meta.update(
            train_seconds=round(time.time() - t0, 1),
            n_train=int(len(theta_t)),
            density_estimator=self.density_estimator,
            param_names=cfg.PARAM_NAMES,
            feature_names=npe.FEATURE_NAMES,
            prior=cfg.PRIOR,
            seed=self.seed,
        )
        return self.posterior

    # -- persistence --------------------------------------------------------------------------
    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else cfg.SNPE_POSTERIOR
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"posterior": self.posterior, "meta": self.meta}, fh)
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SNPETrainer":
        path = Path(path) if path else cfg.SNPE_POSTERIOR
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        obj = cls()
        obj.posterior = blob["posterior"]
        obj.meta = blob.get("meta", {})
        obj.seed = obj.meta.get("seed", cfg.SEED)
        return obj


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the SNPE-C posterior artifact.")
    ap.add_argument("--n", type=int, default=50_000, help="number of training simulations")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--out", type=str, default=str(cfg.SNPE_POSTERIOR))
    ap.add_argument("--density-estimator", type=str, default="maf")
    args = ap.parse_args()

    cfg.ensure_dirs()
    tr = SNPETrainer(n_simulations=args.n, seed=args.seed, density_estimator=args.density_estimator)
    print(f"[snpe] generating {args.n} simulations (seed={args.seed}) ...", flush=True)
    tr.generate()
    print(f"[snpe]   kept {tr.meta['n_kept']} rows in {tr.meta['gen_seconds']}s; training ...",
          flush=True)
    tr.train()
    out = tr.save(args.out)
    print(f"[snpe] trained in {tr.meta['train_seconds']}s -> saved {out}", flush=True)


if __name__ == "__main__":
    main()
