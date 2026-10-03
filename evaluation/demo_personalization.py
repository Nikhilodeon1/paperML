"""Demo: how a user's metabolic projection sharpens as they log data (brief 3).

Shows the two things the brief insists are distinct:
  - the RMR-multiplier POSTERIOR narrowing as weigh-ins accumulate, and
  - the resulting projection confidence interval tightening — automatically, per user.

This is illustrative output, not a pass/fail test. Run:
    python -m evaluation.demo_personalization
"""

from __future__ import annotations

import numpy as np

from knowledge_base import load_system
from modules.metabolic import _simulate_weight, project_weight
from personalization.metabolic import Weighin, estimate_rmr_multiplier

HEIGHT, AGE, SEX, ACT = 178.0, 35.0, "male", "sedentary"
TRUE_M = 0.87          # this user's real metabolism runs 13% below the population mean
START_W, INTAKE = 82.0, 2200.0


def _make_logs(n, spacing, noise, seed=7):
    kb = load_system("metabolic")
    pal = kb[f"pal_{ACT}"].value
    kcal = kb["kcal_per_kg_fat"].value
    rng = np.random.default_rng(seed)
    days = np.arange(1, n * spacing + 1)
    traj = _simulate_weight(START_W, HEIGHT, AGE, SEX, INTAKE, TRUE_M, pal, kcal, days, kb)
    logs = [Weighin(0.0, START_W + rng.normal(0, noise), INTAKE)]
    for i in range(1, n):
        logs.append(Weighin(float(i * spacing),
                            float(traj[i * spacing - 1] + rng.normal(0, noise)), INTAKE))
    return logs


def main():
    print(f"True (hidden) RMR multiplier for this user: {TRUE_M}")
    print(f"{'weigh-ins':>10} {'posterior m':>14} {'post. sd':>10} "
          f"{'90d proj CI width (kg)':>24}")
    print("-" * 64)
    for n in (1, 3, 6, 12, 24):
        logs = _make_logs(n, spacing=14, noise=0.5)
        post = estimate_rmr_multiplier(logs, HEIGHT, AGE, SEX, ACT, weight_noise_kg=0.5)
        from modules.metabolic import MetabolicPersonalParams
        pp = MetabolicPersonalParams(rmr_multiplier=post.mean,
                                     rmr_multiplier_sd=post.sd, n_observations=n - 1)
        proj = project_weight(START_W, HEIGHT, AGE, SEX, daily_intake_kcal=2000,
                              horizon_days=90, personal=pp)
        ci_w = proj.weight_upper[-1] - proj.weight_lower[-1]
        print(f"{n:>10} {post.mean:>14.3f} {post.sd:>10.3f} {ci_w:>24.2f}")
    print("-" * 64)
    print("Posterior converges toward the true multiplier and the projection CI\n"
          "narrows automatically — no retraining, just Bayesian updating (Layer 3).")


if __name__ == "__main__":
    main()
