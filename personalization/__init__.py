"""Layer 3 — Personalization (Bayesian parameter updating).

NOT per-user model fine-tuning (brief 3). Each system module has parameters with
population-level priors from Layer 1. As a user logs data, we update the POSTERIOR
distribution of THEIR specific parameters from their actual outcomes. This is cheap
enough to run on-device and is how real clinical risk models personalize.

Two consequences fall directly out of doing this correctly (brief 3):
  - Confidence intervals tighten over time: more observations => tighter posterior.
  - "Gets better over time" here means THIS USER's personalization, automatically —
    distinct from Layer 1 knowledge improving (which needs manual research / cohort
    work). Do not conflate the two.

Implemented as lightweight recursive Gaussian updating (a 1-D Kalman filter), per
the brief's "Kalman filter / particle filter for online updating" guidance — no
deep-learning framework required.
"""
