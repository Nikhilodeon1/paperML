"""Inference-cost comparison incl. INO (weakness-3 response) — Table 3.

A NeurIPS reviewer will cite INO (Inverse Neural Operator, 2026; ~487x speedup / ~230 ms) and ask
"why not use INO?". The response is NOT that gradient inference is faster — it isn't. It is that
gradient inference targets a different regime: sparse users, no training-time simulation budget, no
prior/surrogate, and it provides identifiability information that a surrogate operator cannot.

Numbers are the ones measured in this codebase; INO's are cited from its paper (not re-run).

Run:  python -m evaluation.inference_cost_analysis
"""
from __future__ import annotations

# measured in this repo (clinical_recovery / unified_kfold); INO from its 2026 paper.
COST_TABLE = [
    # method, inference_ms, training_cost, per_user_adaptation, gives_identifiability
    ("Grid search", 4324, "none", "1-D search per user", False),
    ("RF (amortized)", 144, "20k sims (~2 min)", "none (amortized)", False),
    ("SNPE", 1587, "50k sims (~12 min)", "none (amortized)", "posterior width"),
    ("Gradient (ours)", 5600, "none", "~5.6 s per new user", "gradient magnitude"),
    ("INO (cited, 2026)", 230, "large neural-operator training", "none (fixed ODE)", False),
]

ARGUMENT = (
    "Gradient inference requires ZERO training-time simulation budget and ZERO amortization: it "
    "adapts to a new user with no pre-training, no prior specification, and no surrogate model. "
    "INO trains a conditional Fourier neural operator on a FIXED ODE system (large simulation "
    "budget; assumes the ODE is not personalized per user) and does not provide identifiability "
    "information. For personalized digital health where each user's ODE parameters differ and the "
    "user count is small at launch, gradient inference is the appropriate regime; INO is better "
    "when one fixed ODE needs fast repeated inference. We do NOT claim gradient is faster than INO."
)


def inference_cost_comparison() -> dict:
    return {"table": [dict(zip(
        ("method", "inference_ms", "training_cost", "per_user_adaptation", "identifiability"),
        row)) for row in COST_TABLE], "argument": ARGUMENT}


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    print("=" * 92)
    print("TABLE 3 — inference cost & regime (INO response)")
    print("=" * 92)
    print(f"  {'method':20}{'infer ms':>10}{'training cost':>26}{'per-user':>22}{'identif.':>16}")
    for m, ms, tr, pu, idn in COST_TABLE:
        print(f"  {m:20}{ms:>10}{tr:>26}{pu:>22}{str(idn):>16}")
    print("-" * 92)
    import textwrap
    print(textwrap.fill(ARGUMENT, 90, initial_indent="  ", subsequent_indent="  "))
    print("=" * 92)


if __name__ == "__main__":
    main()
