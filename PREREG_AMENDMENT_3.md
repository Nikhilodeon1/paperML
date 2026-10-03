# Pre-registration amendment 3 (revised)

Amends `PREREG.md` and amendments 1 and 2. Written after the Phase 2.4 structural-identifiability runs had
been seen (`julia/`), and before any profile-likelihood, ladder or cross-validation result exists. No
threshold changes. This revision replaces the first version of this file, committed earlier the same day
before any new result existed; the change is in the H1 label and in the scope of the sensitivity run.

## H1 is a deviation, not a met hypothesis

The planned rule: report the tool output verbatim; if `k_e` and `k_a` are globally identifiable, delete all
swap framing. **That rule fired under the planned formulation** (the meal as an initial condition `s(0)=D`,
with the intestinal compartment generic): the tool reports `k_e` and `k_a` globally identifiable.

H1 is therefore recorded as **formulation-dependent, not met**, and the paper says so in these words:
the tool output under the planned formulation triggered the deletion rule; the empty-gut model that the
engine runs satisfies the exchange symmetry analytically and numerically.

* Analytically: with an empty gut the transfer function from meal rate to the rate of appearance is
  `f k_e k_a / ((s + k_e)(s + k_a))`, symmetric in `k_e` and `k_a` (sympy check in
  `evaluation/gut_sweep.py`). A nonzero intestinal initial content `Q0` adds `f k_a Q0 / (s + k_a)`, which
  is not symmetric; initial stomach content is symmetric and cannot break the exchange.
* Numerically: exchanging the rates changes simulated glucose by at most 0.009 mg/dL for an empty gut
  (`evaluation/swap_check.py`).
* The tool gives "global" because it treats the gut initial state as a generic known value and cannot be
  told it is exactly zero. Every run, with its inputs, tool version and raw output, is tabulated in
  `julia/SI_RUNS.md` and reported in the paper.

**Initial-gut-content sweep.** Free-living data carries residual gut content from earlier meals. The
engine is run with the intestinal compartment starting at `Q0/D` of the meal mass in {0, 0.5, 1, 2, 5, 10}
percent and the swap-induced glucose change is reported (maximum and median), to show how fast the
symmetry breaks relative to CGM noise (`evaluation/gut_sweep.py`, `A2_gut_sweep`).

**Terminology.** The generic rank of the observable Jacobian is full (3) for every observable (`A3`).
iAUC is therefore **locally identifiable but ill-conditioned**. The exchange symmetry of the empty-gut
model is the only structural claim. `evaluation/lint_terms.py` fails the build if "structurally
non-identifiable" or "structural non-identifiability" appears outside text about the exchange symmetry,
and if the word "pre-registered" is used. The paper says "analysis plan fixed in version control before
the analyses were run", and an appendix lists Amendments 1 to 3 with reasons (`paper/appendix_deviations.tex`).

## Optimizer budget

The primary A9 protocol is the submitted one (Adam, learning rate 0.02, 150 steps). Two checks, in this
order, both declared before any cross-validation result exists:

1. **Inference gap, first.** From the primary A9 results: for every fold, the relative difference between
   the 150-step gradient fit's training loss and the exhaustive-grid minimum of the same regularized loss
   (`grad3` against `grid3`, `grad1` against `grid1`), tolerance one percent. If the gradient fit reaches
   the grid loss within tolerance for nearly every fold, under-convergence is ruled out
   (`prediction_stats.inference_gap`).
2. **500-step check, only after the primary cross-validation and the ladder are finished.** The `grad3`
   and `grad1` iAUC cells only, three repeats, 500 steps:
   `--set steps=500 --set repeats=3 --set 'cells=["grad3","grad1"]'`. It is a sanity check, not a re-run.
   **If it changes the equivalence verdict for H7 or H8, the conclusion is that the result is not robust
   to optimization budget, and the paper's optimizer-irrelevance wording is softened accordingly.** The
   primary verdict is still reported as the pre-specified one, but it is not presented as robust.
