# Claims ledger and reviewer-objection ledger

Labels only; nothing here rewrites the abstract. Status date 2026-10-04, before the Phase 8 pod results
were merged. Entries marked **pending** are decided by a result file that has not been produced or merged
yet, and are updated by `evaluation/phase8_summary.py` plus a re-read of this table.

## A. Sentences of the submitted abstract and TL;DR

The text of the abstract and TL;DR has not been supplied, so the sentences cannot be labelled one by one.
Below are the five known suspects, each with the label the current evidence supports for the CLAIM as
named in the task description. They must be re-checked against the exact wording.

| claim as named | provisional label | deciding result |
| --- | --- | --- |
| "S_I reliably recoverable" | **contradicted** as an identifiability claim; **pending** as a recoverability claim | H5 S_I clause not met at any box: bounded interval in 0 of 15, 3 of 22, 24 of 40 interior subjects at 0.5x, 1x, 2x (`results/phase2_summary.json`, `profile_likelihood`); H13 replica (pending) decides whether the weakness is inherent or model-related |
| "residual gap attributed primarily to the fitting objective" | **contradicted** | H9 not met: the trace objective is worse on held-out iAUC by +32 [+2, +67]; the objective changes what is predicted well (trace and peak time) but not the iAUC error. H14 (pending) replaces the attribution with a noise budget |
| "three cohorts" | **pending** | CGMacros analysed in full. Shanghai and Hall replication (H15) queued; the cached cohorts exist. If H15 does not complete, the claim is **unsupported-in-the-revision** |
| "two model classes" | **unsupported-in-the-revision** | the Dalla Man-style model was examined only in the earlier draft; no result in this revision. The reparameterized and tied variants are the same model, not a second class |
| "peak timing shows increased sensitivity" | **pending** (wording) | trace-objective fit improves peak-time error (41.6 vs 47.5 min) but the personal mean is better still (41.0); the area is more sensitive to timing than to S_I at a 90 minute window (ratio 1.1 to 1.7) and not at three hours or more (H2 met) |

## B. Reviewer objections (the first six are the ones in the project record)

| # | objection | status | where it is handled |
| --- | --- | --- | --- |
| 1 | gradient magnitude and diagonal Fisher are local heuristics, not proof of structural non-identifiability | **addressed** | full Fisher with eigenvectors and the Schur complement, profile likelihood, generic rank, structural-identifiability runs, an analytic exchange symmetry (`REPORTS/phase2.md` sections 1 to 5, `revision_writeup.md` sections 4 and 6). The wording is restricted to "locally identifiable but ill-conditioned" for iAUC |
| 2 | ceiling confounded with the iAUC target | **addressed**, one rung pending | three-rung ladder in Fisher and held-out prediction; H9 shows the ceiling persists for iAUC, the trace helps its own targets. The trace and centroid profile rung is pending the merge of the node archive |
| 3 | mathematical specification missing (model, loss, observation operators, identifiability criterion) | **partially addressed** | model, bounds, loss and constants exported to `SPEC_EXPORT.md`; observation operators in `simulation/jax_observables.py`; identifiability criterion defined (profile interval bounded inside the box at a drop of 1.92, interior per parameter, exchange direction). The manuscript text that states them has not been written |
| 4 | error decomposition 79/21/0 undefined and not additive | **addressed, pending the replica results** | replaced by the well-specified replica (H13, H14) reported as a table of what-if noise budgets, no additive shares (`evaluation/subject_source.py`, `phase8_summary.py`) |
| 5 | comparison uncontrolled; one fold assignment; no repeated-split uncertainty; optimizer, parameter set and information not isolated | **addressed** | five repeats of five-fold cross-validation with SHA-256 seeded folds identical for every method; paired bootstrap intervals, Holm correction; optimizer isolated (gradient vs grid, H7), parameter set isolated (three parameters vs S_I only, H8), information isolated (ladder, H9); 500-step check |
| 6 | S_I gradient may reflect an active bound or non-convergence | **addressed** | box sweep at three widths, per-parameter interior definition, bound-projected gradient diagnostic (separation survives), multistart L-BFGS check of the maximum-likelihood fits (gap below 1.92 everywhere). The S_I clause of H5 fails, which is itself the answer |
| 7 | abstract said "equivalent prediction" while Table 2 showed a 403 MAE spread with "decisive" win rates; no equivalence criterion | **addressed in the analysis; wording pending** | equivalence is now a stated criterion (90% interval inside +/-150, margins 90 and 300 also reported) and applies only to the pairs it was tested on (gradient vs grid, three vs one parameter). The 395 gap between gradient and SNPE is a difference, not an equivalence |
| 8 | joint, not coordinate-wise, identifiability: need sensitivity-matrix rank or profile analysis | **addressed** | generic rank of the stacked sensitivity matrix (full rank, ill-conditioned), full Fisher spectrum, joint profile likelihood with the other parameters re-optimized |
| 9 | model-class test covered one modified Dalla Man reconstruction only | **not addressed** | not re-run; the paper says the second model class was examined only in the earlier draft |
| 10 | peak-timing result reads as evidence against a modality-wide ceiling | **partially addressed** | the revision reframes the ceiling as a property of the iAUC target: the trace objective improves peak-time and trace error, not iAUC error. The claim of a modality-wide ceiling is not made |
| 11 | novelty: differentiable ODE inversion is established (Kersting et al. 2020; Beck et al. 2024); missing references GlucoFM, HIPNO | **not addressed** | a writing task: cite and position against these |
| 12 | other explanations not excluded: shared model misspecification, meal-recording error, preprocessing | **partially addressed, pending** | replica (noise, carbohydrate error) H13/H14; carbohydrate-scale sensitivity (T5 ii); area-deficit check on pinned subjects (pinned subjects are under-predicted by about 1,000); window dependence (H2). Preprocessing choices other than the window are not varied |
| 13 | suggestions: sensitivity-guided design; subject-level recoverability; cohort-and-protocol table; clinical correlations as a case study | **partially addressed** | subject-level correlates computed (`stored_analyses.py`): S_I boundedness is unrelated to meal count, carbohydrate range, mean iAUC or the estimate; timing pinning tracks mean iAUC (rho 0.65). Cohort table in `REPORTS/phase8.md`. Design sweep (A15) not run; clinical correlations not re-run |
| 14 | presentation defects in the old draft | **partially addressed** | figures and tables are regenerated from stored results. The new text must state the effective number of Bergman states and define every ratio: the glucose dynamics in the active regime are driven by five states (glucose, insulin, insulin action, stomach, gut), not three; the ratio `abs(d iAUC / d log k) / abs(d iAUC / d log S_I)` is defined in `evaluation/moment_checks.py` |
