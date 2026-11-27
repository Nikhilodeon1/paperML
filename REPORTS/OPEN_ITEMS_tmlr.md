# Open items for the TMLR paper

Updated 26 November 2026, after the work on the reviewer's second-round plan (`reviewer_plan_2026-11-26.md`).
The user's ordered to-do list is `your_steps.md`. Claims are tracked in `claim_ledger.md`.

## Status of the reviewer's plan

| item | state |
|---|---|
| Stage 0: completeness manifest | done; also fails on a bounded interval with no in-box point under the threshold |
| Stage 0: unfinished runs exit nonzero | done |
| Stage 0: test failures | shown to predate this work (`test_status.md`); not fixed |
| Stage 0: hand-typed numbers | done: the checker no longer scans generated files and reports 0; numbers in math mode were reviewed and are settings |
| Stage 0: frozen index covers every macro file; tagged commit | **after the reruns** |
| Stage 1: outcome rules written before results | done; R1 applied once on the earlier solver and will be reapplied |
| Stage 1: pending runs | replaced by the reruns (the earlier solver stopped early at some points) |
| Stage 2: one protocol | code and pod queue ready; **runs not started** |
| Stage 2: exact-domain results deleted | done in the appendix |
| Stage 3: calibration, isolated meals, basal glucose, sensor lag, Dalla Man rerun | code, tests, one end-to-end unit each, and pod queue ready; **runs not started** |
| Stage 4: rules applied by code | `python -m evaluation.rules` |
| Stage 4: claim ledger | written; last column waits for results |
| Stage 4: Table 2 split and wording, interval score column, vocabulary, figure caption | done |
| Stage 4: true-parameter reference with bootstrap intervals; rank correlation against CV | done |
| Stage 4: appendix cuts and keeps; register moved out of the paper | done (27 pages, from 39) |
| Stage 4: design schematic | **not done** |
| Stage 4: Hovorka-type identifiability reference | added Garcia-Tirado et al. 2018 (abstract checked). It covers three control-oriented type 1 models, not the Hovorka model by name; I found no Hovorka-specific identifiability study |
| Stage 5: authors' hand rewrite, verification statement, reference checks | **authors** |
| Stage 6: submission gate | after all of the above |

## Still to write once results arrive (not experiments)

- A summary that writes LaTeX macros from the reruns, and the repointing of every iAUC, centroid, residual-gut, window,
  Dalla Man, calibration, isolated-meal, basal, and Shanghai number in the paper. `rules.py` already computes them.
- Rewrite of Section 5.5 around the Dalla Man rerun, and of Section 5.2's statement that a perturbation test was not
  run. A short remedy paragraph for the calibration result, and one-sentence results for R14 to R16.
- The appendix still prints numbers from the old protocol in the timing, cohort, window, Dalla Man real-fit, and
  solver-history parts. Each is replaced by its rerun or cut.
- The paper must not be sent out in its current state: Sections 5.1, 5.2, and 5.5 are old-protocol, and Section 5.4 uses
  the solver without restarts.

## Found 27 November: a partial sweep in Table 3

The four-times-box row came from 29 of 45 subjects while the caption said 45. The summary now leaves a partial width
out (`tests/test_box_table.py`), the row prints `[pending]`, and the introduction says "up to twice". When `iauc_box4`
is complete, restore "four times" in the introduction only if rule R5 allows it.

## Not run, stated as limitations

Dalla Man fitted to its own replica; a second Dalla Man parameter distribution; per-meal k_e; posterior contraction;
a free `p2`.

## Record keeping and anonymity

- `amendment16_macros.tex` and `amendment17_macros.tex` are not yet in the frozen number index.
- Tracked old failure files, `results_return/logs`, and `workshop_caleai/build` contain a local user path. The anonymous
  package must leave them out, with git history. The compiled PDF was checked: no author or user name.
- Running the test suite rewrites `evaluation/figures/fig4_clinical_recovery.pdf`; restore it afterwards.

## References checked for existence only

bellman1970structural, rothenberg1971identification, venzon1988method, efron1987better, holm1979simple,
wilson1927probable, byrd1995limited, white1982maximum, dallaman2002oral, brazeau2013carbohydrate, cook1994simulation,
albers2017personalized.

## Elsewhere

The CALE-AI workshop paper still uses old-protocol coverage numbers.
