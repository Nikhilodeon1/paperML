# Test status

## Four failing tests predate the paper work (checked 26 November 2026)

The same four tests fail on a clean checkout of commit `68fb52d`, which is older than every change on the
`tmlr-revision` branch, and on the branch itself, with the same interpreter and packages:

- `tests/test_api.py::test_ask_cross_system_vodka_to_sleep`
- `tests/test_pipeline.py::test_chain_carries_grounding_label_and_citations`
- `tests/test_sleep.py::test_predict_returns_all_metrics_with_intervals`
- `tests/test_stress.py::test_drivers_ranked_and_hr_or_eda_lead`

They exercise application modules (cross-system answers, sleep, stress). They are not fixed. Before the anonymized
package is built, either fix them or leave those modules and tests out of the package.

## Latest full run (26 November 2026, branch `tmlr-revision`)

At commit `9405e39`: 717 passed, 6 skipped, 5 failed. Four are the ones above. The fifth,
`tests/test_phase15.py::test_replica_arguments_separate_residual_and_perturbation`, was broken by this work: it read
replica arguments by position, and two arguments had been added. It now reads them by name and its file passes. The
run before it also caught two other breakages from this work (the specification exporter and a similar test), fixed in
`9405e39`. A full run on the final tagged commit is still owed.

## Known side effect

Running the suite rewrites the tracked file `evaluation/figures/fig4_clinical_recovery.pdf`. Restore it with
`git checkout -- evaluation/figures/fig4_clinical_recovery.pdf` after a run.
