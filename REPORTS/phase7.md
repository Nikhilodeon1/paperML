# Phase 7 report: paper assets and packaging (in progress)

## Built and tested

| piece | module | what it does |
| --- | --- | --- |
| verdicts | `evaluation/hypotheses.py` | one function per hypothesis; status is a comparison of the observed number with the threshold in the plan |
| macros | `evaluation/paper_numbers.py` | turns stored analyses into 230+ alphabetic TeX macros (cells, boxes, objectives spelled in words) through `freeze_results` and `make_macros` |
| figures | `evaluation/figs_aistats.py` | F1 spectra, F2 window ratio, F3 profile curves and bounded fractions, F4 prediction pairs; appendix: generic rank, gut sweep, box sweep. Skips a figure whose inputs are missing |
| tables | `evaluation/tables_aistats.py` | hypotheses, bounds, profiles, ladder, prediction, comparisons, written to `paper/generated/` |
| assets | `evaluation/make_paper_assets.py` | summary, numbers, freeze, macros, figures, tables, then the number lint and the terminology lint; fails if any step fails |
| final report | `evaluation/final_report.py` | `REPORTS/final.md` from the same verdict functions |
| reproduction | `evaluation/reproduce_all_aistats.py` | full run in priority order, and a `--quick` three-subject smoke test in a temporary directory |
| package | `evaluation/package_anonymized.py` | allow-list archive; scans every included text file for names, addresses, home directories, machine names and key shapes, and writes nothing if any is found |

Tests: `tests/test_phase7.py` (14), `tests/test_lint_terms.py` (6), and the Phase 2 modules (15). All pass.

## Open items

* The assets in `paper/` and `REPORTS/final.md` currently reflect a **partial** moment-check run (H2 shows
  a status computed from the subjects finished so far). Re-run `python -m evaluation.make_paper_assets`
  and `python -m evaluation.final_report` after the moment checks and the pod profile runs are merged.
* `reproduce_all_aistats --quick` has not been run end to end yet.
* The anonymized package was built once with no findings, using a forbidden-word list supplied in the
  environment. It must be rebuilt after the last results are merged, extracted somewhere clean, and its
  tests run there.
* Figure 5 of the plan (diagnostic against profile likelihood) is absent because Phase 5 was not run.
* Figures have not been inspected by eye.
