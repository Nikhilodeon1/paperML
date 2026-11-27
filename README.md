# Identifiability of a Bergman-type glucose digital twin

Code and results for a study of which parameters of a Bergman-type glucose simulator are determined by continuous
glucose monitoring after meals, and whether held-out prediction reveals the ones that are not.

## Layout

| path | contents |
|---|---|
| `simulation/`, `knowledge_base/` | the differentiable simulator (JAX, Diffrax) and the constants it loads |
| `personalization/` | likelihoods, parameterizations, and the estimators used as baselines |
| `evaluation/` | analyses, reruns, summaries, figures, tables, and checks |
| `cohort_cache/` | parsed, hash-checked cohorts, so no dataset download is needed |
| `results/` | stored results behind the numbers in the paper |
| `paper/` | the journal manuscript; every result in it is a generated macro |
| `workshop_caleai/` | the workshop paper |
| `julia/` | symbolic identifiability runs |
| `PREREG*.md`, `REPORTS/` | analysis plan and amendments, outcome rules, status |
| `tests/` | tests |

## Setup

```bash
pip install -r requirements-aistats.txt
tar xzf a4_fisher.tgz                      # stored first-stage fits used by several analyses
python -m evaluation.cohort_cache verify
```

## Reruns

All profile sweeps use one protocol (L-BFGS-B at every grid point, restarted where its line search fails, a 29-point
grid, and polished estimates). They are defined in `evaluation/manifest.py` and split over three machines:

```bash
python -m evaluation.stage2_queue --pod 1 --workers 8     # then --pod 2 and --pod 3
python -m evaluation.manifest                             # fails if any sweep is missing a subject
python -m evaluation.rules                                # applies the outcome rules written before the runs
```

## Paper

```bash
python -m evaluation.tmlr_build --compile                 # paper/main.pdf
python -m evaluation.lint_numbers                         # no hand-typed results
python -m pytest -q tests
```

Workshop paper:

```bash
python workshop_caleai/ws_numbers.py                      # add --draft before the reruns exist
python workshop_caleai/build.py
python workshop_caleai/package.py                         # refuses draft numbers
```

## Anonymous code package

```bash
ANON_FORBIDDEN="names,to,forbid" python -m evaluation.package_anonymized
```

The package is built from an allow-list, and the build fails if it finds a forbidden string.

This is research code. The simulator is not validated for clinical use.
