# Phase 0 report — foundations

Gate **G0**. Everything below is measured, not estimated, except where it says "projected".

## 1. Pre-registration

`PREREG.md` is committed at **`421b657`** on branch `paper/aistats-revision`. H1–H10, their decision
rules, their thresholds and the primary endpoints (H1, H3, H5 under iAUC, H7, H9) are frozen from
that commit onward.

### Version control had to be created first

The brief assumed a branch `paper/differentiable-engine` to work from. **It did not exist, and could
not have**: `paperML/` is listed in the product repository's `.gitignore` (line 5, `/paperML`), so
none of the paper work — the JAX engine, the gradient fit, the Dalla Man model, the loaders, the
k-fold harness — had ever been committed anywhere. There was no commit to hash a configuration
against and nothing to branch from.

Resolved by giving `paperML/` its own repository, which is also what Phase 7.4 wants (an
anonymized ZIP without a `.git`), and by not touching the product repository's history:

| commit | what |
| --- | --- |
| `27800a1` | `paper/differentiable-engine` — the state as submitted to AAAI 2027 and rejected |
| `421b657` | the pre-registration |
| `abb426b` | Phase 0 foundations |

Two consequences worth stating. First, `results_io.git_commit()` appends `-dirty` when the tree has
uncommitted changes, so a result produced from an edited tree says so rather than claiming a commit
it cannot be reproduced from. Second, `.env` (live Gemini and Groq API keys) was caught in the first
`git add -A`; it was removed from the index, the commit amended, the reflog expired and the objects
pruned, and `.env` is now in `paperML/.gitignore`. The repository has no remote and was never
pushed.

## 2. Determinism

| check | result |
| --- | --- |
| 3-subject, 500-step iAUC fit, two processes, `PYTHONHASHSEED` 0 vs 98765 | **max absolute difference 0.0**, no structural mismatches |
| Fold assignment for three subjects, two processes, different hash seeds | identical |
| Repeated in-process calls to `make_folds` | identical |

Tolerance was 1e-6; the observed difference is exactly zero, which is what a seeded pipeline on one
machine should give.

**What was broken.** `evaluation/unified_kfold.py` seeded each subject's fold draw with
`abs(hash(sid)) % 100000`. Python salts `str.__hash__` per process, so the published held-out
comparison used a fold assignment that no later run could reproduce — reviewer point 5, and worse
than the review stated. `evaluation/cv_utils.make_folds` replaces it: the fold vector is a pure
function of `(subject_id, seed)` through SHA-256, and folds are balanced by permutation rather than
by independent uniform draws, which also removes the chance of an empty fold on a subject with few
meals. The old module is untouched; new analyses use the new one.

## 3. Toy validation — the correctness gate

`tests/test_identifiability_toys.py`: **18 passed, 0 failed.**

| toy | assertion | result |
| --- | --- | --- |
| A: `h = 1/k_e + 1/k_a` | Fisher rank 1, eigenvalue ratio < 1e-8 | pass |
| A | informative direction parallel to `(-1/k_e, -1/k_a)`, cosine > 0.999 | pass |
| A | null direction parallel to `(1/k_a, -1/k_e)`, cosine > 0.999 | pass |
| A | Schur complement after eliminating one rate vanishes, while the raw diagonal entry is large | pass |
| B: add `1/k_e^2 + 1/k_a^2` | full rank away from `k_e = k_a`, rank 1 exactly on it | pass |
| B | conditioning improves monotonically as the rates separate | pass |
| C | profile along A's flat direction stays flat, classified `flat` | pass |
| C | profile of a regression coefficient matches the analytic quadratic to 1e-4, and the 95% half-width equals `1.96 * se` to within 2% | pass |

One test was added beyond the brief because the first version of toy C failed for an instructive
reason: **an active bound can fake identifiability.** With a narrow box on the compensating
parameter, the profile rises at the far end of the grid even though the direction is structurally
flat — not information, just the box. The test now asserts that every grid point where the profile
rose is a point where the compensating parameter was pinned at a bound. This is reviewer point 6 in
miniature and is why 2.2 splits subjects into an interior-converged set and the full set.

## 4. Measured unit costs

AMD Ryzen 7 8845HS, **8 physical cores / 16 logical**, CPU only, float64, diffrax Tsit5 with
`rtol = atol = 1e-4`. Three subjects (41, 35, 35 meals); medians. Both columns are measured; the
difference between them is the padding fix described in section 10.

| operation | before the padding fix | **after** |
| --- | --- | --- |
| gradient fit, 500 steps, 3 parameters | 25.2 s | **16.5 s** |
| warm refit, 60 steps | 3.8 s | **2.8 s** |
| profile of one parameter (15 grid points x 60 inner steps, vmapped) | 44.7 s | **24.9 s** |
| exhaustive 20 x 8 x 8 grid (1280 points, vmapped) | 19.9 s | **10.0 s** |
| `jacrev` of the scalar loss | 0.12 s | 0.10 s |
| compilation, once per process (cached on disk thereafter) | 2.9 s | 3.7 s |

Compilation costs slightly more because there are now six padded shapes rather than one; it is paid
once per process and cached to disk, against a 1.5x to 2x saving on every solve.

Parsing the cohorts from their archives took 33 s (CGMacros), 10 s (Shanghai) and 0.1 s (Hall). All
three are now committed as a parsed cache and load in 0.15 s total; see section 11.

### Measured parallel speed-up

Fourteen workers do not do fourteen times the work on eight physical cores, so the plan divides by
the measured speed-up rather than by the worker count:

| worker processes | fits/min | speed-up | efficiency |
| --- | --- | --- | --- |
| 1 | 1.49 | 1.0x | 100% |
| 4 | 5.99 | 4.0x | 100% |
| 8 | 10.24 | 6.9x | 86% |
| 14 | 11.72 | 7.8x | 56% |

All fourteen workers fitting the same subject returned identical parameters, so throughput does not
change the answer. **Recommendation: 8 workers** on this machine. Going to 14 buys 14% more
throughput and takes every core; 8 leaves half the machine free at 86% efficiency. On a rented node
the rule is physical cores minus one, re-measured there with the same three-minute command.

## 5. Compute plan

Total serial work is **90.8 h**, down from 142.8 h before the padding fix. At 8 workers
(6.9x measured):

| job | serial h | projected wall h |
| --- | --- | --- |
| A4 full Fisher, iAUC | 0.00 | 0.00 |
| A5 profile likelihood, iAUC, 3 parameters | 0.93 | 0.14 |
| A6 ladder (2 further objectives: fit + Fisher + profile) | 2.28 | 0.33 |
| A9 grad3 CV, 5 folds x 5 repeats | 5.16 | 0.75 |
| A9 grad1 CV, 5 folds x 5 repeats | 5.16 | 0.75 |
| A9 grid3 CV, 5 folds x 5 repeats | 3.14 | 0.46 |
| A9 multistart grad3, 10 inits x 5 folds x 5 repeats | 51.60 | 7.52 |
| A9b trace-objective CV, 2 cells | 10.32 | 1.51 |
| A10 noise-floor replica, 3 repeats | 3.10 | 0.45 |
| A11 synthetic recovery, 60 virtual subjects x 4 observables | 6.07 | 0.89 |
| A12 replication on Hall and Shanghai | 3.06 | 0.45 |
| **total** | **90.8** | **13.25** |

At 14 workers the total is 11.6 h. On a 64-vCPU node (32 physical cores, expected speed-up about
29x after allowing for lower server clocks) it is roughly 3 h, with multistart at about 1.8 h.

**No job now exceeds the 12-hour ceiling, and the one that was close no longer is.** Multistart fell
from 11.5 h to 7.5 h, so the three reductions proposed earlier are no longer needed on compute
grounds. Nothing has been reduced; the pre-registered 10 initializations x 45 subjects stands.

Two caveats on the projections. The Fisher row is understated as a *method*, not as a cost: 0.10 s is
`jacrev` of the scalar loss, whereas 2.1 needs the stacked per-meal Jacobian. Computing it as
`vmap(jacrev(one_meal))` costs about one gradient over all meals rather than one reverse pass per
meal, and the Phase 2 gate check on 3 subjects will measure it. And the trace objective carries 43
residuals per meal instead of 1; the ODE solve dominates, so A9b should hold, but it is the row most
likely to move.

## 6. Where the specification lives

- **`SPEC_EXPORT.md`** (299 lines) — generated by `evaluation/export_spec.py` from the live modules:
  the 21-entry state vector and which 8 entries this engine stage evolves, every population default,
  the inference bounds, the fixed constants, the meal input model, the solver settings, the
  observation operators and smoothing primitives verbatim, the vector field verbatim with its file
  and line numbers, and the loss with its regularization.
- **`paper/spec_table.tex`** — the parameter and constant tables for the appendix, generated.

The appendix will `\input` these. Nothing in them is transcribed, and
`tests/test_spec_and_cohort.py` fails if the committed export stops matching the code.

## 7. Cohort as loaded

| | CGMacros |
| --- | --- |
| subjects with ≥ 10 usable meals and complete biometrics | 45 |
| meals | 1640 |
| meals per subject | median 36, range 17–64 |
| carbohydrates per meal (g) | median 56, IQR 24–73 |
| sampling | 5 min, window −30 to +180 min, 43 samples |

The busiest subject has 64 meals against a padded width of 72, so no meal is being silently
dropped — asserted in the tests.

## 8. Things found that were not in the brief

1. **All three dataset loaders were broken.** `evaluation/cgmacros.py` resolved
   `parents[2] / "data" / "physionet.org" / "files" / "cgmacros"`, which stopped existing when the
   datasets moved to a shared folder; `hall_loader.py` and `shanghai_loader.py` had absolute paths
   typed into the source, and the shared folder has since been flattened so that `physionet.org/`
   no longer exists. All three now resolve through a new `paperML/data_paths.py`
   (`HORIZON_DATA_DIR` → a local `data/` → the nearest `datasets/` found by search). One line each.
   Nothing in Phase 0 onward could have run otherwise.
2. **The product repository has the same breakage**, in `ml/evaluation/cgmacros.py`, which asks
   `data_paths.dataset("physionet.org", "files", "cgmacros", ...)` while the dataset now sits at
   `datasets/cgmacros/`. Out of scope here (`ml/` is not to be touched) and flagged as a separate
   task.
3. **The engine declares `float32` state and relies on promotion under float64.**
   `simulation/jax_engine.py` builds its state with an explicit `jnp.float32`, so with x64 enabled
   the first write upcasts and JAX emits `FutureWarning: scatter inputs have incompatible types …
   In future JAX releases this will result in an error.` The values are float64 and correct today.
   float64 is not optional — the toy tests assert eigenvalue ratios below 1e-8, which single
   precision cannot represent — so the Phase 1 engine variants will declare their dtype instead of
   inheriting it, and the warning is filtered with a pointer to this note rather than hidden.
4. **Four pre-existing test failures**, confirmed identical at the baseline commit `27800a1` before
   any change here: `test_api.py::test_ask_cross_system_vodka_to_sleep`,
   `test_pipeline.py::test_chain_carries_grounding_label_and_citations`,
   `test_sleep.py::test_predict_returns_all_metrics_with_intervals`,
   `test_stress.py::test_drivers_ranked_and_hr_or_eda_lead`. They are orchestration and physiology
   assertions unrelated to the identifiability work. Suite total: **481 collected, 471 passed,
   4 failed (pre-existing), 6 skipped**. Of the 481, **131 are new in this phase** (18 toys, 11 fold
   assignment, 21 statistics, 21 results pipeline, 11 determinism, 13 specification and cohort,
   14 subject loss and padding, 22 runner failure handling).

## 9. What Phase 0 built

| module | purpose |
| --- | --- |
| `data_paths.py` | one dataset resolver for all three loaders |
| `evaluation/cv_utils.py` | SHA-256 fold assignment, balanced, reproducible across processes |
| `evaluation/results_io.py` | result files carrying commit, config hash, seed and environment; atomic writes; resumability |
| `evaluation/freeze_results.py` | `results/index.json`, per-file and combined SHA-256, macro collection, `--check` |
| `evaluation/make_macros.py` | `paper/results_macros.tex`, one `\newcommand` per scalar, intervals as separate macros |
| `evaluation/lint_numbers.py` | fails the build on a hand-typed number outside math, citations, years and an allowlist |
| `evaluation/stats_utils.py` | BCa bootstrap, Wilcoxon, equivalence by 90% interval against a margin, Holm, Wilson, Clopper-Pearson, Fisher z, Kendall, Spearman |
| `evaluation/identifiability_tools.py` | full Fisher with eigenvectors, Schur complement, profile likelihood, classification into identifiable / one-sided / flat |
| `evaluation/jax_config.py` | float64 and the persistent compilation cache |
| `evaluation/determinism.py` | two-run comparison, structural and numerical, under different hash seeds |
| `evaluation/cohort_data.py` | one cohort loader, sorted, cached to disk |
| `personalization/subject_loss.py` | one definition of the padded arrays, the parameter vector and the iAUC loss |
| `evaluation/export_spec.py` | `SPEC_EXPORT.md` and `paper/spec_table.tex` from the live code |
| `evaluation/benchmark_compute.py`, `benchmark_scaling.py`, `compute_plan.py` | unit costs, measured speed-up, projected plan |
| `evaluation/smoke_fit.py` | the small deterministic fit that gates every long job |

Environment frozen in `requirements-aistats.txt` (116 packages; jax 0.11.0, diffrax 0.7.2,
optax 0.2.8, numpy 2.5.0, scipy 1.18.0, Python 3.14.2). `jaxopt` is not installed, so L-BFGS comes
from `optax.lbfgs` with its zoom line search; both optimizers are already exercised by the toys and
will serve H10.

## 10. A real performance defect, found and fixed

`gradient_fit.py` padded every subject's meals to a fixed width of 72 and masked the unused slots.
The mask makes the padded slots contribute nothing to the loss -- but **it does not stop them being
simulated.** A padded slot still runs a full ODE solve; only afterwards is its residual multiplied by
zero.

Measured cost against padded width, 100 optimizer steps:

| width | 8 | 16 | 24 | 40 | 56 | 72 |
| --- | --- | --- | --- | --- | --- | --- |
| seconds | 1.09 | 1.55 | 2.07 | 3.08 | 4.03 | 4.51 |

Close to linear. The cohort's median subject has 36 meals and a training fold of one has about 29,
so padding everything to 72 was spending roughly half the run solving meals that do not exist.
Applied to the real meal counts:

| scheme | whole-subject fits | 4/5 training folds | compiled shapes |
| --- | --- | --- | --- |
| single pad of 72 (before) | 1.00x | 1.00x | 1 |
| buckets 24/40/56/72 | 1.49x | 1.73x | 4 |
| **buckets 20/28/36/44/56/72 (now)** | **1.58x** | **1.86x** | 6 |
| exact width per subject (unreachable: one compile each) | 1.73x | 2.01x | 45 |

Bucketing is within 8% of the theoretical maximum at six compiled shapes, each costing 3.7 s once
per process and cached to disk.

**It changes no result, and that is asserted rather than assumed.** A padded slot carries
`carbs_g = 0`, whose simulated iAUC is a small positive number rather than zero -- the smooth
positive part of a flat trajectory is not exactly zero -- but the residual is multiplied by
`mask = 0`, and adding exact zeros to a sum does not change it in IEEE arithmetic.
`tests/test_subject_loss.py` asserts the loss is **bit-identical** and the gradient **exactly equal**
across widths 20, 28, 44 and 72, and separately pins the fact that a zero-carbohydrate slot predicts
a nonzero iAUC, so that dropping the mask could never pass silently.

Also fixed: `XLA_FLAGS` carried a token (`intra_op_parallelism_threads=1`) that is not an XLA flag.
XLA ignores unknown tokens silently, so it had no effect and thread control was coming entirely from
`OMP_NUM_THREADS`. Removed rather than left to look like it was doing something.

## 11. Failsafes for a long run

Two pieces, both built because these jobs will run for hours on hardware that can be taken away.

### The cohorts are committed, so a node needs no datasets

`evaluation/cohort_cache.py` parses each cohort once and commits the result. The archives total about
650 MB and the Shanghai parser needs `pandas` plus two Excel engines; none of that has to exist on a
compute node.

| cohort | subjects | meals | cached size | parse time |
| --- | --- | --- | --- | --- |
| CGMacros | 45 | 1640 | 180 KB | 33 s |
| Hall 2018 | 24 | 114 | 7 KB | 0.1 s |
| ShanghaiT2DM | 97 | 2671 | 190 KB | 10 s |

All three load in 0.15 s. The counts match the brief exactly (n=45, n=24, n=97).

Three things worth noting about the cache:

* **It is verified, not trusted.** `manifest.json` records a SHA-256 per file; a cache that fails its
  hash raises rather than being silently re-parsed, because a corrupt cohort would otherwise produce
  a complete set of plausible wrong results.
* **It is byte-reproducible.** The gzip header timestamp is pinned to zero, so rebuilding from the
  same archives gives the same hash instead of churning the repository.
* **It carries the fasting labs.** CGMacros HbA1c, fasting glucose, fasting insulin, HOMA-IR and BMI
  are in it for all 45 subjects, because Phase 5 needs them (the leakage control is a partial
  correlation against fasting glucose) and a node without the archive could not add them later.

`evaluation/cohort_data.py` now normalizes all three cohorts into one record schema, so cross-cohort
code no longer has to know that one loader spelled iAUC `iauc` with a `glucose` block and the others
`observed_iAUC` with a `cgm_curve`. Two cohort differences are carried on the object rather than
smoothed over, and both will matter in Phase 5 and 6:

* **Hall scores a fixed 0-145 min window**, not the 0-180 min window CGMacros uses. Any cross-cohort
  iAUC comparison has to say so.
* **Shanghai sampled every 15 min** and its stored 5-min grid was filled by linear interpolation
  (`grid_is_interpolated = True`). A trace objective there must not treat interpolated points as
  independent observations, or it would inflate n threefold and shrink every interval.

### The runner cannot lose more than the units in flight

`evaluation/runner.py` executes an analysis as resumable work units. The filesystem is the
coordination mechanism, which is what makes all of the following fall out rather than be bolted on:

| event | consequence |
| --- | --- |
| Ctrl-C once | stops dispatching, in-flight units finish, clean exit. Re-run to resume. |
| Ctrl-C twice | stops now; at most the in-flight units are lost |
| node reclaimed, power cut, OOM kill | every finished unit is already on disk |
| a worker dies mid-unit | its claim goes stale, another worker takes the unit, a replacement starts |
| a unit hangs | its worker is killed after `--unit-timeout`; the rest of the run continues |
| a unit raises | retried, then recorded with its traceback and reported; not retried on later runs |
| two runs at once | harmless; claims are exclusive `O_EXCL` locks |
| a changed config | lands in a new directory; results from one setting never pass for another |

Results are written to a temporary file and renamed, so a process killed mid-write leaves either the
old file or the new one, never half of one. Progress, units per hour and an ETA are in
`logs/<analysis>.progress.json`. Only the runner's own children are ever signalled, never every
python process on the machine.

**`tests/test_runner.py` makes each of those happen on purpose: 22 tests** covering resume
recomputing nothing (asserted on file modification times), a unit that calls `os._exit(9)` to
imitate an OOM kill, a unit that never returns, transient failure succeeding on retry, a recorded
failure not being retried, `--restart` retrying failures while keeping successes, claim exclusivity,
and stale-claim recovery.

One performance defect in the runner itself, found while writing those tests and fixed: the
supervisor validated every finished unit on every two-second poll, which means parsing every result
file. At 45 units that is nothing; at the thousands of units a fine-grained analysis would produce,
the monitor would have become the bottleneck. Validated units are now remembered.

`docs/POD.md` has the setup, the flags and the failure table for running this on a rented node.

## Waiting for a decision

1. The repository arrangement above (own repo for `paperML/`, baseline branch created from the
   current state) — confirm or redirect.
2. Worker count: 8 on this laptop, or physical cores minus one on a node,
   re-measured there with `benchmark_scaling`.

The multistart reduction is withdrawn: after the padding fix that job is 7.5 h rather than 11.5 h, so
there is no compute reason to cut it, and the pre-registered 10 initializations x 45 subjects
stands.
