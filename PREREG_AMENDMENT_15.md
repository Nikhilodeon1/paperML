# Pre-registration amendment 15 (TMLR revision, report-only: curvature and counts)

Amends `PREREG.md` and amendments 1 to 14. No hypothesis is added, no threshold changes and no earlier verdict is edited.
Everything below is **report-only** and post hoc, and each item is reported whatever it shows.

## What was seen before this amendment

Everything through Phase 18, and a third outside read of the manuscript. The read asked for three things this amendment adds:

* a direct measure of why the carbohydrate logging error makes the trace profile intervals narrower while coverage falls. The
  stored numbers (Amendment 14) show that the pilot noise estimate barely moves with the error. The account to be measured is
  arithmetic: a logged load that is the true load times a unit-mean multiplier has second moment `1 + CV^2` times the squared
  true load, so the sensitivity of the simulated trace, and with it the Fisher information, grows with the error against an
  unchanged noise level;
* the unperturbed-replica bias of `S_I` (median signed log error -0.19) to be explained: it is -0.20 on the exact-carbohydrate
  replica (Amendment 14), so it does not come from the logging error;
* the overall (not only interior) counts of bounded intervals beside the interior fractions.

## Analyses (A20 and counts from stored results)

* A20 `evaluation/carb_curvature.py`: for the polished trace coordinate profiles of replica seed 0 at logging error 0, 0.25
  and 0.75 (same subjects, same stored estimates and noise models), the Hessian of the negative log-likelihood at the polished
  estimate in `(log S_I, log tau1, log p)`, its eigenvalues, the Schur complement of the `(tau1, p)` block after profiling out
  `S_I`, the pilot noise standard deviation, and the ratio of the mean squared logged load to the mean squared true load.
* From stored results: the median signed log error of `S_I` and the count of estimates on a bound on the exact-carbohydrate
  replica under the iAUC; the overall counts of bounded `S_I`, `k_e` and `k_a` intervals on the real data.

## Rules

None. A20 is a measurement, and its result is written beside the carbohydrate-error sweep whichever way it points.
