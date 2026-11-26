# Pre-registration amendment 17 (TMLR revision: converged profiles, one coverage definition, pseudo-true Dalla Man)

Amends `PREREG.md` and amendments 1 to 16. No threshold of an earlier document changes and no earlier verdict is edited.
Everything below is **post hoc** and exploratory. Each rule is fixed before the computation it governs is read.

## What was seen before this amendment

Everything through Amendment 16, and two further mock reviews. They raised these points, each checked against the stored results:

* The 15-point grid with linear interpolation overstates the profile excess at the true value (Amendment 12: median 5.3 against
  0.2 computed directly for tau1 at a quarter of the noise). Coverage computed from that grid is therefore too low, and the
  carbohydrate-error sweep of Amendment 9 and its seed and Shanghai replications were all read off it.
* The stored profiles of p from before the real-pole fix undid the clamp at upper-tail grid points (the projection after
  resetting the fixed coordinate). Both inner solvers now respect the fixed coordinate.
* Coverage, as defined in the paper, is flattered by wide intervals: an interval open to the box edge covers the truth
  nearly for free (42 of 45 subjects at exact carbohydrate, with 6 bounded).
* The penalty weight of the prediction fit (0.01) is added to a mean squared error in (mg dL^-1 min)^2 whose size is about
  the square of the mean absolute error (3025), so it is negligible.
* Bergman's tau1 is not the Dalla Man mean appearance time, so the coverage of the latter by Bergman intervals compares an
  estimator with a quantity it does not target.

## Analyses

* **R26 (new runs, `scripts/pod_phase21.sh`):** coordinate profiles (hull coordinates, polished estimates) with the converged
  inner solver and a 29-point grid on the real CGMacros data, on replica seed 0 at carbohydrate error 0, 0.1, 0.25, 0.5 and
  0.75 with fitted truths, and with random truths drawn through the box (seeds 0 and 1; error 0, 0.25 and 0.75). Seeds 1 and 2
  with fitted truths, Shanghai (real, replica at 0 and 0.75) and Hall are queued in the same script.
* **Coverage metrics (`evaluation/coverage_metrics.py`):** one definition. The interval is the 95% profile interval clipped to
  the box; coverage is the share of all subjects whose interval contains the truth. Companions: the share bounded; coverage among
  intervals that are bounded and narrower than 0.5 in log units; the median interval score (95%, log scale).
* **R27 (A28, `evaluation/a28_dm_pseudo_true.py`):** the pseudo-true tau1 of the Bergman-type fit to noise-free Dalla Man output
  (same parameters and meal schedules as A27), its bias against the Dalla Man mean appearance time, coverage of the pseudo-true
  value by the A27 intervals, and rank correlations.
* **R28:** the Fisher spectrum at solver tolerance 1e-8 for all 45 subjects.
* **R29:** the iAUC profiles at a box four times as wide, with the primary and double boxes (pod script).
* **Robustness (pod script, not run locally):** iAUC profiles with p1, p2 and n scaled by 0.7 and 1.3 in the generator only.

## Rules

None are graded. Every number is reported whichever way it points, and the text is changed to follow the numbers.

## Not run

Dalla Man generated and fitted by the Dalla Man model (a replica for the second model class), a Bergman variant with sensor lag
and a free p2, a second generator parameter distribution, a latent-carbohydrate or SIMEX remedy for the logging error, meal-level
gastric emptying, a restriction to meals separated by more than 4 hours, subject-specific basal glucose, and posterior
contraction as a box-free complement. The manuscript lists each as a limitation.
