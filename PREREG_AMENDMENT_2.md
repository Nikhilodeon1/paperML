# Pre-registration amendment 2

Amends `PREREG.md` (`421b657`) and `PREREG_AMENDMENT_1.md` (`c1845b6`). Reason and evidence:
`REPORTS/phase2-decision.md` (`31f6ff5`).

## What was seen before this amendment, stated plainly

This amendment is **post hoc**. It was written after the Phase 2.1 Fisher analysis (A4) had been run
and viewed at the pre-registered 1x box and at a 2x box. What had been seen: S_I pinned at a bound in
23/45 subjects at 1x and 5/45 at 2x; `k_e` pinned in 40/45 and 26/45; `k_a` pinned in 39/45 and 34/45;
no subject interior-converged in either box under the original definition; the Fisher eigenvectors and
their cosine with the analytic trade-off direction under the iAUC objective. Nothing from Phase 2.2
onward (bound handling, generic rank, structural identifiability, profile likelihood), and nothing from
Phases 3 to 7, had been run or viewed. No threshold is changed by this amendment.

## The defect

`PREREG.md` evaluates the S_I clause of H5 and the "S_I ranked first" clause of H10 among
**interior-converged** subjects, and the implementation defines that as: converged and no parameter at
a bound. Parameters that are not identifiable are pinned against a bound in any box, so under that
definition the set is empty by construction (0/45 at 1x and at 2x) and neither clause can be
evaluated. That is a defect in the pre-registration, not a property of the data.

## Changes

1. **Interior for parameter j.** Subject i is *interior for parameter j* when `theta_hat_ML,j` lies more
   than 1% of the box width from both bounds, the box width being measured in the coordinates the
   bounds are declared in. The H5 clause for S_I, the H10 clause "S_I ranked first", and the projected
   gradient check in Phase 2.2 use **interior for S_I**. The original global definition is still
   computed and reported alongside it, and the clause as originally written is recorded as
   **unevaluable** (0/45).

2. **Primary box.** The pre-registered 1x box stays primary for every pre-registered endpoint and for
   every cross-method comparison (A9). The random forest and SNPE baselines were trained on that box
   and are not retrained, so promoting a wider box to primary would make the prediction comparison
   unfair to them.

3. **Box sweep is reported, not adjudicated.** Every identifiability analysis (A4, A5, A6, A16) is
   reported at 0.5x, 1x and 2x of the 1x box, scaled in log space about the geometric centre of each
   interval. No box is declared the winner. The sweep is the H10 "3 bound settings" analysis already
   pre-registered. The point of reporting all three is that a bounded profile interval does not depend
   on box width and a flat one stays flat, so a result that holds in every box cannot be attributed to
   the choice of box.

4. **No threshold changes.** Every numeric threshold in `PREREG.md` and amendment 1 stands.

## Exploratory items declared in advance

These are labelled exploratory in every report and none is a pre-registered test.

* The cosine between the weak Fisher eigenvector and the analytic trade-off direction under the **iAUC**
  objective (median 0.9986 at 1x, seen before this amendment). H4 was pre-registered for iAUC+centroid
  and **stays pending** for that objective.
* For `k_e` and `k_a` only, the profile grid in Phase 2.5 is extended to 4x the upper bound as a
  diagnostic labelled outside the physiological range.
* Whether subjects pinned at an upper bound show systematic bias: predicted minus observed iAUC, and
  total carbohydrate, compared with unpinned subjects. This tests the alternative that a pinned timing
  parameter is absorbing an area deficit from carbohydrate under-reporting or from the 180 min window
  rather than reflecting a flat likelihood.
* For H3, each objective is evaluated at its own `theta_hat_ML` **and** at a common reference point per
  subject (the regularized prediction fit). Both are reported. The condition number moved from 4.3e5 to
  7.8e3 between boxes at the same objective, so a comparison across rungs at different locations on a
  flat likelihood is not a clean comparison.

## Reading rule

A pinned parameter is a symptom, not a proof of non-identifiability. The evidence for flatness is the
profile likelihood (Phase 2.5), not the bound pattern, and no report may state that bounds "bite in any
box" as a finding before that analysis exists.
