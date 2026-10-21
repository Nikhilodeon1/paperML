# Phase 10 report (Amendment 6)

Source: `results/phase10_summary.json`, built from the third freeze (`results/index.json`). H23 is computed
from stored profile grids (no new fit). H16 (heteroskedastic noise) was run on a 30-core CPU pod: two
configurations of 45 units each, all done, 0 failed (about 9 minutes each). Amendment 6 was committed
(commit `1f84686`, together with the Phase 10 code) before any Phase 10 result existed; no threshold or rule
was changed afterwards.

## H23: recalibrated cutoff: **not met**

Cutoff = 95th percentile of the profile rise at the true value on replica subjects (bootstrap over real
subjects).

| quantity | S_I (iAUC, replica seeds 0 to 4, n = 225) | tau1 (trace, coordinates, seed 0, n = 45) | p (same) |
| --- | --- | --- | --- |
| calibrated cutoff (nominal 1.92) | 2.75 [2.05, 3.87] | 22.65 [4.98, 43.61] | 6.69 [2.19, 12.96] |
| true value inside the nominal interval (replica) | 91% | 62% | 82% |
| true value inside the calibrated interval (replica) | 95% | 93% | 93% |
| bounded among interior, real data, nominal | 3 of 22 (14%) | 24 of 29 (83%) | 30 of 34 (88%) |
| bounded among interior, real data, calibrated | 1 of 22 (5%) | 12 of 29 (41%) | 21 of 34 (62%) |
| bounded among interior, replica, calibrated | 17 of 157 (11%) | 4 of 34 (12%) | 5 of 33 (15%) |

Rule: part (a) S_I below 50%, **met** (5%); part (b) tau1 at least 70% and p at least 50%: tau1 41%, p 62%,
**not met** on the tau1 clause. So H23 is not met.

Reading. The nominal cutoff undercovers for the coordinates (62% and 82% of true values inside), as the H18
coverage numbers suggested, so part of the earlier trace result was a cutoff effect. Recalibrating lowers the
real-data fractions from 83% and 88% to 41% and 62%. It does not rescue the combination claim (tau1 fails its
70%). Two things stay unexplained and are in the paper: real data remain well above the replica at the calibrated
cutoff (41% and 62% against 12% and 15%); and the coordinate cutoffs rest on 45 replica subjects with very wide
intervals, so the calibrated fractions are uncertain.

## H16: heteroskedastic noise, S_I profile (exploratory): **reported**

45 CGMacros subjects, iAUC, primary box, real data; `root` (sd proportional to sqrt of the observed area) and
`proportional` (sd proportional to the observed area), floor at 10% of the subject's mean observed area.

| model | S_I bounded among interior | bounded over all 45 | changes against baseline | median fitted sigma |
| --- | --- | --- | --- | --- |
| homoskedastic (plan) | 3 of 22 (14%) | 3 | | |
| root | 0 of 15 (0%) | 0 | 0 more, 3 fewer | 75.7 |
| proportional | 0 of 2 (0%) | 0 | 0 more, 3 fewer | 3.05 |

The H5 S_I clause (80%) is unmet under every variant, so the S_I result does not depend on the variance
structure tried. Caveat stated in the paper: under the proportional model the S_I estimate is on a bound for 43
of 45 subjects, so the interior fraction is nearly uninformative, and the large fitted relative scale (3.05)
says proportional errors describe the residuals poorly. iAUC objective only.

## Verdict tally after Phase 10

20 hypotheses with a pass-or-fail rule among H1 to H23 (H1 formulation-dependent, H14 and H16 reported
without a rule): 9 met, 11 not met. New: H23 not met.

## Not run

Design sweep (A15); replica or synthetic study for the Dalla Man model; heteroskedastic models for the trace
objective.

## Still to do before submission

Official `tmlr.sty` and `tmlr.bst` (not yet in `paper/`), rebuild and fix overfull boxes under it, anonymized
package, final read-through of claims, user commits (the manuscript and report changes from this session are
uncommitted).
