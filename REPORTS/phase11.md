# Phases 11 and 12 report (Amendments 7 and 8)

Source: `results/phase11_summary.json` (both phases), built from the fifth freeze (`results/index.json`).
Amendments 7 and 8 were committed before the corresponding runs; no threshold or rule was changed afterwards.
Compute: 30-core CPU pod; Phase 11 six profile configurations (45 units each) and Phase 12 four, all done, 0 failed.

## H24: is the replica CGM noise too large? **not met**

Replica seed 0 with the CGM noise scaled (same random draws; carbohydrate error CV 0.25 unchanged).

| CGM noise | tau1 bounded | p bounded | nominal interval covers true tau1 | true p | calibrated cutoff tau1 | S_I bounded (iAUC) |
| --- | --- | --- | --- | --- | --- | --- |
| x1 (H13, H18) | 65% | 48% | 62% | 82% | 22.6 | 26% (9 of 34) |
| x0.5 | 73% | 71% | 33% | 51% | 68.5 | 37% (13 of 35) |
| x0.25 | 69% | 81% | 11% | 22% | 183.8 | 49% (17 of 35) |

Real data (polished): tau1 83%, p 88%. Rule (some scale with tau1 >= 73%, p >= 78%, and nominal coverage >= 90%
for both) fails at both scales. Lowering the noise raises the bounded fractions but lowers the coverage of the
nominal interval.

## H25: is the undercoverage caused by the carbohydrate logging error? **met**

Replica seed 0 with exact carbohydrate (CV 0).

| CGM noise | tau1 bounded | p bounded | nominal covers tau1 | p | cutoff tau1 / p | S_I bounded (iAUC) |
| --- | --- | --- | --- | --- | --- | --- |
| x1, exact carbohydrate | 17% | 23% | 93% | 96% | 1.9 / 1.9 | 15% (4 of 26) |
| x0.25, exact carbohydrate | 61% | 60% | 16% | 20% | 12.0 / 6.8 | 27% (10 of 37) |

Rule: at scale 1 with exact carbohydrate the nominal interval covers the true value for at least 90% of subjects
for tau1 and p: 93% and 96%, **met**.

## Reading

1. The nominal 1.92 threshold is adequate (empirical cutoff 1.9) when the inputs are exact and the noise is the
   estimated one. The undercoverage of H18 and H24 is attributable to the carbohydrate logging error.
2. With exact inputs the trace bounds tau1 and p in only 17% and 23% of interior subjects; with the logging error
   65% and 48%; real data 83% and 88%. The input error makes the trace intervals more often bounded and less often
   right. The real-data trace fractions are consistent with the same mechanism; this is not shown.
3. A second source of undercoverage appears at low noise even with exact carbohydrate (16% and 20% at a quarter
   noise). It is not identified (optimizer or likelihood shape).
4. Under iAUC the S_I weakness is not a consequence of the logging error (15% bounded with exact carbohydrate against
   26%), but it depends on the noise level (26%, 37%, 49% as the noise falls to a quarter): "inherent to the
   observable" holds at the estimated noise, which is an upper bound on sensor noise.
5. Pooled calibration, report only (replica seeds 0 to 2, 135 subjects): cutoffs tau1 15.4 (8.3 to 19.3), p 6.3 (4.0
   to 9.1); real-data fractions at them 48% (14 of 29) and 62% (21 of 34). H23 keeps its verdict (not met).

## Tally

22 hypotheses with a pass-or-fail rule among H1 to H25 (H1 formulation-dependent, H14 and H16 reported without a
rule): 10 met, 12 not met. New: H24 not met, H25 met.

## Not run

Design sweep (A15); replica or synthetic study for the Dalla Man model; heteroskedastic models for the trace; the
second source of undercoverage at low noise was not investigated.
