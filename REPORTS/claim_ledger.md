# Claim ledger (Stage 4)

Every claim made in the abstract, introduction, and conclusion, the evidence for it, the protocol that produced the
number now in the paper, and what decides whether it survives the reruns. "New" is the one converged protocol
(L-BFGS-B with restarts, 29-point grid, polished estimates). "Old" is anything earlier. **No claim marked old or
pending may stay in the submitted paper.** Fill the last column from `REPORTS/rule_outcomes.md` after the reruns, then
freeze results.

| # | claim (where made) | evidence | protocol now | sweeps that replace it | decided by | after reruns |
|---|---|---|---|---|---|---|
| C1 | Under the area target no interval for k_e or k_a is bounded (abstract, contribution 1) | Table 3 | **old** | `iauc_box1`, `iauc_box2`, `iauc_box4` | R5 | |
| C2 | The same holds at windows from 60 to 180 min (contribution 1) | Appendix window paragraph | **old** | `window60`, `window90`, `window120` | window rule | |
| C3 | The S_I interval is bounded for a minority and depends on the box (Section 5.2) | Table 3 | **old** | box sweeps, `iauc_rep_*`, `iauc_synth_*` | R6, R7 | |
| C4 | Held-out area error does not show the unidentified rates: wrong-corner rates stay within 1.03 of the true-parameter reference (abstract, contribution 1, conclusion) | Table 4 | no profiles involved; stored prediction runs | none | stands as computed; interval now reported | |
| C5 | Only peak time separates wrong rates, and only in part (contribution 1, recommendations) | Table 4 | no profiles involved | none | stands as computed | |
| C6 | With exact inputs the trace recovers tau1 and p: intervals near nominal (abstract, contribution 2, conclusion). State it as a same-model best case | Table 6 | new, but solver without restarts: **pending** | `rand0.0_s0`, `rand0.0_s1` | R19 | |
| C7 | Logging error shifts estimates toward slower absorption without widening intervals; coverage falls (abstract, contribution 2, conclusion) | Table 6, Figure 3 | **pending** (same reason) | `rand*`, `fit*` | R1, R2, R19 | |
| C8 | The bounded count can rise or fall under the same error (abstract, contribution 2, conclusion) | Section 5.4 | **pending** | `rand*`, `fit*` | R1, R19 | |
| C9 | The shift is the mechanism of the coverage loss, as opposed to only accompanying it | not claimed yet | not run | `calibrated*` | R13 | |
| C10 | Subject ordering degrades less than coverage; tau1 suits ordering in simulation under the assumed error (abstract, recommendations). Never recommend it for real data | Table 6, Figure 3d | **pending** | `rand*` | R19, R18 | |
| C11 | The result does not depend on overlapping meals | not claimed yet | not run | `gap4_*`, `gap6_*` | R14 | |
| C12 | The result does not depend on the fixed basal glucose | not claimed yet | not run | `basal_*` | R15 | |
| C13 | Fixed constants matter for S_I | Section 5.2 | **old** (and one sentence says a test was not run) | `iauc_pert_*` | R8 | |
| C14 | The information limit comes first and the symmetry second (Discussion) | centroid, Dalla Man real fit, residual gut | **old** | `centroid_*`, `dm_real_iauc`, `residual*` | R9, R10, R11 | |
| C15 | A different gut model moves the target: pseudo-true bias against the generator (contribution 3) | Section 5.5 | **old** | `dm_clean`, `dm_cgm` | R11 | |
| C16 | Sensor lag explains part of that bias | not claimed yet | not run | `dm_*_lag` | R16 | |
| C17 | Shanghai replicates the trace pattern | not claimed yet | pending | `shanghai_*` | R3 | |
| C18 | Ten hypotheses were fixed before analysis; of five primary endpoints three were not met, one met, one inconclusive (Table 2) | Table 2 | stored verdicts | none | stands; timing is author-reported | |

## Claims that are cut or limited whatever the reruns show

- Exact-domain results: deleted (R12). Done in the appendix on 26 November.
- Gradient diagnostic analysis, HbA1c check, learned baselines, heteroskedastic variants, recalibrated cutoff: removed
  from the manuscript on 26 November. Their outcomes in Table 2 remain, as pre-registered rows.
- Dalla Man on Dalla Man, a second generator distribution, per-meal k_e, posterior contraction: limitations, not run.
