# Decision rules for the pending and planned runs

Written on 26 November 2026, after the reviewer's second-round plan (`REPORTS/reviewer_plan_2026-11-26.md`) and before
any result named below was summarised. Each rule says what the paper will claim under each outcome. A rule may be
tightened later only if the change is recorded here with its date and reason; no rule is loosened after its result is
seen.

## What had already been seen

- Complete and read: real CGMacros and standardized-meal trace sweeps; fitted truths (seed 0) at CV 0 and 0.25;
  random truths (seeds 0 and 1) at CV 0, 0.25 and 0.75; the unperturbed iAUC replica; all old-protocol results already
  in the paper.
- On disk but only counted, never summarised or opened: partial units of fitted truths seed 0 at CV 0.75 (11 of 45,
  run on the authors' machine), Shanghai real traces (30 of 87) and the `glucose_effectiveness` 0.7 perturbation
  (23 of 45). Run-time lines of those logs were seen; no parameter value, interval or coverage was.
- Not yet downloaded: everything else.

## Common definitions

`tau1`, coverage, bounded, narrow, interior and signed log error are as defined in the paper and in
`evaluation/coverage_metrics.py`. "Shift" is the median over all subjects of log estimate minus log truth for `tau1`.
A sweep is used only when `evaluation.manifest` reports it complete.

## Stage 1: pending runs

**R1. Fitted truths, seed 0, CV 0.75.** If the shift is larger than at CV 0.25 (0.064), the account stands: one shift
toward slower absorption closes intervals near the lower edge and opens them mid-range. Otherwise the paragraph that
explains the two directions is cut, and only the observation that bounded counts moved in opposite directions between
CV 0 and 0.25 is kept. Separately, if coverage at CV 0.75 is not lower than with exact inputs, the sentence "coverage
falls on every truth set" is replaced by the counts per truth set.

**R2. Fitted truths, seeds 1 and 2, CV 0 and 0.75.** Each seed agrees if coverage is lower and the shift is larger at
CV 0.75 than at CV 0. If both agree, the paper says the pattern held on every fitted-truth seed run. If either does not,
the paper reports the disagreement as seed dependence, with the counts.

**R3. Shanghai replica, CV 0 and 0.75.** It replicates if coverage is lower and the shift larger at CV 0.75 than at
CV 0; then one sentence says so. If not, it is reported as a limit on generality in Limitations. In either case, if
coverage with exact inputs is below 80%, the near-nominal best case is stated for CGMacros only. Shanghai real-data
counts are reported whatever they are.

**R4. Pod perturbation runs (old protocol).** These ran with the 15-point grid and fixed-step solver. They are not
reported; the converged rerun under R8 replaces them.

## Stage 2: one protocol (L-BFGS-B at every point, 29-point grid, polished estimates)

A note on direction: at a fixed reference value, solving the inner problem further can only lower a profile, which can
only remove bounded intervals. The rerun also changes the estimate (polishing), the grid, and the reference, so counts
can move either way. These are reruns, not relabels.

**R5. iAUC, rates, boxes 1x, 2x, 4x.** If no interval for k_e or k_a is bounded at any width, the abstract keeps its
statement. If any is bounded at 2x or 4x only, the abstract says "at most X of 45" and keeps the zero statement for the
primary box. If any is bounded at the primary box, the abstract states the count and the zero statement is removed.

**R6. S_I counts.** The claim stays "box-dependent" if the bounded count differs between the primary box and either wider
box. If the three counts are equal, "box-dependent" is removed. All counts are reported.

**R7. S_I replica (seeds 0 to 4) and noise variants.** Reported as they come out. The sentence that the bias is also
present with exact carbohydrate stays only if the exact-input shift has the same sign and at least half the size of
the shift at CV 0.25. The bias is attributed to boundary pinning only if the median shift among estimates not on a bound
is below one third of the overall median shift in absolute value; otherwise it is listed as unexplained.

**R8. Fixed-constant perturbations (p1, p2, n at 0.7 and 1.3; gamma and p3 as before).** The paper says a constant
materially affects S_I if its perturbation moves S_I coverage by 15 percentage points or more from the unperturbed
replica, or changes the median signed log error of S_I by 0.2 or more. If any perturbation yields a bounded k_e or k_a
interval, the count is reported next to the claim under R5.

**R9. Centroid (coordinates).** If tau1 is bounded for at least half of interior subjects, the sentence that the
centroid did not deliver what the moment argument promises is removed and the information-limit reading is rewritten.
Otherwise it stays, with the new count.

**R10. Residual gut content (10%, and the 0%, 5%, 20% dose points).** The information-limit reading is rewritten if at
10% both k_e and k_a are bounded for at least half of interior subjects with coverage of at least 90% (the original
positive-control criterion). Otherwise it stays, with the new counts. If this rerun cannot be completed, the residual-gut
argument is cut rather than labelled.

**R11. Dalla Man generator.** Bounded counts, coverage of the pseudo-true tau1 and coverage of the generator's value are
reported. "No interval covers the generator's value" stays only if the count is zero; otherwise the count is given. The
sentence that intervals fall short about the estimator's own target stays only if pseudo-true coverage among bounded
intervals is below 90%. The Dalla Man model fitted to real data under iAUC is rerun if the code allows it; otherwise its
claim is cut.

**R12. Exact-domain results** are deleted whatever any rerun would show.

## Stage 3: experiments that can change claims

**R13. Regression calibration on the random-truth replica (seeds 0 and 1, CV 0.25 and 0.75).** The calibrated carbohydrate
value is E[X | W] under a lognormal model for the true load X, with the mean and variance of log X estimated per subject
from the logged loads W and the error CV taken as known (variance of log X floored at 1e-4). Intervals are not otherwise
corrected. The attenuation account is called **supported** if, at CV 0.75 on both seeds, calibration removes at least half
of the absolute shift and recovers at least half of the coverage lost relative to exact inputs. In any other case the paper
says the mechanism was not established, and reports the counts. Knowing the CV makes this a best case for the remedy, and
the paper says so.

**R14. Isolated meals** (more than 4 h since the previous logged meal; 6 h as a second cut; subjects keep at least 10
meals). Overlap is called material if any k_e or k_a iAUC interval becomes bounded, or if the share of interior subjects
with a bounded trace tau1 interval differs by 15 percentage points or more from the same subjects with all meals.
Otherwise one sentence reports that the result did not depend on it.

**R15. Subject-specific basal glucose.** Same criterion as R14, against the 90 mg/dL runs, for CGMacros and Shanghai.

**R16. Sensor lag in the Bergman fit.** Run only if it needs a small code change. If run, the change in the Dalla Man
bias is reported. If not run, it is stated as a limitation.

## Reporting rules

**R17. The 0.99 ratio.** "Floor" is renamed "true-parameter reference", because a fit can absorb input error through S_I
and so can beat the true parameters. A bootstrap interval is reported with the ratio.

**R18. Rank correlation** is shown as a function of CV, and no recommendation to rank subjects is made for real data.

## Recorded changes

- 26 November 2026, before any Stage 2 sweep was run or inspected (applies to R5 to R10): Stage 2 profile grids stop at
  the edge of the box (`extension_points` 0). Earlier iAUC runs added six points beyond the upper bound for k_e and k_a,
  and the reference value of the in-box interval was taken over those points too. With the extension a lower value
  outside the box can change the in-box class. This is a tightening of the protocol, not of any outcome rule. One
  solver-health test of one subject was run before this note; it printed solver status and run time, no class.

## Outcomes

- **R1 (applied 26 November 2026; sweep complete, 45/45).** The shift of `tau1` at CV 0.75 is 0.187, larger than 0.064 at
  CV 0.25: **the two-direction account stands.** Coverage at CV 0.75 is 6 of 45 against 44 of 45 with exact inputs, so
  "coverage falls on every truth set" stays. Fitted truths, seed 0, CV 0 / 0.1 / 0.25 / 0.5 / 0.75: coverage 44, 43, 32,
  15, 6; bounded 6, 13, 22, 28, 28; shift 0.016, 0.030, 0.064, 0.126, 0.187.
- A solver change was recorded before any Stage 2 sweep ran: a grid point where L-BFGS-B stops without success is now
  also solved from two other starting points and the lowest value kept (`lbfgsb_restarts` = 2). On one test subject all
  39 such points had stopped on a failed line search; restarting in place gained nothing, and another start lowered one
  point by 0.31. The solver test printed solver status and loss differences only.

- **The trace sweeps are rerun too (recorded 26 November 2026, before the rerun).** The solver test found that the problem
  is not specific to iAUC: on one unit of a trace sweep already run without restarts, 3 flagged points could be lowered
  by up to 0.94 by an independent method, and with restarts none remained. So every number of Section 5.4 is replaced by
  the rerun with `lbfgsb_restarts` = 2, whatever it shows. The trace results seen so far (including the R1 outcome
  above) came from the solver without restarts; R1 to R3 are applied again to the rerun. The fitted-truth runs of seeds
  1 and 2 and the Shanghai runs without restarts are not used for any claim.

**R19. Statements of Section 5.4, applied to the rerun.**
"With exact inputs the intervals are close to nominal" stays if the `tau1` interval covers the truth in at least 41 of 45
subjects on both random-truth seeds; otherwise the counts are stated without that phrase. "Coverage falls with logging
error on every truth set" stays if coverage at CV 0.75 is below coverage with exact inputs on every truth set run.
"Without widening the intervals" stays if the median log width at CV 0.75 is not larger than with exact inputs on both
random-truth seeds. "A bounded count moves in opposite directions" stays if bounded `tau1` counts rise from CV 0 to 0.25
on fitted truths (seed 0) and fall from CV 0 to 0.75 on both random-truth seeds. The ordering statement stays if rank
correlation at CV 0.75 is at least 0.6 on both random-truth seeds.
- **Definitions for the new experiments (recorded 26 November 2026, before any of them ran).**
  *R11:* the Dalla Man control is run through the same pipeline as every other replica (coordinate trace fit, search
  rectangle, converged profiles) in place of the exact-domain code of the earlier control. One parameter distribution and
  one random stream, the same sampler as that control; sensor noise is the subject's own, as in the other replicas; fat
  and fibre are set to zero in the fitted model because the Dalla Man gut takes only the carbohydrate load. The
  pseudo-true `tau1` is the polished estimate on the noise-free output.
  *R13:* only the logged carbohydrate given to the fit is replaced; the truth and the noise are unchanged.
  *R14:* gaps are hours since the previous logged meal of any kind. The comparison is with the same subjects using all
  their meals; for the iAUC it is made at the primary box.
  *R15:* a subject's basal glucose is the median over their meals of the mean glucose in the 30 minutes before the meal,
  used in the fitted model's basal term, insulin-secretion threshold and starting state.
  *R16:* the lag is first order with a fixed time constant of 10 minutes (not the generator's per-subject delay), applied
  inside the fitted model. It is run on the noise-free and noisy Dalla Man replicas only. `p2` is not freed.
  *Window sweep:* iAUC at 60, 90 and 120 minutes, primary box; 180 minutes is the main run. If any `k_e` or `k_a`
  interval is bounded at a shorter window, the statement "at windows from 60 to 180 minutes" is replaced by the counts.
