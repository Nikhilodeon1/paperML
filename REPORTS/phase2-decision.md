# Phase 2 decision memo: the bound-pinning, and what it actually is

A pre-registered stop rule fired ("more than 20 percent of subjects are on a bound for S_I"). This memo
is the diagnosis and the decision it forces. No threshold has been changed and no result discarded.

## What fired

A4 complete on all 45 CGMacros subjects at `theta_hat_ML` under the iAUC objective:

| | 1x box (pre-registered) | 2x box (H10 sweep) |
| --- | --- | --- |
| S_I pinned | **23/45 (51.1%)** — 12 lower, 11 upper | **5/45 (11.1%)** — 5 lower, 0 upper |
| gastric emptying `k_e` pinned | 40/45 (88.9%) — 8 lower, 32 upper | 26/45 (57.8%) — 0 lower, 26 upper |
| carb absorption `k_a` pinned | 39/45 (86.7%) — 6 lower, 33 upper | 34/45 (75.6%) — 0 lower, 34 upper |
| interior-converged | **0/45** | **0/45** |
| timing-block condition number (median) | 4.32e5 | 7.83e3 |
| median S_I | 0.6615, range 0.3000–1.6000 | 0.7635, range 0.1299–3.6196 |

The 2x box is part of the H10 bounds sweep that `PREREG.md` already requires, so running it is a planned
robustness analysis rather than a change of specification.

## The diagnosis: two different causes, which the sweep separates

**S_I was a box problem.** At 1x the subjects pile up at *both* ends, 12 against 11, which is the
signature of an interval too narrow for the cohort rather than of a one-sided misspecification. CGMacros
spans normal, prediabetic and diabetic subjects, and the 1x interval [0.30, 1.60] was inherited from
`npe.py` for comparability with the random-forest baseline, not derived from physiology. Double it and
pinning collapses from 51.1% to 11.1%, all of it at the lower edge. This is below the 20% stop threshold.

**`k_a` was not a box problem.** 86.7% to 75.6%, and every pinned case moves to the *upper* bound. Double
the interval and three quarters of subjects still run to the new wall. The optimizer is drifting along a
direction the likelihood does not constrain until something stops it, and widening the box only moves
where that is. `k_e` sits in between, 88.9% to 57.8%: part box, most not.

That contrast is the cleanest available answer to reviewer point 6. The objection was that the headline
S_I gradient might merely reflect an active bound. The sweep shows the S_I bound was indeed biting and is
fixable, while the timing bounds bite in any box — which is the paper's claim, not an artifact of it.

**The weak direction is unchanged by the box**, which is the other half of the answer. Median weak
eigenvector (−0.016, 0.810, −0.587) in (log S_I, log k_e, log k_a), with inter-quartile widths of 0.017,
0.028 and 0.038 across 45 subjects. Its cosine with the analytic trade-off direction `(1/k_a, −1/k_e)` has
a median of **0.9986** at 1x and **0.9996** at 2x, at or above 0.8 for **100%** of subjects in both. The
geometry is a property of the data, not of the prior.

## The decision

It is not really about the box. It is about a definition.

`PREREG.md` evaluates the H5 clause for S_I "among interior-converged subjects", and `interior` as
implemented means *no parameter at a bound* and converged. The timing parameters are pinned by
non-identifiability in any box, so that set is **empty by construction** and the clause can never be
evaluated as written. This is a defect in the pre-registration, not in the data, and it was not
foreseeable before the first run.

**Option A (recommended) — make `interior` per-parameter, and adopt 2x as primary.** A subject is
interior *for S_I* when S_I is off its own bounds, irrespective of the timing parameters. That is the
minimal reading of what the clause was for: reviewer point 6 asked whether the S_I result reflects an
active *S_I* bound, not whether `k_a` happened to be pinned. The denominator becomes 40/45 at 2x and
22/45 at 1x. Adopting 2x as primary also stops every downstream S_I number being box-limited for half the
cohort, and [0.1299, 3.6950] sits closer to the engine's own clamp of [0.05, 3.0] than the 1x interval
does. Both boxes get reported throughout, which the H10 sweep requires anyway. Cost: an amendment, and
re-running A4 is 13 minutes.

**Option B — keep the global definition, report H5's S_I clause as unevaluable.** Costs nothing and
changes no specification, but discards part of a primary endpoint over a definitional accident and leaves
"your prior was too tight for half the cohort" visible and unaddressed.

**Option C — widen until nothing is pinned.** Not available. `k_a` stays pinned at 75.6% at 2x, and going
wider makes the interval unphysical. Worth stating explicitly so it is clear the pinning is not a
tractable nuisance.

Option A changes a definition after seeing data, so it needs to be declared as amendment 2 with this
memo as the reason, and the 1x numbers reported alongside the 2x ones everywhere. No threshold moves.

## A defect found and fixed on the way

The first 2x run used a box scaled linearly about its centre, which put the S_I lower bound at **−0.35**
— a negative insulin sensitivity — and let a fit report 0.0088 as though it were interior. Bounds are now
scaled in **log** space, which keeps positive parameters positive and leaves the 1x box exactly
unchanged, so the 1x results stand. The linearly-scaled run was never committed and was deleted rather
than left to be selected by accident.

Separately, `config_hash` includes the git commit by design, which meant a result set became unfindable
by its own summary function as soon as anything was committed. `results_io.load_all` and
`largest_result_set` now walk the directories instead of recomputing the hash, and the summary names the
set it used.

## Status

Done: 2.1 (full Fisher, both boxes). Ready: Julia 1.13.1 with StructuralIdentifiability v0.5.34 for 2.4.
Blocked on this decision: 2.2 (bound handling is exactly this question), and therefore 2.5 and gate G1.
Unaffected and runnable now: 2.3 (generic rank) and 2.4 (structural identifiability).
