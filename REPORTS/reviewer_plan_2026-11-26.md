# Reviewer's second-round plan (received 26 November 2026, verbatim)

I only have your update file, not the new PDF (the attached PDF is the old draft). I'm taking the status as reported, and the final audit has to check it against the build.
What decides the odds
On TMLR's criterion (are the claims supported by the evidence), the draft's main risks are:

1. The headline iAUC claims run on a protocol you have said you won't defend.
2. The appendix reads as a lab log.
3. The prose is AI-drafted and the verification statement is unbacked.
4. No numbers can be trusted until the run records are complete.

Missing experiments matter less than these, if the limitations are stated honestly. So the order is: trust the numbers, then unify the protocol, then add only the experiments that change claims, then write.
Stage 0: Make the numbers trustworthy

* Completeness manifest: list the expected units per sweep and fail loudly on any gap. Lock files skipped units silently before, so a summary should never be produced from a partial sweep.
* Run records: add the two new macro files to the frozen index and commit the untracked files. Tag one commit that everything reports from.
* Test failures: run the suite on a clean checkout of that tag. Either show the 4 failures predate this work, or fix them, or drop the unused code from the package.
* Hand-typed numbers: review the 22-file list. Nothing in the abstract, tables or results text may be hand-typed.

Exit: one tagged commit, an index covering every reported number, and a documented test status.
Stage 1: Close pending runs
Merge the fitted-truth CV 0.75 run, the fitted seeds 1 and 2, Shanghai, and the constant perturbations, with the manifest check.
Before looking at results, write the rule for each outcome:

* If the fitted truths at CV 0.75 move toward slower absorption, the two-direction account stands.
* If they don't, drop the mechanism paragraph and keep only the observation.
* If Shanghai replicates, say so in one sentence.
* If it doesn't, report it as a limit on generality.

Stage 2: One protocol everywhere

* iAUC profiles: rerun all three box widths, the S_I replica and the perturbations with the converged solver and 29-point grid.
   * Converging the inner solve can only lower profile values, so bounded counts cannot rise at a fixed estimate. Your 0 of N for ke and ka is therefore unlikely to flip.
   * The interior sets and S_I counts can move because the estimates and grids change, so this is still a rerun, not a relabel.
   * Rules to write first: if some ke or ka interval bounds at 4x, change the abstract to "at most X of N" and keep the claim about the primary box. If S_I counts move, the claim is only "box-dependent", which survives either way.
* Centroid and residual-gut: rerun both, since they carry the "information limit first" argument. If a rerun is not feasible, cut the argument instead of labeling it.
* Dalla Man intervals: rerun. The 26 of 44 pseudo-true coverage is understated. Zero coverage of the generator value should hold (bias 26.7 min against width about 10), but that has to be shown.
* Exact-domain results: delete.

Exit: a single protocol paragraph in Section 3 and no per-result protocol labels.
Stage 3: Only the experiments that change claims
Do these:

1. Regression-calibration or SIMEX on the random-truth replica. This is your remedy and a direct test of the attenuation story in one run. If calibrated inputs remove the shift and restore coverage, the mechanism is supported. If not, say it is untested.
2. Isolated-meal restriction (gaps over 4 to 6 h), compute only.
3. Subject-specific basal glucose, likely a small code change since the Dalla Man code already does it. It matters most for Shanghai.
4. Sensor-lag Bergman variant, only if the code is small. It would explain part of the remaining 15.5 min bias.

State as limitations: Dalla Man-on-Dalla-Man, a second generator distribution, per-meal ke, and posterior contraction. The 4x box row already shows ke and ka unbounded without the wall effect.
Stage 4: Claim ledger, then restructure

* Build one table: each claim in the abstract, intro and conclusion, its evidence table, its protocol and its status. Anything unsupported gets cut or softened. Freeze new results after this.
* Table 2: split prediction hypotheses H7 to H9 from the exploratory ones. Say in words that of five primary endpoints three were not met, one was met and one was inconclusive. Replace "formulation-dependent" with "the rule fires under the plan's formulation but not under the simulator's". Add that preregistration timing is author-reported and cannot be checked under blinding.
* Interval score: add a column to the 5.4 table, since the values are stored (otherwise delete the promise).
* Remove "polished" and fix the Figure 3 caption.
* Appendix cuts:
   * gradient diagnostic (F.7)
   * the HbA1c/SNPE check (one sentence to keep)
   * RF and SNPE baselines
   * gamma scaling
   * heteroskedastic variants
   * recalibrated-cutoff section
   * the register and all hypothesis codes (move to the repo)
* Appendix keep:
   * solver and grid checks (including Table 15)
   * Fisher tolerance
   * Shanghai table
   * window sweep, short
   * area-deficit note
* Unexplained results:
   * The 0.99 ratio is probably legitimate. The fit can absorb logging error through S_I, so true parameters under noisy inputs are a reference, not a lower bound. Rename "floor" to "true-parameter reference" and give a bootstrap CI.
   * Explain the S_I bias of about -0.2 only if boundary pinning accounts for it (19 of 45 pinned earlier). Otherwise list it as unexplained.
   * The direction of the shift gets the same treatment, per Stage 3 item 1.

Stage 5: Authors' pass

* Write the abstract, intro, related work, discussion, conclusion and 5.4 by hand from the ledger. Don't edit the AI draft.
* Restore the verification sentence in the AI-use statement only after the number check (index) and reference check (primary sources) are done. Otherwise state what was done.
* Reference pointers: cite them only for what the metadata supports, or read them. Add Hovorka-type identifiability.

Stage 6: Submission gate
All must hold:

* zero `[pending]`
* no undefined references
* linter clean in the main text
* full number audit against the index
* anonymization check, including PDF metadata and repo history
* one outside read of the abstract and Section 5 by someone off the project

Your seven questions

1. Recommendation: major revision, close to minor once Stages 0 to 2 and the rewrite are done. The blockers are the mixed protocol, the hand rewrite and the record keeping.
2. Phases A, B, D plus limitations: nearly enough. The minimum from Phase C is the errors-in-variables remedy as the mechanism test. Basal glucose and isolated meals are cheap extras.
3. Old-protocol results: iAUC, centroid, residual-gut and Dalla Man must be rerun. Exact-domain is deleted. A label alone is not enough for any claim in the abstract.
4. Random vs fitted truths: this resolves the wall confound, not the mechanism. The known-slope simulation is not required if the calibration run and the text say the mechanism is untested.
5. Positive result: fair if labeled a same-model best case, not real-data calibration. Keep the width-aware companions beside it, since fitted truths cover 44 with 6 bounded.
6. τ1 for ordering: supported only conditionally. The claim is that it ranked subjects in simulation under assumed logging error, with ρ falling to about 0.7 at CV 0.75. Show ρ as a function of CV and avoid recommending it for real data.
7. Appendix cuts: listed in Stage 4. Nothing on the earlier cut list needs rescuing, apart from the Table 15 grid finding, which should stay.
