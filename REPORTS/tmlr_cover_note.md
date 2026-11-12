# TMLR submission: cover note and claims-and-evidence statement (draft; the author edits and confirms)

Working title: Prediction Is Not Identification: Structural and Practical Identifiability of a Glucose Digital Twin
from Postprandial Monitoring.

## What the paper claims

1. The gastric-emptying and carbohydrate-absorption rates of a two-compartment gut cannot be separated from the
   incremental area under the CGM curve. Their exchange is an exact symmetry of the empty-gut model (proved), the weakest
   Fisher direction lies along it in every subject of three cohorts, and no profile interval for either rate is bounded
   under iAUC at any parameter range tried. This is the glucose counterpart of the flip-flop ambiguity of oral
   pharmacokinetics.
2. Insulin sensitivity is only conditionally recoverable: bounded in 14% of subjects whose estimate is interior at the
   primary parameter range, 20% when the model is exactly true, and about half at a quarter of the estimated noise; the
   weakness is not caused by carbohydrate logging error and not changed by heteroskedastic noise.
3. Held-out prediction does not see either problem: gradient fitting and grid search, and three parameters and one, are
   equivalent within a pre-set margin in two cohorts; a full-trace objective does not lower held-out iAUC error.
4. Whether the symmetric combination of the rates is recoverable from the full trace is not established: the
   real-data result fails a replica control, and the controls localize why (a carbohydrate input error makes trace
   intervals more often bounded and less often right).
5. In a second gut model (Dalla Man) the gut parameters are not recovered under iAUC either.

What the paper does not claim: a new fitting method, a clinical result, or recovery of the combination.

## Evidence for each claim

Section 4 and Appendix D (proofs); Sections 5.1 to 5.5 and Appendix F (analysis trail); the register and scorecard
(Appendix A) list 25 hypotheses with their rules, origin and verdicts (10 met, 12 not met among 22 graded). The plan was
fixed in version control before the analyses (H1 to H10); H11 and H12 were declared before their analyses and H13
to H25 after earlier results were seen, with thresholds fixed before each run, and are labelled post hoc throughout.
No threshold was changed after any run. Every number in the text is generated from frozen result files by code.

## Reproducibility

The supplementary archive contains the code, the stored results with provenance and a SHA-256 index, the analysis
plan and its eight amendments, and the pipeline that rebuilds every number, figure and table. The three cohorts are
public; the archive contains only parsed, checksummed copies.

## Use of AI assistance (to be confirmed by the author before submission)

The analysis code, the result-generation pipeline, the compute job scripts and first drafts of the manuscript text
were produced by a large-language-model coding agent (Claude Code, Anthropic) working in the authors' repository on
their instructions. The authors posed the questions, decided which analyses to run, approved the analysis plan and each
amendment, and take responsibility for the content. The paper states this in its Discussion.

## Items for the author to fill in

* Suggested action editors (Editorial Board page; statistics, probabilistic modeling, ML for health), excluding
  conflicts with the author's OpenReview history.
* Confirm that no other venue holds a live submission of this work (an earlier conference version was reviewed and is
  not under review).
* Preprint: post at submission, with the author's name; the submission itself must never link to the named version.
