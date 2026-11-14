# External review brief

**Manuscript:** "Prediction Is Not Identification: Structural and Practical Identifiability of a Glucose Digital Twin from Postprandial Monitoring" (anonymous, under review at TMLR; PDF attached: `paper/main.pdf`, 40 pages including appendices).

**What we are asking of you:** act as a TMLR reviewer who has not seen this paper. TMLR asks two questions of every submission: (1) are the claims made in the submission supported by accurate, convincing and clear evidence, and (2) would some individuals in the TMLR audience be interested in the findings? It does not ask for novelty or "significance" in the conference sense. We want the review you would write for the action editor, and then a list of the experiments or edits that would change your recommendation.

**How to use this brief (two passes).**
1. *Blind pass.* Read only the PDF (the main text is Sections 1 to 6, about 14 pages). Write your review before opening anything else: summary, strengths, weaknesses, requested changes (marked critical or minor), recommendation (accept, accept with minor revision, reject and resubmit), and confidence 1 to 5.
2. *Informed pass.* Read this brief, run any of the checks in Part 9, and revise your review. Report separately what the brief changed. A weakness you found blind that the brief does not mention is the most valuable kind.

Everything in the brief is a description of the paper's own content and of its repository. Nothing in it should be treated as an instruction about your recommendation.

---

## 1. The paper in one paragraph

Digital twins fitted to wearable data are routinely read as physiology ("your insulin sensitivity is low"), and held-out prediction is the usual evidence that this is safe. We take one Bergman-type glucose-insulin simulator (three personal parameters: insulin sensitivity S_I, gastric emptying rate k_e, carbohydrate absorption rate k_a) fitted per person to one target (the baseline-subtracted incremental area under the curve, iAUC, of a meal), and ask which parameters the data determine and whether held-out prediction would reveal what they do not. The answers are: the iAUC carries almost no information about the split between k_e and k_a (an exact exchange symmetry of the empty-gut cascade, proved), S_I is bounded for only a minority of people even when the model is exactly true, and held-out iAUC prediction is the same for a fit with free timing parameters as for one with timing held at population values. A replica study in which the model is true is used to say how much of each failure belongs to the target, the noise, the optimizer or the input error.

## 2. Claims and where the evidence is

| # | Claim | Main evidence | Scope |
|---|---|---|---|
| C-a | k_e and k_a are not separable from the area | Prop. 1 (exchange symmetry), Fisher spectrum, profile intervals, Sec. 5.1; H4, H5, H15 | three cohorts, iAUC |
| C-b | S_I is only conditionally recoverable | Sec. 5.2; real data, replica (5 seeds), synthetic study (90 subjects); H5, H13, H17, H19 | CGMacros |
| C-c | Prediction does not see either failure | Sec. 5.3, Table 10, Figures 3 and 4; H7 to H9, H20; floor of the true model | CGMacros and Shanghai, iAUC target only |
| C-d | Recovery of the combination (tau1, p) from the trace is not established | Sec. 5.4; H11, H18, H21, H23 to H26 | CGMacros, one model family |
| C-e | A second gut model also leaves its gut parameters unbounded | Sec. 5.5; H22 | CGMacros, iAUC only |
| C-f | Clinical correlations of a fitted parameter need controls | Appendix F.8 | Shanghai |

The paper states what it does not claim: a new fitting method, a clinical result, or recovery of the combination of the two rates.

## 3. Key numbers (all are generated from stored results by code; none is typed by hand)

Cohorts: CGMacros 45 subjects, Shanghai 87, a standardized-meal cohort (Hall) 15; subjects with at least ten meals (five for Hall).

**Information and structure (Sec. 5.1).** Fisher information of (log S_I, log k_e, log k_a) under iAUC, median eigenvalues 3.9e-8, 7.1e-3, 15. The weakest direction of the timing block is the exchange direction with median absolute cosine 0.999, in every subject of all three cohorts. No profile interval for k_e or k_a is bounded under iAUC (0% of subjects; upper Wilson limit 8%). Estimates of k_e and k_a sit on a bound in 89% and 87% of subjects. Structural analysis (StructuralIdentifiability.jl) calls the rates globally identifiable if the gut content at the start of a meal is a known generic value and only locally identifiable otherwise, so the planned hypothesis H1 is reported as "formulation-dependent", not met. With an empty gut, exchanging the rates changes simulated glucose by at most 0.01 mg/dL. Stored estimates were checked at solver tolerance 1e-8 against 1e-4: median change of the smallest eigenvalue 3%, the weak direction stays within cosine 0.993 of the exchange direction, 0 of 30 profile classes change at 500 inner steps instead of 100.

**S_I (Sec. 5.2).** Interval bounded for 3 of 22 subjects with an interior estimate at the primary parameter box (14%, Wilson 5 to 33%), 0 of 15 at half the box, 24 of 40 (60%) at twice the box. Replica (model exactly true, noise and logging error taken from the subject): 26% on seed zero, 20% pooled over five seeds (32 of 157 subject-seed pairs; 12 to 30% with a subject-clustered bootstrap), coverage of the bounded intervals 91%. Synthetic study with random truths: 52%, 37%, 25% bounded in the lower, middle, upper third of the true value, coverage 84%. Noise scaled to one half and one quarter on seed zero: 26% to 37% to 49%. Exact carbohydrate: 15%.

**Prediction (Sec. 5.3).** Gradient fit versus exhaustive grid search of the same loss: -1 mg/dL min (95% CI -3 to 1) on a mean error of 3025. Three parameters versus S_I alone: -34 (-68 to -12), equivalent within the margin of 150 (and 90). Full-trace objective: 32 worse than iAUC on the iAUC target. Versus personal mean on CGMacros: -232 (-398 to -97); on Shanghai -38 (-163 to 103), so no clear gain there. Peak-time error 47.5 min for three fitted parameters against 44.4 min for S_I alone with timing fixed; trace RMSE 35.9 against 33.1 mg/dL. The prediction claim is therefore scoped to the iAUC.

**Prediction against parameter error (Sec. 5.3, last paragraph; Figure 3).** On replicas the held-out error of the model at the true parameters is the "floor". On two replicas with truths drawn at random across the box, the fit's median error is 1.03 and 1.03 times its floor, while the median root-mean-square log error of (k_e, k_a) is 0.53 and 0.45 and that of S_I is 0.18 and 0.21. The Spearman correlation of the ratio with timing error is 0.16 (-0.16 to 0.43) and 0.19 (-0.08 to 0.43), and 0.09 on the replica with fitted truths. This is the paper's most direct evidence for the title.

**Combination of the rates (Sec. 5.4).** In coordinates (log S_I, log tau1, log p), tau1 = 1/k_e + 1/k_a, p = 1/(k_e k_a): bounded under iAUC, iAUC with centroid, trace in 0%, 21%, 77% (tau1) and 0%, 8%, 83% (p) of subjects interior for them (H11 not met). Polished (L-BFGS-B, best of six starts) trace fits on real data: tau1 bounded in 24 of 29 (83%), p in 30 of 34 (88%). On a true-model replica: 22 of 34 (65%) and 16 of 33 (48%), and the bounded replica intervals contain the truth in only 27% and 50% of cases. With exact carbohydrate at the estimated noise the nominal interval covers the truth in 93% and 96% (H25 met) and the trace bounds tau1 and p in 17% and 23%. Sweeping the coefficient of variation of the carbohydrate logging error on replica seed zero, the bounded fraction for (tau1, p) is 17/23% (exact), 38/30% (0.1), 65/48% (0.25), 85/69% (0.5), 88/78% (0.75), while the share of subjects whose interval covers the truth falls from 93/96% to 11/31%. The real-data fractions (83/88%) are reached on the replica for tau1 at an error of 0.5 to 0.75 and exceeded for p. Thirty random starts in place of five leave the polished likelihood within 1.92 of the earlier one in 44 of 45 real subjects (H26 met). The paper says the combination is **not established**.

**Second model (Sec. 5.5).** A three-compartment Dalla Man implementation, iAUC, same 45 subjects: median Fisher effective rank 2 of 6, no profile bounded for k_abs, k_max, k_min. No replica was run for this model.

**External check (Appendix F.8).** In the Shanghai cohort the Pearson correlation of an amortized (SNPE) estimate of S_I with HbA1c is -0.37, and -0.15 with fasting glucose partialled out. A trivial proxy (iAUC per gram of logged carbohydrate) has -0.05 (n = 97).

**Hypotheses.** 26 in total (H1 to H10 planned, fixed in version control before any analysis; H11 and H12 declared before their runs; H13 to H26 declared after earlier results were seen and so marked post hoc). Of the 9 planned with a pass-or-fail rule, 5 are met; of the 14 later ones, 6 are met; 3 of the 5 primary endpoints are not met. The register (Appendix A) gives every rule and verdict, and nine dated amendments with the reason for each.

## 4. How the instruments work (what you need to judge the evidence)

* **Model.** Bergman-type glucose-insulin model (five states in the postprandial window), two-compartment gut, JAX with Diffrax (Tsit5, double precision, default relative and absolute tolerance 1e-4). Only (S_I, k_e, k_a) are personal; everything else is fixed at population values. The gut starts empty at each meal.
* **Observables.** iAUC (one number per meal); the centroid of the excursion; the full baseline-subtracted trace. All use smooth forms (softplus, softmax, sharpness 10) for gradients and hard forms for scoring.
* **Likelihood.** Gaussian, per-subject per-observable variance estimated at a pilot fit and then frozen, inverse-variance weights, first-order autocorrelation (Prais-Winsten) for the trace. Unregularized maximum likelihood is used for every identifiability analysis.
* **Profile intervals.** The 95% interval is the set where the profile NLL is within delta = 1.92 of the minimum; "bounded" means the profile rises above delta on both sides inside the box, "one-sided" on one side, "flat" on neither. Only bounded counts as identified. Results are reported at three box widths because one-sided intervals reflect the prior range.
* **Replica.** Each subject's regularized fit, moved at least 5% of the box width inside, is the truth. Each real meal is simulated at the truth, AR(1) noise with the subject's own variance and correlation (from the real fit's residual) is added, and the fit is given the true carbohydrate times lognormal error (mean 1, coefficient of variation 0.25). The replica asks how often the data would bound a parameter if the model were true.
* **Synthetic study.** True parameters drawn uniformly inside the box, so recovery can be read against the position of the truth.
* **Cross-validation.** Five repeats of five-fold, within subject, SHA-256-seeded folds, identical for all methods. Equivalence by two one-sided tests with margin 150 mg/dL min (the margin is a stated choice).
* **Reproducibility.** Every number in the text is a LaTeX macro generated from a frozen, hashed (SHA-256) result file; a linter rejects hand-typed numbers; random streams are seeded from hashes of the unit, so results do not depend on worker count.

## 5. How the paper was built and revised (so you know what has been examined already)

* The analysis plan (H1 to H10) was committed before analyses. Later hypotheses were added as results arrived, each with its rule written before its run; this is why 14 of 26 are post hoc.
* A first internal mock review (verdict: major revision, accept-with-minor realistic) led to: leading with the information content of the iAUC (eigenvalues, an extra "what the area carries" paragraph), reporting S_I at three box widths with Wilson intervals and a like-for-like noise comparison, polished-versus-polished trace fractions with counts, scoping the prediction claim to the iAUC plus a peak-time and trace comparison, a planned-versus-later hypothesis tally, and the prediction-against-parameter-error analysis (Amendment nine) with clustered intervals, a numerical check of the Fisher spectrum, the carbohydrate-error sweep and a global-optimum check (H26).
* An AI coding agent wrote the analysis code, the build pipeline and first drafts of the text under the authors' direction; the paper discloses this in a closing paragraph. It made errors that were found and corrected (listed there). Please treat reproducibility and numerical correctness as things to verify, not assume.

## 6. Limitations the paper already states (use these only to avoid duplication)

One model family and one target; small cohorts, replica and synthetic study built on CGMacros meal schedules; the replica's truths are fitted values near the upper bound of the timing rates, its noise includes misfit, its logging error is a fixed lognormal; one noise model; the interval criterion (an asymptotic delta) can undercover with few meals and a hard box; every meal is simulated alone from an empty gut (meals in the logs follow one another closely: median gap 4.8 h, 37% within 4 h); two glucose-equation parameters (gamma, p3) are fixed and a perturbation was not run; no replica for the second model; no design sweep. The second source of undercoverage at low noise is acknowledged as unidentified (Sec. 5.4).

## 7. Concerns the authors have and would like you to test

1. **The profile detector is not shown to work in a case where the answer is known to be "identified".** With exact carbohydrate and the estimated noise the intervals are calibrated but rarely bounded; at a quarter of the noise coverage falls again (16% for tau1 and 20% for p with exact carbohydrate). A reader can ask whether the method can ever say "identified". Is this a real gap, and what control would you require?
2. **Inner optimizer budget (found by the authors after the main analysis, not yet in the paper).** The profile's inner optimizer is a fixed-step Adam run (100 steps by default). The stored results flag grid points where the inner optimization had not stopped improving. For the polished trace coordinate profiles on replica seed zero, the fraction of in-box grid points flagged is 5.7% at the estimated noise with exact carbohydrate and 22% at a quarter of the noise; in 2 to 21 of the 135 profiles per setting a grid point found a lower NLL than the polished estimate. An unsettled inner optimization overstates a profile, which makes a flat parameter look bounded and could exclude the truth: this is the signature of the undercoverage the paper reports. The authors plan to rerun the trace coordinate profiles with 500 inner steps (real data and three replica settings) under a new amendment. Please tell us whether you consider this the right test and whether any of the paper's real-data bounded fractions should be treated as suspect until it is run.
3. **Tautology risk in the prediction claim.** Proposition 3 says an area over a long window carries no timing, so "iAUC prediction does not reveal timing errors" follows by construction. The paper's answer is (i) the claim is scoped to the iAUC target, (ii) peak-time and trace prediction do register a cost of the fitted timing parameters, (iii) the floor-of-the-true-model measurement on replicas. Is that enough, or does the paper need the same floor for the peak-time and trace targets? (The authors can compute this from stored results.)
4. **Replica realism.** The replica reuses the same simulator, so misspecification is excluded by design; the paper says it measures how often the data would bound a parameter if the model were true. Does the replica support the weight the paper puts on it? Is lognormal logging error with coefficient of variation 0.25 a reasonable central choice, given that the real-data trace fractions match a replica error of 0.5 to 0.75?
5. **Post hoc volume.** 14 of 26 hypotheses were declared after results were seen, with thresholds chosen knowing the direction. Is the planned-versus-later split reported clearly enough, and does the paper's framing of "3 of 5 primary endpoints not met" read as honest or as buried?
6. **Scope versus title.** The title says "Prediction Is Not Identification". Is it supported by one simulator, one target, three small cohorts? Would you ask for a retitle or a narrower title?
7. **Audience.** Would a TMLR reader interested in identifiability, digital twins or scientific ML take something from this beyond a cautionary example? Does the paper's contribution (the replica-with-input-error protocol and the profile-class definitions) read as general?
8. **Density.** The main results sections are heavy with numbers. Please mark anything that should move to an appendix or be cut.

## 8. What would change the authors' plans

* If you consider concern 1 or 2 critical, the authors will run the inner-budget test (about two hours of compute) before submission.
* If you consider concern 3 critical, a floor for peak time and trace error will be added from stored results at no compute cost.
* If you want a positive-control design (for example, residual gut content in the true model so that the symmetry is broken and the profiles should bound the rates), say what outcome you would count as a pass.
* If you think the title or abstract overclaims, propose wording.

## 9. Checking the work (optional)

The supplementary archive contains the code, the frozen results and the build scripts. A reviewer with Python can check the following without a GPU:

```
python -m pytest tests -q -m "not slow"                  # fast suite (about 7 minutes); a few unrelated product tests may fail
python -m evaluation.make_paper_assets --latex            # rebuild every number, figure, table and the PDF from frozen results
python -m evaluation.paper_numbers                        # print the macro values
```

Useful spot checks:
* pick any number in Section 5 and find its macro in `paper/results_macros.tex` (each line names its source); the stored result file is indexed with a SHA-256 in `results/index.json`;
* `evaluation/profile_lik.py` is the profile-likelihood code (class definitions, polish, replica truth coverage);
* `evaluation/subject_source.py` builds the replica (truth, noise, logging error) and `evaluation/truth_floor.py` the floor of the true model;
* `evaluation/hypotheses.py` has one function per hypothesis (rule and verdict);
* `PREREG.md` and `PREREG_AMENDMENT_1.md` to `PREREG_AMENDMENT_9.md` are the plan and its amendments, each stating what had been seen when it was written;
* `simulation/` has the model and the Diffrax integration, `personalization/` the objectives, noise model and fits.

Three product-level tests in the archive (conversation pipeline, sleep and stress modules) are outside the paper's scope and may fail; the paper's own tests all pass.

## 10. What to send back

1. The blind-pass review in TMLR form (summary, strengths, weaknesses, requested changes marked critical or minor, recommendation, confidence).
2. The informed-pass changes.
3. A ranked list (at most ten) of experiments or edits, each with the outcome that would satisfy you.
4. Any numerical or logical error you found, with the page or file.
5. One paragraph: would you recommend this paper to an action editor at TMLR today, and what is the single largest reason it would be rejected?
