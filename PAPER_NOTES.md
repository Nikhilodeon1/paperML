# Horizon SBI paper — results ledger

Every number below was produced on **paperML/** (Python 3.14, seeds locked to 42 in
`paper_config.py`). Regenerate everything with `python -m evaluation.reproduce_all`.

## PIVOT — NeurIPS 2027 "Differentiable physiological simulator + gradient-based inference" (2026-07-20)
The paper's method is now **gradient-based inference through a differentiable JAX/diffrax engine**
(`simulation/jax_engine.py`), which has NO prior and therefore no OOD/prior-mismatch problem. SNPE/
RF/grid stay as baselines. Result — gradient inference is best-of-both:

| metric (n=45, CGMacros) | grid | RF | SNPE | **gradient** |
|---|---|---|---|---|
| Spearman(Si,HbA1c) | −0.694 | −0.707 | −0.733 | **−0.729** [−0.85,−0.54] |
| Pearson(Si,HbA1c) | −0.686 | −0.663 | −0.412 | **−0.664** [−0.78,−0.54] |
| held-out iAUC MAE (5-fold, ALIGNED) | 3007 | 3150 | 3404 | **3001** |
| % beats personal-mean | 59% | 51% | 44% | **61%** |
| PAIRED vs gradient (gradient wins) | 49% | 69% | 76% | — |
| identifiability: Si vs gastric/carb | — | — | posterior-width | **∇-norm ~60× (Fig 7)** |

**HONEST held-out result (aligned 5-fold, `evaluation/unified_kfold.py`): gradient TIES grid on
prediction (3001 vs 3007 MAE; 49% paired — a coin flip) and BEATS the amortized methods (RF 69%,
SNPE 76% paired).** The earlier "gradient beats grid (2902/64%)" was a chronological-split artifact
that did NOT survive protocol alignment — DROP that claim. This is **Scenario B, not A.** But the
tie is COHERENT with the thesis: grid fits only Si (1-D search), gradient fits Si+gastric+carb and
ties *because gastric/carb are non-identifiable* (extra params don't help) — the same story the
gradient diagnostic shows. Contribution = differentiable engine + gradient identifiability
diagnostic + the only method recovering BOTH Pearson and Spearman; prediction MATCHES the classical
baseline while beating amortized SBI.

**ADAPTIVE PARAM SELECTION (venue-deciding experiment, `unified_kfold --methods adaptive,...`) →
OUTCOME B.** Use the gradient diagnostic to pick which params to fit per subject (probe 50 steps,
keep ≥10% of max gradient-norm, freeze rest), then fit only those. Aligned 5-fold n=45: adaptive
MAE 3053, gradient 3045, grid 3065 — all tie. PAIRED adaptive-vs-grid **51%** (does NOT beat grid);
adaptive-vs-gradient 27% (freezing non-identifiable params slightly HURT — regularization already
handled overfitting). Adaptive kept Si 100%, gastric 44%, carb 51%. CONCLUSION: **the Bergman model
is the prediction bottleneck, not the inference method** — no method beats a 1-D Si grid because Si
is the only identifiable parameter, and even per-subject identifiability-based selection can't break
the tie. Clean, stress-tested Outcome B → **AAAI 2026 as-is; Dalla Man as the NeurIPS follow-up.**

**Robustness within CGMacros (`evaluation/gradient_robustness.py`):** Si-vs-HbA1c holds within
subgroups — diabetic Spearman −0.770 (n=14, strongest, where Bergman is most strained), non-diabetic
−0.577 (n=31); group-mean Si normal 1.14 > prediabetic 0.75 > diabetic 0.43. Robust to population
shift without needing OhioT1DM (which would require porting the exogenous-insulin bolus to the JAX
engine). Engine validated: numpy↔jax glucose 1.5% max err, `jax.grad` exact vs finite-diff. See
`SMOOTHING_NOTES.md`.

## PEAK-TIME TARGET — observable-dependent identifiability (2026-07-25, `evaluation/peaktime_kfold.py`)

Same aligned 5-fold / four methods / paired comparison as `unified_kfold`, but the target is
minutes-to-glucose-peak (a TIMING observable governed by gastric/carb) instead of iAUC. Gradient
fits Si+gastric+carb to peak time (`gradient_fit_peaktime.py`); grid fits Si to peak time; RF/SNPE
use native iAUC-Si. RESULT (n=45): held-out peak-time MAE gradient **40.7±1.5** < grid 42.1 < snpe
43.3 < rf 45.6 (persist 55.6). **PAIRED gradient-vs-grid 56%** (vs 49% on iAUC), vs-rf 71%, vs-snpe
60%. RANKING CHANGES: gradient ties grid on iAUC (49%) but EDGES it on peak-time (56%, lowest MAE) —
the timing params non-identifiable/useless for iAUC become MODESTLY useful when the target rewards
timing. CAVEAT: peak-time is a NOISY observable (MAE ~40 min, near resolution; fold-std overlaps), so
the edge is directional/modest, not a decisive win. Suggests the prediction tie is observable-
specific and a multi-observable target could help. Commits 6097cef+.

## THEORETICAL GROUNDING — Fisher information (2026-07-23, `evaluation/fisher_information.py`)

Connects the gradient-norm diagnostic to the diagonal Fisher information FI(θ_i) = (1/σ²)·
E_meal[(∂iAUC/∂θ_i)²] (iAUC observable, reverse-mode grad — diffrax's custom_vjp forbids the
brief's jacfwd). **HONEST RESULT: the clean per-subject correlation grounding FAILED for the key
param** — r(FI, grad_norm) across 45 subjects = Si **+0.03**, gastric +0.57, carb +0.60. Cause: the
trajectory-averaged gradient norm decouples from the fixed-point Fisher for the parameter (Si) that
moves most during optimization. **The grounding that WORKS is the RANKING agreement:** both
quantities rank Si as the sole identifiable parameter — Fisher Si/gastric **16.2×**, Si/carb 11.6×;
gradient-norm Si/gastric 5.6×, Si/carb 4.4×. PAPER SENTENCE must be about ranking, NOT "r=0.X
correlation": e.g. "the diagonal Fisher information independently ranks insulin sensitivity as the
sole identifiable parameter (12–16× the Fisher of the gut/timing parameters), matching the
gradient-magnitude diagnostic and grounding it in the Fisher-information identifiability criterion."

**Dalla Man identifiability ratio (parity, Fig 8):** Vmx/kabs **27.6×**, Vmx/kmax 85× (Vmx 2.25e6 >>
f 4.4e5 > kmin 1.4e5 > kabs 8.1e4 > Td 6.4e4 > kmax 2.7e4). Comparable to Bergman Si/gut → the
identifiability structure is model-INDEPENDENT (replicates in Bergman AND Dalla Man).

## DALLA MAN MODEL UPGRADE (2026-07-22) — does a more expressive model break the tie? NO.

`simulation/dalla_man.py` (11-state, 3-compartment gut, subcutaneous CGM compartment; validation
gates in `dalla_man_validation.py` PASS: OGTT peak 170.3@50min, CGM lag 10min, steady-state drift
0.08). Gradient fit `gradient_fit_dalla_man.py`; comparison `dalla_man_comparison.py`.

**RESULT (aligned 5-fold, n=45): Dalla Man does NOT beat grid — it is WORSE.** held-out iAUC MAE:
bergman 3051, grid 3068, **dalla_man 3288**; paired dalla-vs-grid **40%**, dalla-vs-bergman 38%.
=> the CGM observation (baseline-subtracted iAUC) is the bottleneck, NOT Bergman's expressiveness.

**IDENTIFIABILITY REPLICATES in the more expressive model** (strengthens the central claim to a
model-CLASS property, not a Bergman artifact): normalized |grad| Vmx 3.37e6 (dominant) >> kmin 1.8e6
> kabs/f ~9e5 > Td 1.6e5 > kmax 1.2e5 (~29x spread). Insulin-sensitivity param identifiable;
gut/timing/lag params not.

**CORRECTION forced by this result:** the error decomposition's "79% Bergman model-expressiveness
gap" was mis-interpreted — upgrading to a more expressive model recovered NONE of it, so that bucket
is "NOT recoverable from baseline-subtracted iAUC", not "fixable with a better model". Arithmetic
stands; the model-upgrade framing was too optimistic. This experiment caught it.

CAVEATS: (1) reduced+CALIBRATED Dalla Man-style model (kmin/beta_sec/kcr calibrated to the OGTT;
counter-regulation added after omitting glucagon undershot basal), not verbatim Table I. (2)
meal-composition confound: Bergman fit gets fat/fibre blunting, Dalla Man got carbs-only — a PARITY
run (Dalla Man carbs blunted identically) RESOLVED this: MAE gap shrinks 3288->3019 (part of
the raw gap WAS the confound), but all three still TIE (bergman 2996 / grid 3004 / dalla_man 3019)
and Dalla Man still does not win (paired 40% vs grid, 24% vs bergman). Conclusion airtight: even the
FDA gold-standard model with 6 params ties a 1-D Si grid -- the CGM observation is the ceiling.
(3, still true) 6 fitted params vs 3 → overfits non-identifiable directions (consistent with the identifiability
finding). Do NOT write the Dalla Man claim until the parity number is in.

## PART 1 WEAKNESS FIXES (2026-07-20) — external validation + decompositions

**Second external cohort — Hall 2018 (`evaluation/hall_loader.py` + `hall_validation.py`).**
Standardized-meal CGM (Bar/CF/PB, carbs from USDA), n=24 with HbA1c (mostly non-diabetic, range
4.9-6.4). RESULT: gradient Si vs HbA1c Spearman **-0.351** [-0.67,+0.06] (correct sign, moderate, but
CI crosses 0 = NOT significant); Pearson -0.506. Si vs **SSPG** (the direct insulin-resistance
measure) Spearman **-0.527** but n=17 < 20 -> NOT citable per protocol. Identifiability replicates on
a 3rd cohort: Si/gastric 4.5x (weaker than CGMacros/Shanghai). READING: directionally supportive but
UNDERPOWERED — the narrow HbA1c range (healthy cohort) limits power for HbA1c; SSPG varies widely and
gives the stronger signal but lacks n. Standardized meals are a STRENGTH (controlled carbs, only Si
varies -> no carb-estimation confound, unlike Shanghai). Overall pattern fits the cohort-dependence
thesis: CGMacros (wide Si range) -0.73 significant; Shanghai (T2D, fasting-dominated) -0.14 null;
Hall (non-diabetic, narrow HbA1c range) -0.35 directional-but-underpowered.

**Weakness 1 — ShanghaiT2DM external validation (n=97 T2D, 2671 meals; frozen CGMacros artifacts,
pure transfer).** `evaluation/shanghai_loader.py` + `cross_dataset_validation.py`. Si vs HbA1c:
- gradient Spearman **-0.14** [-0.34,+0.05] (weak, CI crosses 0) — does NOT replicate
- grid -0.04, rf -0.09 (null)
- snpe -0.44 [-0.58,-0.28] — replicates, BUT **TAUTOLOGICAL** (see below)
- identifiability REPLICATES: Si grad-norm 10.7x gastric / 8.3x carb (vs ~20-60x CGMacros)

**TAUTOLOGY CHECK (`evaluation/tautology_check.py`) — SNPE's Shanghai result is NOT real
generalization.** partial r(SNPE-Si, HbA1c | fasting glucose) = **-0.15** (raw was -0.37 → collapses
after controlling fasting glucose; r(fasting,HbA1c)=+0.49). Leakage r(SNPE baseline-feature, SNPE-Si)
= **-0.54** — SNPE routes fasting glucose into its Si. So DO NOT cite SNPE's Shanghai -0.44 as
generalization. Gradient partial r = -0.11 ≈ raw -0.13 → gradient is CLEAN (fits iAUC = baseline-
subtracted, no fasting contamination), which makes its CGMacros -0.73 more credible.

HONEST READING: the paper's CORE claim (identifiability structure) replicates across two cohorts;
the SECONDARY claim (gradient clinical Si-recovery) does NOT externally. CAUSE — CONFIRMED, and NOT
what we first guessed:
- Carb-estimation error is REFUTED (`carb_sensitivity.py`): injecting ±0/15/30/45% carb noise into
  CGMacros leaves gradient Spearman flat at -0.72 → -0.71. Per-subject Si averages out per-meal carb
  error. (Also not range restriction: CGMacros diabetic subgroup gradient was -0.77.)
- The REAL cause: gradient fits ONLY baseline-subtracted iAUC. On Shanghai the HbA1c signal lives in
  the ABSOLUTE glucose level, not the incremental response — Spearman(HbA1c) = mean-glucose +0.575,
  fasting-baseline +0.55, peak +0.57, but iAUC only +0.24. iAUC subtracts away the chronic-
  hyperglycemia signal; SNPE uses absolute-baseline features so it keeps it (SNPE meanSi 0.40 correct
  for T2D; gradient/grid meanSi 0.88/1.03 too high). FIXABLE: fit the ABSOLUTE glucose curve (not
  iAUC) — a concrete, testable extension that should restore the Shanghai correlation.

**Weakness 2 — error decomposition (`model_expressiveness_analysis.py`, Table 4):** held-out iAUC MAE
3001 = CGM-noise 632 (21%) + **Bergman model gap 2360 (79%)** + inference residual **9 (0%)**. The
tie is model-limited: inference is ~perfect, model is the bottleneck. (Shanghai carb-estimate iAUC
floor ±25% = 576, ~= the CGM-noise floor — supports the carb attribution.)

**Weakness 3 — inference-cost/INO (`inference_cost_analysis.py`, Table 3):** grid 4324ms, RF 144ms,
SNPE 1587ms, gradient ~5600ms, INO(cited) 230ms. Argument: NOT faster than INO; appropriate for a
different regime (sparse users, zero training budget, no prior/surrogate, gives identifiability).

## Paper framing (reframed 2026-07-19, per strategy review)
The contribution is **methodological, not "SNPE wins"**. SNPE does NOT beat grid/RF on absolute
accuracy or held-out prediction — and that is a *finding*, not a failure. Restated claim:

> We compare grid search, amortized point estimation (RF), and calibrated posterior estimation
> (SNPE) for physiological parameter inference from real meal-glucose data, and show: (1) insulin
> sensitivity is structurally identifiable from postprandial CGM while gastric emptying and carb
> absorption are not; (2) only calibrated posteriors reveal this structure; (3) all three methods
> recover clinical insulin-resistance status comparably by **rank** correlation; (4) amortized
> point estimation achieves the best held-out predictive utility; and (5) SNPE underperforms on
> absolute recovery due to a measurable **simulator-reality gap**, which we diagnose *before*
> inference. Identifiability characterization via posterior width should precede parameter
> interpretation in physiological digital twins.

Writing rules: **lead with Spearman** (rank), not Pearson (where SNPE loses and grid/RF's edge is
partly a shrinkage/ceiling artifact). Lead the paper with identifiability + the SBC calibration
gate. Present grid/RF as the accuracy baselines. Cite robust-SBI-under-misspecification (NeurIPS
2024) for the OOD framing. Venue: reframed paper fits **ICLR** (methodological) over AAAI; but
ICLR 2026 + AAAI 2026 deadlines have passed (today 2026-07-19) — realistic targets are the next
cycle (ICLR 2027 / NeurIPS 2026 / AAAI 2027) with npj Digital Medicine as the clinical parallel.
Confirm exact deadlines.

## Estimators
- **grid / SMC** — classical per-subject 1-D iAUC grid fit (`cgmacros.fit_subject_si`). This is the
  established r≈-0.59 method. (The brief called it "SMC / particle_fit"; `particle_fit` needs
  full-day CGM, which CGMacros' per-meal windows don't provide — so the reproducible classical
  baseline is the grid fit.)
- **RF** — amortized RandomForest point estimator (`personalization/npe.py`).
- **SNPE** — amortized calibrated SNPE-C posterior (`personalization/snpe_trainer.py` +
  `snpe_infer.py`). 50k training sims, artifact at `evaluation/artifacts/snpe_posterior.pkl`.

All three use the SAME simulator, summary stats (iAUC, peak, peak-time, baseline, early-slope),
and 10-dim feature vector (+ carbs + weight/height/age/sex). Prior standardized to Si∈[0.30,1.60],
gastric∈[0.015,0.040], carb∈[0.012,0.032].

## Table 1 — clinical recovery (in-sample, n=45 CGMacros subjects)
`evaluation/clinical_recovery.py`, `snpe_vs_smc_vs_rf.py`

| metric | grid/SMC | RF | SNPE |
|---|---|---|---|
| r(Si, HbA1c) Pearson | **-0.686** [-0.80,-0.56] | -0.663 [-0.79,-0.49] | -0.412 [-0.56,-0.29] |
| Spearman(Si, HbA1c) | -0.694 | -0.707 | **-0.733** |
| r(Si, HOMA-IR) | -0.429 | -0.427 | **-0.482** |
| group mean Si (normal/pre/diab) | 1.29/0.81/0.44 | 1.05/0.82/0.65 | 0.61/0.39/0.31 |
| one-way ANOVA p | 3.1e-5 | 1.8e-4 | 3.3e-3 |
| inference / subject | ~3980 ms | ~140 ms | ~1490 ms |

**Honest reading:** all three recover clinical status (Spearman -0.69…-0.73, monotonic groups,
significant ANOVA). SNPE's **Pearson is lower** than grid/RF but its **Spearman is the strongest**.
Grid/RF's higher Pearson is partly the shrinkage/ceiling artifact (grid rails to 1.6 for healthy;
RF shrinks extremes) which spreads the linear range; SNPE reports honest, compressed absolute Si
(real CGM is out-of-distribution vs the synthetic training set), so it recovers a strong *rank*
relationship with attenuated linear r. **Do NOT claim SNPE beats the others on correlation** — its
contribution is calibration + identifiability, not a higher r. Lead the SNPE story with
Spearman + group separation.

## Table 2 — held-out k-fold predictive utility
`evaluation/snpe_kfold.py` (5-fold, iAUC MAE, lower = better)

| method | iAUC MAE | gap vs personal-mean | % beats personal-mean |
|---|---|---|---|
| **grid/SMC** | **2998±188** | **+233±113** | **60%** |
| RF | 3134±239 | +96±144 | 56% |
| SNPE | 3416±231 | -185±171 | 41% |
| personal-mean | 3231±158 | — | — |
| persistence | 4380±242 | — | — |
| population | 3335±223 | — | — |

**Honest reading:** the **grid fit carries the predictive-utility claim** (beats a personal
constant, +233 MAE, 60% of subjects — reproduces the established +256±72 / 61% k-fold result). RF
is close. **SNPE does NOT beat baselines on prediction** (its low/compressed absolute Si biases the
iAUC predictions) — per the brief's own contingency, RF/grid is the utility result and SNPE's value
is elsewhere. This is a clean division of labour, not a failure.

## Simulator-reality gap — WHY SNPE loses utility (new Figure 6)
`evaluation/ood_analysis.py`, `figures.py::fig_ood_gap`

Distribution of each summary statistic, SNPE training simulations vs 1640 real CGMacros meals.
**98% of real meals are out-of-distribution on ≥1 statistic.** Standardized median shifts:

| summary stat | sim median | real median | shift (z) | % real outside sim central-90% |
|---|---|---|---|---|
| baseline (fasting) | 90 | 119 | **+7.1** | 91% |
| peak_time (min) | 50 | 80 | **+2.0** | 57% |
| iAUC | 4650 | 3199 | −0.6 | 50% |
| early_slope | 1.2 | 0.5 | −1.2 | 49% |
| peak | 159 | 169 | +0.4 | 32% |

**Reading:** the engine simulates a healthy fasting baseline (~90) and an early (~50 min) glucose
peak; the real cohort (many pre/diabetic) has elevated fasting glucose (~119) and later peaks
(~80 min). The amortized estimator is therefore queried far off its training support, which
compresses its absolute Si — the mechanistic cause of the Table 2 utility gap. This is a
*pre-inference* diagnostic: you can detect that amortized SBI will underperform before running it.
Do NOT patch this by retraining on real-matched stats (overfits to 45 subjects) — report it.

## Table 3 — identifiability (the novel finding)
`evaluation/identifiability_analysis.py`

Posterior std as % of prior width, over 45 real subjects: **Si 3%**, gastric 14%, carb 13%.
Convergence (synthetic, true Si=0.7, n_meals 1→32): **Si std 0.089→0.024 (monotone shrink)**;
gastric std flat ~0.006, carb flat ~0.005 (no shrink with more data).

**Reading:** Si is structurally identifiable (uses data), gastric/carb are structurally
non-identifiable (a ridge — additional meals do not narrow them). This is the distinction between
"needs more data" and "structurally unidentifiable", shown directly. RF hides this via shrinkage;
SNPE reveals it via calibrated width. `rf_shrinkage_analysis` gives Figure 3 (RF bias at true
Si=1.6 vs SNPE posterior width).

## Appendix — SBC calibration (the gate)
`evaluation/snpe_calibration.py` — PASSED. Credible-interval coverage 50/90/95% = 49/50/49,
89/91/86, 95/95/91 (near-nominal); KS-uniformity p all > 0.05. Posterior widths are trustworthy.

## Demographic baseline (reviewer confound check)
`evaluation/baselines.py` — |R| demographics-only (age+BMI+sex → HbA1c) = 0.50; full model
(+Si) = 0.60; **partial r(Si, HbA1c | demographics) = -0.38**. Si carries signal beyond body size.

## Claim → script map
1. "SNPE recovers Si from real meal-glucose without label supervision" → `clinical_recovery.py`
   (SNPE Spearman -0.73, monotonic groups).
2. "Recovered Si correlates with independent clinical measures" → `clinical_recovery.py` (r + CIs).
3. "RF hides non-identifiability via shrinkage; SNPE reveals it via posterior width" →
   `identifiability_analysis.py::rf_shrinkage_analysis` (Fig 3).
4. "Si identifiable; gastric/carb structurally non-identifiable from meal-glucose" →
   `identifiability_analysis.py` (Table 3, Fig 2).
5. "Personalized Si improves held-out iAUC over baselines" → carried by **grid/RF** in
   `snpe_kfold.py` (SNPE does not; state this honestly).
5b. "SNPE underperforms on absolute recovery due to a measurable simulator-reality gap, detectable
   before inference" → `ood_analysis.py` (98% real meals OOD; baseline +7.1z) + Figure 6.
6. "Framework rejects plausible-but-ineffective modifications" → the circadian null result in
   `evaluation/circadian_validation.py` (existing; Fig 6).
7. "Inference fast enough for real-time" → Table 1 timings (SNPE ~1.5s vs SMC ~40s; RF ~140ms).

## Datasets / licensing (Block 7d)
- **CGMacros** (PhysioNet) — all core paper claims. Check the PhysioNet CGMacros license permits
  academic redistribution of derived results (standard PhysioNet credentialed/open terms).
- **WESAD** — NOT part of any core paper claim. It backs only the standalone stress classifier
  (`modules/stress.py`, 87.9% balanced acc), which is out of scope here. WESAD is research-license;
  demote to a supplement or drop from the paper to avoid the commercial-use question. Safe.
- OhioT1DM, D1NAMO, NHANES, Sleep-EDF — supporting/appendix only.
