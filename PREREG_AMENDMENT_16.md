# Pre-registration amendment 16 (TMLR revision: negative control, window profiles, standardized-meal trace, and sandwich remedy)

Amends `PREREG.md` and amendments 1 to 15. No threshold of an earlier document is changed and no earlier verdict is edited. Everything below is **post hoc**, declared and fixed before its computation is run.

## What was seen before this amendment

Everything through Phase 19 (Amendment 15), plus two mock TMLR reviews of the 12-page manuscript. The reviews identified five concrete empirical gaps that can be tested directly:

1. **Mechanism of the carb-error boundedness jump (R25, stored profiles only):** On replica seed 0, bounded $\tau_1$ intervals rose from 6/45 at exact carbohydrates to 28/45 at $\mathrm{CV}=0.75$ even though Fisher curvature and max profile rise did not increase. We check on the already-stored profiles whether the trend holds on a fixed subject set (overall counts and subjects interior at all five CVs) and measure the verdict transition matrix (`one-sided` to `bounded`) and signed log shift of $(\hat\tau_1, \hat p)$.
2. **Negative control and power for the iAUC prediction claim (H39):** On random-truth replicas, the fitted model achieves $1.03\times$ the true-parameter iAUC floor, while the personal mean is near $1.1$--$1.2\times$. Without deliberately wrong timing rates, it is unclear how much iAUC error bad gut rates can produce—either with $\SI$ held at truth or with $\SI$ re-fitted to compensate.
3. **iAUC profile likelihood across postprandial windows (H40):** Linearized timing sensitivity exceeds $\SI$ sensitivity at $90$\,min, so the claim that iAUC leaves $(\ke,\ka)$ unbounded should be verified in profile likelihood across $T\in\{60, 90, 120, 180\}$\,min on CGMacros (the cached meal window extends to $180$\,min).
4. **Exploratory real-data trace check on standardized meals (H41, report-only):** The Hall cohort ($n=15$, $5$--$6$ meals/subject, $145$\,min window) has known carbohydrates by construction. Running polished full-trace coordinate profiles provides an exploratory real-data check without carbohydrate logging error (low power due to small meal count and shorter window).
5. **Tested coverage remedy under carbohydrate logging error (H42):** Standard profile likelihood assumes exact inputs and severely undercovers $(\tau_1, p)$ under $\mathrm{CV}=0.25$ and $\mathrm{CV}=0.75$. Because logging error is independent across meals, we test a meal-clustered Huber--White sandwich adjustment (both Wald intervals and Kent/Chandler--Bate sandwich-rescaled profile intervals using $c_j = \hat V_{\text{sand},jj} / \hat V_{\text{naive},jj}$) on replica seed 0.

## Analyses (A21--A24 and R25)

* **R25 (from stored results):** Fixed-subject boundedness across $\mathrm{CV}\in\{0, 0.1, 0.25, 0.5, 0.75\}$ for $\tau_1$ and $p$ (all 45 subjects and common-interior subset), verdict transitions (`one-sided` $\to$ `bounded`), and median signed log shift $\log(\hat\theta_{\text{CV}=0.75}/\hat\theta_{\text{CV}=0})$.
* **A21 (`evaluation/a21_neg_control.py`, H39):** On CGMacros random-truth replicas (seeds 0 and 1, $\mathrm{CV}=0.25$), compute held-out iAUC, peak-time, and trace error ratios to the true-parameter floor under two deliberately wrong timing settings: slowest box corner $(\ke,\ka)=(0.015, 0.012)$ and fastest box corner $(0.040, 0.032)$, both (a) with $\SI$ fixed at the true value and (b) with $\SI$ re-fitted on each training fold (matching the cross-validation protocol).
* **A22 (`evaluation/a22_window_profiles.py`, H40):** On all 45 CGMacros subjects (primary box, rate coordinates), run unregularized iAUC profile likelihoods for $(\SI, \ke, \ka)$ at post-meal windows $T\in\{60, 90, 120, 180\}$\,min.
* **A23 (`evaluation/a23_hall_trace.py`, H41):** On all 15 Hall standardized-meal subjects (primary box, symmetric coordinates $(\log\SI, \log\tau_1, \log p)$, L-BFGS-B polished, $145$\,min window), run full-trace profile likelihoods.
* **A24 (`evaluation/a24_sandwich_coverage.py`, H42):** On replica seed 0 ($\mathrm{CV}\in\{0.0, 0.25, 0.75\}$, polished trace coordinate fits), compute the meal-clustered sandwich covariance $V_{\text{sand}} = \frac{M}{M-3} H^{-1}(\sum_{m=1}^M g_m g_m^\top) H^{-1}$ at $\hat\phi$, evaluate $95\%$ sandwich Wald coverage and sandwich-rescaled profile coverage (threshold $c_j \delta$ on the stored profiles), and report resulting bounded fractions and median widths.

## Rules (fixed before running)

* **H39 (post hoc):** Under deliberately wrong corner gut rates with $\SI$ re-fitted to iAUC on random-truth replica seed 0, median held-out iAUC ratio to floor stays $\le 1.10$ for both slow and fast corners (confirming $\SI$ compensates on iAUC even for extreme timing error), while peak-time and trace ratios exceed the iAUC ratio.
* **H40 (post hoc):** Across $T\in\{60, 90, 120, 180\}$\,min on CGMacros, bounded fractions for $\ke$ and $\ka$ remain $\le 25\%$ of all subjects at every window.
* **H41 (post hoc, exploratory / report-only):** No pass/fail threshold; report bounded fractions (interior and overall) and interior counts on the Hall cohort ($n=15$).
* **H42 (post hoc):** On replica seed 0 at $\mathrm{CV}=0.25$, meal-clustered sandwich rescaling of the trace coordinate profiles raises nominal coverage to $\ge 85\%$ for both $\tau_1$ and $p$ (and report the resulting bounded fraction and width at $\mathrm{CV}=0.0, 0.25, 0.75$).
