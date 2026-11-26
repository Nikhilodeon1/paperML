# Open items for the TMLR paper (updated 2026-11-26, after the front-matter rewrite)

## Blocking submission

1. **Pending pod results.** `[pending]` cells remain in the coordinate table (Shanghai, real traces) and in the
   coverage table (fitted truths, CV 0.75). Needed: `phase21_a_results.tgz`, `phase21_gapfill_b.tgz`,
   `phase21_gapfill_perturb.tgz`. Pod C's archive is merged. After merging: `python -m evaluation.phase21_summary`,
   `python -m evaluation.figs_tmlr`, `python -m evaluation.tmlr_build --compile`.
2. **Text that depends on those results.**
   - Section 5.2 says perturbing p1, p2 and n in the generator "was not run". The pods run it. Rewrite from the
     `resPb*` macros, then recheck the limitation "One model family with fixed constants".
   - Limitation "Replica design" says fitted truths come from one seed. Seeds 1 and 2 (CV 0 and 0.75) are running;
     the summary reads them but no table shows them yet.
   - Shanghai replicas (CV 0 and 0.75) are summarised as `resCShanghaiFit*` and not yet in the text.
   - If fitted CV 0.75 or the extra seeds disagree with Section 5.4, change the abstract and discussion too.
3. **Appendix cleanup not done.** The linter still reports amendment mentions, hypothesis codes, inline status tags
   and number walls, mostly in the appendices. The side analyses the review flagged (F.7, F.9, the random-forest and
   SNPE comparison) are still there. `evaluation.lint_numbers` reported hand-typed numbers in 22 files (mostly
   settings and the specification table); not reviewed.
4. **AI-use statement.** The sentence claiming the authors verified every number and reference was removed because
   nothing in the record shows it. Add it back only after doing the check. Also confirm that the statement and the
   first-page footnote describe what the assistant did.
5. **References checked for existence only** (metadata, not content): bellman1970structural,
   rothenberg1971identification, venzon1988method, efron1987better, holm1979simple, wilson1927probable,
   byrd1995limited, white1982maximum, dallaman2002oral, brazeau2013carbohydrate, cook1994simulation,
   albers2017personalized. They are cited only as pointers. Read them or leave the wording as it is.
   The four arXiv entries that a lookup could not find are no longer cited and were removed from the bibliography.

## Author pass (rewrite in your own words, keep the facts)

Introduction paragraphs 1 to 3 (motivation, closest prior work, scope); Section 5.3 opening (what the real-data
prediction comparison can and cannot show); Section 5.4 "A bounded interval is not a sign of better data";
Section 5.5 second paragraph (pseudo-true value); Discussion "What limits identification", "What the replicas add"
and "Recommendations"; Limitations; Conclusion; Broader Impact.

## Not run (stated in Limitations or the text)

SIMEX or another input-error remedy; a Bergman fit with sensor lag and free p2; a Dalla Man replica fitted by Dalla
Man; a second generator distribution; overlapping meals; restriction to meals with gaps over 4 h; subject-specific
basal glucose; per-meal k_e; posterior contraction.

## Weak points a reviewer can press

- **All iAUC profiles (box table, Section 5.2 S_I results, replica, synthetic, perturbations) use the older 15-point
  Adam protocol and Sections 5.1 and 5.2 do not say so.** The abstract's unbounded-rates claim rests on them. Rerun with
  `inner_solver=lbfgsb`, `grid_points=29`, or label them.
- Centroid, residual-gut, exact-domain and Dalla Man profiles also use the older protocol. The text says so.
- Why logging error moves estimates toward slower absorption is not established.
- Held-out iAUC ratio 0.99 (fast corner, S_I fixed at truth, appendix) is below the floor of 1.00, unexplained.
- `amendment16_macros.tex` and `amendment17_macros.tex` are not in the frozen number index or ledger.

## Elsewhere

The CALE-AI workshop paper still uses coverage numbers from the older protocol.
