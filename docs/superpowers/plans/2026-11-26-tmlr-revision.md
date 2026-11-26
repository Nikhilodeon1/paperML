# TMLR Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the TMLR paper to a state where every claim rests on complete, one-protocol, traceable results.

**Architecture:** Trust the numbers first (manifest, frozen index, test status), then rerun old-protocol results with the
converged solver on three Colab pods, then add only the experiments that can change claims, then restructure and hand
the prose to the authors. Existing `evaluation/` conventions are followed; new code gets a test first.

**Tech Stack:** Python, JAX/Diffrax, SciPy L-BFGS-B, pytest, LaTeX (latexmk), `tools_lint/paperlint.py`.

**Spec:** `REPORTS/reviewer_plan_2026-11-26.md`. **Outcome rules:** `REPORTS/decision_rules_2026-11-26.md` (R1 to R18).

## Global Constraints

- No number from a sweep that `evaluation.manifest` does not report complete.
- No outcome rule is loosened after its result is seen.
- One protocol for every reported profile: L-BFGS-B at every point, 29-point grid, polished estimates.
- No number in the abstract, tables or results text is typed by hand.
- Main text at most 12 pages. No hypothesis codes, amendment mentions or script names in the final PDF.
- The AI-use statement says only what was done. The authors write the final prose.
- Running pod jobs are never interrupted.

## Review Focus

- A sweep with the right count but the wrong subjects: reported incomplete (Task 1 test).
- A run whose units are held by dead claim files: exits nonzero (Task 1 test).
- A profile whose lowest in-box point is already above the threshold but is classed bounded: checked after every sweep
  (Task 3).
- A grid point whose solver reports failure although a lower loss exists: re-solve check (Task 3).
- A summary silently reading an old-protocol result where a converged one is expected: configuration match must include
  solver and grid (Task 5).

---

### Task 1: Completeness manifest and unfinished-run exit code (Stage 0)
- [x] Tests `tests/test_manifest.py`, `tests/test_runner.py::test_exit_code_fails_when_units_are_left_unfinished`
- [x] `evaluation/manifest.py`; `runner.exit_code`; `phase21_summary` strict unless `--draft`
- [ ] Commit

### Task 2: Test status (Stage 0)
- [x] The 4 failures reproduce on clean `68fb52d` (`REPORTS/test_status.md`)
- [ ] Full suite on the final tagged commit

### Task 3: Solver health for iAUC rate profiles (gate for Task 4)
- [ ] Re-solve every point flagged unconverged on at least 3 subjects and boxes; accept if the largest loss improvement is
      below 0.05 (the tolerance the pipeline already uses), otherwise fix the solver first
- [ ] Add to `evaluation/manifest.py` a check that no stored profile is classed bounded while its lowest in-box point is
      above the threshold; test with a constructed row

### Task 4: Stage 2 pod queue, existing code only
- [ ] `scripts/pod_stage2_lib.sh`, `pod_stage2_1.sh`, `_2.sh`, `_3.sh`: iAUC (boxes, replicas, noise, synthetic,
      perturbations, Shanghai, standardized meal), centroid, residual gut
- [ ] Add every Stage 2 sweep to the manifest (`stage2_sweeps()`), with a test that pod job names and manifest names agree
- [ ] User pushes the branch; each pod starts its queue only after its current job has finished

### Task 5: Stage 1 merge and rules R1 to R3
- [ ] Download archives, merge without overwriting, delete no result; `python -m evaluation.manifest` passes for phase 21
- [ ] `python -m evaluation.phase21_summary` (strict); apply R1 to R3 and record each outcome in the rules file

### Task 6: Stage 2 summary and claims (R5 to R10, R12)
- [ ] `evaluation/stage2_summary.py` writing macros only from complete converged sweeps; point every iAUC, centroid and
      residual-gut number in the paper at them; delete exact-domain results

### Task 7: New code for the second pod queue
- [ ] Dalla Man as a replica generator inside `evaluation/subject_source.py`, profiled by the standard converged pipeline;
      noise-free run gives the pseudo-true value (R11)
- [ ] Window length as a profile configuration key (window sweep, kept short)
- [ ] Converged solver option for the six-parameter Dalla Man fit to real data (R11)
- [ ] Regression calibration of logged carbohydrate as a replica key (R13)
- [ ] Minimum gap since the previous meal as a cohort key (R14)
- [ ] Subject-specific basal glucose as a context in the engine (R15)
- [ ] Sensor lag, only if a small change (R16)

### Task 8: Number traceability (Stage 0, finished after the last result)
- [ ] Freeze index covers all macro files including amendment 16 and 17; review the hand-typed list; zero hand-typed
      numbers in abstract, tables and results

### Task 9: Claim ledger and restructure (Stage 4)
- [ ] `REPORTS/claim_ledger.md`: claim, evidence table, protocol, status for abstract, introduction, conclusion
- [ ] Table 2 split and wording; interval score column; remove "polished"; Figure caption; reference-ratio rename with
      bootstrap interval (R17); rank correlation against CV (R18)
- [ ] Appendix cuts and keeps exactly as listed in the spec; register and hypothesis codes moved to the repository
- [ ] Design schematic; Hovorka-type identifiability reference, checked at its primary source

### Task 10: Authors' pass and submission gate (Stages 5 and 6)
- [ ] Authors write abstract, introduction, related work, discussion, conclusion and Section 5.4 from the ledger, without
      editing the AI draft
- [ ] Reference check against primary sources and number check against the index, then the AI-use statement
- [ ] Zero pending, no undefined references, linter clean in main text, number audit, anonymization (PDF metadata and
      repository history), one outside read
