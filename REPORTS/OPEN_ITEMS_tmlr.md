# Open items for the TMLR paper (mock review of 2026-11-24)

1. AI-use statement (paper, end of Section 6): it says the authors "verified all reported numbers against stored outputs and all references against primary sources" and "reviewed and edited the final manuscript". Confirm both are true, or narrow them.
2. Held-out iAUC ratio 0.99 (fast corner, SI fixed at truth, Section 5.3) is below the oracle floor of 1.00. The paper reports it without a reason. Decide whether to add one after checking the A21 output.
3. The paper's "shift away from the box wall" pattern for the carbohydrate-error sweep is an observation on replica seed 0. A test of it (for example a replica whose truths lie mid-box) was not run.
4. Section 5.4 still carries about 40 numbers in four paragraphs. Table 3 and Table 4 hold most of them; consider cutting the in-text copies further if the main text must stay at 12 pages after the next edit.
5. `amendment16_macros.tex` is generated from `results/phase20_amendment16/*.json` and is not part of the frozen P7 index or ledger. Add it to the freeze step so every number in the paper has a hash.
6. The linter used here is the earlier `paperlint.py` copy (it is not in the current skill folder). Its macro-seam, log-structure and venue-copy checks were therefore not run.
7. Commit: `paper/` sources, `evaluation/tmlr_build.py`, this file. Amendment 16 files are still untracked.
