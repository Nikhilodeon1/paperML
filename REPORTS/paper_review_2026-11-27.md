# Mock review of the TMLR draft, 27 November 2026

**How it was done.** A separate reader agent was given only the compiled PDF and told to read as a tired, non-expert
reviewer, front to back. Its findings were then checked against the source. The `ml-paper-writer` linter and its
cold-read checklist were run as well.

## Verdict

**Not ready to submit.** The reader's verdict was major revision: the question matters and the known-truth coverage
experiment is more convincing than expected, but the draft has placeholders, mixed protocols, and is hard to follow
outside the first two pages, the figures of Section 5.4, and the Discussion.

## What got through, and what did not

- Got through: the area does not determine the gut rates; held-out error does not warn of it; the trace works with exact
  inputs and logging error breaks it.
- Did not get through: the Dalla Man control (the reader forgot it), that the real-data twin is only a modest predictor,
  and that the CV 0.75 counts differ between seeds.

## Errors confirmed and fixed

| finding | note |
|---|---|
| **Table 3, four-times row came from 29 of 45 subjects** (found while checking the reader's question about S_I falling from 40 to 28) | serious; the summary now withholds partial widths, the row prints `[pending]`, and the introduction claims "up to twice" |
| The prediction figure still plotted the removed random-forest and SNPE comparisons | left over from the appendix cut |
| A hypothesis code in a figure caption, and a "rerun is pending" note inside the manuscript | notes do not belong in the manuscript |
| The specification table caption said "Left/Right" for stacked blocks | |
| The specification appendix described the 15-point protocol without saying which results use it | scope now stated; Section 5.1 also says its profiles use it |
| A garbled sentence in 5.3 that presented a seed-1 corner value as the fitted rates' value | |
| "The trace does not discriminate" against Table 4's higher rank correlation for the trace | both facts are now stated |
| "Within a factor of 1.03" omitted that it is a median; the conclusion said "accurate prediction" for a modest real-data predictor | abstract, introduction, conclusion |
| An appendix sentence about an estimator the paper never introduces; a solver-history paragraph that the restart finding contradicts | the paragraph was removed |
| "8 to 22" with no CV; a coordinate edge named before the coordinates were introduced; "model bias" not said to be a median of per-subject differences | |

## Open: needs the reruns

- `[pending]`: Shanghai trace row, the four-times box row.
- The iAUC results, the centroid, residual gut, windows, and Dalla Man are old-protocol; Section 5.4 uses the solver
  without restarts.
- Appendix D.5 (solver history) should be rewritten around the new protocol.

## Open: reading experience (authors, during the hand rewrite)

In the reader's order of priority:

1. A plain-language setup early on: a model diagram, a short glossary (box, interior, bounded, covers, the replica
   kinds, tau1, p), and a table of experiments. Introduce tau1 and p early, with intuition.
2. Say which protocol each result used (this disappears once the reruns are in).
3. Section 3.4 loses the reader: too many new ideas in two paragraphs. Sections 5.2 and 5.5 are walls of numbers: one
   question and answer per paragraph.
4. Figures 1 and 2 are too small. Figure 3 (the sweep) is the best item; make it prominent. Consider a graphic of the
   wrong-corner experiment.
5. Style: repeated clipped contrasts ("These counts say ... They do not say ..."), slogan headings, and appendix prose
   that reads like project notes.
6. Check Table 11 (Shanghai): the share "on a bound" and the interior counts look inconsistent. They use different
   tolerances; say so in the caption or use one definition.

## Linter

0 errors. About 35 warnings, mostly disclaimer density, number walls in the appendix, citations checked for existence
only, and four script names that sit in comment lines of generated files.
