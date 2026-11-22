# CALE-AI 2027 submission: checklist

Venue facts (read from https://caleai.github.io/ and https://easychair.org/cfp/CALE-AI2027 on 2026-11-21): submission deadline 27 November 2026 (timezone not stated); notification 2 December; camera-ready 10 December; full papers up to 8 pages on the workshop site and 9 on EasyChair, references excluded; non-archival; at least two reviews per submission; submission through EasyChair (`https://easychair.org/my/conference?conf=caleai2027`). Not stated by the venue: double-blind or single-blind review, a generative-AI policy, the template version, dual-submission rules.

## Open items for the authors

1. **AI-use statement.** The Ethical Statement holds a placeholder (`[AUTHOR: ...]`). Fill in the tool, the tasks, and what the authors checked. Remove the bracket before submitting.
2. **Confirm the venue's rules by email** (contact on the workshop site): that a concurrent submission to a journal is allowed for a non-archival workshop paper, and whether review is double-blind. The paper is anonymous either way.
3. **Abstract registration.** EasyChair lists an abstract registration date of 27 November; register the title and abstract first.
4. **Metadata.** `pdfinfo` shows no title, author or subject, only the TeX producer. If you want it cleared further, run a metadata cleaner on the final PDF.
5. **Figure fonts.** The figures use Arial (embedded TrueType) as the Helvetica stand-in. The kit asks for Times or Helvetica in figures. If a strict check objects, redraw with a Helvetica clone; the script is `evaluation/figs_workshop.py`.
6. **Overlap with the TMLR manuscript.** This is a condensed version of the same work. Keep the two consistent, and disclose the relationship wherever the venue or TMLR asks.
7. **Author-pass.** Rewrite the passages listed in `author_pass.md` in your own words.

## AAAI author kit rules (AuthorKit27/AnonymousSubmission2027.tex), and status

| rule | status |
|---|---|
| `\documentclass[letterpaper]{article}`, `\usepackage[submission]{aaai2027}`, url, graphicx, natbib (no options), caption, `\frenchspacing`, `\pdfinfo` as in the kit | done, unchanged lines |
| No forbidden packages (authblk, babel, balance, cjk, epsf, epsfig, euler, float, fullpage, geometry, graphics, hyperref, layout, lmodern, navigator, pdfcomment, pgfplots, psfig, pstricks, t1enc, times, titlesec, tocbibind, ulem) | none loaded; only booktabs is added (the kit recommends it) |
| No forbidden commands (`\abovecaption`, `\addtolength`, `\baselinestretch`, `\clearpage`, `\columnsep`, `\float`, `\linespread`, `\newpage`, `\pagebreak`, `\setlength`, `\textheight`, `\tiny`, `\trim`, `\clip`, negative `\vspace` or `\vskip`) | none present (searched) |
| Single .tex file, no `\input`; only the .bib is separate | done: `build.py` inlines every macro |
| Anonymous: author "Anonymous Submission", no affiliations, no identifying links or self-citations, no copyright footer | done (`submission` option; no self-citation; no links block) |
| Title in Title Case, 16 pt bold from the style file | done |
| Two-column format, US letter, margins from the style file, no page numbers or headers | done (`pdfinfo`: letter, 612 x 792 pt) |
| Times-like text font from the style file, no font packages, no type 3 fonts | done; `pdffonts` shows Type 1 text fonts and embedded TrueType figure fonts, no Type 3 |
| Abstract without references; `begin{abstract}` as in the kit | done |
| Section order: main content, (appendices), Ethical Statement, (Acknowledgments), References | done; no acknowledgments in an anonymous paper |
| Numbered sections (`secnumdepth` 1, allowed by the kit); Ethical Statement unnumbered | done |
| Figures: PDF files, cropped outside LaTeX, no clip or trim, captions under the figure, labels at least 9 pt at inclusion size, lines at least 0.5 pt, readable in grayscale, no minipage grouping | done; figure sizes checked against inclusion width (scale at least 1.0) |
| Tables: 10 pt or 9 pt, no `resizebox`, caption under the table | done (`\small` table, caption below) |
| No overfull boxes | none in the log (a few underfull boxes only) |
| References: BibTeX with `aaai2027.bst` (set by the style file), complete fields, consistent format | done; 16 entries from verified rows |
| Page limit: 8 pages excluding references | about 6 pages of body plus references (7 pages in all) |
| File name uses the first author's family name | not applicable to an anonymous file; the source is `Anonymous.tex` |
| Compiles with standard TeX Live and matches the PDF | the package in `submission/` contains the exact source, the .bbl, the .bib, style files and the figures used |

## Style checks (ml-paper-writer)

`python workshop_caleai/style_check.py`: no dashes, banned words or contractions; no density flags; first-person plural 11.6 per 1,000 words. Every number comes from a macro (`EVIDENCE.md`); every citation is a row of `CITATIONS_VERIFIED.md`.
