"""Assemble `workshop_caleai/submission/` (only files the paper uses) and write the evidence and citation logs.

Run after `python workshop_caleai/build.py`.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402

SUB = HERE / "submission"


def evidence() -> str:
    src = (HERE / "paper_src.tex").read_text(encoding="utf-8")
    used = sorted(set(re.findall(r"\\((?:res|fig)[A-Za-z]+)", src)))
    values = {**build.result_macros(), **build.figure_macros()}
    rows = []
    for name in used:
        origin = ("`workshop_caleai/figure_numbers.json`, written by `evaluation/figs_workshop.py` from the stored results"
                  if name.startswith("fig") else "`paper/results_macros.tex`, written by `evaluation/paper_numbers.py` from `results/index.json`")
        rows.append(f"| `{name}` | {values[name]} | {origin} |")
    return ("# Evidence ledger for the workshop paper\n\nEvery number in the paper is a macro. The result macros come from the frozen result "
            "index of the main manuscript; the figure macros are computed from the same stored results when the figures are drawn. "
            f"{len(used)} macros are used.\n\n| macro | value | source |\n|---|---|---|\n" + "\n".join(rows) + "\n")


def citations() -> str:
    log = (ROOT / "paper" / "ledger" / "CITATIONS_VERIFIED.md").read_text(encoding="utf-8").splitlines()
    head = [line for line in log if line.startswith("| key") or line.startswith("|---")]
    rows = [line for line in log if any(line.startswith(f"| `{k}`") for k in build.KEYS)]
    return ("# Citations used in the workshop paper\n\nThe rows are copied from the citation ledger of the main manuscript "
            "(`paper/ledger/CITATIONS_VERIFIED.md`), where each source was opened and its metadata and supporting passage checked "
            "on the access date given in the row. Re-open each URL once more before submission.\n\n" + "\n".join(head[:2] + rows) + "\n")


def main() -> int:
    if SUB.exists():
        shutil.rmtree(SUB, ignore_errors=True)
    (SUB / "figures").mkdir(parents=True, exist_ok=True)
    b = HERE / "build"
    for name in ("Anonymous.tex", "Anonymous.bbl", "Anonymous.pdf", "caleai.bib", "aaai2027.sty", "aaai2027.bst"):
        shutil.copy(b / name, SUB / name)
    for fig in (b / "figures").glob("fig_*.pdf"):
        shutil.copy(fig, SUB / "figures" / fig.name)
    (HERE / "EVIDENCE.md").write_text(evidence(), encoding="utf-8")
    (HERE / "CITATIONS_VERIFIED.md").write_text(citations(), encoding="utf-8")
    print("wrote", SUB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
