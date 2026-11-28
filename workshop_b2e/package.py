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
    used = sorted(set(re.findall(r"\\((?:res|fig|ws)[A-Za-z]+)", src)))
    values = {**build.result_macros(), **build.figure_macros(), **build.workshop_numbers()["macros"]}
    rows = []
    for name in used:
        if name.startswith("ws"):
            origin = "`workshop_caleai/ws_numbers.py`, from complete sweeps of the converged protocol"
        elif name.startswith("fig"):
            origin = "`workshop_caleai/figure_numbers.json`, written by `evaluation/figs_workshop.py` from the stored results"
        else:
            origin = "a generated macro file of the main manuscript (`paper/*macros.tex`)"
        rows.append(f"| `{name}` | {values[name]} | {origin} |")
    return ("# Evidence ledger for the workshop paper\n\nEvery result in the paper is a macro written by a script from stored results. "
            f"{len(used)} macros are used.\n\n| macro | value | source |\n|---|---|---|\n" + "\n".join(rows) + "\n")


def citations() -> str:
    log = (ROOT / "paper" / "ledger" / "CITATIONS_VERIFIED.md").read_text(encoding="utf-8").splitlines()
    head = [line for line in log if line.startswith("| key") or line.startswith("|---")]
    rows = [line for line in log if any(line.startswith(f"| `{k}`") for k in build.KEYS)]
    return ("# Citations used in the workshop paper\n\nThe rows are copied from the citation ledger of the main manuscript "
            "(`paper/ledger/CITATIONS_VERIFIED.md`). Each row states its status: VERIFIED rows record a passage that was read, "
            "METADATA-ONLY rows only that the source exists. Re-open each source before submission.\n\n" + "\n".join(head[:2] + rows) + "\n")


def main() -> int:
    if build.output_name() != "Anonymous":
        raise SystemExit("the numbers are a draft: run `python workshop_caleai/ws_numbers.py` (without --draft) and rebuild")
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
