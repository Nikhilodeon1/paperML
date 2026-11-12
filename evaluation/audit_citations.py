"""End-of-draft citation audit: every cited key exists, every entry is cited, and the ledger covers every entry.

The ledger is `paper/ledger/CITATIONS_VERIFIED.md`: one row per bibliography key with the status the
`paper-citation-verification` protocol defines (VERIFIED, METADATA-ONLY, UNVERIFIED). An UNVERIFIED entry, or an
entry missing from the ledger, fails the audit; so does a bibliography entry nothing cites and a citation key that
the bibliography does not define.

Run:  python -m evaluation.audit_citations
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from evaluation.results_io import ROOT

PAPER = ROOT / "paper"
LEDGER = PAPER / "ledger" / "CITATIONS_VERIFIED.md"
SKIP = {"results_macros.tex", "pending_macros.tex"}
CITE = re.compile(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")


def bib_keys() -> list[str]:
    return re.findall(r"^@\w+\{([^,]+),", (PAPER / "references.bib").read_text(encoding="utf-8"), re.M)


def cited_keys() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for path in sorted(PAPER.glob("*.tex")):
        if path.name in SKIP:
            continue
        for match in CITE.finditer(path.read_text(encoding="utf-8")):
            for key in match.group(1).split(","):
                out.setdefault(key.strip(), set()).add(path.name)
    return out


def ledger_status() -> dict[str, str]:
    if not LEDGER.exists():
        return {}
    status = {}
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0].startswith("`") and cells[0].endswith("`"):
            status[cells[0].strip("`")] = cells[-1]
    return status


def audit() -> list[str]:
    keys, cited, status = bib_keys(), cited_keys(), ledger_status()
    problems = []
    for key in cited:
        if key not in keys:
            problems.append(f"cited but not in the bibliography: {key}")
    for key in keys:
        if key not in cited:
            problems.append(f"in the bibliography but never cited: {key}")
        if key not in status:
            problems.append(f"missing from the citation ledger: {key}")
        elif not status[key].startswith(("VERIFIED", "METADATA-ONLY")):
            problems.append(f"not verified in the ledger: {key} ({status[key]})")
    return problems


def main() -> int:
    keys, cited, status = bib_keys(), cited_keys(), ledger_status()
    counts = {s: sum(1 for k in keys if status.get(k, "").startswith(s)) for s in ("VERIFIED", "METADATA-ONLY")}
    problems = audit()
    for p in problems:
        print("    " + p)
    print(f"    {len(keys)} references, {counts['VERIFIED']} VERIFIED, {counts['METADATA-ONLY']} METADATA-ONLY, "
          f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
