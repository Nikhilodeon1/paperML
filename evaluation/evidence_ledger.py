"""Evidence ledger: every number the manuscript quotes, with its value, where it is used and where it comes from.

The manuscript never contains a typed result. Each number is a macro in `paper/results_macros.tex`, written by
`evaluation/make_macros.py` from the frozen result index (`results/index.json`, SHA-256 per file), whose declarations
come from `evaluation/paper_numbers.py`, which reads the summaries `results/phase{2,8,9,10,11}_summary.json`, which
are computed from the stored per-unit result files. This module lists the macros the text actually uses.

Run:  python -m evaluation.evidence_ledger
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict

from evaluation.results_io import ROOT

PAPER = ROOT / "paper"
OUT = PAPER / "ledger" / "EVIDENCE_LEDGER.md"
SKIP = {"results_macros.tex", "pending_macros.tex", "preamble_defs.tex"}
USE = re.compile(r"\\(res[A-Za-z]+)")
DEFINE = re.compile(r"\\newcommand\{\\(res[A-Za-z]+)\}\{(.*?)\}\s*(?:%\s*(\S+))?\s*$", re.M)

# the analysis each macro family comes from (prefix matches, first hit wins)
FAMILY = [("resVerdict", "hypothesis verdicts (evaluation/hypotheses.py)"), ("resCount", "verdict tallies"),
          ("resNoise", "H24, noise-scaled replica (phase 11)"), ("resExact", "H25, exact-carbohydrate replica (phase 12)"),
          ("resPooled", "pooled coordinate calibration (phase 11)"), ("resRecal", "H23, recalibrated cutoff (phase 10)"),
          ("resHetero", "H16, heteroskedastic noise (phase 10)"), ("resSeeds", "H17, replica seeds (phase 9)"),
          ("resSynth", "H19, synthetic study (phase 9)"), ("resShCv", "H20, Shanghai prediction (phase 9)"),
          ("resDm", "H22, second model class (phase 9)"), ("resPolish", "H21, polished estimates (phase 9)"),
          ("resReplicaCoords", "H18, coordinate replica (phase 9)"), ("resTen", "H10, robustness (phase 9)"),
          ("resReplica", "H13-H14, replica (phase 8)"), ("resCoords", "H11, coordinate fits (phase 8)"),
          ("resLeak", "leakage check (phase 8)"), ("resCarbScale", "carbohydrate-scale sensitivity"),
          ("resDiff", "H7-H9, CGMacros prediction (phase 2)"), ("resProfile", "H5, profile intervals (phase 2)"),
          ("resRep", "H15, replication cohorts (phase 8)"), ("resLevel", "interval-level sensitivity (phase 9)")]


def family(name: str) -> str:
    for prefix, label in FAMILY:
        if name.startswith(prefix):
            return label
    return "phase 2 summary"


def main() -> int:
    values = {n: (v.strip(), s) for n, v, s in DEFINE.findall((PAPER / "results_macros.tex").read_text(encoding="utf-8"))}
    used: dict[str, set[str]] = defaultdict(set)
    for path in sorted(PAPER.glob("*.tex")):
        if path.name in SKIP:
            continue
        for name in USE.findall(re.sub(r"(?m)(?<!\\)%.*$", "", path.read_text(encoding="utf-8"))):
            used[name].add(path.name)
    index = json.loads((ROOT / "results" / "index.json").read_text(encoding="utf-8"))
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    rows = []
    missing = []
    for name in sorted(used):
        if name not in values:
            missing.append(name)
            continue
        rows.append(f"| `{name}` | {values[name][0]} | {family(name)} | {', '.join(sorted(used[name]))} |")
    head = (f"# Evidence ledger\n\nFrozen result index: `results/index.json`, {index.get('n_files')} files, frozen "
            f"{index.get('frozen_at')}, combined SHA-256 `{index.get('sha256_over_all')}`; repository commit `{commit}` "
            "when this file was generated. Every value below is a macro written from that index by "
            "`evaluation/make_macros.py`; the code that computes each macro is in `evaluation/paper_numbers.py` and the "
            "summaries under `results/`. The ledger lists the macros that the manuscript text uses.\n\n"
            f"{len(rows)} macros used, {len(missing)} used but undefined.\n\n"
            "| macro | value | analysis | used in |\n|---|---|---|---|\n")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(head + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"    {len(rows)} macros in the ledger, {len(missing)} undefined")
    for name in missing:
        print("    undefined: " + name)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
