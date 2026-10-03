"""Fail the build when a result number is typed into the paper instead of coming from a macro.

The rule being enforced: any quantity that an analysis produced must reach the text through a macro
from `paper/results_macros.tex`. Hand-typed numbers are how the rejected submission ended up with a
79/21/0 decomposition nobody could define and a table that no longer matched the code.

Numbers that are NOT results are allowed, and are recognized rather than listed one by one:

* anything inside math mode (`$...$`, `\\[...\\]`, `equation`, `align`, `gather`, `split`) -- the
  model equations are full of constants and belong in the text;
* citation, label, reference and include commands, bibliography keys, package options;
* years from 1900 to 2099, which appear in prose when discussing prior work;
* a comment line;
* anything in `paper/number_allowlist.txt`, one literal per line with an explanation after `#`.

Run:  python -m evaluation.lint_numbers                 # lint paper/*.tex
      python -m evaluation.lint_numbers file.tex ...    # lint specific files
Exit code 1 and a file:line listing on any violation.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from evaluation.results_io import ROOT

PAPER = ROOT / "paper"
ALLOWLIST = PAPER / "number_allowlist.txt"

# Environments whose bodies are mathematics, not prose.
_MATH_ENVIRONMENTS = ("equation", "equation*", "align", "align*", "gather", "gather*",
                      "multline", "multline*", "split", "eqnarray", "eqnarray*", "array",
                      "cases", "pmatrix", "bmatrix", "tikzpicture")

# Commands whose arguments are identifiers or options, never quantities.
_OPAQUE_COMMANDS = re.compile(
    r"\\(?:cite[a-zA-Z]*|ref|eqref|autoref|pageref|label|input|include(?:graphics)?|"
    r"usepackage|documentclass|bibliography(?:style)?|newcommand|renewcommand|def|"
    r"hspace|vspace|setlength|addtolength|columnwidth|textwidth|linewidth|arraystretch|"
    r"url|href|caption\s*\*?\s*\[|begin|end|color|rowcolor|cmidrule|midrule|toprule|bottomrule|"
    r"multicolumn|multirow|fontsize|selectfont|pdfoutput|pdfminorversion)"
    # Repeated optional brackets: \citep[see][p.~3]{key} carries two of them. At most two brace
    # groups, so \multicolumn{2}{c}{Held-out MAE 1270} loses its column spec but keeps its
    # content -- a number in a table cell is exactly what should be caught.
    r"\s*(?:\[[^\]]*\]\s*)*(?:\{[^{}]*\}\s*){0,2}")

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_DIGITS = re.compile(r"\d")
# A run of digits with optional sign, decimals, thousands separators, percent or exponent.
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?")


def load_allowlist(path: Path = ALLOWLIST) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        literal = raw.split("#", 1)[0].strip()
        if literal:
            out.add(literal)
    return out


def _strip_math(line: str) -> str:
    """Blank out inline math so its constants are not treated as prose numbers."""
    line = re.sub(r"\$\$.*?\$\$", " ", line)
    line = re.sub(r"(?<!\\)\$[^$]*\$", " ", line)
    line = re.sub(r"\\\(.*?\\\)", " ", line)
    line = re.sub(r"\\\[.*?\\\]", " ", line)
    return line


def _blank_commands(line: str) -> str:
    return _OPAQUE_COMMANDS.sub(" ", line)


def scan_text(text: str, allow: set[str], name: str = "<text>") -> list[dict]:
    """Every hand-typed number in `text`, as `{file, line, number, context}` records."""
    findings: list[dict] = []
    in_math_env = 0
    in_verbatim = False
    open_dollar = False

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw

        if re.search(r"\\begin\{(?:verbatim|lstlisting|minted)\}", line):
            in_verbatim = True
        if in_verbatim:
            if re.search(r"\\end\{(?:verbatim|lstlisting|minted)\}", line):
                in_verbatim = False
            continue

        begins = sum(1 for e in _MATH_ENVIRONMENTS if f"\\begin{{{e}}}" in line)
        ends = sum(1 for e in _MATH_ENVIRONMENTS if f"\\end{{{e}}}" in line)
        if begins:
            in_math_env += begins
        if in_math_env:
            in_math_env -= ends
            continue
        if ends:
            in_math_env = max(0, in_math_env - ends)
            continue

        line = line.split("%")[0] if not line.lstrip().startswith("%") else ""
        if not line.strip():
            continue

        # A display-math block opened on an earlier line with a bare $$.
        if open_dollar:
            if "$$" in line:
                open_dollar = False
            continue
        if line.count("$$") % 2 == 1:
            open_dollar = True
            continue

        cleaned = _blank_commands(_strip_math(line))
        cleaned = _YEAR.sub(" ", cleaned)
        if not _DIGITS.search(cleaned):
            continue

        for match in _NUMBER.finditer(cleaned):
            literal = match.group(0)
            if literal in allow or literal.lstrip("+-") in allow:
                continue
            findings.append({"file": name, "line": lineno, "number": literal,
                             "context": raw.strip()[:160]})
    return findings


def scan_files(paths: list[Path]) -> list[dict]:
    allow = load_allowlist()
    findings: list[dict] = []
    for path in paths:
        findings.extend(scan_text(path.read_text(encoding="utf-8"), allow, name=str(path)))
    return findings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", type=Path)
    args = ap.parse_args()
    paths = args.files or sorted(PAPER.glob("*.tex"))
    paths = [p for p in paths if p.name != "results_macros.tex"]
    if not paths:
        print("no .tex files to lint")
        return
    findings = scan_files(paths)
    for f in findings:
        print(f"{f['file']}:{f['line']}: hand-typed number {f['number']!r} -- {f['context']}")
    print(f"{len(paths)} files, {len(findings)} hand-typed numbers")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
