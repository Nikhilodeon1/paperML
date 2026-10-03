"""Fail the build when the paper uses a word the analyses do not support.

Two rules, both about claims a reviewer would test:

1. **"structurally non-identifiable" / "structural non-identifiability"** may appear only in text about
   the exchange symmetry of the empty-gut model, the one structural statement the analyses support. The
   generic rank of the iAUC Jacobian is full, so iAUC is *locally identifiable but ill-conditioned*, and
   calling it structurally non-identifiable would be wrong. A paragraph that uses the phrase must also
   mention the exchange (swap) or the empty gut.
2. **"pre-registered" / "preregistered"** is not used. The analysis plan was fixed in version control
   before the analyses ran, which a reviewer cannot verify, and three amendments have followed. The
   accurate wording is "analysis plan fixed in version control before the analyses were run".

The unit of context is the paragraph (text between blank lines).

Run:  python -m evaluation.lint_terms                 # lint paper/*.tex
      python -m evaluation.lint_terms file.tex ...
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from evaluation.results_io import ROOT

PAPER = ROOT / "paper"

STRUCTURAL = re.compile(r"structural(?:ly)?[\s~-]+non[\s-]?identifiab(?:le|ility)", re.I)
PREREGISTERED = re.compile(r"pre[\s-]?regist(?:ered|ration)", re.I)
EXCHANGE_CONTEXT = re.compile(r"\b(exchange|swap|empty[\s-]gut|interchange)\b", re.I)


def _paragraphs(text: str):
    line = 1
    for block in re.split(r"(\n\s*\n)", text):
        yield line, block
        line += block.count("\n")


def lint_text(text: str, name: str = "<text>") -> list[str]:
    problems = []
    for start, block in _paragraphs(text):
        body = "\n".join(l for l in block.splitlines() if not l.lstrip().startswith("%"))
        if STRUCTURAL.search(body) and not EXCHANGE_CONTEXT.search(body):
            problems.append(f"{name}:{start}: 'structural non-identifiability' outside text about the "
                            f"empty-gut exchange symmetry; iAUC is locally identifiable but "
                            f"ill-conditioned")
        if PREREGISTERED.search(body):
            problems.append(f"{name}:{start}: do not write 'pre-registered'; write 'analysis plan "
                            f"fixed in version control before the analyses were run'")
    return problems


def lint_files(paths) -> list[str]:
    out = []
    for path in paths:
        out.extend(lint_text(Path(path).read_text(encoding="utf-8"), str(path)))
    return out


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(a) for a in args] or sorted(PAPER.glob("*.tex"))
    problems = lint_files(paths)
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s) in {len(paths)} file(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
