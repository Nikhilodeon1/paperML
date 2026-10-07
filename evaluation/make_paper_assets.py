"""Build every machine-generated paper asset, in order, and fail loudly if any step does.

    1. summaries  results/phase2_summary.json, phase8_summary.json, phase9_summary.json from stored results
    2. numbers    the macro declarations (P7_paper_numbers)
    3. freeze     results/index.json with a SHA-256 per result file
    4. macros     paper/results_macros.tex, and placeholders for macros not yet produced
    5. figures    paper/figures/tmlr/*.pdf, *.png
    6. tables     paper/generated/*.tex
    7. lint       hand-typed numbers in prose, and banned terminology
    8. latex      paper/main.pdf (optional, `--latex`)

Nothing is fitted. The point of the order is that a number cannot reach the paper without passing
through a frozen, hashed result file.

Run:  python -m evaluation.make_paper_assets
      python -m evaluation.make_paper_assets --skip-lint     # while the prose is still being written
      python -m evaluation.make_paper_assets --latex         # and compile the manuscript
      python -m evaluation.make_paper_assets --no-freeze     # rebuild everything but leave results/index.json
"""
from __future__ import annotations

import argparse
import sys
import time


def _step(name, fn) -> bool:
    started = time.perf_counter()
    print(f"[{name}] ...", flush=True)
    try:
        code = fn()
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, str):
            print(f"[{name}] {code}")
    except Exception as exc:
        print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
        return False
    ok = code is None or (isinstance(code, int) and not isinstance(code, bool) and code == 0)
    print(f"[{name}] {'ok' if ok else 'FAILED'} ({time.perf_counter() - started:.1f}s)")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-lint", action="store_true")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--no-freeze", action="store_true",
                    help="do not rewrite results/index.json (the macros are then read from the existing index)")
    args = ap.parse_args(argv)

    from evaluation import (figs_tmlr, freeze_results, lint_numbers, lint_terms, make_macros, paper_numbers,
                            phase2_summary, phase8_summary, phase9_summary, tables_tmlr, tmlr_build)

    def freeze():
        if args.no_freeze:
            print("    skipped (--no-freeze)")
            return 0
        freeze_results.freeze()
        return 0

    def macros():
        text, count = make_macros.build()
        make_macros.OUTPUT.write_text(text, encoding="utf-8")
        print(f"    {count} macros")
        missing = tmlr_build.pending_macros()
        print(f"    {len(missing)} result macros used by the manuscript are not yet defined")
        return 0

    def numbers_lint():
        paths = [q for q in tmlr_build.manuscript_files()]
        findings = lint_numbers.scan_files(paths) if paths else []
        for f in findings:
            print(f"    {f['file']}:{f['line']}: hand-typed number {f['number']!r}")
        print(f"    {len(paths)} files, {len(findings)} hand-typed numbers")
        return 1 if findings else 0

    def terms_lint():
        paths = tmlr_build.manuscript_files()
        problems = lint_terms.lint_files(paths)
        for p in problems:
            print(f"    {p}")
        return 1 if problems else 0

    def todos():
        found = tmlr_build.check_todos()
        print(f"    {len(found['todo'])} to-do markers, {len(found['verifycite'])} citations to verify")
        return 0

    steps = [("summary 2", phase2_summary.main), ("summary 8", phase8_summary.main),
             ("summary 9", phase9_summary.main), ("numbers", paper_numbers.main), ("freeze", freeze),
             ("macros", macros), ("figures", figs_tmlr.main), ("tables", tables_tmlr.main)]
    if not args.skip_lint:
        steps += [("lint numbers", numbers_lint), ("lint terms", terms_lint)]
    steps.append(("to-dos", todos))
    if args.latex:
        steps.append(("latex", tmlr_build.compile_latex))
    failed = [name for name, fn in steps if not _step(name, fn)]
    if failed:
        print("FAILED steps: " + ", ".join(failed))
        return 1
    print("all steps ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
