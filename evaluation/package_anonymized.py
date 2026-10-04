"""Build the anonymized code package: a ZIP of what a reviewer needs, nothing that identifies a person.

The package is built from an ALLOW-LIST of paths, not from "everything except": a new file anywhere in
the repository is excluded until someone decides it belongs. Then every text file that is going in is
scanned for identifying strings (names, e-mail addresses, machine names, user home directories, API-key
shapes), and the build FAILS, writing no archive, if any is found.

Identifying strings are read from `ANON_FORBIDDEN` (comma separated, case-insensitive) in addition to a
built-in list of generic shapes, so the author name and machine name can be supplied without being
written into this file, which is itself part of the package.

Run:  ANON_FORBIDDEN="surname,firstname,machinename" python -m evaluation.package_anonymized
      python -m evaluation.package_anonymized --check      # scan only, write nothing
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

from evaluation.results_io import ROOT

INCLUDE_DIRS = ("evaluation", "personalization", "simulation", "cohort_cache", "julia", "paper", "scripts",
                "REPORTS", "results")
INCLUDE_FILES = ("data_paths.py", "paper_config.py", "pytest.ini", "requirements-aistats.txt",
                 "PREREG.md", "PREREG_AMENDMENT_1.md", "PREREG_AMENDMENT_2.md",
                 "PREREG_AMENDMENT_3.md", "SPEC_EXPORT.md", "docs/POD.md")
INCLUDE_TESTS = ("test_prediction_cv.py", "test_phase7.py", "test_phase1.py", "test_phase2_modules.py", "test_lint_terms.py",
                 "test_runner.py", "test_results_pipeline.py", "test_cv_utils.py", "test_subject_loss.py",
                 "test_identifiability.py", "test_identifiability_toys.py", "test_observables.py",
                 "test_stats_utils.py", "test_determinism.py", "test_spec_and_cohort.py")
EXCLUDE_PARTS = {"__pycache__", ".git", ".venv", ".jax_cache", ".cache", "node_modules", "data",
                 "user_data", "logs", "sbi-logs"}
# Product code that lives beside the analyses in this repository and that no analysis imports.
EXCLUDE_FILES = ("personalization/auth.py", "personalization/user_store.py", "personalization/mailer.py",
                 "personalization/dexcom.py", "personalization/store.py", "personalization/time_machine.py")
EXCLUDE_SUFFIXES = (".pyc", ".claim", ".heartbeat", ".partial", ".tgz", ".zip", ".failed.json", ".log")
TEXT_SUFFIXES = {".py", ".md", ".tex", ".json", ".txt", ".toml", ".cfg", ".yaml", ".yml", ".sh", ".jl",
                 ".ini", ".csv", ".bib", ".gz"}

# Shapes that identify a person or a machine without naming anyone.
GENERIC = {
    "e-mail address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "windows home directory": re.compile(r"[A-Za-z]:\\+Users\\+[^\\\s\"']+", re.I),
    "unix home directory": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+"),
    "api key shape": re.compile(r"\b(?:AIza[0-9A-Za-z_-]{20,}|gsk_[0-9A-Za-z]{20,}|sk-[0-9A-Za-z]{20,})\b"),
    "hosted notebook name": re.compile(r"jupyter-[a-z0-9-]+", re.I),
}


def _forbidden_words() -> list[str]:
    return [w.strip().lower() for w in os.environ.get("ANON_FORBIDDEN", "").split(",") if w.strip()]


def candidate_files() -> list[Path]:
    """Every file that would go into the package, before scanning."""
    files: list[Path] = []
    for name in INCLUDE_DIRS:
        base = ROOT / name
        if base.is_dir():
            files.extend(p for p in base.rglob("*") if p.is_file())
    for name in INCLUDE_FILES:
        path = ROOT / name
        if path.is_file():
            files.append(path)
    for name in INCLUDE_TESTS:
        path = ROOT / "tests" / name
        if path.is_file():
            files.append(path)
    keep = []
    for path in files:
        parts = set(path.relative_to(ROOT).parts)
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if parts & EXCLUDE_PARTS or path.name.endswith(EXCLUDE_SUFFIXES) or rel in EXCLUDE_FILES:
            continue
        keep.append(path)
    return sorted(set(keep))


def _read_text(path: Path) -> str | None:
    if path.suffix not in TEXT_SUFFIXES:
        return None
    if path.suffix == ".gz":
        import gzip
        try:
            return gzip.open(path, "rt", encoding="utf-8", errors="replace").read()
        except OSError:
            return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def scan(paths: list[Path]) -> list[str]:
    words = _forbidden_words()
    findings = []
    for path in paths:
        text = _read_text(path)
        if text is None:
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        lowered = text.lower()
        for word in words:
            if word in lowered:
                line = lowered[:lowered.index(word)].count("\n") + 1
                findings.append(f"{rel}:{line}: contains forbidden word {word!r}")
        for label, pattern in GENERIC.items():
            match = pattern.search(text)
            if match:
                line = text[:match.start()].count("\n") + 1
                findings.append(f"{rel}:{line}: {label}: {match.group(0)[:60]!r}")
        if rel.lower() in (w + ".md" for w in words):
            findings.append(f"{rel}: file name is forbidden")
    return findings


README = """# Code and results for the revision of the identifiability analysis

This archive reproduces the analyses of the paper from stored data. It contains no raw participant data:
the parsed cohorts are in `cohort_cache/` (checksummed in `cohort_cache/manifest.json`).

    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements-aistats.txt      # install the CPU build of PyTorch first on a node
    python -m evaluation.cohort_cache verify
    python -m pytest tests -q
    python -m evaluation.make_paper_assets       # rebuild macros, figures and tables from results/

Layout: `evaluation/` analyses and the resumable runner, `personalization/` objectives and fitting,
`simulation/` the differentiable engine and its observables, `julia/` the structural-identifiability
runs with their raw output, `results/` every stored result with provenance, `paper/` generated assets,
`PREREG*.md` the analysis plan and its amendments, `REPORTS/` phase reports.
"""


def build(out: Path, check_only: bool = False) -> int:
    paths = candidate_files()
    findings = scan(paths)
    print(f"{len(paths)} files considered, {len(findings)} findings")
    for f in findings[:50]:
        print("  " + f)
    if len(findings) > 50:
        print(f"  ... and {len(findings) - 50} more")
    if findings:
        print("NOT WRITTEN: remove or rewrite the files above, then run again.")
        return 1
    if check_only:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)).replace("\\", "/"),
                                   date_time=(2026, 1, 1, 0, 0, 0))   # no timestamps of ours
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
        archive.writestr(zipfile.ZipInfo("README.md", date_time=(2026, 1, 1, 0, 0, 0)), README)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT.parent / "anonymous_code.zip")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    return build(args.output, args.check)


if __name__ == "__main__":
    sys.exit(main())
