"""Where results are written, and what provenance travels with them.

Every number that reaches the paper comes from a JSON file written through this module, and every
such file records the git commit, the configuration it was produced under, and a hash of that
configuration. Three properties follow, each of which a reviewer asked for:

* A result can be traced to the exact code and settings that produced it.
* A long job is resumable. One file per work unit (usually one subject), named by the unit, inside
  a directory named by the configuration hash. A unit whose file already exists is skipped, so an
  interrupted run resumes instead of restarting, and a changed configuration lands in a different
  directory instead of silently overwriting the old one.
* Results are append-only in practice. Nothing here deletes or rewrites a file from a previous
  configuration; `save_result` refuses to overwrite unless explicitly told to.

The `macros` key of a payload is special: `freeze_results` collects it into `results/index.json`
and `make_macros` turns it into the LaTeX macros the paper uses, so a scalar quoted in the text is
mechanically the one the analysis produced.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Overridable by environment so a run can write somewhere other than the checkout: a mounted volume
# on a rented node whose root filesystem is ephemeral, or a temporary directory in the tests. The
# default is inside the repository, which is what makes results reviewable alongside the code.
RESULTS = Path(os.environ.get("HORIZON_RESULTS_DIR") or (ROOT / "results"))
LOGS = Path(os.environ.get("HORIZON_LOGS_DIR") or (ROOT / "logs"))

__all__ = [
    "ROOT", "RESULTS", "LOGS", "git_commit", "config_hash", "analysis_dir", "unit_path",
    "unit_done", "save_result", "load_result", "load_units", "log", "env_summary",
]


@lru_cache(maxsize=1)
def git_commit() -> str:
    """The commit this code is at, with `-dirty` appended when the tree has uncommitted changes.

    A dirty marker matters more than it looks: a result produced from an edited working tree cannot
    be reproduced from the commit alone, and the paper should not claim otherwise.
    """
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=30)
        if rev.returncode != 0:
            return "no-git"
        commit = rev.stdout.strip()[:12]
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                text=True, timeout=60)
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else False
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.SubprocessError):
        return "no-git"


def _canonical(obj: Any) -> Any:
    """A JSON-serializable, order-independent view of a config, for hashing.

    Sets and dict keys are sorted so that two logically identical configurations hash the same.
    Floats are left alone: a config that differs in the 15th decimal place IS a different config.
    """
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_canonical(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def config_hash(config: dict, seed: int | None = None, length: int = 12) -> str:
    """Hash of (git commit, config, seed) -- the name of the directory results land in.

    The commit is part of it on purpose: the same configuration run against changed code is a
    different result and must not share a directory with the old one.
    """
    payload = {"commit": git_commit(), "config": _canonical(config), "seed": seed}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def analysis_dir(analysis_id: str, config: dict, seed: int | None = None) -> Path:
    """`results/<analysis_id>/<config_hash>/`, created if absent."""
    path = RESULTS / analysis_id / config_hash(config, seed)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_unit(unit: str) -> str:
    """A filesystem-safe unit name. Subject identifiers can contain separators and spaces."""
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(unit))
    return keep or "unit"


def unit_path(analysis_id: str, config: dict, unit: str, seed: int | None = None) -> Path:
    return analysis_dir(analysis_id, config, seed) / f"{_safe_unit(unit)}.json"


def unit_done(analysis_id: str, config: dict, unit: str, seed: int | None = None) -> bool:
    """Whether this work unit already has a result under this exact configuration.

    A truncated file (a job killed mid-write) counts as not done, so a resumed run repairs it
    rather than carrying a half-written result into the paper.
    """
    path = unit_path(analysis_id, config, unit, seed)
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except (json.JSONDecodeError, OSError):
        return False


def env_summary() -> dict:
    """The parts of the environment that can change a numerical result."""
    versions = {}
    for name in ("jax", "jaxlib", "diffrax", "optax", "numpy", "scipy", "statsmodels"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = "absent"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": versions}


def save_result(analysis_id: str, payload: dict, config: dict, unit: str = "result",
                seed: int | None = None, overwrite: bool = False) -> Path:
    """Write one work unit of one analysis, with its provenance, and return the path.

    The written document is `{"analysis_id", "unit", "commit", "config", "config_hash", "seed",
    "timestamp", "environment", "payload"}`. Reading code should take numbers from `payload` and
    read the rest as provenance.
    """
    path = unit_path(analysis_id, config, unit, seed)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Results are not overwritten: either pass overwrite=True "
            f"deliberately, or change the config so it lands in a new directory.")
    document = {
        "analysis_id": analysis_id,
        "unit": str(unit),
        "commit": git_commit(),
        "config": _canonical(config),
        "config_hash": config_hash(config, seed),
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": env_summary(),
        "payload": payload,
    }
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)   # atomic: a killed job never leaves a half-written result
    return path


def load_result(analysis_id: str, config: dict, unit: str = "result",
                seed: int | None = None) -> dict:
    return json.loads(unit_path(analysis_id, config, unit, seed).read_text(encoding="utf-8"))


def load_units(analysis_id: str, config: dict, seed: int | None = None) -> dict[str, dict]:
    """Every completed unit of one analysis configuration, keyed by unit name.

    Unreadable files are skipped and counted under the key `_unreadable`, never silently dropped.
    """
    out: dict[str, dict] = {}
    unreadable: list[str] = []
    directory = analysis_dir(analysis_id, config, seed)
    for path in sorted(directory.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            unreadable.append(path.name)
            continue
        # The runner writes config.json and <unit>.failed.json into the same directory; neither is a
        # result, and both lack a payload. Skipped by shape rather than by filename so a future
        # bookkeeping file does not have to be added to a list here.
        if "payload" not in doc:
            continue
        out[doc.get("unit", path.stem)] = doc
    if unreadable:
        out["_unreadable"] = {"files": unreadable}
    return out


def load_all(analysis_id: str) -> dict[str, dict]:
    """Every result of an analysis, grouped by config hash, regardless of the current commit.

    `config_hash` deliberately includes the git commit, so results are tied to the code that produced
    them. The side effect is that `load_units` stops finding a result set the moment anything is
    committed, which makes a summary unrunnable minutes after the run that produced it. This walks the
    directories instead of recomputing the hash, and the caller says which set it used.
    """
    out: dict[str, dict] = {}
    base = RESULTS / analysis_id
    if not base.is_dir():
        return out
    for directory in sorted(base.iterdir()):
        if not directory.is_dir():
            continue
        units: dict[str, dict] = {}
        for path in sorted(directory.glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if "payload" not in doc:
                continue
            units[doc.get("unit", path.stem)] = doc
        if units:
            out[directory.name] = units
    return out


def largest_result_set(analysis_id: str) -> tuple[str, dict]:
    """The config directory of an analysis holding the most completed units, and those units."""
    groups = load_all(analysis_id)
    if not groups:
        return "", {}
    name = max(groups, key=lambda k: len(groups[k]))
    return name, groups[name]


def log(analysis_id: str, message: str, echo: bool = False) -> None:
    """Append a timestamped line to `logs/<analysis_id>.log`.

    Long jobs print nothing until they finish, so this file is the only way to see that one is
    making progress; `echo=True` also writes to stdout for interactive runs.
    """
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{analysis_id}] {message}"
    with (LOGS / f"{analysis_id}.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if echo:
        print(line, flush=True)
