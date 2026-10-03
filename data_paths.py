"""Where the datasets live, for the paper branch.

Three loaders each resolved the dataset root themselves: `evaluation/cgmacros.py` by walking up
`parents[2] / "data"`, `evaluation/hall_loader.py` and `evaluation/shanghai_loader.py` by an
absolute path typed into the source. When the datasets moved to a folder shared across research
projects, the relative one broke, and the absolute ones will break at the next reorganization.

Resolution order, most explicit first:

  1. `HORIZON_DATA_DIR` — an explicit override, for CI or another machine.
  2. `<paperML>/data` — a copy inside the branch, if populated. Checked before the shared folder
     so that a deliberately pinned dataset is not silently overridden.
  3. The nearest `datasets/` directory walking up from this file. Found by search rather than by
     counting `parents[n]`, so moving the project does not break it.

Raises with every location named when none of them exist: a bare
`FileNotFoundError: .../cgmacros/1.0.0/CGMacros_dateshifted365.zip` does not say which root was
expected.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# Any one of these marks a candidate as the real dataset root rather than an empty directory of
# the right name. Not every checkout holds every dataset.
_MARKERS = ("cgmacros", "hall2018", "diabetes_datasets", "cdc")


def _looks_populated(path: Path) -> bool:
    return path.is_dir() and any((path / m).exists() for m in _MARKERS)


def _shared_candidates() -> list[Path]:
    return [parent / "datasets" for parent in _ROOT.parents]


@lru_cache(maxsize=1)
def data_root() -> Path:
    override = os.environ.get("HORIZON_DATA_DIR", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"HORIZON_DATA_DIR points at {path}, which does not exist.")
        return path

    local = _ROOT / "data"
    if _looks_populated(local):
        return local

    for candidate in _shared_candidates():
        if _looks_populated(candidate):
            return candidate

    searched = "\n  ".join(str(p) for p in [local, *_shared_candidates()[:3]])
    raise FileNotFoundError(
        "Could not find the research datasets. Set HORIZON_DATA_DIR, or put them in one of:\n"
        f"  {searched}")


def dataset(*parts: str) -> Path:
    """A path inside the dataset root. Existence is not checked here: each loader reports its own
    dataset as missing, which is a more useful message than a generic one from this module."""
    try:
        root = data_root()
    except FileNotFoundError:
        # No raw datasets on this machine, which is the normal state of a compute node: the parsed
        # cohorts are committed in `cohort_cache/`. Several modules build their file paths at import
        # time, so raising here would make them unimportable even for code that never opens a raw
        # file (the amortized estimators import such a module). Return a path that cannot exist
        # instead; a loader that really needs the file fails on open with a message naming it.
        root = _ROOT / "data" / "__datasets_not_found__"
    return root.joinpath(*parts)


def available() -> bool:
    """Whether a dataset root was found at all. For tests that skip rather than fail without one."""
    try:
        data_root()
        return True
    except FileNotFoundError:
        return False
