"""Tiny .env loader (no dependency) for API keys and config.

Reads `ml/.env` (project root) once and copies any KEY=VALUE lines into the process
environment WITHOUT overwriting variables already set in the real environment. This
gives you one place to drop secrets like GEMINI_API_KEY. `.env` is gitignored.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


@lru_cache(maxsize=1)
def load_env() -> None:
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)  # real env wins over .env


def gemini_api_key() -> str | None:
    load_env()
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def gemini_model(default: str) -> str:
    load_env()
    return os.getenv("GEMINI_MODEL", default)


def groq_api_key() -> str | None:
    load_env()
    return os.getenv("GROQ_API_KEY")


def groq_model(default: str) -> str:
    load_env()
    return os.getenv("GROQ_MODEL", default)
