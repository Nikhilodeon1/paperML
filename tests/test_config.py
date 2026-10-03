"""Tests for the .env config loader."""

from __future__ import annotations

import importlib


def _fresh_config():
    import orchestration.config as c
    importlib.reload(c)
    c.load_env.cache_clear()
    return c


def test_env_file_loads_without_overriding_real_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=from_dotenv\nGEMINI_MODEL=gemini-2.5-flash\n")
    c = _fresh_config()
    monkeypatch.setattr(c, "_ENV_PATH", env)
    c.load_env.cache_clear()

    # Clear any values a real ml/.env already seeded into the process environment.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert c.gemini_api_key() == "from_dotenv"
    assert c.gemini_model("gemini-2.0-flash") == "gemini-2.5-flash"


def test_real_env_wins_over_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=from_dotenv\n")
    c = _fresh_config()
    monkeypatch.setattr(c, "_ENV_PATH", env)
    c.load_env.cache_clear()

    monkeypatch.setenv("GEMINI_API_KEY", "from_real_env")
    assert c.gemini_api_key() == "from_real_env"


def test_missing_env_file_is_noop(tmp_path, monkeypatch):
    c = _fresh_config()
    monkeypatch.setattr(c, "_ENV_PATH", tmp_path / "nope.env")
    c.load_env.cache_clear()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert c.gemini_api_key() is None
