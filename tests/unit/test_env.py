from __future__ import annotations

import os
from pathlib import Path

import pytest

from bossyk_sandbox.env import bossyk_root, load_project_env


def test_load_project_env_does_not_override_a_shell_exported_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole point of override=False: a key exported in the shell wins,
    # .env only fills gaps -- so a run's config never silently changes under
    # you depending on load order.
    env_file = tmp_path / ".env"
    env_file.write_text("SHARED_KEY=from_dotenv\nONLY_IN_FILE=file_value\n")
    monkeypatch.setenv("SHARED_KEY", "from_shell")
    monkeypatch.delenv("ONLY_IN_FILE", raising=False)

    load_project_env(env_file)

    assert os.environ["SHARED_KEY"] == "from_shell"
    assert os.environ["ONLY_IN_FILE"] == "file_value"


def test_load_project_env_with_a_missing_file_is_a_noop(tmp_path: Path) -> None:
    # Fresh checkouts / CI may have no .env -- loading must not raise.
    load_project_env(tmp_path / "does-not-exist.env")


def test_bossyk_root_defaults_to_projects_bossyk_under_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BOSSYK_ROOT", raising=False)

    assert bossyk_root() == Path("~/Projects/bossyk").expanduser()


def test_bossyk_root_honors_bossyk_root_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_root = tmp_path / "custom-bossyk-checkout"
    monkeypatch.setenv("BOSSYK_ROOT", str(custom_root))

    assert bossyk_root() == custom_root


def test_bossyk_root_expands_a_tilde_prefixed_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOSSYK_ROOT", "~/somewhere/bossyk")

    assert bossyk_root() == Path("~/somewhere/bossyk").expanduser()
    assert "~" not in str(bossyk_root())
