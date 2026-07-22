from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# src/bossyk_sandbox/env.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = REPO_ROOT / ".env"


def load_project_env(env_path: Path = ENV_PATH) -> None:
    """Load the project `.env` into `os.environ` for CLI / billable-run
    entrypoints, WITHOUT overriding variables already set in the shell
    (`override=False`) -- an exported value wins, `.env` only fills gaps, so a
    run's config never silently changes depending on load order. A missing
    file is a no-op (fresh checkouts / CI may have none).

    Only script `main()`s call this. Library and test code stay hermetic --
    they never read `.env`, so tests don't depend on developer secrets."""
    load_dotenv(env_path, override=False)
