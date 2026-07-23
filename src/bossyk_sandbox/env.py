from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# src/bossyk_sandbox/env.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = REPO_ROOT / ".env"

# Default location of the sibling `bossyk` checkout this sandbox reuses
# (instruments/policy.py's PolicyAwareJudge, its `data/policies/` YAML) --
# overridable via BOSSYK_ROOT so a checkout elsewhere on disk doesn't
# require editing source. Resolved at call time (not a module-level
# constant) so tests can override it via monkeypatch.setenv/delenv.
BOSSYK_ROOT_ENV_VAR = "BOSSYK_ROOT"
DEFAULT_BOSSYK_ROOT = "~/Projects/bossyk"


def bossyk_root() -> Path:
    """The single knob for where the `bossyk` checkout lives on disk --
    `BOSSYK_ROOT` env var if set, else `~/Projects/bossyk`. Consumed by
    `instruments/policy.py` (policy YAML path, bossyk `src/` for sys.path)
    and `domains.py` (per-domain default policy paths)."""
    return Path(os.environ.get(BOSSYK_ROOT_ENV_VAR, DEFAULT_BOSSYK_ROOT)).expanduser()


def load_project_env(env_path: Path = ENV_PATH) -> None:
    """Load the project `.env` into `os.environ` for CLI / billable-run
    entrypoints, WITHOUT overriding variables already set in the shell
    (`override=False`) -- an exported value wins, `.env` only fills gaps, so a
    run's config never silently changes depending on load order. A missing
    file is a no-op (fresh checkouts / CI may have none).

    Only script `main()`s call this. Library and test code stay hermetic --
    they never read `.env`, so tests don't depend on developer secrets."""
    load_dotenv(env_path, override=False)
