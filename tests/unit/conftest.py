"""Fixtures for tests of the private-bossyk seam (see `_bossyk.py`)."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.unit._bossyk import FakeBossyk, FakePolicy


@pytest.fixture
def fake_bossyk(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeBossyk]:
    """Stand-ins for bossyk's `policy.load_policy` and `judge.PolicyAwareJudge`.
    `load_policy` reads the path it is GIVEN (like the real one), so a TOCTOU
    bug in the caller would still show; no network, no API key."""
    record = FakeBossyk()

    def load_policy(path: str | Path) -> FakePolicy:
        record.loaded_paths.append(Path(path))
        data = yaml.safe_load(Path(path).read_text())
        return FakePolicy(str(data["policy_id"]))

    class PolicyAwareJudge:
        def __init__(self, policy: FakePolicy, api_key: str | None = None) -> None:
            self._policy = policy

    policy_module: Any = types.ModuleType("policy")
    policy_module.load_policy = load_policy
    judge_module: Any = types.ModuleType("judge")
    judge_module.PolicyAwareJudge = PolicyAwareJudge
    monkeypatch.setitem(sys.modules, "policy", policy_module)
    monkeypatch.setitem(sys.modules, "judge", judge_module)
    yield record
