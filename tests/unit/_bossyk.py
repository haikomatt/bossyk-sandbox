"""Shared helpers for tests that touch the private bossyk checkout.

The policy YAMLs and the `judge` / `policy` modules live in the private bossyk
repo (BOSSYK_ROOT), which exists on dev machines but not in CI. Tests that only
need SOME policy bytes use `SYNTHETIC_POLICY` (a committed placeholder). Tests
of `build_default_policy_instrument` use `fake_bossyk` to stand in for the
private loader and judge, so the hash-of-loaded-bytes behaviour still runs in
CI. Only a test that needs the REAL loader is gated by
`requires_bossyk_checkout`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bossyk_sandbox.env import bossyk_root

requires_bossyk_checkout = pytest.mark.skipif(
    not bossyk_root().exists(),
    reason="needs the private bossyk checkout at BOSSYK_ROOT",
)

SYNTHETIC_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "policies" / "synthetic-support-v1.yaml"
)
SYNTHETIC_POLICY: bytes = SYNTHETIC_POLICY_PATH.read_bytes()


class FakePolicy:
    def __init__(self, policy_id: str) -> None:
        self.policy_id = policy_id


class FakeBossyk:
    """What the fake `policy` and `judge` modules record."""

    def __init__(self) -> None:
        self.loaded_paths: list[Path] = []
