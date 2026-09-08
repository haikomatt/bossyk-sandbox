"""Identity -> policy pack mapping (round-2 Phase 3). RED-phase skeleton:
importable names, no behaviour -- the implementation lands in Green."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bossyk_sandbox.gateproxy.policies import PolicyPack
from bossyk_sandbox.instruments.base import Instrument


@dataclass(frozen=True)
class ResolvedPack:
    pack: PolicyPack
    instruments: list[Instrument]
    pack_path: Path
    pack_sha256: str
    matched: bool


@dataclass(frozen=True)
class IdentityPacks:
    default: ResolvedPack
    by_identity: dict[str, ResolvedPack]

    @classmethod
    def single(cls, pack_path: Path, *, workspace_root: Path) -> IdentityPacks:
        raise NotImplementedError

    def for_identity(self, spiffe_id: str | None) -> ResolvedPack:
        raise NotImplementedError


def load_identity_packs(path: Path, *, workspace_root: Path) -> IdentityPacks:
    raise NotImplementedError
