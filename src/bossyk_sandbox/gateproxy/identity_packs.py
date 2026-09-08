"""Identity -> policy pack mapping (round-2 Phase 3).

One gate, several packs. The mapping file:

```yaml
version: 1
default: deny-pack.yaml            # required: unknown/absent identity runs under this
packs:
  spiffe://example.org/coding-agent: coding-pack.yaml
  spiffe://example.org/research-agent: research-pack.yaml
```

Pack paths are relative to the mapping file. Every pack is loaded,
compiled and hashed once at startup -- a missing file, an unparseable
pack or a malformed identity key fails the gate at start, not at the
first request from that caller -- so a resolved pack carries the same
`pack_sha256` provenance the single-pack gate stamps today. The
single-pack gate is the degenerate mapping (`IdentityPacks.single`):
one code path serves both.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bossyk_sandbox.gateproxy.identity import InvalidSpiffeId, parse_spiffe_id
from bossyk_sandbox.gateproxy.policies import PolicyPack, compile_instruments, load_policy_pack
from bossyk_sandbox.instruments.base import Instrument

_SUPPORTED_MAPPING_VERSION = 1


@dataclass(frozen=True)
class ResolvedPack:
    pack: PolicyPack
    instruments: list[Instrument]
    pack_path: Path
    pack_sha256: str
    matched: bool  # False when this is the default standing in for an unknown caller


@dataclass(frozen=True)
class IdentityPacks:
    default: ResolvedPack
    by_identity: dict[str, ResolvedPack]

    @classmethod
    def single(cls, pack_path: Path, *, workspace_root: Path) -> IdentityPacks:
        """A gate with one pack for every caller (the pre-Phase-3 gate)."""
        return cls(default=_load(pack_path, workspace_root, matched=True), by_identity={})

    @property
    def identity_aware(self) -> bool:
        return bool(self.by_identity)

    def for_identity(self, spiffe_id: str | None) -> ResolvedPack:
        if spiffe_id is not None and spiffe_id in self.by_identity:
            return self.by_identity[spiffe_id]
        return self.default


def _load(pack_path: Path, workspace_root: Path, *, matched: bool) -> ResolvedPack:
    pack = load_policy_pack(pack_path)
    return ResolvedPack(
        pack=pack,
        instruments=compile_instruments(pack, workspace_root=workspace_root),
        pack_path=pack_path,
        # The file's own bytes, exactly as the single-pack gate hashes them.
        pack_sha256=hashlib.sha256(pack_path.read_bytes()).hexdigest(),
        matched=matched,
    )


def load_identity_packs(path: Path, *, workspace_root: Path) -> IdentityPacks:
    raw: Any = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"identity-pack mapping {path} is not a mapping")
    version = raw.get("version")
    if version != _SUPPORTED_MAPPING_VERSION:
        raise ValueError(
            f"unsupported identity-pack mapping version {version!r} "
            f"(supported: {_SUPPORTED_MAPPING_VERSION})"
        )
    if not raw.get("default"):
        raise ValueError(
            f"identity-pack mapping {path} declares no default pack; "
            "an unknown caller must never run unpoliced"
        )
    base = Path(path).resolve().parent
    default = _load(base / str(raw["default"]), workspace_root, matched=False)
    by_identity: dict[str, ResolvedPack] = {}
    for key, pack_file in (raw.get("packs") or {}).items():
        try:
            spiffe_id = parse_spiffe_id(str(key))
        except InvalidSpiffeId as exc:
            raise ValueError(f"identity-pack mapping key {key!r} is not a SPIFFE ID") from exc
        by_identity[spiffe_id] = _load(base / str(pack_file), workspace_root, matched=True)
    return IdentityPacks(default=default, by_identity=by_identity)
