"""Red-phase contract for identity -> policy pack mapping (round-2 Phase 3).

One gate, several packs: a mapping file binds SPIFFE IDs to pack files.
An unknown identity, or no identity at all, gets the mapping's declared
`default` pack, never silently unpoliced; a mapping without a default is
refused at load. Each pack is loaded and hashed once, at startup, so a
resolved pack carries the same provenance (`pack_sha256`) the single-pack
gate stamps today.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bossyk_sandbox.gateproxy.identity_packs import IdentityPacks, load_identity_packs
from tests.unit.gateproxy.conftest import POLICY_PACK_YAML

CODING_PACK_YAML = """\
version: 1
policies:
  - id: no-network-egress
    trap: T3
    class: A
    type: network-egress
    tools: [bash]
    commands: [curl, wget]
"""

RESEARCH_PACK_YAML = """\
version: 1
policies:
  - id: no-package-installs
    trap: T6
    class: A
    type: package-install
    tools: [bash]
    allowed_packages: []
"""

DENY_PACK_YAML = """\
version: 1
policies:
  - id: unknown-caller-no-shell
    trap: T3
    class: A
    type: network-egress
    tools: [bash]
    commands: [curl, wget, ssh, pip, python, sh, bash]
"""


@pytest.fixture()
def packs_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "packs"
    directory.mkdir()
    (directory / "coding-pack.yaml").write_text(CODING_PACK_YAML)
    (directory / "research-pack.yaml").write_text(RESEARCH_PACK_YAML)
    (directory / "deny-pack.yaml").write_text(DENY_PACK_YAML)
    (directory / "identity-packs.yaml").write_text(
        "version: 1\n"
        "default: deny-pack.yaml\n"
        "packs:\n"
        "  spiffe://example.org/coding-agent: coding-pack.yaml\n"
        "  spiffe://example.org/research-agent: research-pack.yaml\n"
    )
    return directory


@pytest.fixture()
def identity_packs(packs_dir: Path, workspace_root: Path) -> IdentityPacks:
    return load_identity_packs(packs_dir / "identity-packs.yaml", workspace_root=workspace_root)


class TestLoading:
    def test_paths_resolve_relative_to_the_mapping_file(
        self, identity_packs: IdentityPacks, packs_dir: Path
    ) -> None:
        resolved = identity_packs.for_identity("spiffe://example.org/coding-agent")
        assert resolved.pack_path == packs_dir / "coding-pack.yaml"
        assert resolved.pack_sha256 == hashlib.sha256(CODING_PACK_YAML.encode()).hexdigest()
        assert [p.id for p in resolved.pack.policies] == ["no-network-egress"]
        assert len(resolved.instruments) == 1

    def test_missing_default_is_refused(self, packs_dir: Path, workspace_root: Path) -> None:
        path = packs_dir / "no-default.yaml"
        path.write_text("version: 1\npacks:\n  spiffe://example.org/a: coding-pack.yaml\n")
        with pytest.raises(ValueError, match="default"):
            load_identity_packs(path, workspace_root=workspace_root)

    def test_missing_pack_file_is_refused_at_load(
        self, packs_dir: Path, workspace_root: Path
    ) -> None:
        path = packs_dir / "dangling.yaml"
        path.write_text("version: 1\ndefault: nope.yaml\npacks: {}\n")
        with pytest.raises(FileNotFoundError):
            load_identity_packs(path, workspace_root=workspace_root)

    def test_malformed_identity_key_is_refused_at_load(
        self, packs_dir: Path, workspace_root: Path
    ) -> None:
        path = packs_dir / "bad-key.yaml"
        path.write_text(
            "version: 1\ndefault: deny-pack.yaml\npacks:\n  not-a-spiffe-id: coding-pack.yaml\n"
        )
        with pytest.raises(ValueError, match="not-a-spiffe-id"):
            load_identity_packs(path, workspace_root=workspace_root)

    def test_unknown_version_is_refused(self, packs_dir: Path, workspace_root: Path) -> None:
        path = packs_dir / "v9.yaml"
        path.write_text("version: 9\ndefault: deny-pack.yaml\npacks: {}\n")
        with pytest.raises(ValueError, match="version"):
            load_identity_packs(path, workspace_root=workspace_root)

    def test_bad_pack_content_is_refused_at_load(
        self, packs_dir: Path, workspace_root: Path
    ) -> None:
        (packs_dir / "broken.yaml").write_text("version: 1\npolicies:\n  - id: x\n    type: nope\n")
        path = packs_dir / "with-broken.yaml"
        path.write_text("version: 1\ndefault: broken.yaml\npacks: {}\n")
        with pytest.raises(ValueError, match="nope"):
            load_identity_packs(path, workspace_root=workspace_root)


class TestResolution:
    def test_known_identities_get_their_own_packs(self, identity_packs: IdentityPacks) -> None:
        coding = identity_packs.for_identity("spiffe://example.org/coding-agent")
        research = identity_packs.for_identity("spiffe://example.org/research-agent")
        assert coding.matched and research.matched
        assert coding.pack_sha256 != research.pack_sha256

    def test_unknown_identity_gets_the_default_pack(self, identity_packs: IdentityPacks) -> None:
        resolved = identity_packs.for_identity("spiffe://example.org/stranger")
        assert resolved.matched is False
        assert [p.id for p in resolved.pack.policies] == ["unknown-caller-no-shell"]

    def test_no_identity_gets_the_default_pack(self, identity_packs: IdentityPacks) -> None:
        resolved = identity_packs.for_identity(None)
        assert resolved.matched is False
        assert resolved.pack_path.name == "deny-pack.yaml"

    def test_single_pack_gate_is_the_degenerate_mapping(
        self, policy_pack_path: Path, workspace_root: Path
    ) -> None:
        """A gate started with just --policy-pack behaves as a mapping whose
        only pack is the default, so one code path serves both."""
        packs = IdentityPacks.single(policy_pack_path, workspace_root=workspace_root)
        resolved = packs.for_identity("spiffe://example.org/anyone")
        assert resolved.pack_path == policy_pack_path
        assert resolved.pack_sha256 == hashlib.sha256(POLICY_PACK_YAML.encode()).hexdigest()
        assert len(resolved.pack.policies) == 5
