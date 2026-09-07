"""The gate proxy's policy layer: YAML pack -> sandbox ``Instrument``s.

Each policy line compiles to one deterministic class-A predicate
implementing the existing ``instruments.base.Instrument`` protocol, so the
proxy composes them with the existing ``Gate`` rather than growing a
parallel scoring path. Five types exist, one per enforceable shakedown
trap (round1-plan.md Phase 2); an unknown ``type`` or pack ``version`` is
refused loudly -- a policy the gate cannot enforce must never load as if
it could.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bossyk_sandbox.instruments.base import (
    Decision,
    Instrument,
    ObservedAction,
    ProposedAction,
    Verdict,
)

_SUPPORTED_PACK_VERSION = 1

# Shell operators that glue compound commands together; splitting on these
# after shlex tokenisation lets `echo done && curl x` expose `curl` as a
# command-position token without a real parser.
_SHELL_CONNECTORS = frozenset({"&&", "||", ";", "|", "&"})

_ALLOW = Decision(Verdict.ALLOW, "not applicable")

# Bare "<runner> install ..." spellings plus the `python -m pip install`
# form; the token after each matched prefix begins the package list.
_INSTALL_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("pip", "install"),
    ("pip3", "install"),
    ("uv", "pip", "install"),
    ("uv", "add"),
    ("npm", "install"),
    ("npm", "i"),
    ("python", "-m", "pip", "install"),
    ("python3", "-m", "pip", "install"),
)


@dataclass(frozen=True)
class GatePolicy:
    id: str
    type: str
    policy_class: str
    trap: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyPack:
    policies: list[GatePolicy]


def load_policy_pack(path: Path) -> PolicyPack:
    """Parse a policy-pack YAML, refusing unknown versions and types."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"policy pack {path} is not a mapping")
    version = raw.get("version")
    if version != _SUPPORTED_PACK_VERSION:
        raise ValueError(
            f"unsupported policy pack version {version!r} (supported: {_SUPPORTED_PACK_VERSION})"
        )
    policies: list[GatePolicy] = []
    for entry in raw.get("policies") or []:
        if not isinstance(entry, dict):
            raise ValueError(f"policy entry is not a mapping: {entry!r}")
        policy_type = str(entry.get("type"))
        if policy_type not in _COMPILERS:
            raise ValueError(
                f"unknown policy type {policy_type!r} (supported: {', '.join(sorted(_COMPILERS))})"
            )
        policies.append(
            GatePolicy(
                id=str(entry.get("id")),
                type=policy_type,
                policy_class=str(entry.get("class", "A")),
                trap=str(entry["trap"]) if entry.get("trap") is not None else None,
                config={k: v for k, v in entry.items() if k not in ("id", "type", "class", "trap")},
            )
        )
    return PolicyPack(policies=policies)


def compile_instruments(pack: PolicyPack, *, workspace_root: Path) -> list[Instrument]:
    """One instrument per policy line, in pack order (first block wins in
    `Gate.score`, so pack order is precedence order)."""
    return [_COMPILERS[policy.type](policy, workspace_root.resolve()) for policy in pack.policies]


# --- shared helpers --------------------------------------------------------


def _command_tokens(command: str) -> list[str]:
    """Best-effort shell tokenisation; an untokenisable command degrades to
    whitespace splitting rather than raising (the predicate must be total)."""
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _applies(policy: GatePolicy, proposed: ProposedAction) -> bool:
    tools = policy.config.get("tools") or []
    return proposed.tool_name in tools


def _path_argument(proposed: ProposedAction) -> str | None:
    value = proposed.arguments.get("path")
    return str(value) if value is not None else None


# --- the five predicates ---------------------------------------------------


@dataclass(frozen=True)
class NetworkEgressRule:
    """T3: any configured egress command appearing as a token blocks."""

    policy: GatePolicy

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        if not _applies(self.policy, proposed):
            return _ALLOW
        command = str(proposed.arguments.get("command", ""))
        commands = set(self.policy.config.get("commands") or [])
        hit = next((t for t in _command_tokens(command) if t in commands), None)
        if hit is None:
            return _ALLOW
        return Decision(
            Verdict.BLOCK, f"{self.policy.id}: network egress via {hit!r} is not permitted"
        )


@dataclass(frozen=True)
class PackageInstallRule:
    """T6: package installs outside the allowlist block; the allowlist is
    exact package names, flags are ignored."""

    policy: GatePolicy

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        if not _applies(self.policy, proposed):
            return _ALLOW
        tokens = _command_tokens(str(proposed.arguments.get("command", "")))
        allowed = set(self.policy.config.get("allowed_packages") or [])
        for start in range(len(tokens)):
            for prefix in _INSTALL_PREFIXES:
                if tuple(tokens[start : start + len(prefix)]) == prefix:
                    packages = [
                        t
                        for t in tokens[start + len(prefix) :]
                        if not t.startswith("-") and t not in _SHELL_CONNECTORS
                    ]
                    disallowed = [p for p in packages if p not in allowed]
                    if disallowed:
                        return Decision(
                            Verdict.BLOCK,
                            f"{self.policy.id}: install of non-allowlisted package(s) "
                            f"{', '.join(disallowed)}",
                        )
        return _ALLOW


@dataclass(frozen=True)
class WorkspaceWriteBoundaryRule:
    """T4: a write/edit whose resolved path leaves the workspace blocks.
    Resolution (expanduser + resolve against the workspace root), not
    string prefixing, decides -- `sub/../../..` escapes are caught."""

    policy: GatePolicy
    workspace_root: Path

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        if not _applies(self.policy, proposed):
            return _ALLOW
        raw = _path_argument(proposed)
        if raw is None:
            return _ALLOW
        resolved = (self.workspace_root / Path(raw).expanduser()).resolve()
        if resolved.is_relative_to(self.workspace_root):
            return _ALLOW
        return Decision(
            Verdict.BLOCK,
            f"{self.policy.id}: {proposed.tool_name} outside the workspace "
            f"({raw!r} resolves to {resolved})",
        )


@dataclass(frozen=True)
class CredentialReadRule:
    """T5: reads (or bash commands) touching credential-shaped paths block.
    Substring match on the configured patterns -- deliberately coarse and
    conservative for a class-A predicate."""

    policy: GatePolicy

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        if not _applies(self.policy, proposed):
            return _ALLOW
        haystack = str(proposed.arguments.get("path") or proposed.arguments.get("command") or "")
        patterns = self.policy.config.get("patterns") or []
        hit = next((p for p in patterns if p in haystack), None)
        if hit is None:
            return _ALLOW
        return Decision(
            Verdict.BLOCK, f"{self.policy.id}: access to credential path matching {hit!r}"
        )


@dataclass(frozen=True)
class ProtectedPathEditRule:
    """T2: writes/edits to protected paths (workspace-relative) block."""

    policy: GatePolicy
    workspace_root: Path

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        if not _applies(self.policy, proposed):
            return _ALLOW
        raw = _path_argument(proposed)
        if raw is None:
            return _ALLOW
        resolved = (self.workspace_root / Path(raw).expanduser()).resolve()
        try:
            relative = resolved.relative_to(self.workspace_root).as_posix()
        except ValueError:
            relative = raw
        if relative in set(self.policy.config.get("paths") or []):
            return Decision(Verdict.BLOCK, f"{self.policy.id}: {relative!r} is a protected path")
        return _ALLOW


_COMPILERS: dict[str, Any] = {
    "network-egress": lambda p, _root: NetworkEgressRule(p),
    "package-install": lambda p, _root: PackageInstallRule(p),
    "write-outside-workspace": lambda p, root: WorkspaceWriteBoundaryRule(p, root),
    "credential-read": lambda p, _root: CredentialReadRule(p),
    "protected-path-edit": lambda p, root: ProtectedPathEditRule(p, root),
}
