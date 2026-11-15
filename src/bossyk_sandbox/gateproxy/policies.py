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
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import yaml

from bossyk_sandbox.instruments.base import (
    Decision,
    Instrument,
    ObservedAction,
    ProposedAction,
    Verdict,
)

_SUPPORTED_PACK_VERSION = 1

# `on_match` is what a matching policy line asks the gate for; `on_hold`
# is how a HOLD resolves when NO approver is configured. Both default to
# block: a pack must opt in to holding. A configured approver that does not
# answer is always a block, whatever `on_hold` says (decision D2).
_ON_MATCH_VALUES = frozenset({"block", "hold"})
_ON_HOLD_VALUES = frozenset({"block", "allow"})
_POLICY_META_KEYS = frozenset({"id", "type", "class", "trap", "on_match", "on_hold"})

# Shell operators that glue compound commands together; splitting on these
# after shlex tokenisation lets `echo done && curl x` expose `curl` as a
# command-position token without a real parser.
_SHELL_CONNECTORS = frozenset({"&&", "||", ";", "|", "&"})


def _not_applicable(source: str) -> Decision:
    return Decision(Verdict.ALLOW, "not applicable", source=source)


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
    on_match: str = "block"
    on_hold: str = "block"


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
                config={k: v for k, v in entry.items() if k not in _POLICY_META_KEYS},
                on_match=_choice(entry, "on_match", _ON_MATCH_VALUES),
                on_hold=_choice(entry, "on_hold", _ON_HOLD_VALUES),
            )
        )
    return PolicyPack(policies=policies)


def _choice(entry: dict[str, Any], key: str, allowed: frozenset[str]) -> str:
    value = str(entry.get(key, "block"))
    if value not in allowed:
        raise ValueError(
            f"policy {entry.get('id')!r}: {key} {value!r} not one of {', '.join(sorted(allowed))}"
        )
    return value


def compile_instruments(pack: PolicyPack, *, workspace_root: Path) -> list[Instrument]:
    """One instrument per policy line, in pack order (first block wins in
    `Gate.score`, so pack order is precedence order)."""
    root = workspace_root.resolve()
    return [
        _with_on_match(policy, _COMPILERS[policy.type].compile(policy, root))
        for policy in pack.policies
    ]


def _with_on_match(policy: GatePolicy, rule: Instrument) -> Instrument:
    return HoldOnMatch(rule) if policy.on_match == "hold" else rule


@dataclass(frozen=True)
class HoldOnMatch:
    """Wraps any predicate so its BLOCK becomes a HOLD with the same reason
    and source (so the proxy still attributes it to its policy). The
    predicate itself stays unchanged; only what a match asks the gate for
    differs."""

    inner: Instrument

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        decision = self.inner.score(proposed, history)
        if decision.verdict is Verdict.BLOCK:
            return Decision(Verdict.HOLD, decision.reason, source=decision.source)
        return decision


# --- the rule bases --------------------------------------------------------


def _rule_source(rule_class: type, policy: GatePolicy) -> str:
    """`"<RuleClass>:<policy.id>"`: the one place a policy rule's source is
    spelled, used by the rule itself and by `policy_source`."""
    return f"{rule_class.__name__}:{policy.id}"


@dataclass(frozen=True)
class _PolicyRule(ABC):
    """One compiled policy line. Every decision it returns carries
    `source`, which names this rule class and its policy."""

    policy: GatePolicy

    @property
    def source(self) -> str:
        return _rule_source(type(self), self.policy)

    @classmethod
    def compile(cls, policy: GatePolicy, workspace_root: Path) -> Self:
        return cls(policy)

    @abstractmethod
    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision: ...


@dataclass(frozen=True)
class _WorkspaceRule(_PolicyRule):
    """A policy rule that resolves paths against the workspace root."""

    workspace_root: Path

    @classmethod
    def compile(cls, policy: GatePolicy, workspace_root: Path) -> Self:
        return cls(policy, workspace_root)


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


def _non_string_governed(
    rule: _PolicyRule, proposed: ProposedAction, keys: tuple[str, ...]
) -> Decision | None:
    """BLOCK when a governed argument is present but not a string: a value
    the predicate cannot read must not be read as "no match"."""
    for key in keys:
        if key in proposed.arguments and not isinstance(proposed.arguments[key], str):
            return Decision(
                Verdict.BLOCK,
                f"{rule.policy.id}: governed argument {key!r} is not a string (fail closed)",
                source=rule.source,
            )
    return None


def _path_argument(proposed: ProposedAction) -> str | None:
    value = proposed.arguments.get("path")
    return str(value) if value is not None else None


# --- the five predicates ---------------------------------------------------


@dataclass(frozen=True)
class NetworkEgressRule(_PolicyRule):
    """T3: any configured egress command appearing as a token blocks."""

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        if not _applies(self.policy, proposed):
            return _not_applicable(self.source)
        if refused := _non_string_governed(self, proposed, ("command",)):
            return refused
        command = str(proposed.arguments.get("command", ""))
        commands = set(self.policy.config.get("commands") or [])
        hit = next((t for t in _command_tokens(command) if t in commands), None)
        if hit is None:
            return _not_applicable(self.source)
        return Decision(
            Verdict.BLOCK,
            f"{self.policy.id}: network egress via {hit!r} is not permitted",
            source=self.source,
        )


@dataclass(frozen=True)
class PackageInstallRule(_PolicyRule):
    """T6: package installs outside the allowlist block; the allowlist is
    exact package names, flags are ignored."""

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        if not _applies(self.policy, proposed):
            return _not_applicable(self.source)
        if refused := _non_string_governed(self, proposed, ("command",)):
            return refused
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
                            source=self.source,
                        )
        return _not_applicable(self.source)


@dataclass(frozen=True)
class WorkspaceWriteBoundaryRule(_WorkspaceRule):
    """T4: a write/edit whose resolved path leaves the workspace blocks.
    Resolution (expanduser + resolve against the workspace root), not
    string prefixing, decides -- `sub/../../..` escapes are caught."""

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        if not _applies(self.policy, proposed):
            return _not_applicable(self.source)
        if refused := _non_string_governed(self, proposed, ("path",)):
            return refused
        raw = _path_argument(proposed)
        if raw is None:
            return _not_applicable(self.source)
        resolved = (self.workspace_root / Path(raw).expanduser()).resolve()
        if resolved.is_relative_to(self.workspace_root):
            return _not_applicable(self.source)
        return Decision(
            Verdict.BLOCK,
            f"{self.policy.id}: {proposed.tool_name} outside the workspace "
            f"({raw!r} resolves to {resolved})",
            source=self.source,
        )


@dataclass(frozen=True)
class CredentialReadRule(_PolicyRule):
    """T5: reads (or bash commands) touching credential-shaped paths block.
    Substring match on the configured patterns -- deliberately coarse and
    conservative for a class-A predicate."""

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        if not _applies(self.policy, proposed):
            return _not_applicable(self.source)
        if refused := _non_string_governed(self, proposed, ("path", "command")):
            return refused
        haystack = str(proposed.arguments.get("path") or proposed.arguments.get("command") or "")
        patterns = self.policy.config.get("patterns") or []
        hit = next((p for p in patterns if p in haystack), None)
        if hit is None:
            return _not_applicable(self.source)
        return Decision(
            Verdict.BLOCK,
            f"{self.policy.id}: access to credential path matching {hit!r}",
            source=self.source,
        )


@dataclass(frozen=True)
class ProtectedPathEditRule(_WorkspaceRule):
    """T2: writes/edits to protected paths (workspace-relative) block."""

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        if not _applies(self.policy, proposed):
            return _not_applicable(self.source)
        if refused := _non_string_governed(self, proposed, ("path",)):
            return refused
        raw = _path_argument(proposed)
        if raw is None:
            return _not_applicable(self.source)
        resolved = (self.workspace_root / Path(raw).expanduser()).resolve()
        try:
            relative = resolved.relative_to(self.workspace_root).as_posix()
        except ValueError:
            relative = raw
        if relative in set(self.policy.config.get("paths") or []):
            return Decision(
                Verdict.BLOCK,
                f"{self.policy.id}: {relative!r} is a protected path",
                source=self.source,
            )
        return _not_applicable(self.source)


# The one table: each policy type and the rule class it compiles to.
_COMPILERS: dict[str, type[_PolicyRule]] = {
    "network-egress": NetworkEgressRule,
    "package-install": PackageInstallRule,
    "write-outside-workspace": WorkspaceWriteBoundaryRule,
    "credential-read": CredentialReadRule,
    "protected-path-edit": ProtectedPathEditRule,
}


def policy_source(policy: GatePolicy) -> str:
    """The `Decision.source` every decision of this policy's rule carries."""
    return _rule_source(_COMPILERS[policy.type], policy)
