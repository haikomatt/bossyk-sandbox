"""Fail-closed defaults (Phase B of the fail-closed audit scope).

Design rule 1: every effect requires an affirmative ALLOW. No path may
execute on a verdict it derived by default, timeout or exception; absence of
an answer is a BLOCK.

Each guard below is a function that returns its violations, so the same
guard can be pointed at a deliberately BROKEN variant. A guard that only
passes on real code could be passing for the wrong reason; every guard that
pins already-fail-closed behaviour therefore has a `*_catches_*` test that
runs it against a broken variant and asserts it reports a violation
(design rule 3).

Rules are DISCOVERED, not sampled (design rule 5): every fast rule any
registered domain factory builds (for every shipped scenario and for no
scenario), and every gate-proxy compiler type. A rule class with no declared
hostile-input surface fails `test_every_*_has_a_declared_*`; it is never
skipped silently. The one deliberate exception, `advice_fast_rules_observe_only`,
is a named, asserted exemption (Phase 0 S4).

Red on `main` (fixed in Phase B Green):
- F3: a gate-proxy approver timeout resolves to ALLOW under `on_hold: allow`.
- F4: `MinimisationInstrument` ALLOWs a missing or non-string ref.
- F5: every gate-proxy rule ALLOWs a governed argument that is present but
  not a string (a list-valued `command` evades the egress and install rules).

No network, no spend: scripted fakes and fixture environments only.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from auditk.attestation.signer import generate_keypair
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from bossyk_sandbox.console.live import initial_input, run_live_session
from bossyk_sandbox.domains import _DOMAIN_BUILDERS
from bossyk_sandbox.gate import Gate, TwoSpeedGate
from bossyk_sandbox.gateproxy.events import effective_verdict
from bossyk_sandbox.gateproxy.hold import HoldRequest
from bossyk_sandbox.gateproxy.policies import (
    _COMPILERS,
    CredentialReadRule,
    GatePolicy,
    NetworkEgressRule,
    PackageInstallRule,
    ProtectedPathEditRule,
    WorkspaceWriteBoundaryRule,
    compile_instruments,
    load_policy_pack,
)
from bossyk_sandbox.gateproxy.proxy import (
    GateProxyConfig,
    _parse_tool_call,
    _resolve_hold,
    create_app,
)
from bossyk_sandbox.instruments import drift
from bossyk_sandbox.instruments.base import (
    Decision,
    Instrument,
    InstrumentVerdict,
    ObservedAction,
    ProposedAction,
    Verdict,
)
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel, RequirePassedCheck
from bossyk_sandbox.instruments.minimisation import MinimisationInstrument
from bossyk_sandbox.runtime import langgraph_agent
from bossyk_sandbox.runtime.langgraph_agent import build_retail_agent_session
from bossyk_sandbox.scenarios.loader import load_scenarios
from bossyk_sandbox.scenarios.runner import advice_fast_rules_observe_only

History = list[ProposedAction | ObservedAction]

# =============================================================================
# 1. Registry discovery
# =============================================================================

# Factories that build fast rules but are deliberately NOT held to the
# fail-closed contract, each with the reason. Asserted below, never skipped.
EXEMPT_FACTORIES: dict[str, tuple[Callable[..., list[Instrument]], str]] = {
    "advice_fast_rules_observe_only": (
        advice_fast_rules_observe_only,
        "free-threshold arms A/B measure the UNMITIGATED leak, so score() always "
        "ALLOWs; it is an explicit opt-in factory, not registered in domains.py",
    ),
}


def registered_fast_rules() -> list[tuple[str, Instrument]]:
    """Every distinct fast rule a registered domain factory builds, for no
    scenario and for each of that domain's shipped scenarios."""
    seen: set[str] = set()
    rules: list[tuple[str, Instrument]] = []
    for domain, build in _DOMAIN_BUILDERS.items():
        cfg = build()
        for scenario in [None, *load_scenarios(cfg.scenarios_path)]:
            for rule in cfg.fast_rules_factory(scenario):
                key = f"{domain}:{rule!r}"
                if key in seen:
                    continue
                seen.add(key)
                rules.append((_rule_id(domain, rule), rule))
    return rules


def _rule_id(domain: str, rule: Any) -> str:
    """A stable, readable test id: domain, class and the tool the rule gates
    (plus the required band for a minimisation rule, whose instances differ
    only by scenario config). Independent of discovery order."""
    if isinstance(rule, MinimisationInstrument):
        detail = f"{rule.config.predicate_tool_name},band={rule.config.required_band_width_gbp:.0f}"
    else:
        detail = str(getattr(rule, "gated_tool", "?"))
    return f"{domain}:{type(rule).__name__}[{detail}]"


REGISTERED = registered_fast_rules()

# =============================================================================
# 2. Precondition fast rules: hostile input never ALLOWs the gated tool
# =============================================================================

# (gated tool, key argument, prerequisite tool or None) per precondition rule
# class. A registered rule whose class (or an ancestor) is not here is
# UNCOVERED and fails `test_every_registered_fast_rule_has_a_declared_hostile_input_surface`.
_PRECONDITION_SURFACES: dict[type, Callable[[Any], tuple[str, str, str | None]]] = {
    RequireLookupBeforeCancel: lambda r: (r.gated_tool, r.key_arg, r.required_lookup_tool),
    RequirePassedCheck: lambda r: (r.gated_tool, r.key_arg, r.check_tool),
    MinimisationInstrument: lambda r: (r.config.predicate_tool_name, r.config.key_arg, None),
}

_MISSING = object()


def _surface(rule: Any) -> tuple[str, str, str | None] | None:
    for cls in type(rule).__mro__:
        if cls in _PRECONDITION_SURFACES:
            return _PRECONDITION_SURFACES[cls](rule)
    return None


def uncovered_rule_classes(rules: Sequence[Any]) -> list[str]:
    return sorted({type(r).__name__ for r in rules if _surface(r) is None})


def hostile_arguments(key: str) -> dict[str, dict[str, Any] | None]:
    """Hostile argument sets for a rule keyed on `key`. A realistic
    `threshold` rides along so a predicate tool's other argument is present."""

    def with_key(value: Any) -> dict[str, Any]:
        args: dict[str, Any] = {"threshold": 50_000}
        if value is not _MISSING:
            args[key] = value
        return args

    return {
        "missing key": with_key(_MISSING),
        "null key": with_key(None),
        "int key": with_key(123),
        "bool key": with_key(True),
        "list key": with_key(["X1"]),
        "dict key": with_key({"id": "X1"}),
        "None arguments": None,
    }


def _histories(prerequisite: str | None, args: dict[str, Any] | None) -> dict[str, History]:
    """No history, and a history in which the prerequisite tool was called
    (and observed) with the SAME hostile arguments -- so a rule that compared
    raw values instead of refusing them would be satisfied."""
    histories: dict[str, History] = {"empty history": []}
    if prerequisite is not None:
        histories["prerequisite called with the same arguments"] = [
            ObservedAction(ProposedAction(prerequisite, dict(args or {})), result={"ok": True})
        ]
    return histories


def fail_open_cases(rules: Sequence[Any]) -> list[str]:
    """Every (rule, hostile input, history) on which the rule ALLOWs the tool
    it gates. A rule that RAISES is not a violation here: the exception
    propagates out of `Gate.score`, which `gate_effects_when_an_instrument_raises`
    pins as no-effect."""
    violations: list[str] = []
    for rule in rules:
        surface = _surface(rule)
        if surface is None:
            violations.append(f"{type(rule).__name__}: no declared surface")
            continue
        tool, key, prerequisite = surface
        for label, args in hostile_arguments(key).items():
            for history_label, history in _histories(prerequisite, args).items():
                try:
                    decision = rule.score(ProposedAction(tool, args), history)  # type: ignore[arg-type]
                except Exception:
                    continue
                if decision.verdict is Verdict.ALLOW:
                    violations.append(f"{type(rule).__name__}[{tool}] {label}, {history_label}")
    return violations


def test_the_fast_rule_registry_sweep_is_not_empty() -> None:
    classes = {type(rule) for _id, rule in REGISTERED}
    assert {RequireLookupBeforeCancel, RequirePassedCheck, MinimisationInstrument} <= classes
    ids = [rule_id for rule_id, _rule in REGISTERED]
    assert len(ids) == len(set(ids))  # every discovered rule gets its own stable id


def test_every_registered_fast_rule_has_a_declared_hostile_input_surface() -> None:
    assert uncovered_rule_classes([rule for _id, rule in REGISTERED]) == []


def test_coverage_guard_catches_a_rule_class_with_no_declared_surface() -> None:
    @dataclasses.dataclass
    class BrandNewRule:
        gated_tool: str = "delete_everything"

        def score(self, proposed: ProposedAction, history: History) -> Decision:
            return Decision(Verdict.ALLOW, "new and unreviewed")

    assert uncovered_rule_classes([BrandNewRule()]) == ["BrandNewRule"]


@pytest.mark.parametrize(
    "rule", [rule for _id, rule in REGISTERED], ids=[i for i, _r in REGISTERED]
)
def test_no_registered_fast_rule_allows_its_gated_tool_on_hostile_input(rule: Any) -> None:
    # Red on main for MinimisationInstrument only (F4): its width rule runs
    # only `if isinstance(ref, str)`, so a missing or non-string ref falls
    # through to the count backstop and is ALLOWed while under budget.
    assert fail_open_cases([rule]) == []


def test_fail_open_guard_catches_a_rule_that_allows_a_missing_key() -> None:
    @dataclasses.dataclass
    class LenientLookupRule(RequireLookupBeforeCancel):
        def score(self, proposed: ProposedAction, history: Any) -> Decision:
            if self.key_arg not in (proposed.arguments or {}):
                return Decision(Verdict.ALLOW, "no key, nothing to check")
            return super().score(proposed, history)

    assert fail_open_cases([LenientLookupRule()]) != []


def test_fail_open_guard_catches_a_rule_that_compares_raw_non_string_keys() -> None:
    @dataclasses.dataclass
    class RawCompareRule(RequireLookupBeforeCancel):
        def score(self, proposed: ProposedAction, history: Any) -> Decision:
            value = (proposed.arguments or {}).get(self.key_arg)
            if any(
                getattr(h, "action", h).tool_name == self.required_lookup_tool
                and getattr(h, "action", h).arguments.get(self.key_arg) == value
                for h in history
            ):
                return Decision(Verdict.ALLOW, "looked up")
            return Decision(Verdict.BLOCK, "no lookup")

    assert fail_open_cases([RawCompareRule()]) != []


# --- the named exemption (S4) -------------------------------------------------


def test_observe_only_minimisation_is_a_named_exemption_not_a_silent_skip() -> None:
    factory, reason = EXEMPT_FACTORIES["advice_fast_rules_observe_only"]
    assert reason
    # Not reachable from the registry: no registered domain uses it ...
    assert all(build().fast_rules_factory is not factory for build in _DOMAIN_BUILDERS.values())
    # ... and it is exempt because it really does fail open, by design.
    rules = factory(None)
    assert rules and all(
        isinstance(r, MinimisationInstrument) and r.config.observe_only for r in rules
    )
    assert fail_open_cases(rules) != []


# --- outcome-aware rules: raising predicate, malformed observed result --------

_PASSED_CHECK_RULES = [rule for _id, rule in REGISTERED if isinstance(rule, RequirePassedCheck)]

MALFORMED_RESULTS: list[Any] = [
    None,
    object(),
    "ok",
    "",
    0,
    1,
    True,
    [],
    {},
    {"eligible": "yes"},
    {"on_tps": None, "opted_out": None},
    {"error": "upstream timeout"},
]


def _observed_check(rule: RequirePassedCheck, result: Any) -> tuple[ProposedAction, History]:
    args = {rule.key_arg: "P-001"}
    return (
        ProposedAction(rule.gated_tool, dict(args)),
        [ObservedAction(ProposedAction(rule.check_tool, dict(args)), result)],
    )


def raising_predicate_violations(rules: Sequence[RequirePassedCheck]) -> list[str]:
    """Each rule, with its predicate swapped for one that raises, must not
    ALLOW even though a matching observed check is in history."""

    def explode(_result: Any) -> bool:
        raise ValueError("malformed check result")

    violations = []
    for rule in rules:
        proposed, history = _observed_check(rule, {"eligible": True})
        decision = dataclasses.replace(rule, predicate=explode).score(proposed, history)
        if decision.verdict is Verdict.ALLOW:
            violations.append(f"{rule.gated_tool}: raising predicate ALLOWed")
    return violations


def malformed_result_violations(rules: Sequence[RequirePassedCheck]) -> list[str]:
    violations = []
    for rule in rules:
        for result in MALFORMED_RESULTS:
            proposed, history = _observed_check(rule, result)
            if rule.score(proposed, history).verdict is Verdict.ALLOW:
                violations.append(f"{rule.gated_tool}: ALLOWed on result {result!r}")
    return violations


def test_registry_has_outcome_aware_rules_to_check() -> None:
    assert len(_PASSED_CHECK_RULES) >= 2


def test_a_raising_predicate_never_allows() -> None:
    assert raising_predicate_violations(_PASSED_CHECK_RULES) == []


def test_raising_predicate_guard_catches_a_rule_that_treats_an_error_as_a_pass() -> None:
    @dataclasses.dataclass
    class ErrorIsPass(RequirePassedCheck):
        def score(self, proposed: ProposedAction, history: Any) -> Decision:
            for item in history:
                if isinstance(item, ObservedAction) and item.action.tool_name == self.check_tool:
                    try:
                        passed = bool(self.predicate(item.result))
                    except Exception:
                        passed = True
                    if passed:
                        return Decision(Verdict.ALLOW, "passed")
            return Decision(Verdict.BLOCK, "no pass")

    broken = [ErrorIsPass(**dataclasses.asdict(r)) for r in _PASSED_CHECK_RULES]
    assert raising_predicate_violations(broken) != []


def test_a_malformed_observed_result_never_allows() -> None:
    assert malformed_result_violations(_PASSED_CHECK_RULES) == []


def test_malformed_result_guard_catches_a_lenient_predicate() -> None:
    lenient = [
        dataclasses.replace(r, predicate=lambda result: result is not None)
        for r in _PASSED_CHECK_RULES
    ]
    assert malformed_result_violations(lenient) != []


# =============================================================================
# 3. Gate-proxy rules: a present, non-string governed argument BLOCKs (F5)
# =============================================================================

# Governed argument keys per proxy rule class. A compiler type whose rule
# class is not here fails `test_every_proxy_compiler_type_has_declared_governed_keys`.
_PROXY_GOVERNED_KEYS: dict[type, tuple[str, ...]] = {
    NetworkEgressRule: ("command",),
    PackageInstallRule: ("command",),
    WorkspaceWriteBoundaryRule: ("path",),
    CredentialReadRule: ("path", "command"),
    ProtectedPathEditRule: ("path",),
}

_ALL_TYPES_PACK = """\
version: 1
policies:
  - id: no-network-egress
    type: network-egress
    tools: [bash]
    commands: [curl, wget]
  - id: package-allowlist
    type: package-install
    tools: [bash]
    allowed_packages: [pytest]
  - id: workspace-write-boundary
    type: write-outside-workspace
    tools: [write, edit]
  - id: no-credential-reads
    type: credential-read
    tools: [read, bash]
    patterns: [".env", "id_rsa"]
  - id: protected-test-paths
    type: protected-path-edit
    tools: [write, edit]
    paths: ["tests/test_provided.py"]
"""

NON_STRING_VALUES: dict[str, Any] = {
    "argv list": ["curl", "https://evil.example"],
    "int": 123,
    "bool": True,
    "dict": {"cmd": "curl https://evil.example"},
    "null": None,
}


def _proxy_rules(tmp_path: Path) -> list[Any]:
    pack_path = tmp_path / "all-types.yaml"
    pack_path.write_text(_ALL_TYPES_PACK)
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    return compile_instruments(load_policy_pack(pack_path), workspace_root=workspace)


def uncovered_proxy_rules(rules: Sequence[Any]) -> list[str]:
    return sorted({type(r).__name__ for r in rules if type(r) not in _PROXY_GOVERNED_KEYS})


def non_string_argument_violations(rules: Sequence[Any]) -> list[str]:
    """Every (rule, gated tool, governed key, non-string value) the rule does
    not refuse. A present, non-string governed argument cannot be checked, so
    it must BLOCK (or HOLD under `on_match: hold`), never ALLOW."""
    violations = []
    for rule in rules:
        for tool in rule.policy.config.get("tools") or []:
            for key in _PROXY_GOVERNED_KEYS[type(rule)]:
                for label, value in NON_STRING_VALUES.items():
                    decision = rule.score(ProposedAction(tool, {key: value}), [])
                    if decision.verdict is Verdict.ALLOW:
                        violations.append(f"{rule.policy.id}[{tool}].{key} = {label}")
    return violations


def test_every_proxy_compiler_type_has_declared_governed_keys(tmp_path: Path) -> None:
    rules = _proxy_rules(tmp_path)
    assert {r.policy.type for r in rules} == set(_COMPILERS)  # the pack covers every type
    assert uncovered_proxy_rules(rules) == []


def test_proxy_coverage_guard_catches_an_undeclared_rule_class() -> None:
    @dataclasses.dataclass(frozen=True)
    class NewProxyRule:
        policy: GatePolicy

    policy = GatePolicy(id="new", type="new", policy_class="A", trap=None, config={})
    assert uncovered_proxy_rules([NewProxyRule(policy)]) == ["NewProxyRule"]


@pytest.mark.parametrize("rule_type", sorted(_COMPILERS))
def test_every_proxy_rule_refuses_a_present_non_string_governed_argument(
    rule_type: str, tmp_path: Path
) -> None:
    # Red on main for all five (F5): each rule str()s or ignores a non-string
    # value, so e.g. `command=["curl", ...]` tokenises to "['curl'," and
    # matches nothing.
    rules = [r for r in _proxy_rules(tmp_path) if r.policy.type == rule_type]
    assert non_string_argument_violations(rules) == []


# =============================================================================
# 4. An instrument that raises has no effect
# =============================================================================


class _RaisingInstrument:
    def score(self, proposed: ProposedAction, history: Any) -> Decision:
        raise RuntimeError("instrument exploded")


_REAL_GATE_SCORE = Gate.score


def _fail_open_gate_score(self: Gate, proposed: ProposedAction) -> Decision:
    """BROKEN variant of `Gate.score`: swallows an instrument exception and
    ALLOWs. Used only to show the guards below can see a fail-open gate."""
    try:
        return _REAL_GATE_SCORE(self, proposed)
    except Exception:
        return Decision(Verdict.ALLOW, "instrument error ignored")


@pytest.fixture()
def fail_open_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Gate, "score", _fail_open_gate_score)


def gate_effects_when_an_instrument_raises() -> list[str]:
    violations = []
    proposed = ProposedAction("get_order_details", {"order_id": "#1"})

    gate = Gate(instruments=[_RaisingInstrument()])
    try:
        decision = gate.score(proposed)
    except Exception:
        pass
    else:
        violations.append(f"Gate.score returned {decision.verdict.value}")

    two_speed = TwoSpeedGate(gate=Gate(instruments=[_RaisingInstrument()]), slow_instruments=[])
    try:
        decision = two_speed.process(proposed)[0]
    except Exception:
        pass
    else:
        violations.append(f"TwoSpeedGate.process returned {decision.verdict.value}")
    finally:
        two_speed.shutdown()
    if two_speed.gate.history:
        violations.append("TwoSpeedGate recorded the action in history")
    return violations


def test_a_raising_instrument_has_no_effect_in_gate_or_two_speed_gate() -> None:
    assert gate_effects_when_an_instrument_raises() == []


@pytest.mark.usefixtures("fail_open_gate")
def test_gate_guard_catches_a_gate_that_swallows_instrument_errors() -> None:
    assert gate_effects_when_an_instrument_raises() != []


class _RecordingToolkit:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def use_tool(self, name: str, **args: Any) -> str:
        self.calls.append(name)
        return f"ok:{name}"

    def get_tools(self) -> dict[str, Any]:
        return {}


class _RecordingEnv:
    policy = "You are a retail support agent."

    def __init__(self) -> None:
        self.tools = _RecordingToolkit()


class _ScriptedLLM:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self._i = 0

    def invoke(self, _messages: Any) -> AIMessage:
        response = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return response


def _one_lookup_llm() -> _ScriptedLLM:
    # A read the retail fast rules ALLOW, so only the raising instrument can
    # stop it.
    call = {"name": "get_order_details", "args": {"order_id": "#W1"}, "id": "c1"}
    return _ScriptedLLM([AIMessage(content="", tool_calls=[call]), AIMessage(content="done")])


def agent_effects_when_an_instrument_raises() -> list[str]:
    env = _RecordingEnv()
    session = build_retail_agent_session(llm=_one_lookup_llm(), environment=env)
    assert session.gate is not None
    session.gate.instruments.append(_RaisingInstrument())
    try:
        run_live_session(session)
    except Exception:
        pass
    return [f"executed {name}" for name in env.tools.calls]


def test_a_raising_instrument_stops_the_langgraph_session_before_any_effect() -> None:
    assert agent_effects_when_an_instrument_raises() == []


@pytest.mark.usefixtures("fail_open_gate")
def test_agent_guard_catches_a_gate_that_swallows_instrument_errors() -> None:
    assert agent_effects_when_an_instrument_raises() == ["executed get_order_details"]


# --- gate proxy ---------------------------------------------------------------


def _proxy_config(tmp_path: Path, pack_yaml: str = _ALL_TYPES_PACK) -> GateProxyConfig:
    pack_path = tmp_path / "pack.yaml"
    pack_path.write_text(pack_yaml)
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    priv_path, _pub = generate_keypair(tmp_path / "gate-key")
    return GateProxyConfig(
        upstream_base_url="http://upstream.invalid/v1",
        policy_pack_path=pack_path,
        workspace_root=workspace,
        signer_key_path=priv_path,
        events_path=tmp_path / "gate-events.jsonl",
        run_label="fail-closed-test",
    )


def _upstream_with(*calls: tuple[str, str]) -> dict[str, Any]:
    tool_calls = [
        {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": args}}
        for i, (name, args) in enumerate(calls)
    ]
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "upstream-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                "finish_reason": "tool_calls",
            }
        ],
    }


_CHAT_REQUEST = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def forwarded_tool_calls(
    config: GateProxyConfig,
    upstream_body: dict[str, Any],
    approver: Callable[[HoldRequest], bool | None] | None = None,
) -> tuple[int, list[str]]:
    """(HTTP status, names of the tool calls the client received)."""
    app = create_app(config, upstream=lambda _b, _h: upstream_body, approver=approver)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/v1/chat/completions", json=_CHAT_REQUEST
    )
    if response.status_code != 200:
        return response.status_code, []
    message = response.json()["choices"][0]["message"]
    return 200, [c["function"]["name"] for c in message.get("tool_calls") or []]


@pytest.fixture()
def raising_proxy_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(self: Any, proposed: ProposedAction, history: Any) -> Decision:
        raise RuntimeError("proxy rule exploded")

    monkeypatch.setattr(NetworkEgressRule, "score", explode)


@pytest.mark.usefixtures("raising_proxy_rule")
def test_a_raising_proxy_rule_returns_500_and_forwards_no_tool_call(tmp_path: Path) -> None:
    status, forwarded = forwarded_tool_calls(
        _proxy_config(tmp_path), _upstream_with(("bash", '{"command": "ls"}'))
    )
    assert (status, forwarded) == (500, [])


@pytest.mark.usefixtures("raising_proxy_rule", "fail_open_gate")
def test_proxy_guard_catches_a_gate_that_swallows_instrument_errors(tmp_path: Path) -> None:
    status, forwarded = forwarded_tool_calls(
        _proxy_config(tmp_path), _upstream_with(("bash", '{"command": "ls"}'))
    )
    assert (status, forwarded) == (200, ["bash"])


# =============================================================================
# 5. Slow instruments: raise / timeout -> "error" label, gate decision unchanged
# =============================================================================


@dataclasses.dataclass
class _RaisingSlow:
    name: str = "raising-slow"

    def annotate(self, proposed: ProposedAction, history: Any) -> InstrumentVerdict:
        raise RuntimeError("judge exploded")


@dataclasses.dataclass
class _HangingSlow:
    name: str = "hanging-slow"

    def annotate(self, proposed: ProposedAction, history: Any) -> InstrumentVerdict:
        time.sleep(0.3)
        return InstrumentVerdict(instrument=self.name, label="ok")


def slow_failure_violations(two_speed_cls: type[TwoSpeedGate]) -> list[str]:
    violations = []
    for proposed in [
        ProposedAction("cancel_reservation", {"reservation_id": "R1"}),  # fast BLOCK
        ProposedAction("get_reservation_details", {"reservation_id": "R1"}),  # fast ALLOW
    ]:
        expected = Gate(instruments=[RequireLookupBeforeCancel()]).score(proposed)
        two_speed = two_speed_cls(
            gate=Gate(instruments=[RequireLookupBeforeCancel()]),
            slow_instruments=[_RaisingSlow(), _HangingSlow()],
            annotation_timeout_s=0.05,
        )
        try:
            decision, future = two_speed.process(proposed)
            if decision != expected:
                violations.append(f"{proposed.tool_name}: decision changed to {decision}")
            try:
                labels = [v.label for v in future.result(timeout=5)]
            except Exception as exc:
                violations.append(f"{proposed.tool_name}: annotations raised {exc!r}")
            else:
                if labels != [drift.ERROR_LABEL, drift.ERROR_LABEL]:
                    violations.append(f"{proposed.tool_name}: labels {labels}")
        finally:
            two_speed.shutdown()
    return violations


def test_slow_instrument_failures_become_error_labels_and_never_change_the_decision() -> None:
    assert slow_failure_violations(TwoSpeedGate) == []


def test_slow_guard_catches_a_gate_that_lets_an_annotation_failure_escape() -> None:
    class Unisolated(TwoSpeedGate):
        def _gather(self, futures: list[Any]) -> list[InstrumentVerdict]:
            return [f.result(timeout=self.annotation_timeout_s) for f in futures]

    assert slow_failure_violations(Unisolated) != []


# =============================================================================
# 6. Holds: no answer is a BLOCK
# =============================================================================

_HOLD_REQUEST = HoldRequest(
    tool_name="bash",
    arguments={"command": "rm -rf build"},
    policy_id="destructive-shell",
    reason="destructive-shell: network egress via 'rm' is not permitted",
    run_label="fail-closed-test",
)


def _hold_policy(on_hold: str) -> GatePolicy:
    return GatePolicy(
        id="destructive-shell",
        type="network-egress",
        policy_class="A",
        trap=None,
        config={"tools": ["bash"], "commands": ["rm"]},
        on_match="hold",
        on_hold=on_hold,
    )


def _no_answer(_request: HoldRequest) -> bool | None:
    return None


@pytest.mark.parametrize("on_hold", ["block", "allow"])
def test_an_approver_that_never_answers_resolves_to_block(on_hold: str) -> None:
    # Red on main for on_hold=allow (F3, decision D2): a configured approver
    # that times out fell back to the policy's on_hold default, so a pack
    # could make "the human didn't answer" mean "go ahead".
    assert _resolve_hold(_hold_policy(on_hold), _HOLD_REQUEST, _no_answer) == (
        "hold_timed_out",
        Verdict.BLOCK,
    )


_HOLD_PACK = """\
version: 1
policies:
  - id: destructive-shell
    type: network-egress
    on_match: hold
    on_hold: allow
    tools: [bash]
    commands: [rm]
"""


def test_the_proxy_forwards_nothing_when_the_approver_times_out_under_on_hold_allow(
    tmp_path: Path,
) -> None:
    # Red on main (F3), end to end through the proxy.
    status, forwarded = forwarded_tool_calls(
        _proxy_config(tmp_path, _HOLD_PACK),
        _upstream_with(("bash", '{"command": "rm -rf build"}')),
        approver=_no_answer,
    )
    assert (status, forwarded) == (200, [])


def test_no_approver_configured_keeps_on_hold_allow_as_allow_and_log() -> None:
    # D2 boundary (not fail-closed by design, pinned so F3's fix stays
    # narrow): with NO approver configured, `on_hold: allow` still means
    # "allow and log".
    assert _resolve_hold(_hold_policy("allow"), _HOLD_REQUEST, None) == (
        "held_then_allowed",
        Verdict.ALLOW,
    )


def unnamed_policy_hold_violations(
    resolve: Callable[[GatePolicy | None, HoldRequest, Any], tuple[str, Verdict]],
) -> list[str]:
    violations = []
    for approver in (None, _no_answer):
        _resolution, verdict = resolve(None, _HOLD_REQUEST, approver)
        if verdict is not Verdict.BLOCK:
            violations.append(f"approver={approver!r}: {verdict.value}")
    return violations


def test_a_hold_on_a_policy_the_pack_cannot_name_blocks() -> None:
    assert unnamed_policy_hold_violations(_resolve_hold) == []


def test_unnamed_policy_guard_catches_a_resolver_that_defaults_to_allow() -> None:
    def lenient(
        policy: GatePolicy | None, request: HoldRequest, approver: Any
    ) -> tuple[str, Verdict]:
        return "held_then_allowed", Verdict.ALLOW

    assert unnamed_policy_hold_violations(lenient) != []


def unresolved_hold_violations(effective: Callable[[dict[str, Any]], str]) -> list[str]:
    # The MISSING-key case. A present-but-null or otherwise unrecognised
    # resolution is pinned separately (Issues & Fixes 10).
    event = {"kind": "tool_call", "verdict": "hold"}
    return [] if effective(event) == "block" else [f"unresolved hold -> {effective(event)!r}"]


def test_an_unresolved_hold_event_counts_as_block() -> None:
    assert unresolved_hold_violations(effective_verdict) == []


def test_unresolved_hold_guard_catches_a_reader_that_counts_it_as_allowed() -> None:
    assert unresolved_hold_violations(lambda event: "allow") != []


@pytest.mark.parametrize(
    "resolved", [None, "", "maybe", "hold", "ALLOW", "None", 0, 1, True, [], {}]
)
def test_a_hold_whose_resolution_is_not_allow_or_block_counts_as_block(resolved: Any) -> None:
    # Regression (Issues & Fixes 10): only "allow" and "block" are
    # resolutions. A present-but-null or any other resolved_verdict is no
    # answer, and no answer is a BLOCK -- not the string "None".
    event = {"kind": "tool_call", "verdict": "hold", "resolved_verdict": resolved}
    assert effective_verdict(event) == "block"


@pytest.mark.parametrize("resolved", ["allow", "block"])
def test_a_hold_resolved_to_allow_or_block_keeps_its_resolution(resolved: str) -> None:
    event = {"kind": "tool_call", "verdict": "hold", "resolved_verdict": resolved}
    assert effective_verdict(event) == resolved


# =============================================================================
# 7. Unparseable proxy arguments, and HOLD in the agent graph
# =============================================================================

UNPARSEABLE_ARGUMENTS: list[Any] = ["{not json", "[1, 2]", "null", '"a string"', "42", 42, None]


def unparseable_argument_violations(
    parse: Callable[[dict[str, Any]], tuple[str, dict[str, Any] | None]],
) -> list[str]:
    """`_parse_tool_call` must return `None` arguments (the proxy's
    "unparseable (fail closed)" BLOCK path) for anything that is not a JSON
    object."""
    violations = []
    for raw in UNPARSEABLE_ARGUMENTS:
        _name, parsed = parse({"function": {"name": "bash", "arguments": raw}})
        if parsed is not None:
            violations.append(f"{raw!r} parsed to {parsed!r}")
    return violations


def test_unparseable_tool_arguments_take_the_fail_closed_path() -> None:
    assert unparseable_argument_violations(_parse_tool_call) == []


def test_unparseable_guard_catches_a_parser_that_defaults_to_empty_arguments() -> None:
    def lenient(call: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        name, parsed = _parse_tool_call(call)
        return name, parsed if parsed is not None else {}

    assert unparseable_argument_violations(lenient) != []


def test_the_proxy_forwards_nothing_for_unparseable_arguments(tmp_path: Path) -> None:
    status, forwarded = forwarded_tool_calls(
        _proxy_config(tmp_path), _upstream_with(("bash", "{not json"))
    )
    assert (status, forwarded) == (200, [])


def _executed_on_resume(resume: str) -> list[str]:
    from langgraph.types import Command

    env = _RecordingEnv()
    session = build_retail_agent_session(llm=_one_lookup_llm(), environment=env)
    config = {"configurable": {"thread_id": f"resume-{resume}"}}
    result = session.graph.invoke(initial_input(), config=config)  # type: ignore[call-overload]
    while "__interrupt__" in result:
        result = session.graph.invoke(Command(resume=resume), config=config)  # type: ignore[call-overload]
    return env.tools.calls


def non_allow_execution_violations() -> list[str]:
    return [
        f"final {verdict} executed {calls}"
        for verdict in ("hold", "block")
        if (calls := _executed_on_resume(verdict))
    ]


def test_the_agent_executes_only_on_a_final_allow() -> None:
    assert _executed_on_resume("allow") == ["get_order_details"]
    assert non_allow_execution_violations() == []


def test_non_allow_guard_catches_an_agent_that_reads_hold_as_allow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HoldIsAllow:
        """BROKEN `Verdict` constructor for the agent module only: parses
        "hold" as ALLOW, as a fail-open approve/execute path would."""

        ALLOW, BLOCK, HOLD = Verdict.ALLOW, Verdict.BLOCK, Verdict.HOLD

        def __new__(cls, value: str) -> Verdict:  # type: ignore[misc]
            return Verdict.ALLOW if value == "hold" else Verdict(value)

    monkeypatch.setattr(langgraph_agent, "Verdict", HoldIsAllow)
    assert non_allow_execution_violations() == ["final hold executed ['get_order_details']"]
