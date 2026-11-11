from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage

from bossyk_sandbox.console.live import LiveResult, order_reader_from_toolkit, run_live_session
from bossyk_sandbox.runtime.langgraph_agent import build_retail_agent_session
from bossyk_sandbox.standing import StandingGrant

# Drives the REAL LangGraph agent graph + REAL retail gate, but with a fake LLM
# and a fake tau2 environment injected -- fully deterministic, no network, no
# API key, no cost. Only swapping the fake LLM for a real Fireworks client is
# billable. This is the non-billable half of Slice B.


class _FakeToolkit:
    """Minimal tau2 toolkit stand-in: `use_tool` returns a canned string for
    the calls the gate ALLOWs (lookups); `get_tools` is never reached because
    an llm is injected (so tool schemas aren't computed)."""

    def use_tool(self, name: str, **args: Any) -> str:
        return f"ok:{name}:{args}"

    def get_tools(self) -> dict[str, Any]:
        return {}


class _FakeEnv:
    policy = "You are a retail support agent."
    tools = _FakeToolkit()


class _FakeLLM:
    """Returns scripted AIMessages: one turn of tool calls, then a plain
    reply so the graph reaches END. `invoke` is the only method the agent node
    calls on an injected llm."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self._i = 0

    def invoke(self, _messages: Any) -> AIMessage:
        response = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return response


def _scripted_llm() -> _FakeLLM:
    calls = [
        {"name": "get_order_details", "args": {"order_id": "#W1"}, "id": "c1"},
        {"name": "cancel_pending_order", "args": {"order_id": "#W2"}, "id": "c2"},
        {"name": "return_delivered_order_items", "args": {"order_id": "#W3"}, "id": "c3"},
        {"name": "get_user_details", "args": {"user_id": "U1"}, "id": "c4"},
        {"name": "modify_pending_order_payment", "args": {"order_id": "#W4"}, "id": "c5"},
    ]
    return _FakeLLM(
        [
            AIMessage(content="", tool_calls=calls),
            AIMessage(content="all done"),
        ]
    )


def test_run_live_session_derives_modes_from_real_gate_verdicts() -> None:
    session = build_retail_agent_session(llm=_scripted_llm(), environment=_FakeEnv())

    result = run_live_session(session)

    assert isinstance(result, LiveResult)
    # gate: allow, block, block, allow, block  ->  derived modes:
    assert result.modes == ["allow", "redirect", "escalate", "step-up", "escalate"]
    assert result.verdicts == ["allow", "block", "block", "allow", "block"]

    # Only the two blocked irreversible writes reach the queue, severity-ordered.
    assert [item["severity"] for item in result.hitl_queue] == ["critical", "high"]

    # Real attested steps were produced by the real graph's execute node.
    assert len(result.steps) == 5


def _standing_scenario_llm() -> _FakeLLM:
    # Three looked-up cancels (each gate-ALLOWed) so the 3rd consumes standing
    # authority beyond max_count=2 -> the live path derives `defer`.
    calls = [
        {"name": "get_order_details", "args": {"order_id": "#A"}, "id": "l1"},
        {"name": "cancel_pending_order", "args": {"order_id": "#A"}, "id": "c1"},
        {"name": "get_order_details", "args": {"order_id": "#B"}, "id": "l2"},
        {"name": "cancel_pending_order", "args": {"order_id": "#B"}, "id": "c2"},
        {"name": "get_order_details", "args": {"order_id": "#C"}, "id": "l3"},
        {"name": "cancel_pending_order", "args": {"order_id": "#C"}, "id": "c3"},
    ]
    return _FakeLLM([AIMessage(content="", tool_calls=calls), AIMessage(content="done")])


def test_run_live_session_defers_the_over_authority_cancel() -> None:
    from bossyk_sandbox.standing import StandingGrant

    session = build_retail_agent_session(llm=_standing_scenario_llm(), environment=_FakeEnv())
    grants = {"cancellation": StandingGrant("cancellation", max_count=2)}

    result = run_live_session(session, grants=grants)

    # All six calls are gate-ALLOWed (each cancel has its prior lookup); the
    # first two cancels are within standing (allow), the third is over -> defer.
    assert result.verdicts == ["allow"] * 6
    assert result.modes == ["allow", "allow", "allow", "allow", "allow", "defer"]
    assert result.hitl_queue == []  # defer is reversible; nothing escalates


def test_run_live_session_without_grants_never_defers() -> None:
    # Back-compat: no standing policy -> the ALLOW path keeps pre-§F behaviour.
    session = build_retail_agent_session(llm=_standing_scenario_llm(), environment=_FakeEnv())
    result = run_live_session(session)  # grants=None
    assert "defer" not in result.modes
    assert result.modes == ["allow"] * 6


def _stepping_clock(step: float) -> Any:
    # A deterministic injected clock: each call advances by `step` seconds, so a
    # test controls how much wall time passes between the six live turns.
    ticks = iter(i * step for i in range(1000))
    return lambda: next(ticks)


def test_windowed_grant_expires_standing_in_the_live_path() -> None:
    # The three cancels land 2000s apart, so by the third (t=10000) the first two
    # (t=2000, t=6000) are older than the 1h window -> the 3rd cancel is back
    # WITHIN standing -> no defer. Same scenario, session-scoped, would defer.
    from bossyk_sandbox.standing import StandingGrant

    session = build_retail_agent_session(llm=_standing_scenario_llm(), environment=_FakeEnv())
    grants = {"cancellation": StandingGrant("cancellation", max_count=2, window="1h")}

    result = run_live_session(session, grants=grants, clock=_stepping_clock(2000.0))

    assert result.modes == ["allow"] * 6  # window expired the earlier cancels
    assert result.hitl_queue == []


def test_windowed_grant_still_defers_within_the_window() -> None:
    # Same window, but the turns are only 100s apart -> all three cancels fall
    # inside the hour -> the 3rd is over standing -> defer (window is applied,
    # not ignored).
    from bossyk_sandbox.standing import StandingGrant

    session = build_retail_agent_session(llm=_standing_scenario_llm(), environment=_FakeEnv())
    grants = {"cancellation": StandingGrant("cancellation", max_count=2, window="1h")}

    result = run_live_session(session, grants=grants, clock=_stepping_clock(100.0))

    assert result.modes == ["allow", "allow", "allow", "allow", "allow", "defer"]


def test_run_live_session_enforces_the_gate_no_blocked_tool_runs() -> None:
    # The driver resumes each interrupt with the gate's own verdict (no human
    # override), so a blocked write is never executed by the tau2 toolkit.
    session = build_retail_agent_session(llm=_scripted_llm(), environment=_FakeEnv())

    result = run_live_session(session)

    # Two lookups were ALLOWed and ran (tool_latency records one per exec);
    # the three writes were blocked and never executed.
    assert len(result.steps) == 5
    blocked = [s for s in result.steps if s.action.payload["gate_verdict"] == "block"]
    assert len(blocked) == 3


# --- amount gating (P3): live wiring ------------------------------------------
# The oracle (`standing_amount.resolve_amount`) needs order state, which lives
# on the tau2 environment's toolkit -- `order_reader_from_toolkit` adapts it to
# the `OrderReader` Protocol so `run_live_session` can resolve + stamp the
# amount of each allowed action and thread it into `evaluate_authority`,
# without this module (or `standing.py`) importing tau2 itself.


def _priced_order(item_id: str, price: float) -> Any:
    # Duck-types the slice of a real tau2 `Order` the adapter reads: `.items`
    # (each with `.item_id`/`.price`) and `.payment_history` (unused by a
    # refund, but present on a real order).
    return SimpleNamespace(
        items=[SimpleNamespace(item_id=item_id, price=price)], payment_history=[]
    )


class _PricedToolkit:
    """Minimal tau2 toolkit stand-in that ALSO carries `.db.orders` (as the
    real `RetailTools` does), so `order_reader_from_toolkit` can resolve
    amounts against it."""

    def __init__(self, orders: dict[str, Any]) -> None:
        self.db = SimpleNamespace(orders=orders)

    def use_tool(self, name: str, **args: Any) -> str:
        return f"ok:{name}:{args}"

    def get_tools(self) -> dict[str, Any]:
        return {}


class _PricedEnv:
    policy = "You are a retail support agent."

    def __init__(self, orders: dict[str, Any]) -> None:
        self.tools = _PricedToolkit(orders)


def _priced_refund_llm() -> Any:
    # Two looked-up refunds (each gate-ALLOWed, same shape as
    # `_standing_scenario_llm`): #R1 returns a $60 item, #R2 returns another
    # $60 item. Count stays well within any generous max_count; the SECOND
    # refund's cumulative amount (60 + 60 = 120) is what crosses a $100 budget.
    calls = [
        {"name": "get_order_details", "args": {"order_id": "#R1"}, "id": "l1"},
        {
            "name": "return_delivered_order_items",
            "args": {
                "order_id": "#R1",
                "item_ids": ["item-a"],
                "payment_method_id": "gift_card_1",
            },
            "id": "r1",
        },
        {"name": "get_order_details", "args": {"order_id": "#R2"}, "id": "l2"},
        {
            "name": "return_delivered_order_items",
            "args": {
                "order_id": "#R2",
                "item_ids": ["item-b"],
                "payment_method_id": "gift_card_1",
            },
            "id": "r2",
        },
    ]
    return _FakeLLM([AIMessage(content="", tool_calls=calls), AIMessage(content="done")])


def test_run_live_session_escalates_a_refund_that_crosses_the_amount_budget() -> None:
    orders = {"#R1": _priced_order("item-a", 60.0), "#R2": _priced_order("item-b", 60.0)}
    env = _PricedEnv(orders)
    session = build_retail_agent_session(llm=_priced_refund_llm(), environment=env)
    grants = {"refund": StandingGrant(boundary="refund", max_count=10, max_amount=100.0)}
    reader = order_reader_from_toolkit(env.tools)

    result = run_live_session(session, grants=grants, reader=reader)

    # Both refunds are gate-ALLOWed (each has its prior lookup) and the count
    # (2nd of a max_count=10) is nowhere near its ceiling -- ONLY the resolved
    # amount (60 + 60 = 120 > 100) pushes the second refund over standing.
    # A refund is irreversible, so over-authority escalates (not defer).
    assert result.verdicts == ["allow", "allow", "allow", "allow"]
    assert result.modes == ["allow", "allow", "allow", "escalate"]
    assert len(result.hitl_queue) == 1
    assert result.hitl_queue[0]["severity"] == "high"
    assert result.hitl_queue[0]["tool_name"] == "return_delivered_order_items"


def test_run_live_session_without_a_reader_never_resolves_amounts() -> None:
    # Back-compat: no reader injected -> amount stays unresolved for every
    # call, so an amount-gated grant would fail-safe to `over` on the very
    # first action. This is the pre-P3 code path exercised elsewhere in this
    # file (no `reader=`) staying green with a count-only grant, unaffected.
    session = build_retail_agent_session(llm=_scripted_llm(), environment=_FakeEnv())
    result = run_live_session(session)  # grants=None, reader=None
    assert result.modes == ["allow", "redirect", "escalate", "step-up", "escalate"]


# --- Phase A: escalation is enforced, not advisory -----------------------------
# `derive_mode` says `escalate` means "a human must decide". With no approver
# in the live loop, that must resolve to BLOCK (decision D1): the action never
# reaches the toolkit, its signed step says `block` with the mode and the
# source of that final verdict, and it does not consume standing authority.
# `defer` (over authority but reversible) is accept + async-approve by design,
# so it must still execute -- the fix cannot over-block.


class _RecordingPricedToolkit(_PricedToolkit):
    """`_PricedToolkit` that also records every executed call, so a test can
    assert which effects actually happened rather than which modes were
    derived."""

    def __init__(self, orders: dict[str, Any]) -> None:
        super().__init__(orders)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def use_tool(self, name: str, **args: Any) -> str:
        self.calls.append((name, args))
        return super().use_tool(name, **args)


class _RecordingPricedEnv:
    policy = "You are a retail support agent."

    def __init__(self, orders: dict[str, Any]) -> None:
        self.tools = _RecordingPricedToolkit(orders)


def _looked_up_refunds_llm(order_ids: list[str]) -> _FakeLLM:
    """One lookup + one refund per order (each refund gate-ALLOWed); order
    `#Rn` returns `item-n`."""
    calls: list[dict[str, Any]] = []
    for n, order_id in enumerate(order_ids, start=1):
        calls.append({"name": "get_order_details", "args": {"order_id": order_id}, "id": f"l{n}"})
        calls.append(
            {
                "name": "return_delivered_order_items",
                "args": {
                    "order_id": order_id,
                    "item_ids": [f"item-{n}"],
                    "payment_method_id": "gift_card_1",
                },
                "id": f"r{n}",
            }
        )
    return _FakeLLM([AIMessage(content="", tool_calls=calls), AIMessage(content="done")])


def _executed_refunds(toolkit: _RecordingPricedToolkit) -> list[str]:
    return [
        str(args["order_id"])
        for name, args in toolkit.calls
        if name == "return_delivered_order_items"
    ]


def _refund_budget_run(prices: list[float]) -> tuple[LiveResult, _RecordingPricedToolkit]:
    order_ids = [f"#R{n}" for n in range(1, len(prices) + 1)]
    orders = {
        order_id: _priced_order(f"item-{n}", price)
        for n, (order_id, price) in enumerate(zip(order_ids, prices, strict=True), start=1)
    }
    env = _RecordingPricedEnv(orders)
    session = build_retail_agent_session(llm=_looked_up_refunds_llm(order_ids), environment=env)
    grants = {"refund": StandingGrant(boundary="refund", max_count=10, max_amount=100.0)}
    result = run_live_session(session, grants=grants, reader=order_reader_from_toolkit(env.tools))
    return result, env.tools


def test_an_escalated_refund_never_reaches_the_toolkit() -> None:
    result, toolkit = _refund_budget_run([60.0, 60.0])

    assert result.modes == ["allow", "allow", "allow", "escalate"]
    assert _executed_refunds(toolkit) == ["#R1"]


def test_an_escalated_refund_is_signed_as_a_standing_block_not_a_manual_override() -> None:
    result, _toolkit = _refund_budget_run([60.0, 60.0])

    escalated = result.steps[-1]
    assert escalated.action.payload["tool_name"] == "return_delivered_order_items"
    assert escalated.action.payload["gate_verdict"] == "block"
    assert escalated.metadata["gate_verdict"] == "block"
    assert escalated.metadata["automatic_verdict"] == "allow"
    assert escalated.metadata["mode"] == "escalate"
    assert escalated.metadata["resolution_source"] == "standing"
    # Standing decided this, not a human: no override claimed anywhere.
    assert escalated.metadata["overridden"] is False
    assert not str(escalated.action.payload["gate_reason"]).startswith("manual override")
    bases = {tag["basis"] for tag in escalated.metadata["bossyk_sandbox_controls"]}
    assert "verdict:blocked" in bases
    assert "verdict:overridden" not in bases


def test_an_escalated_refund_does_not_consume_standing_authority() -> None:
    # 60 is within the 100 budget; the second 60 would make 120 -> escalated
    # (blocked). If that blocked refund still counted, the third refund (30)
    # would make 150 and escalate too; it must instead see 60 + 30 = 90.
    result, toolkit = _refund_budget_run([60.0, 60.0, 30.0])

    assert result.modes == ["allow", "allow", "allow", "escalate", "allow", "allow"]
    assert _executed_refunds(toolkit) == ["#R1", "#R3"]


class _RecordingToolkit(_FakeToolkit):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def use_tool(self, name: str, **args: Any) -> str:
        self.calls.append((name, args))
        return super().use_tool(name, **args)


class _RecordingEnv:
    policy = "You are a retail support agent."

    def __init__(self) -> None:
        self.tools = _RecordingToolkit()


def test_a_deferred_cancel_still_executes() -> None:
    # Over standing but reversible -> defer = accept + async-approve. The fix
    # for escalate must not turn this into a block.
    env = _RecordingEnv()
    session = build_retail_agent_session(llm=_standing_scenario_llm(), environment=env)
    grants = {"cancellation": StandingGrant("cancellation", max_count=2)}

    result = run_live_session(session, grants=grants)

    assert result.modes[-1] == "defer"
    cancelled = [
        args["order_id"] for name, args in env.tools.calls if name == "cancel_pending_order"
    ]
    assert cancelled == ["#A", "#B", "#C"]
    assert result.steps[-1].action.payload["gate_verdict"] == "allow"


def test_a_gate_blocked_escalation_is_not_attributed_to_standing() -> None:
    # Regression (Phase A Issues & Fixes 9): a refund / payment change with no
    # prior lookup is BLOCKed by the gate itself, then derives `escalate`
    # because it is irreversible. Its signed step must not claim standing
    # decided it -- no grants are even configured here.
    session = build_retail_agent_session(llm=_scripted_llm(), environment=_FakeEnv())

    result = run_live_session(session)  # grants=None

    gate_blocked_escalations = [
        step for step, mode in zip(result.steps, result.modes, strict=True) if mode == "escalate"
    ]
    assert len(gate_blocked_escalations) == 2
    for step in gate_blocked_escalations:
        assert step.action.payload["gate_verdict"] == "block"
        assert step.metadata["automatic_verdict"] == "block"
        assert step.metadata.get("resolution_source") != "standing"
