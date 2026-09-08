"""Drive a live LangGraph agent session through the gate, deriving a resolution
mode per proposed tool call.

`run_live_session` is the pure, deterministic core (the analogue of
`replay.drive_replay`): it drives the real agent graph's interrupt chain,
resumes each held call with the gate's OWN verdict (enforcing the gate -- no
human override, so a blocked write never executes), derives the resolution mode
from that verdict + the tool's consequence, and accumulates the hard-cell HITL
queue.

It is billable ONLY when the injected agent session was built with a real LLM
client; built with a fake LLM + fake tau2 env it runs entirely offline (see
tests/unit/test_live_session.py). This module never constructs an LLM itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from auditk.schema import Step
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from bossyk_sandbox.console.modes import SEVERITY_RANK, ModeDecision, derive_mode
from bossyk_sandbox.instruments.base import ProposedAction, Verdict
from bossyk_sandbox.runtime.langgraph_agent import AgentSession, AgentState
from bossyk_sandbox.standing import (
    AuthorityVerdict,
    StandingGrant,
    TimedAction,
    evaluate_authority,
)
from bossyk_sandbox.standing_amount import (
    OrderItemView,
    OrderPaymentView,
    OrderReader,
    OrderView,
    resolve_amount,
)

_DEFAULT_USER_MESSAGE = "Hi, I need help with a couple of orders on my account."


@dataclass(frozen=True)
class LiveResult:
    """The outcome of driving a live session: per-call gate verdicts + derived
    resolution modes, the attested steps the graph produced, and the
    priority-ordered hard-cell HITL queue."""

    verdicts: list[str]
    modes: list[str]
    steps: list[Step]
    hitl_queue: list[dict[str, str]]


def _interrupt_payload(chunk: Any) -> dict[str, object] | None:
    """The approve-node interrupt payload in a stream chunk, or None. LangGraph
    surfaces an interrupt as `{"__interrupt__": (Interrupt(value=...),)}`."""
    interrupts = chunk.get("__interrupt__")
    if not interrupts:
        return None
    payload = interrupts[0].value
    assert isinstance(payload, dict)
    return payload


def initial_input(user_message: str = _DEFAULT_USER_MESSAGE) -> AgentState:
    """The graph's opening state: just the customer's first message."""
    return cast(AgentState, {"messages": [HumanMessage(content=user_message)]})


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def advance_to_interrupt(graph: Any, inp: Any, config: RunnableConfig) -> dict[str, object] | None:
    """Run one stream segment to the next approve-node interrupt; return its
    payload, or None if the graph reached END. Blocking -- the agent's LLM call
    runs here, so the async console wrapper offloads it to a thread. `inp` is the
    initial state on the first segment, then a `Command(resume=...)` thereafter."""
    payload: dict[str, object] | None = None
    for chunk in graph.stream(inp, config):
        found = _interrupt_payload(chunk)
        if found is not None:
            payload = found
    return payload


def _proposed_of(payload: dict[str, object]) -> ProposedAction:
    return ProposedAction(str(payload["tool_name"]), cast(dict[str, Any], payload["arguments"]))


def order_reader_from_toolkit(toolkit: Any) -> OrderReader:
    """Adapt a live tau2 environment's toolkit to the §F amount oracle's
    `OrderReader` Protocol, so `resolve_amount` never has to import tau2
    itself (locked design decision #2). Reads `toolkit.db.orders[order_id]`
    -- the shape every tau2 domain's `ToolKitBase` exposes (`db: DB`) and the
    retail domain's `RetailTools`/`RetailDB` concretely provide (`orders:
    Dict[str, Order]`, each with `.items[].item_id/.price` and
    `.payment_history[].amount`) -- and copies just the fields the oracle
    needs into `OrderView`. Never raises: any missing attribute or unknown
    order id resolves to `None` (the oracle's own fail-safe signal)."""

    class _ToolkitOrderReader:
        def get_order(self, order_id: str) -> OrderView | None:
            orders = getattr(getattr(toolkit, "db", None), "orders", None)
            if not orders:
                return None
            order = orders.get(order_id)
            if order is None:
                return None
            try:
                items = [
                    OrderItemView(item_id=item.item_id, price=item.price) for item in order.items
                ]
                payments = [
                    OrderPaymentView(amount=payment.amount) for payment in order.payment_history
                ]
            except AttributeError:
                return None
            return OrderView(items=items, payment_history=payments)

    return _ToolkitOrderReader()


def authority_for(
    payload: dict[str, object],
    history: Sequence[ProposedAction | TimedAction],
    grants: dict[str, StandingGrant] | None,
    *,
    now: float | None = None,
    amount: float | None = None,
    restrictions: frozenset[tuple[str, str]] = frozenset(),
) -> AuthorityVerdict | None:
    """The §F standing verdict for a held call, or None when no policy is set
    (back-compat: `derive_mode` then keeps its pre-§F ALLOW behaviour). `now`
    (epoch seconds) time-bounds windowed grants; session-scoped grants ignore
    it. `amount` is the call's resolved value (from `resolve_amount`,
    threaded by the caller); a grant with no `max_amount` ignores it entirely.
    `restrictions` is the session's co-occurrence restriction set (composition
    closure); the default empty set changes nothing."""
    if not grants:
        return None
    return evaluate_authority(
        _proposed_of(payload),
        history,
        grants,
        now=now,
        amount=amount,
        restrictions=restrictions,
    )


def resolve_call(
    payload: dict[str, object], *, authority: AuthorityVerdict | None = None
) -> tuple[Verdict, ModeDecision]:
    """The gate verdict and derived resolution mode for one held call."""
    verdict = Verdict(str(payload["auto_verdict"]))
    return verdict, derive_mode(str(payload["tool_name"]), verdict, authority=authority)


def hitl_item(payload: dict[str, object], decision: ModeDecision) -> dict[str, str]:
    """The hard-cell HITL item for an escalated call."""
    assert decision.hitl is not None
    tool = str(payload["tool_name"])
    return {**decision.hitl, "label": tool, "tool_name": tool}


def sort_hitl_queue(queue: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(queue, key=lambda item: SEVERITY_RANK.get(item["severity"], 99))


def resume_after(payload: dict[str, object]) -> Command[Any]:
    """Resume the graph enforcing the gate's own verdict (no human override)."""
    return Command(resume=payload["auto_verdict"])


def run_live_session(
    session: AgentSession,
    *,
    thread_id: str = "live-session-1",
    user_message: str = _DEFAULT_USER_MESSAGE,
    grants: dict[str, StandingGrant] | None = None,
    reader: OrderReader | None = None,
    clock: Callable[[], float] = time.time,
    restrictions: frozenset[tuple[str, str]] = frozenset(),
) -> LiveResult:
    """Drive a live session to completion synchronously (used offline in tests
    with a fake LLM). The console's async wrapper reuses the same helpers.

    `grants` is the §F standing policy (None = no authority model, pre-§F
    behaviour). Authority is consumed by ALLOWed actions: a boundary's count is
    the session's prior *allowed* actions at that boundary. `clock` stamps each
    allowed action with the wall-clock time it was taken so **windowed** grants
    can expire standing; it is injected (defaulting to `time.time`) so tests stay
    deterministic. Session-scoped grants ignore the timestamps.

    `reader` (an `OrderReader`, e.g. `order_reader_from_toolkit(env.tools)`)
    resolves each held call's amount via the §F oracle so **amount-gated**
    grants (`max_amount`) can be evaluated; without one, `amount` stays `None`
    for every call (back-compat: a grant with no `max_amount` is unaffected
    either way). The resolved amount is stamped onto the recorded
    `TimedAction`, exactly like `now`, so a later call's cumulative budget
    check sees it.

    `restrictions` is the co-occurrence restriction set (composition closure,
    `standing.evaluate_authority`): boundary pairs that may not both be
    crossed by ALLOWed actions in one session. Only meaningful alongside
    `grants`; the default empty set leaves the session byte-identical."""
    graph = session.graph
    config = thread_config(thread_id)

    verdicts: list[str] = []
    modes: list[str] = []
    hitl_queue: list[dict[str, str]] = []
    history: list[TimedAction] = []

    inp: Any = initial_input(user_message)
    while True:
        payload = advance_to_interrupt(graph, inp, config)
        if payload is None:
            break  # the graph reached END with no further held call
        now = clock()
        proposed = _proposed_of(payload)
        amount = resolve_amount(proposed, reader) if reader is not None else None
        authority = authority_for(
            payload, history, grants, now=now, amount=amount, restrictions=restrictions
        )
        verdict, decision = resolve_call(payload, authority=authority)
        verdicts.append(verdict.value)
        modes.append(decision.mode)
        if decision.mode == "escalate" and decision.hitl is not None:
            hitl_queue.append(hitl_item(payload, decision))
        if verdict is Verdict.ALLOW:
            history.append(TimedAction(proposed, now, amount=amount))
        inp = resume_after(payload)

    return LiveResult(
        verdicts=verdicts,
        modes=modes,
        steps=list(session.steps),
        hitl_queue=sort_hitl_queue(hitl_queue),
    )
