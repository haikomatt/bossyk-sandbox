"""§F amount oracle -- bossyk-proprietary, part of the standing/authority model.

The last un-built half of §F job #1: gate standing not just by *how many
times* an action is taken (`standing.py`'s count-only model) but by *how much
value* it moves. The decisive fact (grounded against the tau2 retail domain,
`tau2/domains/retail/{tools,data_model}.py`): **the amount is never in a
tool's arguments** -- `cancel_pending_order`, `return_delivered_order_items`
and `modify_pending_order_payment` all carry only IDs. The amount must be
resolved against order state, which is a lookup + sum, not a pricing engine
(`OrderItem.price` and `OrderPayment.amount` are already stored at purchase
time).

`resolve_amount` is that resolver. It depends on a minimal `OrderReader`
Protocol, never on tau2 directly, so `standing.py` and this module stay pure
and unit-testable with no DB/network: the live driver adapts the real tau2
environment to `OrderReader` (see `console.live.order_reader_from_toolkit`);
tests inject a fake reader. `resolve_amount` never raises -- an unresolvable
amount (missing order, invalid item id, no payment on record) is reported as
`None`, which `standing.evaluate_authority` treats as a fail-safe signal to
escalate (never a silent allow). Kept proprietary, not proposed to
auditk-spec (per demonstrator §13), same as the rest of §F.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.standing import boundaries_for_tool

# The consequence boundaries that have a resolvable amount at all.
# `account_change` (modify_user_address) moves no value -- it stays
# PII/step-up gated, never amount-gated (locked design decision #3).
_PRICED_BOUNDARIES = {"cancellation", "refund", "payment_change"}


@dataclass(frozen=True)
class OrderItemView:
    """One order line, priced at purchase time (mirrors tau2's `OrderItem`:
    `item_id` + `price`, nothing this oracle doesn't need)."""

    item_id: str
    price: float


@dataclass(frozen=True)
class OrderPaymentView:
    """One payment/refund transaction against an order (mirrors tau2's
    `OrderPayment`: just the `amount` this oracle needs)."""

    amount: float


@dataclass(frozen=True)
class OrderView:
    """The minimal slice of order state the oracle needs: item prices (to sum
    a partial refund) and payment history (to size the order's total
    exposure). Deliberately NOT tau2's `Order` -- this module must not import
    tau2 (see module docstring); the live driver adapts the real order into
    this shape."""

    items: Sequence[OrderItemView]
    payment_history: Sequence[OrderPaymentView]


class OrderReader(Protocol):
    """Resolves an order id to its priced state, or `None` if it can't
    (missing/invalid id). The oracle's only dependency -- a Protocol, not
    tau2, so it stays swappable between the real live adapter and a fake."""

    def get_order(self, order_id: str) -> OrderView | None: ...


def _order_total(order: OrderView) -> float:
    """Cancellation's exposure: the sum of EVERY payment transaction against
    the order (a cancel refunds everything paid so far, however it was
    paid) -- see the plan's per-boundary formula table."""
    return sum(payment.amount for payment in order.payment_history)


def _payment_change_exposure(order: OrderView) -> float | None:
    """`modify_pending_order_payment`'s exposure = the order's ORIGINAL
    payment amount (decision #3: the full value that would be redirected to
    the new payment method -- exposure, not net P&L). Distinct from
    cancellation's Σ-all-payments: a multi-payment order's payment_change
    amount is `payment_history[0].amount` only. `None` if there's no payment
    on record to redirect -- unresolvable, never a silent zero."""
    if not order.payment_history:
        return None
    return order.payment_history[0].amount


def _refund_amount(order: OrderView, item_ids: Sequence[object]) -> float | None:
    """Sum of `order.items[].price` over `item_ids`, honouring duplicate
    multiplicity -- the tau2 tool explicitly permits repeated ids in one
    return, and each occurrence returns (and refunds) that item again.
    `None` if any id doesn't resolve to a priced item on this order --
    unresolvable, never silently priced at zero."""
    prices = {item.item_id: item.price for item in order.items}
    total = 0.0
    for item_id in item_ids:
        if not isinstance(item_id, str) or item_id not in prices:
            return None
        total += prices[item_id]
    return total


def resolve_amount(proposed: ProposedAction, reader: OrderReader) -> float | None:
    """The value `proposed` moves, resolved against order state via `reader`,
    or `None` if it can't be resolved: not an amount-gated boundary (e.g.
    `account_change` or an ungoverned lookup), a missing/malformed order id,
    a missing order, unresolvable item ids, or a payment_change order with no
    payment on record.

    Never raises. `None` is a signal, not an error: the caller (`standing`'s
    amount-gated `evaluate_authority`) treats it as fail-safe -- escalate,
    never silently allow -- so this oracle can stay conservative rather than
    guessing."""
    priced = boundaries_for_tool(proposed.tool_name) & _PRICED_BOUNDARIES
    if len(priced) != 1:
        # Not an amount-gated boundary, or (never today) a tool spanning two
        # priced boundaries whose amount would be ambiguous -- unresolvable.
        return None
    boundary = next(iter(priced))

    order_id = proposed.arguments.get("order_id")
    if not isinstance(order_id, str):
        return None
    order = reader.get_order(order_id)
    if order is None:
        return None

    if boundary == "cancellation":
        return _order_total(order)
    if boundary == "payment_change":
        return _payment_change_exposure(order)

    item_ids = proposed.arguments.get("item_ids")
    if not isinstance(item_ids, list):
        return None
    return _refund_amount(order, item_ids)
