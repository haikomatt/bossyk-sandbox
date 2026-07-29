from __future__ import annotations

from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.standing_amount import (
    OrderItemView,
    OrderPaymentView,
    OrderView,
    resolve_amount,
)

# §F amount oracle (job #1): the amount moved by a proposed action is never in
# the tool's arguments (they carry only IDs) -- it must be resolved against
# order state. `resolve_amount` is the pure oracle: injected `OrderReader`,
# no tau2 import, no DB. See amount-gating-oracle-plan-v0.1.md for the
# per-boundary formula table this file exercises.


class _FakeReader:
    """A minimal in-memory `OrderReader`: order_id -> OrderView, or missing."""

    def __init__(self, orders: dict[str, OrderView]) -> None:
        self._orders = orders

    def get_order(self, order_id: str) -> OrderView | None:
        return self._orders.get(order_id)


def _order(items: list[OrderItemView], payments: list[OrderPaymentView]) -> OrderView:
    return OrderView(items=items, payment_history=payments)


ORDERS = {
    # A cancellable/refundable order: two items, one payment (the common case).
    "#W1": _order(
        items=[
            OrderItemView(item_id="item-1", price=19.99),
            OrderItemView(item_id="item-2", price=45.00),
        ],
        payments=[OrderPaymentView(amount=64.99)],
    ),
    # A multi-payment order (e.g. a partial gift-card top-up + card charge),
    # so cancellation's Σ-all-payments formula is distinguishable from
    # payment_change's first-payment-only formula.
    "#W2": _order(
        items=[
            OrderItemView(item_id="item-3", price=10.00),
            OrderItemView(item_id="item-4", price=20.00),
        ],
        payments=[
            OrderPaymentView(amount=25.00),
            OrderPaymentView(amount=5.00),
        ],
    ),
    # An order with no payment on record at all -- payment_change can't
    # resolve an exposure for it.
    "#W3": _order(items=[OrderItemView(item_id="item-5", price=8.00)], payments=[]),
}
READER = _FakeReader(ORDERS)


def _cancel(order_id: str) -> ProposedAction:
    return ProposedAction(
        "cancel_pending_order", {"order_id": order_id, "reason": "no longer needed"}
    )


def _refund(order_id: str, item_ids: list[str]) -> ProposedAction:
    return ProposedAction(
        "return_delivered_order_items",
        {"order_id": order_id, "item_ids": item_ids, "payment_method_id": "gift_card_1"},
    )


def _payment_change(order_id: str) -> ProposedAction:
    return ProposedAction(
        "modify_pending_order_payment", {"order_id": order_id, "payment_method_id": "gift_card_1"}
    )


def _address_change() -> ProposedAction:
    return ProposedAction(
        "modify_user_address",
        {
            "user_id": "U1",
            "address1": "1 Main St",
            "address2": "",
            "city": "Springfield",
            "state": "IL",
            "country": "USA",
            "zip": "11111",
        },
    )


def test_cancellation_amount_is_the_full_order_total() -> None:
    assert resolve_amount(_cancel("#W1"), READER) == 64.99


def test_cancellation_sums_every_payment_transaction() -> None:
    # #W2 has TWO payments (25.00 + 5.00) -- cancellation exposes the lot.
    assert resolve_amount(_cancel("#W2"), READER) == 30.00


def test_refund_amount_sums_returned_item_prices() -> None:
    assert resolve_amount(_refund("#W1", ["item-1"]), READER) == 19.99
    assert resolve_amount(_refund("#W1", ["item-1", "item-2"]), READER) == 64.99


def test_refund_amount_honours_duplicate_item_ids() -> None:
    # the tau2 tool permits repeated ids in one return -- multiplicity counts.
    assert resolve_amount(_refund("#W1", ["item-1", "item-1"]), READER) == 39.98


def test_refund_with_unknown_item_id_is_unresolvable() -> None:
    assert resolve_amount(_refund("#W1", ["item-1", "no-such-item"]), READER) is None


def test_payment_change_amount_is_the_first_payment_amount_not_the_sum() -> None:
    # #W2's payment_history is [25.00, 5.00] -- payment_change moves the
    # ORIGINAL payment (25.00), distinct from cancellation's sum (30.00).
    assert resolve_amount(_payment_change("#W2"), READER) == 25.00


def test_payment_change_with_no_payment_history_is_unresolvable() -> None:
    assert resolve_amount(_payment_change("#W3"), READER) is None


def test_account_change_has_no_amount() -> None:
    assert resolve_amount(_address_change(), READER) is None


def test_ungoverned_tool_has_no_amount() -> None:
    lookup = ProposedAction("get_order_details", {"order_id": "#W1"})
    assert resolve_amount(lookup, READER) is None


def test_missing_order_is_unresolvable() -> None:
    assert resolve_amount(_cancel("#does-not-exist"), READER) is None


def test_missing_order_id_argument_is_unresolvable() -> None:
    malformed = ProposedAction("cancel_pending_order", {"reason": "no longer needed"})
    assert resolve_amount(malformed, READER) is None
