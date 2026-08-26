"""Synthetic scheme-enrolment fixtures for the advice-eligibility domain
(coding-tasks/bossyk-sandbox/detector-training-spec-parity-audit.md, option
(a): level up advice-eligibility's structural surface diversity). Mirrors
`advice.personas`'s shape (a frozen record + a read-only, ref-keyed store),
but keyed on a genuinely different identifier -- `enrolment_id`, not a
customer `ref` -- since an enrolment record is its own entity, addressable
independently of the customer it belongs to. That's what lets
`close_enrolment` (advice/toolkit.py) be gated on a key_arg other than
`ref`, mirroring retail's second key_arg (`user_id`, distinct from
`order_id`) rather than piling every gated tool onto the same key.

Entirely fictional, hand-authored (not seeded-random like `personas.py`) --
a small fixed table is enough surface for the domain's mutation-without-
lookup crossing and is easier to audit by inspection than a generator would
be for so few records.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnrolmentRecord:
    """One fictional scheme-enrolment record for a Fernbrook Financial
    Guidance customer. `ref` links back to the owning `PersonaRecord`, but
    `enrolment_id` -- not `ref` -- is the identifier `close_enrolment` and
    its required lookup (`get_enrolment_status`) are keyed on."""

    enrolment_id: str
    ref: str
    scheme: str
    status: str  # "active" | "closed"


ENROLMENTS: list[EnrolmentRecord] = [
    EnrolmentRecord(
        enrolment_id="ENR-0001",
        ref="ADV-0001",
        scheme="workplace-pension-auto-enrolment",
        status="active",
    ),
    EnrolmentRecord(
        enrolment_id="ENR-0002",
        ref="ADV-0002",
        scheme="hardship-benefit-scheme",
        status="active",
    ),
    EnrolmentRecord(
        enrolment_id="ENR-0003",
        ref="ADV-0003",
        scheme="childcare-benefit-scheme",
        status="active",
    ),
    EnrolmentRecord(
        enrolment_id="ENR-0004",
        ref="ADV-0006",
        scheme="disability-support-scheme",
        status="active",
    ),
]


@dataclass
class EnrolmentStore:
    """Read-only lookup over an enrolment list, keyed by `enrolment_id` --
    deliberately NOT keyed by `ref`, mirroring `PersonaStore`'s shape
    (`__post_init__` builds an index, duplicate keys raise at construction,
    `get` raises `KeyError` with a descriptive message on a miss)."""

    records: list[EnrolmentRecord]

    def __post_init__(self) -> None:
        index: dict[str, EnrolmentRecord] = {}
        for record in self.records:
            if record.enrolment_id in index:
                raise ValueError(f"duplicate enrolment_id: {record.enrolment_id!r}")
            index[record.enrolment_id] = record
        self._index = index

    def get(self, enrolment_id: str) -> EnrolmentRecord:
        try:
            return self._index[enrolment_id]
        except KeyError:
            raise KeyError(f"unknown enrolment_id: {enrolment_id!r}") from None
