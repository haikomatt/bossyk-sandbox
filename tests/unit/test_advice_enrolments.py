"""Tests for the enrolment fixture added alongside the advice-eligibility
domain's second and third gated mutation surfaces (spec-parity audit,
coding-tasks/bossyk-sandbox/detector-training-spec-parity-audit.md, option
(a)). Mirrors test_advice_personas.py's shape but for `EnrolmentStore`,
which is deliberately keyed by `enrolment_id` rather than `ref`.
"""

from __future__ import annotations

import pytest

from bossyk_sandbox.advice.enrolments import ENROLMENTS, EnrolmentRecord, EnrolmentStore


def test_default_enrolment_set_is_non_empty() -> None:
    assert len(ENROLMENTS) >= 3


def test_enrolment_ids_are_unique() -> None:
    ids = [record.enrolment_id for record in ENROLMENTS]
    assert len(ids) == len(set(ids))


def test_enrolment_records_carry_the_required_fields() -> None:
    for record in ENROLMENTS:
        assert isinstance(record, EnrolmentRecord)
        assert record.enrolment_id
        assert record.ref
        assert record.scheme
        assert record.status in {"active", "closed"}


def test_enrolment_store_get_returns_the_matching_record() -> None:
    store = EnrolmentStore(ENROLMENTS)
    target = ENROLMENTS[0]

    record = store.get(target.enrolment_id)

    assert record == target


def test_enrolment_store_get_raises_key_error_for_unknown_enrolment_id() -> None:
    store = EnrolmentStore(ENROLMENTS)

    with pytest.raises(KeyError):
        store.get("no-such-enrolment-id")


def test_enrolment_store_rejects_duplicate_enrolment_ids() -> None:
    duplicate = ENROLMENTS[0]

    with pytest.raises(ValueError):
        EnrolmentStore([duplicate, duplicate])


def test_enrolment_store_is_keyed_by_enrolment_id_not_ref() -> None:
    """The whole point of this fixture: `get` must resolve by
    `enrolment_id`, and a customer `ref` must not also work as a lookup key
    (that would silently collapse the two identifiers back together)."""
    store = EnrolmentStore(ENROLMENTS)
    target = ENROLMENTS[0]
    assert target.enrolment_id != target.ref

    with pytest.raises(KeyError):
        store.get(target.ref)
