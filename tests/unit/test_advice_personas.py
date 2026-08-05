from __future__ import annotations

import pytest

from bossyk_sandbox.advice.personas import (
    PERSONAS,
    PersonaRecord,
    PersonaStore,
    generate_personas,
)


def test_default_persona_set_has_between_five_and_ten_records() -> None:
    assert 5 <= len(PERSONAS) <= 10


def test_persona_refs_are_unique() -> None:
    refs = [record.ref for record in PERSONAS]
    assert len(refs) == len(set(refs))


def test_persona_records_carry_the_required_fields() -> None:
    for record in PERSONAS:
        assert isinstance(record, PersonaRecord)
        assert record.ref
        assert record.full_name
        assert record.date_of_birth
        assert record.postcode
        assert record.annual_income_gbp > 0
        assert isinstance(record.health_condition_flag, bool)


def test_generate_personas_is_deterministic_for_the_same_seed() -> None:
    first = generate_personas(seed=12345)
    second = generate_personas(seed=12345)
    assert first == second


def test_generate_personas_differs_across_seeds() -> None:
    first = generate_personas(seed=1)
    second = generate_personas(seed=2)
    assert first != second


def test_generate_personas_spans_multiple_tax_bands() -> None:
    incomes = [record.annual_income_gbp for record in PERSONAS]
    assert min(incomes) <= 50_270
    assert max(incomes) > 125_140


def test_persona_store_get_returns_the_matching_record() -> None:
    store = PersonaStore(PERSONAS)
    target = PERSONAS[0]

    record = store.get(target.ref)

    assert record == target


def test_persona_store_get_raises_key_error_for_unknown_ref() -> None:
    store = PersonaStore(PERSONAS)

    with pytest.raises(KeyError):
        store.get("no-such-ref")


def test_persona_store_rejects_duplicate_refs() -> None:
    duplicate = PERSONAS[0]

    with pytest.raises(ValueError):
        PersonaStore([duplicate, duplicate])


def test_persona_store_refs_lists_every_record_ref() -> None:
    store = PersonaStore(PERSONAS)

    assert store.refs() == [record.ref for record in PERSONAS]
