"""RED-phase tests for the tokenisation proxy (bossyk-sandbox privacy/
minimisation demonstrator, Phase 3): inbound redaction of raw persona
values to stable surrogates, outbound rehydration, and the
`assert_no_raw_pii` trace-content guarantee Phase 4/5 will reuse.

All fixtures here are the committed `PERSONAS` set (deterministic, seeded --
see `advice.personas`) plus small hand-crafted `PersonaStore`s for the
overlapping-match test, where determinism matters more than realism. No
LLM, no console, no live agent -- fully hermetic per the plan.
"""

from __future__ import annotations

import pytest

from bossyk_sandbox.advice.personas import PERSONAS, PersonaRecord, PersonaStore
from bossyk_sandbox.proxy.tokenisation import (
    RawPiiLeakError,
    RedactedText,
    SurrogateMap,
    TokenisationProxy,
    assert_no_raw_pii,
    find_raw_pii,
    rehydrate,
)

STORE = PersonaStore(records=PERSONAS)


def _canonical_text(record: PersonaRecord) -> str:
    """Text using each field's exact store-format rendering -- the
    round-trip guarantee's scope (see the module docstring's
    "round-trip guarantee" convention)."""
    return (
        f"{record.full_name} lives at {record.postcode}, born {record.date_of_birth}, "
        f"earns {record.annual_income_gbp} a year."
    )


# --- round trip ---------------------------------------------------------


def test_round_trip_single_persona_all_fields() -> None:
    record = PERSONAS[0]
    proxy = TokenisationProxy(store=STORE)
    text = _canonical_text(record)

    redacted = proxy.redact(text)

    assert isinstance(redacted, RedactedText)
    assert record.full_name not in redacted.text
    assert record.postcode not in redacted.text
    assert record.date_of_birth not in redacted.text
    assert str(record.annual_income_gbp) not in redacted.text
    assert proxy.rehydrate(redacted.text) == text


def test_round_trip_multiple_personas_and_repeated_mentions() -> None:
    a, b = PERSONAS[0], PERSONAS[1]
    proxy = TokenisationProxy(store=STORE)
    text = (
        f"{a.full_name} asked about {a.full_name} again, and separately "
        f"{b.postcode} came up for {b.full_name}."
    )

    redacted = proxy.redact(text)

    # repeated mention of a.full_name -> same surrogate both times
    first_person_surrogate = "<PERSON_1>"
    assert redacted.text.count(first_person_surrogate) == 2
    assert proxy.rehydrate(redacted.text) == text


def test_round_trip_pii_free_text_is_identity() -> None:
    proxy = TokenisationProxy(store=STORE)
    text = "What is the current basic-rate tax band threshold?"

    redacted = proxy.redact(text)

    assert redacted.text == text
    assert proxy.rehydrate(redacted.text) == text


# --- no raw PII in redacted output --------------------------------------


def test_redacted_output_has_no_raw_pii_for_every_seeded_persona() -> None:
    proxy = TokenisationProxy(store=STORE)
    for record in PERSONAS:
        redacted = proxy.redact(_canonical_text(record))
        assert_no_raw_pii(redacted.text, STORE)  # must not raise


# --- surrogate stability + distinctness ---------------------------------


def test_surrogate_stability_within_a_session() -> None:
    record = PERSONAS[0]
    proxy = TokenisationProxy(store=STORE)

    first = proxy.redact(record.full_name)
    second = proxy.redact(f"and again, {record.full_name}")

    assert first.text == "<PERSON_1>"
    assert "<PERSON_1>" in second.text


def test_distinct_values_get_distinct_surrogates() -> None:
    a, b = PERSONAS[0], PERSONAS[1]
    proxy = TokenisationProxy(store=STORE)

    first = proxy.redact(a.full_name)
    second = proxy.redact(b.full_name)

    assert first.text != second.text
    assert first.text == "<PERSON_1>"
    assert second.text == "<PERSON_2>"


# --- normalisation -------------------------------------------------------


def test_name_case_variant_is_caught_and_shares_the_canonical_surrogate() -> None:
    record = PERSONAS[0]
    proxy = TokenisationProxy(store=STORE)

    canonical = proxy.redact(record.full_name)
    variant = proxy.redact(record.full_name.upper())

    assert canonical.text == variant.text  # same underlying value -> same surrogate
    assert_no_raw_pii(variant.text, STORE)


def test_postcode_spacing_variants_are_both_caught() -> None:
    record = PERSONAS[0]
    proxy = TokenisationProxy(store=STORE)
    compact = record.postcode.replace(" ", "")

    spaced = proxy.redact(record.postcode)
    unspaced = proxy.redact(compact)

    assert spaced.text == unspaced.text
    assert_no_raw_pii(unspaced.text, STORE)


def test_dob_ddmmyyyy_rendering_is_caught() -> None:
    record = PERSONAS[0]
    proxy = TokenisationProxy(store=STORE)
    year, month, day = record.date_of_birth.split("-")
    ddmmyyyy = f"{day}/{month}/{year}"

    iso = proxy.redact(record.date_of_birth)
    alt = proxy.redact(ddmmyyyy)

    assert iso.text == alt.text
    assert_no_raw_pii(alt.text, STORE)


@pytest.mark.parametrize("render", ["plain", "commas", "currency"])
def test_income_renderings_are_all_caught(render: str) -> None:
    record = PERSONAS[0]
    proxy = TokenisationProxy(store=STORE)
    rendered = {
        "plain": str(record.annual_income_gbp),
        "commas": f"{record.annual_income_gbp:,}",
        "currency": f"£{record.annual_income_gbp:,}",
    }[render]

    redacted = proxy.redact(f"income is {rendered} this year")

    assert str(record.annual_income_gbp) not in redacted.text
    assert_no_raw_pii(redacted.text, STORE)
    assert "<INCOME_1>" in redacted.text


# --- repr / str guard against accidental logging -------------------------


def test_surrogate_map_does_not_appear_in_proxy_repr_or_str() -> None:
    record = PERSONAS[0]
    proxy = TokenisationProxy(store=STORE)
    proxy.redact(_canonical_text(record))

    rendered_repr = repr(proxy)
    rendered_str = str(proxy)

    for leaked in (
        record.full_name,
        record.postcode,
        record.date_of_birth,
        str(record.annual_income_gbp),
        "SurrogateMap(",
    ):
        assert leaked not in rendered_repr
        assert leaked not in rendered_str


def test_surrogate_map_repr_does_not_expose_raw_values() -> None:
    record = PERSONAS[0]
    surrogate_map = SurrogateMap()
    surrogate_map.surrogate_for("PERSON", record.full_name)

    assert record.full_name not in repr(surrogate_map)
    assert record.full_name not in str(surrogate_map)


# --- assert_no_raw_pii / find_raw_pii guarantee helper --------------------


def test_assert_no_raw_pii_passes_on_clean_text() -> None:
    assert_no_raw_pii("nothing sensitive here", STORE)  # must not raise


@pytest.mark.parametrize(
    "make_text",
    [
        lambda r: r.full_name,
        lambda r: r.full_name.upper(),
        lambda r: r.postcode,
        lambda r: r.postcode.replace(" ", ""),
        lambda r: r.date_of_birth,
        lambda r: "/".join(reversed(r.date_of_birth.split("-"))),
        lambda r: str(r.annual_income_gbp),
        lambda r: f"{r.annual_income_gbp:,}",
        lambda r: f"£{r.annual_income_gbp:,}",
    ],
)
def test_assert_no_raw_pii_catches_a_deliberate_leak_in_each_rendering(make_text) -> None:  # type: ignore[no-untyped-def]
    record = PERSONAS[0]
    text = make_text(record)

    with pytest.raises(RawPiiLeakError):
        assert_no_raw_pii(text, STORE)

    assert find_raw_pii(text, STORE)


def test_health_condition_flag_is_documented_as_out_of_scope() -> None:
    # Booleans have no distinctive literal rendering to pattern-match --
    # see the module docstring's "documented boundary" note. "True"/"False"
    # must not be treated as PII (that would be an unusable false-positive
    # generator), so plain boolean text is never flagged.
    assert_no_raw_pii("health_condition_flag: True", STORE)


# --- overlapping matches: longest-match-first ----------------------------


def _crafted_store() -> PersonaStore:
    return PersonaStore(
        records=[
            PersonaRecord(
                ref="CRAFT-0001",
                full_name="Alex Fenwick",
                date_of_birth="1980-01-01",
                postcode="ZZ1 1AA",
                annual_income_gbp=40_000,
                health_condition_flag=False,
            ),
            PersonaRecord(
                ref="CRAFT-0002",
                full_name="Fenwick",
                date_of_birth="1990-02-02",
                postcode="ZZ2 2BB",
                annual_income_gbp=60_000,
                health_condition_flag=False,
            ),
        ]
    )


def test_overlapping_names_resolve_longest_match_first() -> None:
    store = _crafted_store()
    proxy = TokenisationProxy(store=store)

    redacted = proxy.redact("Alex Fenwick called today.")

    # "Alex Fenwick" (persona 1, the longer literal) wins whole; "Fenwick"
    # (persona 2) is NOT separately matched inside it.
    assert redacted.text == "<PERSON_1> called today."
    assert_no_raw_pii(redacted.text, store)


def test_overlapping_names_shorter_value_still_matches_standalone() -> None:
    store = _crafted_store()
    proxy = TokenisationProxy(store=store)

    redacted = proxy.redact("Fenwick called today.")

    assert redacted.text == "<PERSON_1> called today."
    assert_no_raw_pii(redacted.text, store)


def test_rehydrate_leaves_unknown_surrogate_tokens_untouched() -> None:
    assert rehydrate("<PERSON_99> was never minted", SurrogateMap()) == "<PERSON_99> was never minted"
