"""Tests for the eligibility surface added to `AdviceToolkit` (advice-eligibility
domain spec, build order step 2): a required-lookup tool (`verify_eligibility`)
and a gated mutation tool (`submit_eligibility_decision`), added alongside the
existing raw/derived read tools. Unlike the raw-vs-derived split, the
eligibility tools are offered under BOTH `minimisation=True` and `False` --
they are an orthogonal concern (structural mutation-without-lookup), not part
of the privacy/minimisation toggle.
"""

from __future__ import annotations

import pytest

from bossyk_sandbox.advice.personas import PersonaRecord, PersonaStore
from bossyk_sandbox.advice.toolkit import ELIGIBILITY_TOOL_NAMES, AdviceToolkit

_RECORD = PersonaRecord(
    ref="ADV-TEST-1",
    full_name="Test Persona",
    date_of_birth="1980-01-01",
    postcode="ZZ1 1AA",
    annual_income_gbp=80_000,
    health_condition_flag=True,
)
_STORE = PersonaStore([_RECORD])


def _toolkit(*, minimisation: bool = False) -> AdviceToolkit:
    return AdviceToolkit(store=_STORE, minimisation=minimisation)


def test_eligibility_tool_names_are_verify_and_submit() -> None:
    assert ELIGIBILITY_TOOL_NAMES == {"verify_eligibility", "submit_eligibility_decision"}


def test_minimisation_off_offers_the_eligibility_tools_too() -> None:
    toolkit = _toolkit(minimisation=False)

    offered = set(toolkit.get_tools())

    assert ELIGIBILITY_TOOL_NAMES <= offered


def test_minimisation_on_still_offers_the_eligibility_tools() -> None:
    """The eligibility surface is orthogonal to the raw/derived minimisation
    toggle -- it must be reachable regardless of `minimisation`."""
    toolkit = _toolkit(minimisation=True)

    offered = set(toolkit.get_tools())

    assert ELIGIBILITY_TOOL_NAMES <= offered


def test_verify_eligibility_returns_ref_and_a_qualifying_band() -> None:
    toolkit = _toolkit()

    result = toolkit.verify_eligibility(ref=_RECORD.ref)

    assert result["ref"] == _RECORD.ref
    assert "eligible" in result
    assert isinstance(result["eligible"], bool)
    assert "qualifying_income_band" in result
    assert isinstance(result["qualifying_income_band"], str)


def test_verify_eligibility_never_discloses_the_raw_income() -> None:
    toolkit = _toolkit()

    result = toolkit.verify_eligibility(ref=_RECORD.ref)

    assert _RECORD.annual_income_gbp not in result.values()


def test_submit_eligibility_decision_returns_the_submitted_fields() -> None:
    toolkit = _toolkit()

    result = toolkit.submit_eligibility_decision(
        ref=_RECORD.ref, scheme="workplace-pension-topup", decision="enrol"
    )

    assert result["ref"] == _RECORD.ref
    assert result["scheme"] == "workplace-pension-topup"
    assert result["decision"] == "enrol"


def test_submit_eligibility_decision_raises_for_an_unknown_ref() -> None:
    toolkit = _toolkit()

    with pytest.raises(KeyError):
        toolkit.submit_eligibility_decision(ref="no-such-ref", scheme="s", decision="enrol")


def test_use_tool_can_invoke_both_eligibility_tools_by_name() -> None:
    toolkit = _toolkit()

    verify_result = toolkit.use_tool("verify_eligibility", ref=_RECORD.ref)
    submit_result = toolkit.use_tool(
        "submit_eligibility_decision", ref=_RECORD.ref, scheme="childcare-benefit", decision="enrol"
    )

    assert verify_result["ref"] == _RECORD.ref
    assert submit_result["scheme"] == "childcare-benefit"
