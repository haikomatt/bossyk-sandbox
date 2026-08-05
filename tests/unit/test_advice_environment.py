from __future__ import annotations

import pytest

from bossyk_sandbox.advice.environment import AdviceEnvironment, get_advice_environment
from bossyk_sandbox.advice.personas import PersonaRecord, PersonaStore
from bossyk_sandbox.advice.tax import TaxBand, contribution_headroom_band, tax_band
from bossyk_sandbox.advice.toolkit import ELIGIBILITY_TOOL_NAMES, RAW_TOOL_NAMES, AdviceToolkit

_RECORD = PersonaRecord(
    ref="ADV-TEST-1",
    full_name="Test Persona",
    date_of_birth="1980-01-01",
    postcode="ZZ1 1AA",
    annual_income_gbp=80_000,
    health_condition_flag=True,
)
_STORE = PersonaStore([_RECORD])


def _toolkit(*, minimisation: bool) -> AdviceToolkit:
    return AdviceToolkit(store=_STORE, minimisation=minimisation)


def test_get_advice_environment_returns_policy_and_tools() -> None:
    env = get_advice_environment()

    assert isinstance(env, AdviceEnvironment)
    assert isinstance(env.policy, str)
    assert env.policy.strip()
    assert isinstance(env.tools, AdviceToolkit)


def test_get_advice_environment_defaults_to_minimisation_off() -> None:
    env = get_advice_environment()

    assert env.tools.minimisation is False


def test_minimisation_off_offers_both_raw_and_derived_tools() -> None:
    toolkit = _toolkit(minimisation=False)

    offered = set(toolkit.get_tools())

    assert "get_customer_profile" in offered
    assert "get_tax_position" in offered
    assert "get_contribution_headroom" in offered
    assert "is_income_above" in offered


def test_minimisation_on_offers_only_derived_tools_plus_the_eligibility_surface() -> None:
    # Test-integrity note (advice-eligibility domain spec, build order step
    # 2): this used to assert an exact 3-tool set. That assumption broke
    # when the eligibility surface (verify_eligibility,
    # submit_eligibility_decision) was added -- it is orthogonal to the
    # raw/derived minimisation toggle (a different boundary, structural
    # mutation-without-lookup) and so is offered under minimisation=True
    # too. Updated to assert what minimisation actually still governs: the
    # raw tool stays excluded, the derived tools stay included.
    toolkit = _toolkit(minimisation=True)

    offered = set(toolkit.get_tools())

    assert (
        offered
        == {
            "get_tax_position",
            "get_contribution_headroom",
            "is_income_above",
        }
        | ELIGIBILITY_TOOL_NAMES
    )
    assert offered.isdisjoint(RAW_TOOL_NAMES)


def test_use_tool_executes_an_offered_tool() -> None:
    toolkit = _toolkit(minimisation=False)

    result = toolkit.use_tool("get_tax_position", ref=_RECORD.ref)

    assert result == tax_band(_RECORD.annual_income_gbp).value


def test_use_tool_raises_when_the_tool_is_not_offered_under_minimisation() -> None:
    toolkit = _toolkit(minimisation=True)

    with pytest.raises(KeyError):
        toolkit.use_tool("get_customer_profile", ref=_RECORD.ref)


def test_raw_and_derived_surfaces_read_the_same_store() -> None:
    toolkit = _toolkit(minimisation=False)

    profile = toolkit.get_customer_profile(_RECORD.ref)
    band = toolkit.get_tax_position(_RECORD.ref)

    assert profile["annual_income_gbp"] == _RECORD.annual_income_gbp
    assert band == tax_band(_RECORD.annual_income_gbp).value


def test_get_customer_profile_returns_the_full_record_including_health_flag() -> None:
    toolkit = _toolkit(minimisation=False)

    profile = toolkit.get_customer_profile(_RECORD.ref)

    assert profile["ref"] == _RECORD.ref
    assert profile["full_name"] == _RECORD.full_name
    assert profile["date_of_birth"] == _RECORD.date_of_birth
    assert profile["postcode"] == _RECORD.postcode
    assert profile["health_condition_flag"] == _RECORD.health_condition_flag


def test_get_tax_position_returns_the_band_value_not_the_raw_income() -> None:
    toolkit = _toolkit(minimisation=False)

    band = toolkit.get_tax_position(_RECORD.ref)

    assert band == TaxBand.HIGHER.value


def test_get_contribution_headroom_matches_the_pure_function() -> None:
    toolkit = _toolkit(minimisation=False)

    headroom = toolkit.get_contribution_headroom(_RECORD.ref)

    assert headroom == contribution_headroom_band(_RECORD.annual_income_gbp)


def test_is_income_above_is_true_just_below_income_and_false_at_or_above() -> None:
    toolkit = _toolkit(minimisation=False)

    assert toolkit.is_income_above(_RECORD.ref, _RECORD.annual_income_gbp - 1) is True
    assert toolkit.is_income_above(_RECORD.ref, _RECORD.annual_income_gbp) is False
    assert toolkit.is_income_above(_RECORD.ref, _RECORD.annual_income_gbp + 1) is False


def test_unknown_ref_raises_key_error_through_every_tool() -> None:
    toolkit = _toolkit(minimisation=False)

    with pytest.raises(KeyError):
        toolkit.get_customer_profile("no-such-ref")
    with pytest.raises(KeyError):
        toolkit.get_tax_position("no-such-ref")
    with pytest.raises(KeyError):
        toolkit.get_contribution_headroom("no-such-ref")
    with pytest.raises(KeyError):
        toolkit.is_income_above("no-such-ref", 1000)
