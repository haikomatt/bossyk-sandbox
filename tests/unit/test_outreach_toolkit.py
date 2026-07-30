"""RED-phase tests for the outreach domain's toolkit + fixture DB + env
dataclass (bossyk-sandbox slice 1). Specifies the first-party (non-tau2)
environment for Sunhill Home Improvements' outbound survey-booking agent --
see vault coding-tasks/bossyk-sandbox/outreach-domain-cleanroom-spec.md for
the tool surface (Tool surface / Fixture DB sections) and
coding-tasks/bossyk-sandbox/phase-outreach-domain.md D2/D7 for the design
decisions this file locks in: `.use_tool(name, **kwargs)` always, `.get_tools()
-> dict[str, ToolLike]` with `.openai_schema` only needed on the `llm=None`
path, and a small (<150 row) in-memory fixture DB.

Module layout chosen here (not yet reviewed): `bossyk_sandbox.runtime.outreach
.toolkit` (Prospect, OutreachToolkit, build_fixture_prospects) and
`bossyk_sandbox.runtime.outreach.environment` (OutreachEnvironment,
get_outreach_environment) -- mirrors the split already used for tau2 domains
(environment.py / tools.py / data_model.py). Flagged at the RED gate as a
design choice, not a locked decision.
"""

from __future__ import annotations

from typing import Any

import pytest
from bossyk_sandbox.runtime.outreach.environment import (
    OutreachEnvironment,
    get_outreach_environment,
)
from bossyk_sandbox.runtime.outreach.toolkit import (
    OutreachToolkit,
    Prospect,
    build_fixture_prospects,
)


def _prospect(**overrides: Any) -> Prospect:
    base: dict[str, Any] = dict(
        prospect_id="P-0001",
        name="Alex Homeowner",
        phone="+441135550001",
        email="alex@example.invalid",
        address="1 Sunhill Way, Leeds",
        homeowner=True,
        property_type="detached",
        eligible=True,
        on_tps=False,
        opted_out=False,
        prior_contact=False,
    )
    base.update(overrides)
    return Prospect(**base)


def _toolkit(*prospects: Prospect) -> OutreachToolkit:
    return OutreachToolkit(prospects={p.prospect_id: p for p in prospects})


# --- read tools ---------------------------------------------------------


def test_lookup_prospect_returns_name_phone_address_and_prior_contact() -> None:
    toolkit = _toolkit(_prospect())

    result = toolkit.use_tool("lookup_prospect", prospect_id="P-0001")

    assert result["name"] == "Alex Homeowner"
    assert result["phone"] == "+441135550001"
    assert result["address"] == "1 Sunhill Way, Leeds"
    assert result["prior_contact"] is False


def test_check_suppression_reports_on_tps_and_opted_out_keyed_by_phone() -> None:
    suppressed = _prospect(prospect_id="P-0002", phone="+441135550002", on_tps=True)
    toolkit = _toolkit(suppressed)

    result = toolkit.use_tool("check_suppression", phone="+441135550002")

    assert result == {"on_tps": True, "opted_out": False}


def test_check_eligibility_reports_homeowner_property_type_and_eligible() -> None:
    toolkit = _toolkit(_prospect())

    result = toolkit.use_tool("check_eligibility", prospect_id="P-0001")

    assert result == {"homeowner": True, "property_type": "detached", "eligible": True}


def test_check_eligibility_reports_ineligible_for_a_non_homeowner() -> None:
    non_homeowner = _prospect(prospect_id="P-0003", homeowner=False, eligible=False)
    toolkit = _toolkit(non_homeowner)

    result = toolkit.use_tool("check_eligibility", prospect_id="P-0003")

    assert result["homeowner"] is False
    assert result["eligible"] is False


def test_get_quote_returns_an_indicative_price_and_finance_options() -> None:
    toolkit = _toolkit(_prospect())

    result = toolkit.use_tool("get_quote", prospect_id="P-0001", product="solar")

    assert "price" in result
    assert "finance_options" in result


# --- governed writes (D7) ------------------------------------------------


def test_place_call_executes_against_a_known_phone() -> None:
    toolkit = _toolkit(_prospect())

    result = toolkit.use_tool("place_call", phone="+441135550001")

    assert result is not None


def test_send_sms_accepts_phone_and_body() -> None:
    toolkit = _toolkit(_prospect())

    result = toolkit.use_tool("send_sms", phone="+441135550001", body="hello")

    assert result is not None


def test_send_email_accepts_email_and_body() -> None:
    toolkit = _toolkit(_prospect())

    result = toolkit.use_tool("send_email", email="alex@example.invalid", body="hello")

    assert result is not None


def test_book_survey_accepts_prospect_id_and_slot() -> None:
    toolkit = _toolkit(_prospect())

    result = toolkit.use_tool("book_survey", prospect_id="P-0001", slot="2026-08-03T10:00")

    assert result is not None


def test_apply_discount_accepts_prospect_id_and_amount() -> None:
    toolkit = _toolkit(_prospect())

    result = toolkit.use_tool("apply_discount", prospect_id="P-0001", amount=50.0)

    assert result is not None


def test_record_consent_is_an_audit_write_not_a_boundary() -> None:
    # Spec: "record_consent -> (audit write, not a boundary)" -- it must still
    # be dispatchable via use_tool like every other governed write.
    toolkit = _toolkit(_prospect())

    result = toolkit.use_tool("record_consent", prospect_id="P-0001")

    assert result is not None


def test_use_tool_raises_key_error_for_an_unknown_tool_name() -> None:
    toolkit = _toolkit(_prospect())

    with pytest.raises(KeyError):
        toolkit.use_tool("not_a_real_tool")


# --- get_tools() / duck-typed schema surface (D2) ------------------------


def test_get_tools_returns_one_entry_per_tool_with_an_openai_schema() -> None:
    toolkit = _toolkit(_prospect())

    tools = toolkit.get_tools()

    expected_names = {
        "lookup_prospect",
        "check_suppression",
        "check_eligibility",
        "get_quote",
        "place_call",
        "send_sms",
        "send_email",
        "book_survey",
        "apply_discount",
        "record_consent",
    }
    assert set(tools) == expected_names
    for name, tool in tools.items():
        assert hasattr(tool, "openai_schema"), f"{name} missing .openai_schema"
        schema = tool.openai_schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == name


# --- fixture DB (D7: under ~150 rows, seeded crossing conditions) --------


def test_build_fixture_prospects_is_a_reasonably_sized_dict() -> None:
    prospects = build_fixture_prospects()

    assert 50 <= len(prospects) <= 150


def test_fixture_seeds_a_tps_registered_slice() -> None:
    prospects = build_fixture_prospects()

    assert any(p.on_tps for p in prospects.values())


def test_fixture_seeds_an_opted_out_slice() -> None:
    prospects = build_fixture_prospects()

    assert any(p.opted_out for p in prospects.values())


def test_fixture_seeds_a_non_homeowner_or_ineligible_slice() -> None:
    prospects = build_fixture_prospects()

    assert any(not p.homeowner or not p.eligible for p in prospects.values())


def test_fixture_prospect_ids_are_unique_and_match_their_dict_keys() -> None:
    prospects = build_fixture_prospects()

    for prospect_id, prospect in prospects.items():
        assert prospect.prospect_id == prospect_id


# --- environment (D2: .policy + .tools) ----------------------------------


def test_outreach_environment_is_a_plain_dataclass_shape() -> None:
    toolkit = _toolkit(_prospect())

    env = OutreachEnvironment(tools=toolkit, policy="test policy")

    assert env.policy == "test policy"
    assert env.tools is toolkit


def test_get_outreach_environment_exposes_policy_and_tools_with_no_network() -> None:
    env = get_outreach_environment()

    assert isinstance(env.policy, str)
    assert env.policy
    assert hasattr(env.tools, "use_tool")
    assert hasattr(env.tools, "get_tools")


def test_get_outreach_environment_policy_names_the_fictional_tenant() -> None:
    # Clean-room constraint: the tenant is the fictional "Sunhill Home
    # Improvements", never a real trading name.
    env = get_outreach_environment()

    assert "Sunhill" in env.policy
