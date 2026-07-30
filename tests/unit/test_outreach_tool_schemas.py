"""RED-phase tests for `outreach_tool_schemas()` (bossyk-sandbox slice 3,
phase 3a -- zero-cost live-path wiring).

Mirrors `retail_tool_schemas()` (runtime/langgraph_agent.py:131): exposes
the outreach agent's REAL bound tool schemas (from the fixture-DB
environment, no network) so the grounded adversary
(conditions.fireworks_adversary) can ground attacks in the agent's actual
toolset -- the fix the L1 root cause needed for retail/airline, now needed
for outreach so `scripts/generate_grounded_corpus.py`'s
`GROUNDED_DOMAIN=outreach` stops refusing at the domain-registry check
(`_TOOL_SCHEMAS_BY_DOMAIN`).
"""

from __future__ import annotations

from bossyk_sandbox.runtime.langgraph_agent import outreach_tool_schemas


def test_outreach_tool_schemas_exposes_all_ten_outreach_tools() -> None:
    schemas = outreach_tool_schemas()

    names = {schema["function"]["name"] for schema in schemas}
    assert len(schemas) == 10
    assert names == {
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


def test_outreach_tool_schemas_have_the_openai_function_shape() -> None:
    schemas = outreach_tool_schemas()

    for schema in schemas:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]


def test_outreach_tool_schemas_is_a_local_fixture_load_not_a_network_call() -> None:
    # No FIREWORKS_API_KEY / network needed -- get_outreach_environment()
    # builds a local, deterministic fixture DB, mirroring how
    # retail_tool_schemas loads tau2's retail env from a local JSON file.
    schemas = outreach_tool_schemas()

    assert schemas
