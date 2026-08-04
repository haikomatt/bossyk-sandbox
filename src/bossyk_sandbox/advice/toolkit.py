"""Dual tool surface over the advice persona store (bossyk-sandbox privacy/
minimisation demonstrator, Phase 1): a raw profile lookup and three derived,
never-raw tools, over the SAME `PersonaStore`.

Mirrors the shape `runtime.outreach.toolkit.OutreachToolkit` established for
a first-party (non-tau2) domain -- `ToolLike.openai_schema`,
`get_tools()`/`use_tool()` -- so a later phase could wire this into
`runtime.langgraph_agent._build_agent_session(environment=...)` the same
way `OutreachEnvironment` does, without this module importing tau2 at all
(console/live-agent wiring is out of Phase 1 scope).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bossyk_sandbox.advice.personas import PersonaStore
from bossyk_sandbox.advice.tax import contribution_headroom_band, tax_band

RAW_TOOL_NAMES = frozenset({"get_customer_profile"})
DERIVED_TOOL_NAMES = frozenset(
    {"get_tax_position", "get_contribution_headroom", "is_income_above"}
)


@dataclass(frozen=True)
class AdviceTool:
    """Duck-typed `ToolLike` (mirrors `runtime.outreach.toolkit.OutreachTool`):
    the only thing a future live-agent builder reads off a bound tool is
    `.openai_schema`, an OpenAI/tau2-compatible function-call schema."""

    name: str
    description: str
    parameters: dict[str, Any]

    @property
    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _params(*names: str, numeric: frozenset[str] = frozenset()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            name: {"type": "number" if name in numeric else "string"} for name in names
        },
        "required": list(names),
    }


_RAW_TOOL_SCHEMAS: dict[str, AdviceTool] = {
    "get_customer_profile": AdviceTool(
        "get_customer_profile",
        "Return a customer's full profile: name, date of birth, postcode, "
        "annual income and health-condition flag.",
        _params("ref"),
    ),
}

_DERIVED_TOOL_SCHEMAS: dict[str, AdviceTool] = {
    "get_tax_position": AdviceTool(
        "get_tax_position",
        "Return a customer's UK income-tax band (basic, higher or "
        "additional) without disclosing their exact income.",
        _params("ref"),
    ),
    "get_contribution_headroom": AdviceTool(
        "get_contribution_headroom",
        "Return a customer's pension contribution headroom as a coarse "
        "band (high, medium, low), derived from their tax band only.",
        _params("ref"),
    ),
    "is_income_above": AdviceTool(
        "is_income_above",
        "Return whether a customer's annual income is strictly above a "
        "given GBP threshold, without disclosing the exact income.",
        _params("ref", "threshold", numeric=frozenset({"threshold"})),
    ),
}

_ALL_TOOL_SCHEMAS: dict[str, AdviceTool] = {**_RAW_TOOL_SCHEMAS, **_DERIVED_TOOL_SCHEMAS}


@dataclass
class AdviceToolkit:
    """Persona-store backed toolkit exposing BOTH tool surfaces over the
    same store. `minimisation` decides which surface `get_tools()` offers:
    `False` (off) offers raw + derived, `True` (on) offers derived only.
    `use_tool` enforces the same restriction -- a tool not currently offered
    cannot be invoked by name either, so the toggle is a real boundary, not
    just an advertising difference."""

    store: PersonaStore
    minimisation: bool = False

    def get_tools(self) -> dict[str, AdviceTool]:
        if self.minimisation:
            return dict(_DERIVED_TOOL_SCHEMAS)
        return dict(_ALL_TOOL_SCHEMAS)

    def use_tool(self, name: str, **kwargs: Any) -> Any:
        if name not in self.get_tools():
            raise KeyError(
                f"{name!r} is not offered by this toolkit (minimisation={self.minimisation})"
            )
        handler = getattr(self, name)
        result: Any = handler(**kwargs)
        return result

    # --- raw surface ------------------------------------------------------

    def get_customer_profile(self, ref: str) -> dict[str, Any]:
        record = self.store.get(ref)
        return {
            "ref": record.ref,
            "full_name": record.full_name,
            "date_of_birth": record.date_of_birth,
            "postcode": record.postcode,
            "annual_income_gbp": record.annual_income_gbp,
            "health_condition_flag": record.health_condition_flag,
        }

    # --- derived surface ----------------------------------------------------

    def get_tax_position(self, ref: str) -> str:
        record = self.store.get(ref)
        return tax_band(record.annual_income_gbp).value

    def get_contribution_headroom(self, ref: str) -> str:
        record = self.store.get(ref)
        return contribution_headroom_band(record.annual_income_gbp)

    def is_income_above(self, ref: str, threshold: float) -> bool:
        record = self.store.get(ref)
        return record.annual_income_gbp > threshold
