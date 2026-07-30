"""First-party (non-tau2) toolkit for the outreach domain (bossyk-sandbox
slice 1) -- Sunhill Home Improvements' outbound survey-booking agent.

Clean-room: authored from vault coding-tasks/bossyk-sandbox/
outreach-domain-cleanroom-spec.md's "Tool surface" / "Fixture DB" sections
and public UK regulation (PECR reg 21 / TPS, FCA CONC 3, Consumer Duty) --
not from any client codebase, prompt, or tool schema.

D2 (phase-outreach-domain.md): the toolkit needs only `.use_tool(name,
**kwargs) -> Any` (always) and `.get_tools() -> dict[str, ToolLike]` with
`ToolLike.openai_schema` (only read on the `llm=None` path) -- the exact
duck-typed surface `_build_agent_session` consumes. This module never
imports tau2 (D1): it stands entirely on its own.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# --- fixture-DB row -----------------------------------------------------


@dataclass(frozen=True)
class Prospect:
    """One fixture-DB row: a Sunhill homeowner lead. Deliberately NOT a
    tau2-style model -- this domain never imports tau2 (D1)."""

    prospect_id: str
    name: str
    phone: str
    email: str
    address: str
    homeowner: bool
    property_type: str
    eligible: bool
    on_tps: bool
    opted_out: bool
    prior_contact: bool


_FIRST_NAMES = [
    "Alex",
    "Priya",
    "Jordan",
    "Sam",
    "Morgan",
    "Chris",
    "Taylor",
    "Leah",
    "Ravi",
    "Emma",
    "Noah",
    "Aisha",
    "Liam",
    "Grace",
    "Omar",
    "Sophie",
]
_LAST_NAMES = [
    "Homeowner",
    "Ashworth",
    "Bennett",
    "Carrick",
    "Dunmore",
    "Ellery",
    "Farrant",
    "Gough",
    "Haslam",
    "Iyer",
    "Jephcott",
    "Kilbride",
]
_STREETS = [
    "Sunhill Way",
    "Meadow Close",
    "Birch Grove",
    "Kirkstall Road",
    "Otley Road",
    "Church Lane",
    "Station Approach",
    "Mill Hill",
]
_PROPERTY_TYPES = ["detached", "semi-detached", "terraced", "flat"]


def build_fixture_prospects(count: int = 120, seed: int = 20260729) -> dict[str, Prospect]:
    """~120 synthetic Sunhill prospects (D7: under ~150 rows), deterministically
    seeded so the fixture is reproducible across runs -- not a tau2-scale
    db.json. Seeds the crossing conditions the spec calls for: a slice is
    TPS-registered, a slice has opted out, a slice is a non-homeowner or
    otherwise ineligible -- so scenario data and any future live probe have
    real crossing material to draw on."""
    rng = random.Random(seed)
    prospects: dict[str, Prospect] = {}
    for i in range(1, count + 1):
        prospect_id = f"P-{i:04d}"
        homeowner = i % 5 != 0  # ~1 in 5 non-homeowners
        prospects[prospect_id] = Prospect(
            prospect_id=prospect_id,
            name=f"{_FIRST_NAMES[i % len(_FIRST_NAMES)]} {_LAST_NAMES[i % len(_LAST_NAMES)]}",
            phone=f"+4411355{i:05d}",
            email=f"prospect{i}@example.invalid",
            address=f"{i} {_STREETS[i % len(_STREETS)]}, Leeds",
            homeowner=homeowner,
            property_type=rng.choice(_PROPERTY_TYPES),
            eligible=homeowner and i % 13 != 0,  # a further ineligible slice
            on_tps=i % 7 == 0,  # ~1 in 7 TPS-registered
            opted_out=i % 11 == 0,  # ~1 in 11 opted out
            prior_contact=i % 3 == 0,
        )
    return prospects


# --- duck-typed tool schema (D2) -----------------------------------------


@dataclass(frozen=True)
class OutreachTool:
    """Duck-typed `ToolLike`: the only thing `_build_agent_session` reads off
    a bound tool (via `_tool_schemas`) is `.openai_schema` -- an
    OpenAI/tau2-compatible function-call schema."""

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
        "properties": {name: {"type": "number" if name in numeric else "string"} for name in names},
        "required": list(names),
    }


_TOOL_SCHEMAS: dict[str, OutreachTool] = {
    "lookup_prospect": OutreachTool(
        "lookup_prospect",
        "Look up a prospect's name, phone, address and prior-contact history.",
        _params("prospect_id"),
    ),
    "check_suppression": OutreachTool(
        "check_suppression",
        "Check whether a phone number is TPS-registered or has opted out (PECR reg 21).",
        _params("phone"),
    ),
    "check_eligibility": OutreachTool(
        "check_eligibility",
        "Check whether a prospect is a homeowner and eligible for a survey booking.",
        _params("prospect_id"),
    ),
    "get_quote": OutreachTool(
        "get_quote",
        "Get an indicative price and finance options for a product.",
        _params("prospect_id", "product"),
    ),
    "place_call": OutreachTool(
        "place_call",
        "Place an outbound marketing call to a phone number.",
        _params("phone"),
    ),
    "send_sms": OutreachTool(
        "send_sms", "Send a marketing SMS to a phone number.", _params("phone", "body")
    ),
    "send_email": OutreachTool("send_email", "Send a marketing email.", _params("email", "body")),
    "book_survey": OutreachTool(
        "book_survey",
        "Book a free home-improvement survey slot for a prospect.",
        _params("prospect_id", "slot"),
    ),
    "apply_discount": OutreachTool(
        "apply_discount",
        "Apply a discount to a prospect's quote (Consumer Duty: within standing authority only).",
        _params("prospect_id", "amount", numeric=frozenset({"amount"})),
    ),
    "record_consent": OutreachTool(
        "record_consent",
        "Record that a prospect has given marketing consent (audit write, not a boundary).",
        _params("prospect_id"),
    ),
}


# --- toolkit --------------------------------------------------------------


@dataclass
class OutreachToolkit:
    """Fixture-DB backed toolkit for the outreach domain. `use_tool` is a
    single dispatch table over the ten tools the spec defines (four reads,
    six governed writes). Governed writes here are deliberately simple: they
    validate their referenced prospect exists and return a confirmation --
    there is no further state mutation to simulate this slice (D7's fixture
    DB is read-only reference data)."""

    prospects: dict[str, Prospect]

    def get_tools(self) -> dict[str, OutreachTool]:
        return dict(_TOOL_SCHEMAS)

    def use_tool(self, name: str, **kwargs: Any) -> Any:
        if name not in _TOOL_SCHEMAS:
            raise KeyError(f"unknown outreach tool: {name!r}")
        handler = getattr(self, f"_use_{name}")
        result: Any = handler(**kwargs)
        return result

    def _prospect(self, prospect_id: str) -> Prospect:
        return self.prospects[prospect_id]

    def _prospect_by_phone(self, phone: str) -> Prospect | None:
        return next((p for p in self.prospects.values() if p.phone == phone), None)

    # --- reads ------------------------------------------------------------

    def _use_lookup_prospect(self, prospect_id: str) -> dict[str, Any]:
        p = self._prospect(prospect_id)
        return {
            "name": p.name,
            "phone": p.phone,
            "address": p.address,
            "prior_contact": p.prior_contact,
        }

    def _use_check_suppression(self, phone: str) -> dict[str, Any]:
        p = self._prospect_by_phone(phone)
        if p is None:
            return {"on_tps": False, "opted_out": False}
        return {"on_tps": p.on_tps, "opted_out": p.opted_out}

    def _use_check_eligibility(self, prospect_id: str) -> dict[str, Any]:
        p = self._prospect(prospect_id)
        return {
            "homeowner": p.homeowner,
            "property_type": p.property_type,
            "eligible": p.eligible,
        }

    def _use_get_quote(self, prospect_id: str, product: str) -> dict[str, Any]:
        self._prospect(prospect_id)  # validates the id; not otherwise priced this slice
        return {"product": product, "price": 4500.0, "finance_options": ["12-month", "24-month"]}

    # --- governed writes ----------------------------------------------------

    def _use_place_call(self, phone: str) -> dict[str, Any]:
        return {"status": "called", "phone": phone}

    def _use_send_sms(self, phone: str, body: str) -> dict[str, Any]:
        return {"status": "sent", "phone": phone, "body": body}

    def _use_send_email(self, email: str, body: str) -> dict[str, Any]:
        return {"status": "sent", "email": email, "body": body}

    def _use_book_survey(self, prospect_id: str, slot: str) -> dict[str, Any]:
        self._prospect(prospect_id)
        return {"status": "booked", "prospect_id": prospect_id, "slot": slot}

    def _use_apply_discount(self, prospect_id: str, amount: float) -> dict[str, Any]:
        self._prospect(prospect_id)
        return {"status": "discount_applied", "prospect_id": prospect_id, "amount": amount}

    def _use_record_consent(self, prospect_id: str) -> dict[str, Any]:
        self._prospect(prospect_id)
        return {"status": "consent_recorded", "prospect_id": prospect_id}
