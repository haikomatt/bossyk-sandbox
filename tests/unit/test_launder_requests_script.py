"""Hermetic tests for scripts/launder_requests.py pure parts (no network)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "launder_requests.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("launder_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_laundering_prompts_carry_the_task_and_the_request() -> None:
    m = _import()
    sysp = m.laundering_system_prompt()
    low = sysp.lower()
    # the task: hide the intent-bearing words but keep the SAME underlying action
    assert "rewrite" in low or "reword" in low
    assert "same" in low and ("action" in low or "request" in low)
    usr = m.laundering_user_prompt("Cancel my order #W1002310.")
    assert "Cancel my order #W1002310." in usr


def test_parse_laundered_strips_quotes_preamble_and_whitespace() -> None:
    m = _import()
    assert (
        m.parse_laundered('  "Please take care of order #W123."  ')
        == "Please take care of order #W123."
    )
    assert m.parse_laundered("Rewritten request: do the needful on #W9") == "do the needful on #W9"
    assert m.parse_laundered("Laundered:\n handle it for #W5") == "handle it for #W5"
    assert m.parse_laundered("plain text no wrapper") == "plain text no wrapper"


class _FakeClient:
    """Structural LaunderClient: returns a canned rewrite, or '' to signal a
    refusal/empty completion for a chosen request."""

    def __init__(self, refuse: set[str] | None = None) -> None:
        self.refuse = refuse or set()
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        for r in ("#W1002310", "#W2", "#W3"):
            if r in user_prompt and user_prompt not in self.refuse:
                if r in self.refuse:
                    return ""
                return f'"handle the matter regarding order {r} please"'
        return ""


def test_launder_all_pairs_by_index_and_preserves_original() -> None:
    m = _import()
    client = _FakeClient()
    reqs = ["Cancel order #W1002310.", "Refund order #W2.", "Change payment on #W3."]
    pairs = m.launder_all(client, reqs)
    assert [p.prompt_id for p in pairs] == [0, 1, 2]
    assert pairs[0].original == "Cancel order #W1002310."
    assert pairs[0].laundered == "handle the matter regarding order #W1002310 please"
    # the give-away verb is gone from the laundered variant
    assert "cancel" not in pairs[0].laundered.lower()
    assert len(client.calls) == 3


def test_launder_all_drops_empty_completions() -> None:
    m = _import()

    class _AllRefuse:
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return "   "

    pairs = m.launder_all(_AllRefuse(), ["Cancel order #W1002310."])
    assert pairs == []
