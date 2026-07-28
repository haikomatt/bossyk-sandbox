"""Render an agent turn's message context to a decision prompt string (stdlib).

Deliberately its own tiny module (no numpy/capture_run import) so
`langgraph_agent` can capture the per-turn prompt without pulling the probe's
numpy stack into the core agent import chain.

The rendered string is the exact input the model saw *before* it acted at a turn
(system/policy + conversation so far) -- the T4 "pre-action" decision context the
nnsight tracer re-tokenises on the pod. This is a readable transcript, not the
model's chat-template tokenisation; applying the real template on the pod (where
the tokenizer lives) is the fidelity refinement, documented in the capture path.
"""

from __future__ import annotations

from typing import Any


def render_prompt(messages: list[Any]) -> str:
    """Serialise LangChain-style messages (duck-typed: `.type`, `.content`,
    optional `.tool_calls`) to a `role: content` transcript. Tool-call messages
    with empty content still render their calls, so the context isn't lossy."""
    lines: list[str] = []
    for message in messages:
        role = getattr(message, "type", "?")
        content = getattr(message, "content", "") or ""
        line = f"{role}: {content}"
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            rendered = "; ".join(f"{call['name']}({call['args']})" for call in tool_calls)
            line = f"{line} [tool_calls: {rendered}]"
        lines.append(line)
    return "\n".join(lines)
