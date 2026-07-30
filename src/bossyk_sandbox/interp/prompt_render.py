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


def render_action(response: Any) -> str:
    """Render an agent turn's OWN response -- its ACTION -- to a compact string:
    the message text plus any tool calls it proposed. This is what the coherence
    judge reads ("what did the agent DO"), distinct from `render_prompt`'s "what
    did the agent SEE". Duck-typed on `.content` / `.tool_calls`; an empty
    response degrades to a sentinel rather than "" so the judge never gets a
    blank input."""
    content = getattr(response, "content", "") or ""
    tool_calls = getattr(response, "tool_calls", None)
    parts: list[str] = []
    if content:
        parts.append(str(content).strip())
    if tool_calls:
        rendered = "; ".join(f"{call['name']}({call['args']})" for call in tool_calls)
        parts.append(f"tool_calls: {rendered}")
    return " | ".join(p for p in parts if p) or "(no action)"
