"""The OpenAI-compatible gate proxy (round-1 Phase 2).

One `/v1/chat/completions` route: forward the request upstream (always
non-streamed -- the gate must see complete tool calls before ruling),
score every proposed tool call through the sandbox ``Gate`` with the
compiled policy instruments, then either forward the upstream response
verbatim (ALLOW) or rewrite it into a valid completion whose content is a
policy refusal naming the violated policies (BLOCK). The refusal therefore
lands in the calling agent's own session file, which is what lets the
auditk trace corroborate the gate log independently.

Every per-tool-call decision (and each plain utterance) is appended to the
Ed25519-signed event log. A ``stream: true`` caller gets a synthesized SSE
stream carrying the same message; buffering-then-deciding is the v1 design
and its latency cost is measured and stated, not hidden.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from bossyk_sandbox.gate import Gate
from bossyk_sandbox.gateproxy.events import EventLog
from bossyk_sandbox.gateproxy.policies import compile_instruments, load_policy_pack
from bossyk_sandbox.instruments.base import Decision, ProposedAction, Verdict

Upstream = Callable[[dict[str, Any]], dict[str, Any]]

_REFUSAL_HEADER = "[bossyk gate] BLOCKED: this response proposed actions that violate policy."


@dataclass
class GateProxyConfig:
    upstream_base_url: str
    policy_pack_path: Path
    workspace_root: Path
    signer_key_path: Path
    events_path: Path
    run_label: str
    listen: str | None = None
    uds: str | None = None


def _default_upstream(base_url: str) -> Upstream:
    """The real httpx path, used when no upstream callable is injected."""
    import httpx

    def call(body: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=600.0)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    return call


def _parse_tool_call(call: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """(tool name, parsed arguments) -- arguments ``None`` when the wire
    JSON does not parse to a dict (the fail-closed case)."""
    function = call.get("function") or {}
    name = str(function.get("name", ""))
    raw = function.get("arguments")
    if isinstance(raw, dict):
        return name, raw
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else None
    except json.JSONDecodeError:
        parsed = None
    return name, parsed if isinstance(parsed, dict) else None


def _sse_chunks(response_body: dict[str, Any]) -> Iterator[str]:
    """Synthesize a chat.completion.chunk stream equivalent to one buffered
    completion: role chunk, content/tool_calls delta, terminal chunk, DONE."""
    choice = response_body["choices"][0]
    message = choice["message"]
    base = {
        "id": response_body.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
        "object": "chat.completion.chunk",
        "created": response_body.get("created", int(time.time())),
        "model": response_body.get("model", ""),
    }

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        payload = {
            **base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant"})
    if message.get("content"):
        yield chunk({"content": message["content"]})
    if message.get("tool_calls"):
        deltas = [{**call, "index": i} for i, call in enumerate(message["tool_calls"])]
        yield chunk({"tool_calls": deltas})
    yield chunk({}, finish_reason=choice.get("finish_reason", "stop"))
    yield "data: [DONE]\n\n"


def create_app(config: GateProxyConfig, upstream: Upstream | None = None) -> FastAPI:
    pack = load_policy_pack(config.policy_pack_path)
    instruments = compile_instruments(pack, workspace_root=config.workspace_root)
    call_upstream = upstream or _default_upstream(config.upstream_base_url)
    events = EventLog(
        path=config.events_path,
        signer_key_path=config.signer_key_path,
        run_label=config.run_label,
    )
    app = FastAPI(title="bossyk gate proxy")

    @app.get("/gate/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "run_label": config.run_label,
            "policies": len(pack.policies),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        wants_stream = bool(body.get("stream"))
        upstream_body = {**body, "stream": False}

        try:
            upstream_response = call_upstream(upstream_body)
        except Exception as exc:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"upstream request failed: {exc}"}},
            )

        message = upstream_response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []

        gate = Gate(instruments=instruments)
        block_reasons: list[str] = []
        if tool_calls:
            for call in tool_calls:
                name, arguments = _parse_tool_call(call)
                if arguments is None:
                    decision = Decision(
                        Verdict.BLOCK,
                        f"unparseable arguments for tool {name!r} (fail closed)",
                    )
                    logged_arguments: Any = {"_raw": (call.get("function") or {}).get("arguments")}
                else:
                    proposed = ProposedAction(
                        tool_name=name,
                        arguments=arguments,
                        declared_intent=message.get("content"),
                    )
                    decision = gate.score(proposed)
                    gate.record(proposed)
                    logged_arguments = arguments
                if decision.verdict is Verdict.BLOCK:
                    block_reasons.append(decision.reason)
                # Every rule prefixes its reason with its own policy id;
                # matching against the loaded pack recovers it as a
                # structured field for the scorecard.
                policy_id = next(
                    (p.id for p in pack.policies if decision.reason.startswith(f"{p.id}:")),
                    None,
                )
                events.append(
                    {
                        "kind": "tool_call",
                        "tool_name": name,
                        "arguments": logged_arguments,
                        "verdict": decision.verdict.value,
                        "policy_id": policy_id,
                        "reason": decision.reason,
                        "model": upstream_response.get("model"),
                    }
                )
        else:
            events.append(
                {
                    "kind": "utterance",
                    "verdict": "allow",
                    "reason": "no tool calls proposed",
                    "model": upstream_response.get("model"),
                }
            )

        if block_reasons:
            refusal = "\n".join([_REFUSAL_HEADER, *(f"- {r}" for r in block_reasons)])
            response_body = {
                **upstream_response,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": refusal},
                        "finish_reason": "stop",
                    }
                ],
            }
        else:
            response_body = upstream_response

        if wants_stream:
            return StreamingResponse(_sse_chunks(response_body), media_type="text/event-stream")
        return JSONResponse(content=response_body)

    return app


def build_config(argv: list[str]) -> GateProxyConfig:
    parser = argparse.ArgumentParser(
        prog="bossyk-gate",
        description="OpenAI-compatible policy gate proxy (bossyk round 1).",
    )
    parser.add_argument("--upstream", required=True, help="Upstream OpenAI-compatible base URL.")
    parser.add_argument("--policy-pack", required=True, help="Policy pack YAML path.")
    parser.add_argument("--workspace", required=True, help="Workspace root the agent runs in.")
    parser.add_argument("--key", required=True, help="Ed25519 private key (PEM) for event signing.")
    parser.add_argument("--events", required=True, help="Signed gate-events JSONL output path.")
    parser.add_argument("--run-label", required=True, help="Label stamped on every event.")
    listener = parser.add_mutually_exclusive_group(required=True)
    listener.add_argument("--listen", help="TCP listen address, host:port.")
    listener.add_argument("--uds", help="Unix domain socket path to listen on.")
    args = parser.parse_args(argv)
    return GateProxyConfig(
        upstream_base_url=args.upstream,
        policy_pack_path=Path(args.policy_pack),
        workspace_root=Path(args.workspace),
        signer_key_path=Path(args.key),
        events_path=Path(args.events),
        run_label=args.run_label,
        listen=args.listen,
        uds=args.uds,
    )


def main(argv: list[str] | None = None) -> None:
    import sys

    import uvicorn

    config = build_config(sys.argv[1:] if argv is None else argv)
    app = create_app(config)
    if config.uds:
        uvicorn.run(app, uds=config.uds)
    else:
        host, _, port = (config.listen or "127.0.0.1:8200").rpartition(":")
        uvicorn.run(app, host=host or "127.0.0.1", port=int(port))
