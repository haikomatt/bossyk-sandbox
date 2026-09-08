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
import hashlib
import json
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from bossyk_sandbox.gate import Gate
from bossyk_sandbox.gateproxy import __version__ as _GATE_VERSION
from bossyk_sandbox.gateproxy.events import EventLog
from bossyk_sandbox.gateproxy.hold import Approver, HoldRequest, command_approver, url_approver
from bossyk_sandbox.gateproxy.policies import GatePolicy, compile_instruments, load_policy_pack
from bossyk_sandbox.gateproxy.sensitive import Marker, detect
from bossyk_sandbox.gateproxy.telemetry import Telemetry, TelemetryConfig
from bossyk_sandbox.instruments.base import Decision, ProposedAction, Verdict

Upstream = Callable[[dict[str, Any], dict[str, str]], dict[str, Any]]

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
    # Set from --upstream-key-env at startup; fills the upstream
    # Authorization header when the client itself sent none.
    upstream_api_key: str | None = None
    # Telemetry is off by default (air-gap friendly); the signed event log
    # is the system of record whether or not these are set.
    otlp_endpoint: str | None = None
    pushgateway_url: str | None = None
    approver_command: list[str] | None = None
    approver_url: str | None = None
    approver_timeout_s: float = 30.0


def _default_upstream(base_url: str) -> Upstream:
    """The real httpx path, used when no upstream callable is injected."""
    import httpx

    def call(body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=body,
            headers=headers,
            timeout=600.0,
        )
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


def _marker_dicts(markers: list[Marker]) -> list[dict[str, str]]:
    """Build B: the wire shape logged into an event -- a plain list of
    {kind, redacted_excerpt}, never the Marker's own richer type, so the
    event log stays plain JSON like every other field on it."""
    return [{"kind": m.kind, "redacted_excerpt": m.redacted_excerpt} for m in markers]


def _configured_approver(config: GateProxyConfig) -> Approver | None:
    if config.approver_command:
        return command_approver(config.approver_command, timeout_s=config.approver_timeout_s)
    if config.approver_url:
        return url_approver(config.approver_url, timeout_s=config.approver_timeout_s)
    return None


def _resolve_hold(
    policy: GatePolicy | None, request: HoldRequest, approver: Approver | None
) -> tuple[str, Verdict]:
    """(resolution, verdict) for a held action. With no approver configured
    the policy's `on_hold` default decides outright; with one, its answer
    decides, and no answer (timeout/failure) falls back to that default.
    A hold on a policy the pack cannot name fails closed."""
    default = Verdict(policy.on_hold) if policy is not None else Verdict.BLOCK
    answer = approver(request) if approver is not None else None
    if approver is not None and answer is None:
        return "hold_timed_out", default
    resolved = default if answer is None else (Verdict.ALLOW if answer else Verdict.BLOCK)
    return ("held_then_allowed" if resolved is Verdict.ALLOW else "held_then_blocked"), resolved


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


def create_app(
    config: GateProxyConfig,
    upstream: Upstream | None = None,
    approver: Approver | None = None,
    telemetry: Telemetry | None = None,
) -> FastAPI:
    pack = load_policy_pack(config.policy_pack_path)
    instruments = compile_instruments(pack, workspace_root=config.workspace_root)
    # Computed once from the pack file's own bytes -- not the parsed
    # PolicyPack -- so it binds to exactly what was on disk (whitespace,
    # comments, key order and all), not merely the pack's declared
    # semantics. Stamped on every event alongside the running gate_version
    # so a scorecard/incident report can tie a decision to precise
    # provenance rather than "some pack this gate loaded at some point."
    pack_sha256 = hashlib.sha256(config.policy_pack_path.read_bytes()).hexdigest()
    call_upstream = upstream or _default_upstream(config.upstream_base_url)
    resolve_hold_with = approver or _configured_approver(config)
    policies_by_id = {p.id: p for p in pack.policies}
    events = EventLog(
        path=config.events_path,
        signer_key_path=config.signer_key_path,
        run_label=config.run_label,
    )
    # Telemetry is a projection of the signed log: `events.append` runs
    # first and returns the exact signed event, which is what gets exported.
    export = telemetry or Telemetry(
        TelemetryConfig(otlp_endpoint=config.otlp_endpoint, pushgateway_url=config.pushgateway_url)
    )
    app = FastAPI(title="bossyk gate proxy")

    @app.get("/gate/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(export.prometheus_text(), media_type="text/plain; version=0.0.4")

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
        # stream_options is only valid alongside stream:true; the gate
        # always asks the upstream non-streamed, so it must go too.
        upstream_body = {k: v for k, v in body.items() if k != "stream_options"}
        upstream_body["stream"] = False
        auth = (
            f"Bearer {config.upstream_api_key}"
            if config.upstream_api_key
            else request.headers.get("authorization")
        )
        upstream_headers = {"Authorization": auth} if auth else {}

        try:
            upstream_response = call_upstream(upstream_body, upstream_headers)
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
                # Every rule prefixes its reason with its own policy id;
                # matching against the loaded pack recovers it as a
                # structured field for the scorecard.
                policy_id = next(
                    (p.id for p in pack.policies if decision.reason.startswith(f"{p.id}:")),
                    None,
                )
                resolution: str | None = None
                resolved: Verdict | None = None
                if decision.verdict is Verdict.HOLD:
                    resolution, resolved = _resolve_hold(
                        policies_by_id.get(policy_id or ""),
                        HoldRequest(
                            tool_name=name,
                            arguments=logged_arguments,
                            policy_id=policy_id,
                            reason=decision.reason,
                            run_label=config.run_label,
                        ),
                        resolve_hold_with,
                    )
                if (resolved or decision.verdict) is Verdict.BLOCK:
                    block_reasons.append(decision.reason)
                # Scanned over this call's OWN arguments only (never the
                # whole request/response), before the event is signed --
                # markers can never expose more of a secret than the
                # `arguments` field above already carries verbatim.
                markers = detect(json.dumps(logged_arguments))
                signed = events.append(
                    {
                        "kind": "tool_call",
                        "tool_name": name,
                        "arguments": logged_arguments,
                        "verdict": decision.verdict.value,
                        "resolution": resolution,
                        "resolved_verdict": resolved.value if resolved else None,
                        "policy_id": policy_id,
                        "reason": decision.reason,
                        "model": upstream_response.get("model"),
                        "pack_sha256": pack_sha256,
                        "gate_version": _GATE_VERSION,
                        "sensitive_markers": _marker_dicts(markers),
                    }
                )
                export.record(signed)
        else:
            # Scanned over this utterance's own content only. Unlike
            # tool_call arguments, the raw content is NOT itself stored
            # on the event -- only the redacted markers are -- so an
            # utterance event never carries more of the raw text than a
            # tool_call event carries of its raw arguments.
            utterance_markers = detect(message.get("content") or "")
            signed = events.append(
                {
                    "kind": "utterance",
                    "verdict": "allow",
                    "resolution": None,
                    "resolved_verdict": None,
                    "reason": "no tool calls proposed",
                    "model": upstream_response.get("model"),
                    "pack_sha256": pack_sha256,
                    "gate_version": _GATE_VERSION,
                    "sensitive_markers": _marker_dicts(utterance_markers),
                }
            )
            export.record(signed)

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
    parser.add_argument(
        "--upstream-key-env",
        default="FIREWORKS_API_KEY",
        help="Env var holding the upstream API key (used when the client sends no Authorization).",
    )
    parser.add_argument(
        "--otlp-endpoint",
        help="OTLP/HTTP collector base URL (e.g. http://collector:4318); off when unset.",
    )
    parser.add_argument(
        "--pushgateway-url",
        help="Prometheus push gateway base URL (e.g. http://pgw:9091); off when unset.",
    )
    listener = parser.add_mutually_exclusive_group(required=True)
    listener.add_argument("--listen", help="TCP listen address, host:port.")
    listener.add_argument("--uds", help="Unix domain socket path to listen on.")
    approver = parser.add_mutually_exclusive_group()
    approver.add_argument(
        "--approver-cmd",
        help="HOLD approver command (held action as JSON on stdin; exit 0 approves).",
    )
    approver.add_argument(
        "--approver-url",
        help='HOLD approver URL (JSON POST; reply {"decision": "allow"|"block"}).',
    )
    parser.add_argument(
        "--approver-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the approver before the policy's on_hold default applies.",
    )
    import os
    import shlex

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
        upstream_api_key=os.environ.get(args.upstream_key_env),
        otlp_endpoint=args.otlp_endpoint,
        pushgateway_url=args.pushgateway_url,
        approver_command=shlex.split(args.approver_cmd) if args.approver_cmd else None,
        approver_url=args.approver_url,
        approver_timeout_s=args.approver_timeout,
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
