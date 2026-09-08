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

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from bossyk_sandbox.gate import Gate
from bossyk_sandbox.gateproxy import __version__ as _GATE_VERSION
from bossyk_sandbox.gateproxy.events import EventLog
from bossyk_sandbox.gateproxy.identity import (
    PEER_CERT_STATE_KEY,
    InvalidSpiffeId,
    PeerCertH11Protocol,
    resolve_identity,
)
from bossyk_sandbox.gateproxy.identity_packs import IdentityPacks, load_identity_packs
from bossyk_sandbox.gateproxy.sensitive import Marker, detect
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
    # Per-identity packs (round 2). With identity_packs_path unset
    # the gate is the single-pack gate it always was.
    identity_packs_path: Path | None = None
    trust_identity_header: bool = False
    client_ca_path: Path | None = None
    ssl_certfile: Path | None = None
    ssl_keyfile: Path | None = None


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
    # One pack for everyone, or one per verified caller identity with a
    # declared default for everyone else. Every pack's sha256 is computed
    # once from its file's own bytes (whitespace, comments, key order and
    # all) and stamped on every event alongside the running gate_version,
    # so a scorecard/incident report can tie a decision to precise
    # provenance rather than "some pack this gate loaded at some point."
    packs = (
        load_identity_packs(config.identity_packs_path, workspace_root=config.workspace_root)
        if config.identity_packs_path is not None
        else IdentityPacks.single(config.policy_pack_path, workspace_root=config.workspace_root)
    )
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
            "policies": len(packs.default.pack.policies),
            "identity_aware": packs.identity_aware,
            "identity_packs": len(packs.by_identity),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        # Identity is resolved before anything is forwarded upstream: a
        # caller the gate cannot identify gets no completion at all.
        peer_cert = getattr(request.state, PEER_CERT_STATE_KEY, None)
        try:
            identity = resolve_identity(
                peer_cert_der=peer_cert,
                headers=request.headers,
                trust_header=config.trust_identity_header,
            )
        except InvalidSpiffeId as exc:
            events.append(
                {
                    "kind": "identity_refused",
                    "verdict": "block",
                    "reason": f"caller identity refused: {exc}",
                    "caller_identity": None,
                    "identity_source": None,
                    "pack_matched": False,
                    "gate_version": _GATE_VERSION,
                }
            )
            return JSONResponse(
                status_code=403,
                content={"error": {"message": f"caller identity refused: {exc}"}},
            )
        resolved = packs.for_identity(identity.spiffe_id if identity else None)
        pack, instruments, pack_sha256 = resolved.pack, resolved.instruments, resolved.pack_sha256
        identity_fields = {
            "caller_identity": identity.spiffe_id if identity else None,
            "identity_source": identity.source if identity else None,
            "pack_matched": resolved.matched,
        }

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
                if decision.verdict is Verdict.BLOCK:
                    block_reasons.append(decision.reason)
                # Every rule prefixes its reason with its own policy id;
                # matching against the loaded pack recovers it as a
                # structured field for the scorecard.
                policy_id = next(
                    (p.id for p in pack.policies if decision.reason.startswith(f"{p.id}:")),
                    None,
                )
                # Scanned over this call's OWN arguments only (never the
                # whole request/response), before the event is signed --
                # markers can never expose more of a secret than the
                # `arguments` field above already carries verbatim.
                markers = detect(json.dumps(logged_arguments))
                events.append(
                    {
                        "kind": "tool_call",
                        "tool_name": name,
                        "arguments": logged_arguments,
                        "verdict": decision.verdict.value,
                        "policy_id": policy_id,
                        "reason": decision.reason,
                        "model": upstream_response.get("model"),
                        "pack_sha256": pack_sha256,
                        "gate_version": _GATE_VERSION,
                        **identity_fields,
                        "sensitive_markers": _marker_dicts(markers),
                    }
                )
        else:
            # Scanned over this utterance's own content only. Unlike
            # tool_call arguments, the raw content is NOT itself stored
            # on the event -- only the redacted markers are -- so an
            # utterance event never carries more of the raw text than a
            # tool_call event carries of its raw arguments.
            utterance_markers = detect(message.get("content") or "")
            events.append(
                {
                    "kind": "utterance",
                    "verdict": "allow",
                    "reason": "no tool calls proposed",
                    "model": upstream_response.get("model"),
                    "pack_sha256": pack_sha256,
                    "gate_version": _GATE_VERSION,
                    **identity_fields,
                    "sensitive_markers": _marker_dicts(utterance_markers),
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
    parser.add_argument(
        "--upstream-key-env",
        default="FIREWORKS_API_KEY",
        help="Env var holding the upstream API key (used when the client sends no Authorization).",
    )
    parser.add_argument(
        "--identity-packs",
        help="Identity -> pack mapping YAML (per-caller packs; --policy-pack stays the "
        "fallback when unset).",
    )
    parser.add_argument(
        "--trust-identity-header",
        action="store_true",
        help="Trust X-Forwarded-Client-Cert / X-SPIFFE-ID when no client certificate is "
        "presented. ONLY behind a proxy you control that sets them.",
    )
    parser.add_argument(
        "--client-ca",
        help="Trust bundle (PEM) for client certificates; requires them on every connection "
        "(mTLS). Needs --ssl-certfile and --ssl-keyfile.",
    )
    parser.add_argument("--ssl-certfile", help="The gate's own TLS certificate (PEM).")
    parser.add_argument("--ssl-keyfile", help="The gate's own TLS private key (PEM).")
    listener = parser.add_mutually_exclusive_group(required=True)
    listener.add_argument("--listen", help="TCP listen address, host:port.")
    listener.add_argument("--uds", help="Unix domain socket path to listen on.")
    import os

    args = parser.parse_args(argv)
    if args.client_ca and not (args.ssl_certfile and args.ssl_keyfile):
        parser.error("--client-ca requires --ssl-certfile and --ssl-keyfile")
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
        identity_packs_path=Path(args.identity_packs) if args.identity_packs else None,
        trust_identity_header=args.trust_identity_header,
        client_ca_path=Path(args.client_ca) if args.client_ca else None,
        ssl_certfile=Path(args.ssl_certfile) if args.ssl_certfile else None,
        ssl_keyfile=Path(args.ssl_keyfile) if args.ssl_keyfile else None,
    )


def uvicorn_config(config: GateProxyConfig, app: FastAPI) -> uvicorn.Config:
    """The listener: TCP or UDS, plain or TLS, and with --client-ca an
    mTLS listener that REQUIRES a client certificate chained to the bundle
    and hands it to the app (`PeerCertH11Protocol`)."""
    import ssl

    kwargs: dict[str, Any] = {}
    if config.uds:
        kwargs["uds"] = config.uds
    else:
        host, _, port = (config.listen or "127.0.0.1:8200").rpartition(":")
        kwargs["host"], kwargs["port"] = host or "127.0.0.1", int(port)
    if config.ssl_certfile:
        kwargs["ssl_certfile"] = str(config.ssl_certfile)
        kwargs["ssl_keyfile"] = str(config.ssl_keyfile) if config.ssl_keyfile else None
    if config.client_ca_path:
        kwargs["ssl_ca_certs"] = str(config.client_ca_path)
        kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED
        kwargs["http"] = PeerCertH11Protocol
    return uvicorn.Config(app, **kwargs)


def main(argv: list[str] | None = None) -> None:
    import sys

    config = build_config(sys.argv[1:] if argv is None else argv)
    uvicorn.Server(uvicorn_config(config, create_app(config))).run()
