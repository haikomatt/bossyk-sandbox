"""End-to-end: two SPIFFE identities on one real mTLS listener.

Not gated: it needs nothing but loopback. A throwaway CA and two
SPIFFE-shaped client certificates (URI SAN `spiffe://example.org/...`,
the shape SPIRE 1.13.2 mints; see tests/fixtures/spiffe/README.md) are
minted per test run, the real uvicorn listener is started with
`--client-ca`-equivalent settings, and two httpx clients present
different certificates for the same proposed action. The point under
test is the listener glue the unit tests cannot reach: the peer
certificate really does travel from the TLS handshake into the request.
"""

from __future__ import annotations

import json
import socket
import ssl
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from bossyk_sandbox.gateproxy.proxy import GateProxyConfig, create_app, uvicorn_config
from tests.unit.gateproxy.conftest import openai_response, tool_call
from tests.unit.gateproxy.test_identity_packs import (
    CODING_PACK_YAML,
    DENY_PACK_YAML,
    RESEARCH_PACK_YAML,
)


def _key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def _write_key(path: Path, key: ec.EllipticCurvePrivateKey) -> None:
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


def _mint(
    directory: Path,
    name: str,
    *,
    ca_key: ec.EllipticCurvePrivateKey,
    ca_cert: x509.Certificate,
    sans: list[x509.GeneralName],
) -> tuple[Path, Path]:
    key = _key()
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(x509.NameOID.ORGANIZATION_NAME, "SPIRE")]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    cert_path, key_path = directory / f"{name}.pem", directory / f"{name}-key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    _write_key(key_path, key)
    return cert_path, key_path


def _pki(directory: Path) -> dict[str, Path]:
    ca_key = _key()
    ca_name = x509.Name([x509.NameAttribute(x509.NameOID.ORGANIZATION_NAME, "SPIFFE test CA")])
    now = datetime.now(UTC)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    bundle = directory / "bundle.pem"
    bundle.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    gate_cert, gate_key = _mint(
        directory,
        "gate",
        ca_key=ca_key,
        ca_cert=ca_cert,
        sans=[
            x509.DNSName("localhost"),
            x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1")),
        ],
    )
    coding_cert, coding_key = _mint(
        directory,
        "coding",
        ca_key=ca_key,
        ca_cert=ca_cert,
        sans=[x509.UniformResourceIdentifier("spiffe://example.org/coding-agent")],
    )
    research_cert, research_key = _mint(
        directory,
        "research",
        ca_key=ca_key,
        ca_cert=ca_cert,
        sans=[x509.UniformResourceIdentifier("spiffe://example.org/research-agent")],
    )
    return {
        "bundle": bundle,
        "gate_cert": gate_cert,
        "gate_key": gate_key,
        "coding_cert": coding_cert,
        "coding_key": coding_key,
        "research_cert": research_cert,
        "research_key": research_key,
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_two_svids_one_listener_different_verdicts(
    workspace_root: Path, signing_keys: tuple[Path, str], tmp_path: Path
) -> None:
    priv, _pub = signing_keys
    pki = _pki(tmp_path)
    packs = tmp_path / "packs"
    packs.mkdir()
    (packs / "coding-pack.yaml").write_text(CODING_PACK_YAML)
    (packs / "research-pack.yaml").write_text(RESEARCH_PACK_YAML)
    (packs / "deny-pack.yaml").write_text(DENY_PACK_YAML)
    mapping = packs / "identity-packs.yaml"
    mapping.write_text(
        "version: 1\ndefault: deny-pack.yaml\npacks:\n"
        "  spiffe://example.org/coding-agent: coding-pack.yaml\n"
        "  spiffe://example.org/research-agent: research-pack.yaml\n"
    )
    port = _free_port()
    config = GateProxyConfig(
        upstream_base_url="http://upstream.invalid/v1",
        policy_pack_path=packs / "deny-pack.yaml",
        workspace_root=workspace_root,
        signer_key_path=priv,
        events_path=tmp_path / "gate-events.jsonl",
        run_label="mtls-e2e",
        listen=f"127.0.0.1:{port}",
        identity_packs_path=mapping,
        client_ca_path=pki["bundle"],
        ssl_certfile=pki["gate_cert"],
        ssl_keyfile=pki["gate_key"],
    )
    body = openai_response(tool_calls=[tool_call("c1", "bash", '{"command": "pip install x"}')])

    def upstream(request_body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        return body

    server = uvicorn.Server(uvicorn_config(config, create_app(config, upstream=upstream)))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    try:
        url = f"https://127.0.0.1:{port}/v1/chat/completions"
        request = {"model": "m", "messages": []}

        def as_caller(name: str) -> dict[str, Any]:
            # An explicit context: httpx 0.28's deprecated `cert=` shortcut
            # silently presents no client certificate under TLS 1.3.
            context = ssl.create_default_context(cafile=str(pki["bundle"]))
            context.load_cert_chain(str(pki[f"{name}_cert"]), str(pki[f"{name}_key"]))
            with httpx.Client(verify=context, timeout=10.0) as client:
                message: dict[str, Any] = client.post(url, json=request).json()["choices"][0][
                    "message"
                ]
                return message

        coding = as_caller("coding")
        research = as_caller("research")
        assert coding["tool_calls"]
        assert "no-package-installs" in research["content"]
        # No client certificate at all: the handshake itself is refused.
        try:
            httpx.post(url, json=request, verify=str(pki["bundle"]), timeout=10.0)
        except httpx.HTTPError:
            pass
        else:
            raise AssertionError("a caller with no client certificate was accepted")
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
    assert [e["caller_identity"] for e in events] == [
        "spiffe://example.org/coding-agent",
        "spiffe://example.org/research-agent",
    ]
    assert {e["identity_source"] for e in events} == {"svid"}
