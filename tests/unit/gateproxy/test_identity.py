"""Red-phase contract for caller identity (round-2 Phase 3).

The gate reads a verified caller identity so different callers on the
same gate run under different policy packs. Preferred source: the SPIFFE
X509-SVID presented on an mTLS connection (its `spiffe://` URI SAN).
Fallback source, only behind an operator's own boundary and only when
explicitly trusted: an `X-Forwarded-Client-Cert` (Envoy XFCC) or
`X-SPIFFE-ID` header. Malformed identities are refused, never guessed.

Phase 0 evidence: tests/fixtures/spiffe/ holds SVIDs minted by a real
SPIRE server (1.13.2). The SPIFFE-ID grammar asserted here is the
SPIFFE-ID spec: `spiffe://<trust-domain>/<path>`, trust domain
`[a-z0-9._-]+`, path segments non-empty, `[A-Za-z0-9._-]` only, no `.`
or `..` segments, no query/fragment/userinfo/port, at most 2048 bytes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding

from bossyk_sandbox.gateproxy.identity import (
    CallerIdentity,
    InvalidSpiffeId,
    parse_spiffe_id,
    resolve_identity,
    spiffe_id_from_der,
    spiffe_id_from_xfcc,
)

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "spiffe"


def _der(name: str) -> bytes:
    pem = (_FIXTURES / f"spire-1.13.2-{name}-svid.pem").read_bytes()
    return x509.load_pem_x509_certificate(pem).public_bytes(Encoding.DER)


def _self_signed(sans: list[x509.GeneralName]) -> bytes:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(x509.NameOID.ORGANIZATION_NAME, "test")])
    now = datetime.now(UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=1))
    )
    if sans:
        builder = builder.add_extension(x509.SubjectAlternativeName(sans), critical=False)
    return builder.sign(key, hashes.SHA256()).public_bytes(Encoding.DER)


class TestParseSpiffeId:
    @pytest.mark.parametrize(
        "value",
        [
            "spiffe://example.org/coding-agent",
            "spiffe://example.org/ns/default/sa/agent",
            "spiffe://trust-domain_1.example/a.b_c-d",
        ],
    )
    def test_valid_ids_round_trip(self, value: str) -> None:
        assert parse_spiffe_id(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "https://example.org/agent",
            "spiffe://example.org",
            "spiffe://example.org/",
            "spiffe://Example.org/agent",
            "spiffe://example.org:8443/agent",
            "spiffe://user@example.org/agent",
            "spiffe://example.org/agent?x=1",
            "spiffe://example.org/agent#frag",
            "spiffe://example.org//agent",
            "spiffe://example.org/agent/",
            "spiffe://example.org/./agent",
            "spiffe://example.org/../agent",
            "spiffe://example.org/agent%20x",
            "spiffe://example.org/agent x",
            "spiffe://exa mple.org/agent",
            "spiffe://example.org/" + "a" * 2048,
        ],
    )
    def test_malformed_ids_are_refused(self, value: str) -> None:
        with pytest.raises(InvalidSpiffeId):
            parse_spiffe_id(value)


class TestSpiffeIdFromSvid:
    def test_real_spire_svid_yields_its_uri_san(self) -> None:
        assert spiffe_id_from_der(_der("coding-agent")) == "spiffe://example.org/coding-agent"
        assert spiffe_id_from_der(_der("research-agent")) == "spiffe://example.org/research-agent"

    def test_certificate_without_spiffe_san_is_refused(self) -> None:
        with pytest.raises(InvalidSpiffeId, match="no spiffe"):
            spiffe_id_from_der(_self_signed([x509.DNSName("agent.example.org")]))
        with pytest.raises(InvalidSpiffeId):
            spiffe_id_from_der(_self_signed([]))

    def test_two_spiffe_sans_are_refused(self) -> None:
        der = _self_signed(
            [
                x509.UniformResourceIdentifier("spiffe://example.org/a"),
                x509.UniformResourceIdentifier("spiffe://example.org/b"),
            ]
        )
        with pytest.raises(InvalidSpiffeId, match="exactly one"):
            spiffe_id_from_der(der)

    def test_malformed_spiffe_san_is_refused(self) -> None:
        der = _self_signed([x509.UniformResourceIdentifier("spiffe://Example.org/agent")])
        with pytest.raises(InvalidSpiffeId):
            spiffe_id_from_der(der)

    def test_garbage_bytes_are_refused(self) -> None:
        with pytest.raises(InvalidSpiffeId):
            spiffe_id_from_der(b"not a certificate")


class TestXfcc:
    def test_envoy_xfcc_uri_element(self) -> None:
        header = (
            'By=spiffe://example.org/gate;Hash=abc;Subject="CN=x,O=y";'
            "URI=spiffe://example.org/coding-agent"
        )
        assert spiffe_id_from_xfcc(header) == "spiffe://example.org/coding-agent"

    def test_first_element_wins_in_a_chain(self) -> None:
        header = "URI=spiffe://example.org/coding-agent,URI=spiffe://example.org/proxy"
        assert spiffe_id_from_xfcc(header) == "spiffe://example.org/coding-agent"

    def test_no_uri_element_is_none(self) -> None:
        assert spiffe_id_from_xfcc("Hash=abc;Subject=x") is None


class TestResolveIdentity:
    def test_svid_wins_over_any_header(self) -> None:
        identity = resolve_identity(
            peer_cert_der=_der("coding-agent"),
            headers={"x-spiffe-id": "spiffe://example.org/research-agent"},
            trust_header=True,
        )
        assert identity == CallerIdentity("spiffe://example.org/coding-agent", "svid")

    def test_header_ignored_unless_explicitly_trusted(self) -> None:
        headers = {"x-spiffe-id": "spiffe://example.org/research-agent"}
        assert resolve_identity(peer_cert_der=None, headers=headers, trust_header=False) is None

    def test_trusted_x_spiffe_id_header(self) -> None:
        headers = {"x-spiffe-id": "spiffe://example.org/research-agent"}
        assert resolve_identity(
            peer_cert_der=None, headers=headers, trust_header=True
        ) == CallerIdentity("spiffe://example.org/research-agent", "header")

    def test_trusted_xfcc_header(self) -> None:
        headers = {"x-forwarded-client-cert": "URI=spiffe://example.org/research-agent"}
        assert resolve_identity(
            peer_cert_der=None, headers=headers, trust_header=True
        ) == CallerIdentity("spiffe://example.org/research-agent", "header")

    def test_no_identity_anywhere_is_none(self) -> None:
        assert resolve_identity(peer_cert_der=None, headers={}, trust_header=True) is None

    def test_malformed_trusted_header_is_refused_not_ignored(self) -> None:
        headers = {"x-spiffe-id": "spiffe://Example.org/agent"}
        with pytest.raises(InvalidSpiffeId):
            resolve_identity(peer_cert_der=None, headers=headers, trust_header=True)

    def test_malformed_svid_is_refused(self) -> None:
        with pytest.raises(InvalidSpiffeId):
            resolve_identity(peer_cert_der=b"junk", headers={}, trust_header=False)
