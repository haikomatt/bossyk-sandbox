"""Caller identity for the gate proxy (round-2 Phase 3).

The gate reads a *verified* caller identity so different callers on one
gate can run under different policy packs (see `identity_packs`). Two
sources, in fixed precedence:

1. **SVID** -- the SPIFFE X509-SVID the caller presented on an mTLS
   connection, i.e. the certificate the TLS layer already verified against
   the operator's trust bundle (`--client-ca`). Its single `spiffe://` URI
   SAN is the identity. This is the source to prefer: it cannot be
   spoofed by the caller.
2. **Header** -- `X-Forwarded-Client-Cert` (Envoy XFCC, `URI=` element) or
   `X-SPIFFE-ID`, read ONLY when `--trust-identity-header` is set. A
   header is whatever the caller says it is; it is only meaningful when a
   proxy the operator controls terminated mTLS and set it, and the gate is
   reachable from nowhere else. That is the operator's boundary to keep,
   and the flag exists so the trust is explicit, never assumed.

Malformed identities are refused (`InvalidSpiffeId`), never coerced or
ignored: an identity the gate cannot read must not become "no identity"
and slide into the default pack. Grammar is the SPIFFE-ID spec.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass

from cryptography import x509
from uvicorn.protocols.http.h11_impl import H11Protocol

logger = logging.getLogger(__name__)

_MAX_SPIFFE_ID_BYTES = 2048
_TRUST_DOMAIN = re.compile(r"^[a-z0-9._-]+$")
_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
_SCHEME = "spiffe://"

# Where the mTLS listener leaves the peer certificate for the request.
PEER_CERT_STATE_KEY = "peer_cert_der"


class InvalidSpiffeId(ValueError):
    """A caller identity that is not a well-formed SPIFFE ID."""


@dataclass(frozen=True)
class CallerIdentity:
    spiffe_id: str
    source: str  # "svid" | "header"


def parse_spiffe_id(value: str) -> str:
    """Validate `value` against the SPIFFE-ID grammar and return it."""
    if not value.startswith(_SCHEME):
        raise InvalidSpiffeId(f"not a spiffe:// URI: {value!r}")
    if len(value.encode()) > _MAX_SPIFFE_ID_BYTES:
        raise InvalidSpiffeId("SPIFFE ID longer than 2048 bytes")
    rest = value[len(_SCHEME) :]
    trust_domain, slash, path = rest.partition("/")
    if not _TRUST_DOMAIN.match(trust_domain):
        raise InvalidSpiffeId(f"invalid trust domain in {value!r}")
    if not slash or not path:
        raise InvalidSpiffeId(f"SPIFFE ID has no path: {value!r}")
    for segment in path.split("/"):
        if segment in ("", ".", "..") or not _PATH_SEGMENT.match(segment):
            raise InvalidSpiffeId(f"invalid path segment {segment!r} in {value!r}")
    return value


def spiffe_id_from_der(der: bytes) -> str:
    """The SPIFFE ID of an X509-SVID: exactly one `spiffe://` URI SAN."""
    try:
        certificate = x509.load_der_x509_certificate(der)
    except ValueError as exc:
        raise InvalidSpiffeId("peer certificate is not parseable DER") from exc
    try:
        san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound as exc:
        raise InvalidSpiffeId("peer certificate has no spiffe URI SAN") from exc
    uris = [u for u in san.get_values_for_type(x509.UniformResourceIdentifier)]
    spiffe_uris = [u for u in uris if u.startswith(_SCHEME)]
    if not spiffe_uris:
        raise InvalidSpiffeId("peer certificate has no spiffe URI SAN")
    if len(spiffe_uris) != 1:
        raise InvalidSpiffeId("an X509-SVID must carry exactly one spiffe URI SAN")
    return parse_spiffe_id(spiffe_uris[0])


def spiffe_id_from_xfcc(header: str) -> str | None:
    """The `URI=` element of the first certificate in an Envoy
    X-Forwarded-Client-Cert header (the immediate client), if any."""
    for element in _xfcc_first_certificate(header).split(";"):
        key, _, value = element.partition("=")
        if key.strip().lower() == "uri":
            return value.strip().strip('"')
    return None


def _xfcc_first_certificate(header: str) -> str:
    """Up to the first comma outside double quotes (XFCC quotes values
    that contain commas, e.g. Subject="CN=x,O=y")."""
    quoted = False
    for index, char in enumerate(header):
        if char == '"':
            quoted = not quoted
        elif char == "," and not quoted:
            return header[:index]
    return header


def resolve_identity(
    *, peer_cert_der: bytes | None, headers: Mapping[str, str], trust_header: bool
) -> CallerIdentity | None:
    """SVID first; a trusted header only when there is no SVID; None when
    neither is present. Raises `InvalidSpiffeId` rather than degrading."""
    if peer_cert_der is not None:
        return CallerIdentity(spiffe_id_from_der(peer_cert_der), "svid")
    if not trust_header:
        return None
    lower = {k.lower(): v for k, v in headers.items()}
    claimed = lower.get("x-spiffe-id")
    if claimed is None and "x-forwarded-client-cert" in lower:
        claimed = spiffe_id_from_xfcc(lower["x-forwarded-client-cert"])
    if claimed is None:
        return None
    return CallerIdentity(parse_spiffe_id(claimed), "header")


class PeerCertH11Protocol(H11Protocol):
    """uvicorn HTTP protocol that surfaces the mTLS peer certificate.

    uvicorn does not expose the TLS peer certificate to the ASGI app. This
    subclass reads it off the transport's SSL object once per connection
    and places it in the per-connection `app_state`, which uvicorn copies
    into every request's `scope["state"]` -- so the app reads
    `request.state.peer_cert_der`. Nothing else about uvicorn is touched.
    """

    def connection_made(self, transport: asyncio.Transport) -> None:  # type: ignore[override]
        super().connection_made(transport)
        ssl_object = transport.get_extra_info("ssl_object")
        der = ssl_object.getpeercert(binary_form=True) if ssl_object is not None else None
        # A fresh dict per connection: the base class shares one app_state
        # across connections, and a peer certificate must never leak from
        # one caller's connection into another's requests.
        self.app_state = {**self.app_state, PEER_CERT_STATE_KEY: der}
