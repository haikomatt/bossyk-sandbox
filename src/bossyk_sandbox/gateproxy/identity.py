"""Caller identity for the gate proxy (round-2 Phase 3). RED-phase
skeleton: importable names, no behaviour -- the implementation lands in
Green."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from uvicorn.protocols.http.h11_impl import H11Protocol


class InvalidSpiffeId(ValueError):
    """A caller identity that is not a well-formed SPIFFE ID."""


@dataclass(frozen=True)
class CallerIdentity:
    spiffe_id: str
    source: str  # "svid" | "header"


def parse_spiffe_id(value: str) -> str:
    raise NotImplementedError


def spiffe_id_from_der(der: bytes) -> str:
    raise NotImplementedError


def spiffe_id_from_xfcc(header: str) -> str | None:
    raise NotImplementedError


def resolve_identity(
    *, peer_cert_der: bytes | None, headers: Mapping[str, str], trust_header: bool
) -> CallerIdentity | None:
    raise NotImplementedError


class PeerCertH11Protocol(H11Protocol):
    """uvicorn HTTP protocol that surfaces the mTLS peer certificate."""
