"""Deterministic sensitive-data marker detection (Build B). RED-phase
skeleton: importable names, no behaviour -- the implementation lands in
Green."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Marker:
    kind: str
    redacted_excerpt: str


def detect(text: str) -> list[Marker]:
    raise NotImplementedError
