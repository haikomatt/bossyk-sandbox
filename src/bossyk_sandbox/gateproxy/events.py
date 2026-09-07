"""The signed gate-event log: one ``{"event", "signature"}`` JSON per line.

Signing reuses auditk's attestation primitives verbatim
(``canonicalize`` + ``LocalEd25519Signer``/``Verifier``) so a gate event
log verifies offline with the same public-key-only story as an evidence
pack -- one signing pattern across both tools, per the round-1 plan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from auditk.attestation.canonical import canonicalize
from auditk.attestation.signer import LocalEd25519Signer, LocalEd25519Verifier


@dataclass
class EventLog:
    """Append-only signed JSONL. The file is created on first append, not
    on construction -- a run that makes no decisions leaves no log."""

    path: Path
    signer_key_path: Path
    run_label: str
    _signer: LocalEd25519Signer | None = field(default=None, repr=False)

    def append(self, event: dict[str, Any]) -> None:
        if self._signer is None:
            self._signer = LocalEd25519Signer(self.signer_key_path)
        full_event = {
            **event,
            "run_label": self.run_label,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        signature = self._signer.sign(canonicalize(full_event))
        line = json.dumps({"event": full_event, "signature": signature.model_dump(mode="json")})
        with self.path.open("a") as handle:
            handle.write(line + "\n")


@dataclass(frozen=True)
class EventLogVerification:
    verified_count: int
    bad_line_numbers: list[int]

    @property
    def ok(self) -> bool:
        return not self.bad_line_numbers


def verify_event_log(path: Path, public_key_pem: str) -> EventLogVerification:
    """Verify every line offline against one public key. A line whose JSON
    is malformed, whose shape is wrong, or whose signature does not check
    out is reported by its 1-based line number -- never skipped silently."""
    verifier = LocalEd25519Verifier(public_key_pem)
    verified = 0
    bad: list[int] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            verifier.verify(canonicalize(entry["event"]), entry["signature"]["signature"])
        except Exception:
            bad.append(line_number)
        else:
            verified += 1
    return EventLogVerification(verified_count=verified, bad_line_numbers=bad)
