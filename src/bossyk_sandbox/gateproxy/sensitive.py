"""Deterministic sensitive-data marker detection (Build B).

`detect(text)` scans free text against a fixed bank of regex (and, for
two kinds, checksum-gated) rules for common secret and PII shapes and
returns one `Marker` per distinct hit. This is pattern matching only --
no entropy scoring, no ML/NER -- so the same input always yields the
same markers and every hit traces to a named, documented rule rather
than a tuned threshold.

v1 is detection-only: markers are informational. They are attached to
gate events and surfaced on the scorecard's "Sensitive data observed"
section, but they NEVER change a `Decision` or block anything -- wiring
markers into policy (e.g. "block if a payment_card_number is proposed")
is explicitly out of scope for this module and is a round-2 decision,
not an oversight.

`Marker.redacted_excerpt` carries only the first four characters of the
match plus its kind label (e.g. "sk-a…openai_api_key") -- never the
match itself and never enough of it to reconstruct the secret. That is
what makes it safe to print on a scorecard that may travel outside the
trust boundary holding the raw event log.

Pattern provenance -- regex TEXT (not code) adapted with attribution per
each source's licence; all sources below are MIT or Apache-2.0, which
this dossier's licence audit confirmed can be folded into this BSL 1.1
codebase with attribution. Two sources with equivalent shapes,
trufflehog and Warp (warpdotdev), were deliberately NOT used as copy
sources because their client code is AGPL-3.0 (copyleft, not safely
mixable into BSL 1.1); their public docs/fixtures were used only to
cross-check that these shapes are the right ones, never to source text.

 - gitleaks (MIT) https://github.com/gitleaks/gitleaks --
   openai_api_key, aws_access_key_id (shape), github_pat,
   github_oauth_token, google_api_key, slack_token (consolidated from
   its bot/user/app token rules), generic_bearer_jwt, pem_private_key
 - detect-secrets (Apache-2.0) https://github.com/Yelp/detect-secrets --
   aws_access_key_id (cross-validated against gitleaks' pattern)
 - Microsoft Presidio (MIT) https://github.com/microsoft/presidio --
   email_address, iban (shape), payment_card_number (shape)
 - HMRC/gov.uk public NINO format specification (a government format
   specification, not independently copyrightable) --
   uk_national_insurance_number, written independently against the
   published format rather than copied from any single implementation
 - ITU-T E.164 public numbering-plan specification (not independently
   copyrightable) -- phone_e164, adapted (see note below) from the bare
   `^\\+[1-9]\\d{1,14}$` whole-string form to a substring-safe form with
   boundary lookarounds and a raised minimum length, since `detect`
   scans arbitrary free text rather than validating an already-isolated
   candidate string

`iban` and `payment_card_number` are checksum-gated: a bare regex hit
for either is discarded unless it also passes its checksum (ISO 7064
mod-97 for IBAN, Luhn/ISO 7812 for card numbers). Presidio treats the
bare shapes the same way (a low-confidence "weak" score bumped by the
checksum) because, alone, both shapes collide heavily with unrelated
numeric/alphanumeric formats (SKUs, batch codes, reference numbers).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Marker:
    kind: str
    redacted_excerpt: str


def _luhn_ok(candidate: str) -> bool:
    """Luhn algorithm (public standard, ISO/IEC 7812), independently
    reimplemented from the public spec, not copied from any one source."""
    digits = [int(d) for d in candidate if d.isdigit()]
    if len(digits) < 12:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _iban_ok(candidate: str) -> bool:
    """ISO 7064 mod-97 check-digit validation (public banking standard,
    ISO 13616). Algorithm re-described from Presidio's IbanRecognizer
    documentation, not copied verbatim."""
    cleaned = candidate.replace(" ", "").replace("-", "").upper()
    if len(cleaned) < 4:
        return False
    rearranged = cleaned[4:] + cleaned[:4]
    try:
        numeric = "".join(str(int(ch, 36)) for ch in rearranged)  # A=10 ... Z=35
    except ValueError:
        return False
    return int(numeric) % 97 == 1


@dataclass(frozen=True)
class _Rule:
    kind: str
    pattern: re.Pattern[str]
    checksum: Callable[[str], bool] | None = None


def _compile(
    kind: str, raw_pattern: str, *, checksum: Callable[[str], bool] | None = None
) -> _Rule:
    # Patterns that already embed an inline (?i) get no extra flag (it
    # would be redundant, not incorrect); everything else is compiled
    # case-insensitively, matching each source's own stated intent.
    flags = 0 if raw_pattern.startswith("(?i)") else re.IGNORECASE
    return _Rule(kind=kind, pattern=re.compile(raw_pattern, flags), checksum=checksum)


_BANK: tuple[_Rule, ...] = (
    _compile(
        "openai_api_key",
        r"\b(sk-(?:proj|svcacct|admin)-(?:[A-Za-z0-9_-]{74}|[A-Za-z0-9_-]{58})T3BlbkFJ"
        r"(?:[A-Za-z0-9_-]{74}|[A-Za-z0-9_-]{58})\b|sk-[a-zA-Z0-9]{20}T3BlbkFJ[a-zA-Z0-9]{20})",
    ),
    _compile(
        "aws_access_key_id",
        r"\b((?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16})\b",
    ),
    _compile("github_pat", r"\bghp_[0-9a-zA-Z]{36}\b"),
    _compile("github_oauth_token", r"\bgho_[0-9a-zA-Z]{36}\b"),
    _compile("google_api_key", r"\bAIza[\w-]{35}\b"),
    _compile(
        "slack_token",
        r"\bxox[baprs]-[0-9]{10,13}(?:-[0-9]{10,13}|-[A-Za-z0-9-]{10,48})[A-Za-z0-9-]*\b",
    ),
    _compile(
        "generic_bearer_jwt",
        r"\b(ey[a-zA-Z0-9]{17,}\.ey[a-zA-Z0-9/\\_-]{17,}\.(?:[a-zA-Z0-9/\\_-]{10,}={0,2})?)\b",
    ),
    _compile(
        "pem_private_key",
        r"(?i)-----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----[\s\S-]{64,}?"
        r"KEY(?: BLOCK)?-----",
    ),
    _compile(
        "email_address",
        r"\b((([!#$%&'*+\-/=?^_`{|}~\w])|([!#$%&'*+\-/=?^_`{|}~\w]"
        r"[!#$%&'*+\-/=?^_`{|}~.\w]{0,}[!#$%&'*+\-/=?^_`{|}~\w]))[@]\w+"
        r"(?:-+\w+)*(?:\.\w+(?:-+\w+)*)+)\b",
    ),
    _compile(
        # ITU-T E.164 is `^\+[1-9]\d{1,14}$` as a whole-string validator;
        # adapted here with boundary lookarounds (so it can fire on a
        # substring inside a larger blob of text) and a raised minimum
        # digit count (8, not 1) to cut false positives against short
        # arbitrary "+N..." numeric sequences that are not phone numbers.
        "phone_e164",
        r"(?<!\d)\+[1-9]\d{7,14}(?!\d)",
    ),
    _compile(
        "uk_national_insurance_number",
        r"(?i)\b(?!BG|GB|NK|KN|TN|NT|ZZ)[ABCEGHJ-PRSTW-Z][ABCEGHJ-NPRSTW-Z]"
        r"[ -]?\d{2}[ -]?\d{2}[ -]?\d{2}[ -]?[A-D]\b",
    ),
    _compile(
        "iban",
        r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b",
        checksum=_iban_ok,
    ),
    _compile(
        "payment_card_number",
        r"\b(?!1\d{12}(?!\d))((4\d{3})|(5[0-5]\d{2})|(6\d{3})|(1\d{3})|(3\d{3}))"
        r"[- ]?(\d{3,4})[- ]?(\d{3,4})[- ]?(\d{3,5})\b",
        checksum=_luhn_ok,
    ),
)


def _excerpt(matched: str, kind: str) -> str:
    return f"{matched[:4]}…{kind}"


def detect(text: str) -> list[Marker]:
    """Scan `text` against the fixed bank, in bank order, and return one
    `Marker` per distinct (kind, matched-text) hit -- repeats of the same
    literal under the same rule are reported once. A checksum-gated
    rule's bare regex hit is silently dropped (not reported at all) when
    the checksum fails, since a bare hit there is not evidence of
    anything on its own (see module docstring)."""
    markers: list[Marker] = []
    seen: set[tuple[str, str]] = set()
    for rule in _BANK:
        for match in rule.pattern.finditer(text):
            candidate = match.group(0)
            if rule.checksum is not None and not rule.checksum(candidate):
                continue
            key = (rule.kind, candidate)
            if key in seen:
                continue
            seen.add(key)
            markers.append(Marker(kind=rule.kind, redacted_excerpt=_excerpt(candidate, rule.kind)))
    return markers
