"""Tokenisation proxy on the chat path (bossyk-sandbox privacy/minimisation
demonstrator, Phase 3): swaps raw persona PII in user-supplied text for
stable surrogates before the agent sees it (`redact`), and swaps surrogates
back for display (`rehydrate`). Console wiring (where this proxy sits on the
live chat path) is Phase 5 -- this module is built and tested hermetically,
with no console/agent dependency.

Detection strategy -- fixtures-first, documented as such: this module does
exact and lightly-normalised matching against the values actually present
in a given `advice.personas.PersonaStore`. It is NOT general-purpose NER.
The demo's personas are known in advance, so recall only needs to cover
those known values plus a handful of common alternate renderings (case,
spacing, date order, currency formatting) -- not open-world free text. That
is the honest, scoped claim: this catches "the fixture values, rendered a
few common ways", not "any PII anyone might type".

Four detected PII categories, one per `advice.personas.PersonaRecord`
field with a distinctive literal rendering:
  - PERSON   -- `full_name`, matched case-insensitively.
  - POSTCODE -- `postcode`, matched with or without the internal space.
  - DOB      -- `date_of_birth` (stored ISO `YYYY-MM-DD`), matched in that
    form plus the common `DD/MM/YYYY` rendering.
  - INCOME   -- `annual_income_gbp`, matched as plain digits, with
    thousands-separator commas, and with an optional leading "£".

`health_condition_flag` is deliberately OUT of scope for `redact`/
`find_raw_pii`: it is a boolean with no distinctive literal rendering --
unlike a name or postcode, the words "true"/"false" are not a safe pattern
to flag (every other sentence would false-positive). This is the same
honest fixtures-first boundary as the NER exclusion, not an oversight.

Persona `ref` values (e.g. "ADV-0001") are also NOT redacted. A ref is
already an opaque, non-identifying lookup key -- see
`advice.personas.PersonaRecord` -- not one of the PII fields above, so
there is nothing to hide. Refs are the system's own stable "identity
reference": a caller can safely keep using them (e.g. as a tool-call
argument) without needing a surrogate at all.

Longest-match-first: PII literals across all personas/categories/renderings
are tried longest-first at every text position, so if one value is a
literal substring of another (e.g. persona A is "Alex Fenwick" and persona
B is "Fenwick"), the longer match wins and consumes the whole span --
"Fenwick" alone is not separately matched inside "Alex Fenwick". See
`test_overlapping_names_resolve_longest_match_first`. Matches are also
boundary-guarded (`(?<![A-Za-z0-9])...(?![A-Za-z0-9])`) so a literal never
matches as a sub-run of a longer alphanumeric token that merely contains it
(e.g. income `47300` inside `473005`).

Round-trip guarantee, scoped: `rehydrate(redact(text).text) == text` holds
when `text` uses each field's CANONICAL (store-format) rendering. Alternate
renderings (comma-formatted income, `DD/MM/YYYY` dates, unspaced postcodes,
different name casing) are still detected and redacted -- the no-raw-PII
guarantee holds for them too -- but because several distinct literal
renderings can share one stable surrogate (a deliberate feature: "the same
underlying value maps to the same surrogate", regardless of how it was
typed), `rehydrate` can only restore ONE rendering per surrogate. It always
restores the canonical one. This is a documented trade-off, not a bug: the
safety property (no raw PII leaks past `redact`) holds unconditionally; the
convenience property (byte-exact round trip) holds for canonical-format
input.

Trace-content guarantee: `assert_no_raw_pii(text, store)` (built on the pure
`find_raw_pii`) is what Phase 4/5 use in tests and evidence-pack assertions
to prove agent context/traces carry surrogates only, never raw persona
values -- built on the exact same detection pattern as `redact`, so it
catches the same normalised renderings.

Repr guard: `SurrogateMap` -- the per-session value<->surrogate mapping --
must never be serialised into a trace or log. `TokenisationProxy.__repr__`
is written by hand (not derived from the dataclass default, and not
delegating to `self.store`'s own repr, which would print every persona
field) so that logging a proxy, deliberately or accidentally, cannot leak
either the map or the underlying store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bossyk_sandbox.advice.personas import PersonaStore

CATEGORY_PERSON = "PERSON"
CATEGORY_POSTCODE = "POSTCODE"
CATEGORY_DOB = "DOB"
CATEGORY_INCOME = "INCOME"

_SURROGATE_TOKEN_RE = re.compile(r"<(?:PERSON|POSTCODE|DOB|INCOME)_\d+>")

# Matches never span into a longer alphanumeric run -- see the module
# docstring's "Longest-match-first" note.
_BOUNDARY_BEFORE = r"(?<![A-Za-z0-9])"
_BOUNDARY_AFTER = r"(?![A-Za-z0-9])"


@dataclass(frozen=True)
class RedactedText:
    """`redact()`'s output: the surrogate-swapped text. A thin wrapper
    (not a bare `str`) so a caller can't accidentally pass unredacted text
    somewhere a `RedactedText` is expected -- the type itself is the
    "this has been through the proxy" marker."""

    text: str


@dataclass(frozen=True)
class PiiMatch:
    """One raw-PII occurrence found by `find_raw_pii`: which persona, which
    field category, the canonical (store-format) value, and the literal
    substring actually matched (which may be a normalised rendering, not
    the canonical form -- e.g. `matched_text="47,300"` with
    `canonical_value="47300"`)."""

    ref: str
    category: str
    canonical_value: str
    matched_text: str


class RawPiiLeakError(AssertionError):
    """Raised by `assert_no_raw_pii` when raw persona PII is found in text
    that should carry surrogates only. An `AssertionError` subclass so it
    behaves naturally in test assertions and pack-building checks alike."""


@dataclass
class SurrogateMap:
    """Session-scoped value<->surrogate mapping. One instance backs one
    `TokenisationProxy` (one chat session): the same `(category,
    canonical_value)` pair always yields the same surrogate for the life of
    the map (stability), and distinct values always yield distinct
    surrogates (per-category counters, never reused).

    Must never be serialised into a trace or log -- see the module
    docstring's "Repr guard" note. `__repr__`/`__str__` are written by hand
    so the map's own logging surface cannot leak raw values either, as a
    second line of defence behind `TokenisationProxy.__repr__`.
    """

    _surrogate_by_value: dict[tuple[str, str], str] = field(default_factory=dict)
    _value_by_surrogate: dict[str, str] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=dict)

    def surrogate_for(self, category: str, canonical_value: str) -> str:
        """Returns the stable surrogate for `(category, canonical_value)`,
        minting a new one (via the per-category counter) the first time
        this exact value is seen."""
        key = (category, canonical_value)
        existing = self._surrogate_by_value.get(key)
        if existing is not None:
            return existing
        next_index = self._counters.get(category, 0) + 1
        self._counters[category] = next_index
        surrogate = f"<{category}_{next_index}>"
        self._surrogate_by_value[key] = surrogate
        self._value_by_surrogate[surrogate] = canonical_value
        return surrogate

    def value_for(self, surrogate: str) -> str | None:
        """The canonical value a surrogate was minted for, or `None` if
        this map never minted it (e.g. it was typed by a user rather than
        produced by `redact`)."""
        return self._value_by_surrogate.get(surrogate)

    def __repr__(self) -> str:
        return f"SurrogateMap({len(self._value_by_surrogate)} surrogate(s))"

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(frozen=True)
class _PiiPattern:
    """One literal rendering to search for, tagged with the persona/field
    it belongs to."""

    literal: str
    category: str
    canonical_value: str
    ref: str


def _postcode_variants(postcode: str) -> list[str]:
    compact = postcode.replace(" ", "")
    return [postcode, compact] if compact != postcode else [postcode]


def _dob_variants(date_of_birth: str) -> list[str]:
    year, month, day = date_of_birth.split("-")
    ddmmyyyy = f"{day}/{month}/{year}"
    return [date_of_birth, ddmmyyyy]


def _income_variants(annual_income_gbp: int) -> list[str]:
    plain = str(annual_income_gbp)
    with_commas = f"{annual_income_gbp:,}"
    variants = [plain]
    if with_commas != plain:
        variants.append(with_commas)
    variants.append(f"£{plain}")
    if with_commas != plain:
        variants.append(f"£{with_commas}")
    return variants


def _patterns_for_store(store: PersonaStore) -> list[_PiiPattern]:
    """Every literal rendering `redact`/`find_raw_pii` recognise, across
    every persona in `store` -- see the module docstring's category list.
    `ref` is deliberately not included: refs are not PII (see the module
    docstring)."""
    patterns: list[_PiiPattern] = []
    for record in store.records:
        patterns.append(
            _PiiPattern(record.full_name, CATEGORY_PERSON, record.full_name, record.ref)
        )
        for literal in _postcode_variants(record.postcode):
            patterns.append(_PiiPattern(literal, CATEGORY_POSTCODE, record.postcode, record.ref))
        for literal in _dob_variants(record.date_of_birth):
            patterns.append(_PiiPattern(literal, CATEGORY_DOB, record.date_of_birth, record.ref))
        canonical_income = str(record.annual_income_gbp)
        for literal in _income_variants(record.annual_income_gbp):
            patterns.append(_PiiPattern(literal, CATEGORY_INCOME, canonical_income, record.ref))
    return patterns


def _compile_matcher(patterns: list[_PiiPattern]) -> tuple[re.Pattern[str], dict[str, _PiiPattern]]:
    """A single compiled pattern (longest-literal-first alternation,
    boundary-guarded, case-insensitive) plus a lowercase-literal ->
    `_PiiPattern` lookup for resolving what a given match was."""
    by_literal_lower: dict[str, _PiiPattern] = {}
    for pattern in patterns:
        by_literal_lower[pattern.literal.lower()] = pattern
    if not by_literal_lower:
        return re.compile(r"(?!x)x"), by_literal_lower  # matches nothing
    ordered_literals = sorted(by_literal_lower, key=len, reverse=True)
    body = "|".join(re.escape(literal) for literal in ordered_literals)
    compiled = re.compile(f"{_BOUNDARY_BEFORE}(?:{body}){_BOUNDARY_AFTER}", re.IGNORECASE)
    return compiled, by_literal_lower


def find_raw_pii(text: str, store: PersonaStore) -> list[PiiMatch]:
    """Scans `text` for any raw persona value from `store` -- exact and
    normalised renderings alike (see the module docstring's category
    list) -- and reports every occurrence found. Returns an empty list for
    clean text. Pure/read-only: never mutates `text` or `store`."""
    patterns = _patterns_for_store(store)
    compiled, by_literal_lower = _compile_matcher(patterns)
    matches: list[PiiMatch] = []
    for m in compiled.finditer(text):
        pii = by_literal_lower[m.group(0).lower()]
        matches.append(PiiMatch(pii.ref, pii.category, pii.canonical_value, m.group(0)))
    return matches


def assert_no_raw_pii(text: str, store: PersonaStore) -> None:
    """Raises `RawPiiLeakError` if `find_raw_pii(text, store)` finds
    anything, listing which persona/category/rendering leaked. This is the
    trace-content guarantee Phase 4/5 use in tests and evidence-pack
    assertions."""
    matches = find_raw_pii(text, store)
    if matches:
        details = ", ".join(
            f"{m.category}:{m.canonical_value} (as {m.matched_text!r}, ref {m.ref})"
            for m in matches
        )
        raise RawPiiLeakError(f"raw PII found in text: {details}")


def redact(text: str, store: PersonaStore, surrogate_map: SurrogateMap) -> RedactedText:
    """Swaps every raw persona value from `store` found in `text` for its
    stable surrogate from `surrogate_map`, minting new surrogates as
    needed. Text with no matches is returned unchanged (identity for
    PII-free text)."""
    patterns = _patterns_for_store(store)
    compiled, by_literal_lower = _compile_matcher(patterns)

    def _replace(m: re.Match[str]) -> str:
        pii = by_literal_lower[m.group(0).lower()]
        return surrogate_map.surrogate_for(pii.category, pii.canonical_value)

    return RedactedText(compiled.sub(_replace, text))


def rehydrate(text: str, surrogate_map: SurrogateMap) -> str:
    """Swaps every surrogate token in `text` back for the value
    `surrogate_map` minted it for. A surrogate-shaped token this map never
    minted (e.g. user-typed text that merely looks like `<PERSON_1>`) is
    left untouched rather than guessed at."""

    def _replace(m: re.Match[str]) -> str:
        value = surrogate_map.value_for(m.group(0))
        return value if value is not None else m.group(0)

    return _SURROGATE_TOKEN_RE.sub(_replace, text)


@dataclass
class TokenisationProxy:
    """The chat-path tokenisation proxy: one instance per session, holding
    one `SurrogateMap`. `redact` is the inbound leg (user text -> surrogate
    text, before the agent sees it); `rehydrate` is the outbound leg
    (surrogate text -> real values, for display to the user). Console
    wiring onto the actual chat path is Phase 5 -- this class is usable
    standalone, with no dependency on it.
    """

    store: PersonaStore
    _surrogate_map: SurrogateMap = field(default_factory=SurrogateMap, repr=False)

    def redact(self, text: str) -> RedactedText:
        return redact(text, self.store, self._surrogate_map)

    def rehydrate(self, text: str) -> str:
        return rehydrate(text, self._surrogate_map)

    def __repr__(self) -> str:
        # Deliberately does NOT interpolate `self.store` (its own default
        # repr would print every persona's raw fields) or the surrogate
        # map -- see the module docstring's "Repr guard" note.
        return f"TokenisationProxy(store=<PersonaStore: {len(self.store.records)} persona(s)>)"

    def __str__(self) -> str:
        return self.__repr__()
