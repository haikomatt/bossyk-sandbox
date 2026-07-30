"""RED-phase tests for `ProhibitedPhraseRule` (bossyk-sandbox slice 2, P6).

Spec ("Regulated-phrase list (boundary 5)" in
outreach-domain-cleanroom-spec.md): a CLOSED, enumerated list of
consumer-credit financial-promotion trigger phrases (FSMA s.21 / FCA CONC 3),
matched case-insensitively as whole phrases. A carve-out (a small set of
preceding-negation patterns) must let a DECLINING utterance ("we can't offer
0% finance") through -- the rule must never become a fuzzy classifier; that
is the whole point of a closed list.

Module target for this RED pass (flagged for review, not silently decided):
`UtteranceInstrument` (the scoring Protocol, `score(text, history) -> Decision`,
a sibling to `Instrument`) lives in `instruments/base.py`; the concrete
`ProhibitedPhraseRule` lives in a NEW module `instruments/utterance_rule.py`,
mirroring how `hardcoded_rule.py` holds the tool-call-shaped concrete rules --
an utterance rule scores free text, a genuinely different scoring domain from
ProposedAction/history.

Simplification flagged: the spec notes "pay monthly" and "spread the cost"
are regulated "when paired with a price" -- this RED pass treats the whole
closed list uniformly (always regulated), deferring that nuance; propose
revisiting only if it produces real false positives.
"""

from __future__ import annotations

from bossyk_sandbox.instruments.base import Verdict
from bossyk_sandbox.instruments.utterance_rule import (
    DEFAULT_CARVE_OUT,
    DEFAULT_REGULATED_PHRASES,
    ProhibitedPhraseRule,
)


def _rule() -> ProhibitedPhraseRule:
    return ProhibitedPhraseRule(terms=DEFAULT_REGULATED_PHRASES, carve_out=DEFAULT_CARVE_OUT)


def test_default_regulated_phrases_match_the_cleanroom_spec_list() -> None:
    expected = {
        "0% finance",
        "0% apr",
        "interest free",
        "interest-free",
        "buy now pay later",
        "bnpl",
        "spread the cost",
        "pay monthly",
        "no deposit",
        "guaranteed acceptance",
        "guaranteed finance",
        "free money",
        "government grant",
    }
    assert set(DEFAULT_REGULATED_PHRASES) == expected


def test_blocks_on_each_regulated_phrase() -> None:
    rule = _rule()
    for phrase in DEFAULT_REGULATED_PHRASES:
        decision = rule.score(f"Great news, we can offer {phrase} on this order.", history=[])
        assert decision.verdict is Verdict.BLOCK, f"expected BLOCK for {phrase!r}"


def test_allows_clean_text_with_no_regulated_phrase() -> None:
    rule = _rule()

    decision = rule.score("Thanks for your interest, I'll book your free survey now.", history=[])

    assert decision.verdict is Verdict.ALLOW


def test_matching_is_case_insensitive() -> None:
    rule = _rule()

    decision = rule.score("We can offer 0% FINANCE on this order.", history=[])

    assert decision.verdict is Verdict.BLOCK


def test_carve_out_negation_allows_a_declining_utterance() -> None:
    rule = _rule()

    decision = rule.score("I'm sorry, we can't offer 0% finance on this product.", history=[])

    assert decision.verdict is Verdict.ALLOW


def test_carve_out_covers_multiple_negation_phrasings() -> None:
    rule = _rule()
    for declining in (
        "We cannot offer interest free credit on this item.",
        "Unfortunately we don't offer buy now pay later here.",
        "We won't offer no deposit terms on this order.",
    ):
        decision = rule.score(declining, history=[])
        assert decision.verdict is Verdict.ALLOW, f"expected ALLOW for {declining!r}"


def test_does_not_false_positive_on_a_bare_substring_of_a_regulated_phrase() -> None:
    # "interest" alone (e.g. "I'm interested in solar panels") must not trip
    # -- only the enumerated whole phrase "interest free"/"interest-free".
    rule = _rule()

    decision = rule.score("I'm interested in solar panels for my home.", history=[])

    assert decision.verdict is Verdict.ALLOW


def test_a_carve_out_far_before_the_phrase_does_not_suppress_an_unrelated_promotion() -> None:
    # The negation must be near the matched phrase, not merely anywhere
    # earlier in a long utterance -- otherwise an early, unrelated "can't"
    # would silently launder a later genuine promotion.
    rule = _rule()

    decision = rule.score(
        "I can't check your address right now, but I can confirm we offer 0% finance "
        "on this whole range, so let's get you booked in today.",
        history=[],
    )

    assert decision.verdict is Verdict.BLOCK
