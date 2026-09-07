"""Red-phase contract for deterministic sensitive-data marker detection
(Build B).

`detect(text)` is a pure function over a fixed regex/checksum bank -- no
entropy scoring, no ML/NER, same input always yields the same markers.
`Marker.redacted_excerpt` must never carry more than the first four
characters of a match plus its kind label, so a caller (the scorecard,
in particular) can print it without leaking the underlying secret.
"""

from __future__ import annotations

from bossyk_sandbox.gateproxy.sensitive import Marker, detect

# A syntactically valid but never-real OpenAI-shaped key: sk- + 20 chars +
# the fixed T3BlbkFJ infix + 20 chars, matching the gitleaks-derived
# openai_api_key pattern without being any actual issued credential.
_FAKE_OPENAI_KEY = "sk-" + "a" * 20 + "T3BlbkFJ" + "b" * 20

# A syntactically valid AWS-access-key-id shape (AKIA + 16 chars from
# [A-Z2-7]) that was never issued by AWS.
_FAKE_AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"

_VALID_TEST_CARD = "4111111111111111"  # standard Luhn-valid Visa test PAN
_INVALID_CHECKSUM_CARD = "4111111111111112"  # same shape, Luhn fails


class TestDetectSecrets:
    def test_openai_key_detected(self) -> None:
        markers = detect(f"export OPENAI_API_KEY={_FAKE_OPENAI_KEY}")
        assert any(m.kind == "openai_api_key" for m in markers)

    def test_aws_key_detected(self) -> None:
        markers = detect(f"aws_access_key_id = {_FAKE_AWS_KEY}")
        assert any(m.kind == "aws_access_key_id" for m in markers)

    def test_github_pat_detected(self) -> None:
        markers = detect("token: ghp_" + "a" * 36)
        assert any(m.kind == "github_pat" for m in markers)

    def test_pem_private_key_detected(self) -> None:
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIB" + "x" * 80 + "\n-----END RSA PRIVATE KEY-----"
        markers = detect(pem)
        assert any(m.kind == "pem_private_key" for m in markers)

    def test_jwt_detected(self) -> None:
        fake_jwt = (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
            ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        )
        markers = detect(f"Authorization: Bearer {fake_jwt}")
        assert any(m.kind == "generic_bearer_jwt" for m in markers)

    def test_email_detected(self) -> None:
        markers = detect("contact matt.lesportif@example.com for details")
        assert any(m.kind == "email_address" for m in markers)

    def test_e164_phone_detected(self) -> None:
        markers = detect("+447911123456")
        assert any(m.kind == "phone_e164" for m in markers)

    def test_uk_nino_detected(self) -> None:
        markers = detect("NINO: AB123456C")
        assert any(m.kind == "uk_national_insurance_number" for m in markers)

    def test_no_markers_in_clean_text(self) -> None:
        assert detect("ls -1 the workspace directory please") == []


class TestChecksumGating:
    """The two checksum-gated kinds: a bare regex hit that fails its
    checksum must not be reported at all -- gating happens inside
    `detect`, not left to the caller."""

    def test_luhn_valid_card_detected(self) -> None:
        markers = detect(_VALID_TEST_CARD)
        assert any(m.kind == "payment_card_number" for m in markers)

    def test_luhn_invalid_card_not_detected(self) -> None:
        markers = detect(_INVALID_CHECKSUM_CARD)
        assert not any(m.kind == "payment_card_number" for m in markers)

    def test_iban_checksum_gates_bare_shape(self) -> None:
        # GB29 NWBK 6016 1331 9268 19 is the canonical Wikipedia IBAN
        # worked example (a real published example, not a live account).
        valid_iban = "GB29NWBK60161331926819"
        invalid_iban = "GB29NWBK60161331926818"  # last digit flipped
        assert any(m.kind == "iban" for m in detect(valid_iban))
        assert not any(m.kind == "iban" for m in detect(invalid_iban))


class TestRedactedExcerpt:
    def test_excerpt_is_first_four_chars_plus_kind_never_full_match(self) -> None:
        markers = detect(f"key={_FAKE_OPENAI_KEY}")
        (marker,) = [m for m in markers if m.kind == "openai_api_key"]
        assert isinstance(marker, Marker)
        assert marker.redacted_excerpt == f"{_FAKE_OPENAI_KEY[:4]}…openai_api_key"
        assert _FAKE_OPENAI_KEY not in marker.redacted_excerpt
        assert _FAKE_OPENAI_KEY[5:] not in marker.redacted_excerpt

    def test_deduplicates_repeated_identical_matches(self) -> None:
        text = f"{_FAKE_AWS_KEY} appears twice: {_FAKE_AWS_KEY}"
        markers = [m for m in detect(text) if m.kind == "aws_access_key_id"]
        assert len(markers) == 1

    def test_deterministic_across_calls(self) -> None:
        text = f"{_FAKE_OPENAI_KEY} and {_FAKE_AWS_KEY} and {_VALID_TEST_CARD}"
        assert detect(text) == detect(text)
