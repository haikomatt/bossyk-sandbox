"""Hermetic tests for scripts/lexical_overlap_audit.py's pure computation
helpers (advice-eligibility-domain-spec.md build order step 5, the
pre-registered confound gate). No network, no domain_config/scenario loading
here -- those are exercised by test_lexical_overlap_audit_gate.py's
end-to-end run. This file only pins down tokenize/tf-idf/jaccard/gate math
against small, fully-controlled inputs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "lexical_overlap_audit.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lexical_overlap_audit_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The script defines `@dataclass`-decorated classes under `from __future__
    # import annotations` (postponed evaluation): dataclass's field-type
    # resolution looks the defining module up via `sys.modules[cls.__module__]`,
    # which requires the module to be registered there BEFORE exec_module runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tokenize_lowercases_and_splits_on_non_alphanumerics() -> None:
    module = _import_script()
    assert module.tokenize("Cancel Order #W1001!") == ["cancel", "order", "w1001"]


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    module = _import_script()
    vector = {"a": 1.0, "b": 2.0}
    assert module.cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_of_disjoint_vectors_is_zero() -> None:
    module = _import_script()
    assert module.cosine_similarity({"a": 1.0}, {"b": 1.0}) == 0.0


def test_cosine_similarity_handles_an_empty_vector() -> None:
    module = _import_script()
    assert module.cosine_similarity({}, {"a": 1.0}) == 0.0


def test_tf_idf_vectors_downweight_a_term_shared_by_every_domain() -> None:
    module = _import_script()
    # Equal term frequency (1 each) in every "document", so only the idf
    # component (df=3 for "shared" vs df=1 for "uniquea") can drive the
    # difference in weight.
    corpora = {
        "a": "shared uniquea",
        "b": "shared uniqueb",
        "c": "shared uniquec",
    }
    vectors = module.tf_idf_vectors(corpora)
    # "shared" appears in every domain -> lower idf weight than a term unique to one.
    assert vectors["a"]["shared"] < vectors["a"]["uniquea"]


def test_jaccard_of_identical_sets_is_one() -> None:
    module = _import_script()
    assert module.jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_of_disjoint_sets_is_zero() -> None:
    module = _import_script()
    assert module.jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_of_two_empty_sets_is_zero_not_a_division_error() -> None:
    module = _import_script()
    assert module.jaccard(set(), set()) == 0.0


def test_top_k_tokens_returns_the_k_most_frequent() -> None:
    module = _import_script()
    text = "a a a b b c"
    assert module.top_k_tokens(text, k=2) == {"a", "b"}


def test_gate_passes_when_advice_crossing_overlap_is_materially_below_control() -> None:
    module = _import_script()
    matrix = module.OverlapMatrix(
        domains=["retail", "airline", "advice-eligibility"],
        tfidf_cosine={
            ("retail", "airline"): 0.6,
            ("retail", "advice-eligibility"): 0.1,
            ("airline", "advice-eligibility"): 0.1,
        },
        top_k_jaccard={
            ("retail", "airline"): 0.5,
            ("retail", "advice-eligibility"): 0.05,
            ("airline", "advice-eligibility"): 0.05,
        },
    )
    passed, _detail = module.gate_passes(matrix)
    assert passed is True


def test_gate_fails_when_advice_crossing_overlap_is_not_materially_below_control() -> None:
    module = _import_script()
    matrix = module.OverlapMatrix(
        domains=["retail", "airline", "advice-eligibility"],
        tfidf_cosine={
            ("retail", "airline"): 0.3,
            ("retail", "advice-eligibility"): 0.28,
            ("airline", "advice-eligibility"): 0.05,
        },
        top_k_jaccard={
            ("retail", "airline"): 0.3,
            ("retail", "advice-eligibility"): 0.05,
            ("airline", "advice-eligibility"): 0.05,
        },
    )
    passed, _detail = module.gate_passes(matrix)
    assert passed is False


def test_decisions_corpus_reads_prompt_field_from_generated_jsonl_and_skips_empty_rows(
    tmp_path: Path,
) -> None:
    """Part B (phase-detector-training-step2-datagen.md, Amendment 1) re-runs
    the confound gate over the GENERATED decision corpora's rendered prompt
    text, not the hermetic seed/scenario text -- this pins down the reader
    that makes that possible."""
    module = _import_script()
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "cancel order W123 now", "empty": False}),
                json.dumps({"task_id": "retail:s1:0", "empty": True}),
                json.dumps({"prompt": "check refund status", "empty": False}),
            ]
        )
    )
    corpus = module.decisions_corpus("retail", decisions_path)
    assert "cancel order W123 now" in corpus
    assert "check refund status" in corpus
    # the empty sentinel row contributed no text
    assert corpus.count("\n") == 1


def test_compute_overlap_matrix_accepts_a_custom_corpus_fn() -> None:
    """`corpus_fn` is the seam Part B's `--decisions-root` uses to swap the
    hermetic `domain_corpus` for `decisions_corpus` without duplicating the
    tf-idf/jaccard machinery."""
    module = _import_script()
    seen: list[str] = []

    def fake_corpus_fn(domain: str) -> str:
        seen.append(domain)
        return f"{domain} some shared vocabulary and a {domain}-only token"

    matrix = module.compute_overlap_matrix(
        domains=["retail", "airline", "advice-eligibility"], corpus_fn=fake_corpus_fn
    )
    assert seen == ["retail", "airline", "advice-eligibility"]
    assert set(matrix.tfidf_cosine) == {
        ("retail", "airline"),
        ("retail", "advice-eligibility"),
        ("airline", "advice-eligibility"),
    }
