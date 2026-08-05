"""End-to-end (but still hermetic -- local scenario JSON + local tau2/advice
environments, no network) test of scripts/lexical_overlap_audit.py's real
domain corpora: confirms the pre-registered confound gate (advice-
eligibility-domain-spec.md build order step 5) actually passes for the
authored retail/airline/advice-eligibility scenario sets, and reports the
overlap matrix so a run of this test doubles as a sanity check on the gate
script itself.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "lexical_overlap_audit.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lexical_overlap_audit_script_gate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # See test_lexical_overlap_audit.py's _import_script for why this is needed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_domain_corpus_is_built_from_hermetic_text_only() -> None:
    module = _import_script()
    corpus = module.domain_corpus("advice-eligibility")
    assert isinstance(corpus, str) and corpus.strip()
    assert "verify_eligibility" in corpus or "submit_eligibility_decision" in corpus


def test_compute_overlap_matrix_covers_all_three_domain_pairs() -> None:
    module = _import_script()
    matrix = module.compute_overlap_matrix()

    assert set(matrix.tfidf_cosine) == {
        ("retail", "airline"),
        ("retail", "advice-eligibility"),
        ("airline", "advice-eligibility"),
    }
    assert set(matrix.top_k_jaccard) == set(matrix.tfidf_cosine)
    for value in {**matrix.tfidf_cosine, **matrix.top_k_jaccard}.values():
        assert 0.0 <= value <= 1.0 + 1e-9


def test_advice_eligibility_overlap_gate_passes_for_the_authored_scenario_sets() -> None:
    """THE pre-registered confound gate. If this fails, the advice-eligibility
    vocabulary must be re-scoped before any downstream training work -- per
    the spec, that is a STOP-and-report condition, not something to route
    around in code."""
    module = _import_script()
    matrix = module.compute_overlap_matrix()

    passed, detail = module.gate_passes(matrix)

    assert passed, f"lexical-overlap confound gate FAILED: {detail}"
