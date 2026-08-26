#!/usr/bin/env python
"""Pre-registered lexical-overlap confound gate (advice-eligibility-domain-spec.md
build order step 5; confound #1 of
a-fine-tuned-violation-detector-transfers-cross-domain.md): computes
cross-domain lexical overlap (tf-idf cosine + top-k token Jaccard) between
the retail, airline, and advice-eligibility scenario corpora.

Every text source is hermetic/offline -- scenario declared_intent + tool
call text, tool descriptions/vocab (from each domain's real toolkit),
deterministically-rendered decision contexts via
`bossyk_sandbox.interp.prompt_render.render_prompt`, and (for
advice-eligibility) the seeded PersonaStore fixture. No live/billable agent
call, no network, no RunPod/GPU work of any kind.

Gate (pre-registered): advice-eligibility's overlap with retail AND with
airline must be MATERIALLY BELOW the retail<->airline overlap (the built-in
similar-domain saturation control), on both metrics. If the gate fails, the
vocabulary must be re-scoped (rename tools/fields, resample personas) before
any downstream training work -- this script only reports the result, it
does not decide to proceed on a failure.

Usage:
    uv run python scripts/lexical_overlap_audit.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tau2.domains.airline.environment import get_environment as get_airline_environment
from tau2.domains.retail.environment import get_environment as get_retail_environment

from bossyk_sandbox.advice.environment import get_advice_environment
from bossyk_sandbox.advice.personas import PERSONAS
from bossyk_sandbox.domains import domain_config
from bossyk_sandbox.interp.prompt_render import render_prompt
from bossyk_sandbox.scenarios.loader import Scenario, load_scenarios

DOMAINS = ["retail", "airline", "advice-eligibility"]

# Each getter is a local, deterministic environment load (tau2's bundled
# fixture JSON for retail/airline; the seeded PersonaStore for
# advice-eligibility) -- no network, no API key, matching the same
# `Callable[[], Any]` shape `runtime.langgraph_agent._build_agent_session`
# already relies on for these domains.
_ENVIRONMENT_GETTERS = {
    "retail": get_retail_environment,
    "airline": get_airline_environment,
    "advice-eligibility": get_advice_environment,
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

OUTPUT_DIR = Path(__file__).parent.parent / "probes" / "detector" / "results"


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _RenderableMessage:
    """Minimal duck-typed stand-in for a LangChain message -- carries only
    what `interp.prompt_render.render_prompt` reads (`.type`, `.content`,
    optional `.tool_calls`), so this audit can reuse the exact renderer the
    live agent's `capture_prompts` path uses, without importing
    langchain_core here."""

    type: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _scenario_text(scenarios: list[Scenario]) -> str:
    """Raw scenario text: each step's tool call plus its declared_intent."""
    parts = [
        f"{step.proposed.tool_name} {step.proposed.arguments} {step.proposed.declared_intent or ''}"
        for scenario in scenarios
        for step in scenario.steps
    ]
    return "\n".join(parts)


def _tool_vocab_text(domain: str) -> str:
    """Every tool name + description the domain's real toolkit exposes."""
    env = _ENVIRONMENT_GETTERS[domain]()
    tools = env.tools.get_tools()
    parts = [
        f"{name}: {tool.openai_schema['function']['description']}" for name, tool in tools.items()
    ]
    return "\n".join(parts)


def _rendered_context_text(scenarios: list[Scenario]) -> str:
    """Deterministically-rendered decision contexts via
    `interp.prompt_render.render_prompt` -- the T4 pre-action context shape
    (human turn + the agent's proposed tool call). Vocabulary-only: no
    policy/system text, so this source doesn't depend on the sibling
    bossyk policy-YAML checkout being present."""
    parts = []
    for scenario in scenarios:
        for step in scenario.steps:
            messages = [
                _RenderableMessage(type="human", content=step.proposed.declared_intent or ""),
                _RenderableMessage(
                    type="ai",
                    content="",
                    tool_calls=[{"name": step.proposed.tool_name, "args": step.proposed.arguments}],
                ),
            ]
            parts.append(render_prompt(messages))
    return "\n".join(parts)


def _persona_text(domain: str) -> str:
    """Persona text is only meaningfully domain-specific for
    advice-eligibility (the fresh, seeded `PersonaStore` fixture);
    retail/airline draw on tau2's own bundled fixtures, out of scope here."""
    if domain != "advice-eligibility":
        return ""
    return "\n".join(f"{persona.full_name} {persona.postcode}" for persona in PERSONAS)


def domain_corpus(domain: str) -> str:
    """The full hermetic text corpus for one domain: scenario text + tool
    vocabulary + rendered decision contexts + (where applicable) persona
    text. Every source is local/offline -- no live agent call."""
    cfg = domain_config(domain)
    scenarios = load_scenarios(cfg.scenarios_path)
    return "\n".join(
        [
            _scenario_text(scenarios),
            _tool_vocab_text(domain),
            _rendered_context_text(scenarios),
            _persona_text(domain),
        ]
    )


def decisions_corpus(domain: str, decisions_path: Path) -> str:
    """Text corpus built from a GENERATED `decisions.jsonl`'s `prompt` field
    (the rendered decision context each row was captured from) -- the
    Part-B confound-gate RE-RUN required by Amendment 1
    (phase-detector-training-step2-datagen.md): the pre-training gate must
    cover the vocabulary the live generator actually produced, not just the
    hermetic seed/scenario text `domain_corpus` builds from. `domain` is
    accepted (unused) so this has the same `Callable[[str], str]` shape as
    `domain_corpus` and can be swapped in via `compute_overlap_matrix`'s
    `corpus_fn`. Empty-task sentinel rows (`empty: true`) contribute no text."""
    del domain  # shape-compatibility with domain_corpus, not read here
    parts = []
    for line in decisions_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("empty"):
            continue
        prompt = row.get("prompt")
        if prompt:
            parts.append(str(prompt))
    return "\n".join(parts)


def tf_idf_vectors(corpora: dict[str, str]) -> dict[str, dict[str, float]]:
    """Smoothed TF-IDF over the domain-level "documents" in `corpora` (one
    document per domain -- this is a cross-CORPUS overlap audit, not a
    per-sentence one). Smoothing (`+1` in both numerator and denominator,
    `+1` on the result) mirrors scikit-learn's default `smooth_idf=True` so
    no term's weight is undefined or negative even with as few as 3
    "documents"."""
    doc_tokens = {domain: tokenize(text) for domain, text in corpora.items()}
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens.values():
        for term in set(tokens):
            doc_freq[term] += 1
    n_docs = len(corpora)

    vectors: dict[str, dict[str, float]] = {}
    for domain, tokens in doc_tokens.items():
        term_freq = Counter(tokens)
        idf = {term: math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0 for term in term_freq}
        vectors[domain] = {term: count * idf[term] for term, count in term_freq.items()}
    return vectors


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    shared = set(a) & set(b)
    dot = sum(a[term] * b[term] for term in shared)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def top_k_tokens(text: str, k: int = 30) -> set[str]:
    counts = Counter(tokenize(text))
    return {term for term, _count in counts.most_common(k)}


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@dataclass
class OverlapMatrix:
    domains: list[str]
    tfidf_cosine: dict[tuple[str, str], float]
    top_k_jaccard: dict[tuple[str, str], float]


def compute_overlap_matrix(
    domains: list[str] | None = None,
    k: int = 30,
    corpus_fn: Callable[[str], str] = domain_corpus,
) -> OverlapMatrix:
    resolved_domains = domains if domains is not None else DOMAINS
    corpora = {domain: corpus_fn(domain) for domain in resolved_domains}
    vectors = tf_idf_vectors(corpora)
    top_tokens = {domain: top_k_tokens(corpora[domain], k=k) for domain in resolved_domains}

    tfidf_cosine: dict[tuple[str, str], float] = {}
    top_k_jaccard: dict[tuple[str, str], float] = {}
    for i, domain_a in enumerate(resolved_domains):
        for domain_b in resolved_domains[i + 1 :]:
            pair = (domain_a, domain_b)
            tfidf_cosine[pair] = cosine_similarity(vectors[domain_a], vectors[domain_b])
            top_k_jaccard[pair] = jaccard(top_tokens[domain_a], top_tokens[domain_b])
    return OverlapMatrix(
        domains=resolved_domains, tfidf_cosine=tfidf_cosine, top_k_jaccard=top_k_jaccard
    )


# "Materially below" is not itself given a specific number by the
# pre-registration (the dossier pins a numeric threshold for the downstream
# detector-vs-baseline AUROC gap, 0.05, not for this confound gate) -- this
# is the operationalisation chosen for this gate: the SAME 0.05 margin,
# applied to both overlap metrics, so a borderline pass/fail isn't a coin
# flip of measurement noise. The full matrix is always reported regardless
# of the verdict, so a human can judge "materially" independently.
GATE_MARGIN = 0.05


def gate_passes(matrix: OverlapMatrix, margin: float = GATE_MARGIN) -> tuple[bool, str]:
    """Pre-registered gate: on BOTH metrics, the worse (max) of
    advice-eligibility's overlap with retail and with airline must sit at
    least `margin` below the retail<->airline control overlap."""
    control_cosine = matrix.tfidf_cosine[("retail", "airline")]
    control_jaccard = matrix.top_k_jaccard[("retail", "airline")]
    advice_cosine = max(
        matrix.tfidf_cosine[("retail", "advice-eligibility")],
        matrix.tfidf_cosine[("airline", "advice-eligibility")],
    )
    advice_jaccard = max(
        matrix.top_k_jaccard[("retail", "advice-eligibility")],
        matrix.top_k_jaccard[("airline", "advice-eligibility")],
    )
    cosine_ok = advice_cosine <= control_cosine - margin
    jaccard_ok = advice_jaccard <= control_jaccard - margin
    passed = cosine_ok and jaccard_ok
    detail = (
        f"control retail<->airline: cosine={control_cosine:.4f} jaccard={control_jaccard:.4f}; "
        f"advice-crossing (max of retail/airline): cosine={advice_cosine:.4f} "
        f"jaccard={advice_jaccard:.4f}; margin={margin} "
        f"(cosine {'OK' if cosine_ok else 'FAIL'}, jaccard {'OK' if jaccard_ok else 'FAIL'})"
    )
    return passed, detail


def _matrix_to_json(matrix: OverlapMatrix, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "domains": matrix.domains,
        "gate_margin": GATE_MARGIN,
        "gate_passed": passed,
        "gate_detail": detail,
        "tfidf_cosine": {f"{a}<->{b}": v for (a, b), v in matrix.tfidf_cosine.items()},
        "top_k_jaccard": {f"{a}<->{b}": v for (a, b), v in matrix.top_k_jaccard.items()},
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-registered lexical-overlap confound gate. Default (no args): "
            "hermetic seed/scenario text (advice-eligibility-domain-spec.md step 5). "
            "Pass --decisions-root for Part B's Amendment-1 RE-RUN over GENERATED "
            "decision corpora (probes/detector/data/<domain>/decisions.jsonl)."
        )
    )
    parser.add_argument(
        "--decisions-root",
        type=Path,
        default=None,
        help=(
            "root dir containing <domain>/decisions.jsonl for each domain; when set, "
            "the gate runs over the GENERATED decisions' rendered prompt text instead "
            "of the hermetic seed/scenario corpus"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.decisions_root is not None:
        root: Path = args.decisions_root

        def corpus_fn(domain: str) -> str:
            return decisions_corpus(domain, root / domain / "decisions.jsonl")

        source_label = f"GENERATED decisions ({root})"
        out_name = "lexical_overlap_matrix_generated.json"
    else:
        corpus_fn = domain_corpus
        source_label = "hermetic seed/scenario text"
        out_name = "lexical_overlap_matrix.json"

    matrix = compute_overlap_matrix(corpus_fn=corpus_fn)
    passed, detail = gate_passes(matrix)

    print(f"Cross-domain lexical overlap matrix (source: {source_label}):")
    print(f"{'pair':<30}{'tfidf_cosine':>14}{'top_k_jaccard':>16}")
    for i, domain_a in enumerate(matrix.domains):
        for domain_b in matrix.domains[i + 1 :]:
            pair = (domain_a, domain_b)
            label = f"{domain_a} <-> {domain_b}"
            print(
                f"{label:<30}{matrix.tfidf_cosine[pair]:>14.4f}{matrix.top_k_jaccard[pair]:>16.4f}"
            )
    print()
    print(detail)
    print(f"GATE {'PASSED' if passed else 'FAILED'}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / out_name
    out_path.write_text(json.dumps(_matrix_to_json(matrix, passed, detail), indent=2))
    print(f"wrote {out_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
