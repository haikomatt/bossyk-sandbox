"""Hermetic tests for interp.logprob_metrics -- pure math, no I/O, no model."""

from __future__ import annotations

import math

import pytest

from bossyk_sandbox.interp.logprob_metrics import (
    StepUncertainty,
    TokenLogprob,
    margin,
    parse_openai_logprobs,
    summarize,
    surprisal,
    top_k_entropy,
)

LN2 = math.log(2.0)


def test_surprisal_is_negative_logprob() -> None:
    assert surprisal(TokenLogprob(logprob=-2.0)) == 2.0
    assert surprisal(TokenLogprob(logprob=0.0)) == 0.0  # certain token, zero surprise


def test_top_k_entropy_uniform_over_two_is_ln2() -> None:
    # p = [0.5, 0.5] -> entropy ln 2
    tok = TokenLogprob(logprob=math.log(0.5), top_logprobs=(math.log(0.5), math.log(0.5)))
    assert top_k_entropy(tok) == pytest.approx(LN2)


def test_top_k_entropy_renormalises_truncated_topk() -> None:
    # Visible mass sums to 0.6, not 1.0; renormalise to [0.5, 0.5] -> ln 2.
    tok = TokenLogprob(logprob=math.log(0.3), top_logprobs=(math.log(0.3), math.log(0.3)))
    assert top_k_entropy(tok) == pytest.approx(LN2)


def test_top_k_entropy_peaked_distribution_is_near_zero() -> None:
    tok = TokenLogprob(logprob=math.log(0.999), top_logprobs=(math.log(0.999), math.log(0.001)))
    assert top_k_entropy(tok) < 0.05


def test_top_k_entropy_no_alternatives_is_zero() -> None:
    assert top_k_entropy(TokenLogprob(logprob=-0.1)) == 0.0


def test_margin_is_top1_minus_top2_regardless_of_input_order() -> None:
    tok = TokenLogprob(logprob=math.log(0.7), top_logprobs=(math.log(0.3), math.log(0.7)))
    assert margin(tok) == pytest.approx(math.log(0.7) - math.log(0.3))


def test_margin_zero_when_fewer_than_two_alternatives() -> None:
    assert margin(TokenLogprob(logprob=-0.1, top_logprobs=(-0.1,))) == 0.0
    assert margin(TokenLogprob(logprob=-0.1)) == 0.0


def test_summarize_empty_turn_is_all_zero() -> None:
    assert summarize([]) == StepUncertainty(0, 0.0, 0.0, 0.0, 0.0)


def test_summarize_uses_localising_aggregations() -> None:
    # One calm token, one high-surprisal near-tie token: max_surprisal and
    # min_margin must reflect the spike, not the average.
    calm = TokenLogprob(logprob=math.log(0.95), top_logprobs=(math.log(0.95), math.log(0.05)))
    spike = TokenLogprob(logprob=math.log(0.10), top_logprobs=(math.log(0.10), math.log(0.09)))
    out = summarize([calm, spike])
    assert out.n_tokens == 2
    assert out.max_surprisal == pytest.approx(-math.log(0.10))
    assert out.max_surprisal > out.mean_surprisal
    # the near-tie spike token has the smaller margin
    assert out.min_margin == pytest.approx(math.log(0.10) - math.log(0.09))


def test_summarize_mean_surprisal_is_the_token_mean() -> None:
    toks = [TokenLogprob(logprob=-1.0), TokenLogprob(logprob=-3.0)]
    assert summarize(toks).mean_surprisal == pytest.approx(2.0)


def test_parse_openai_logprobs_reads_chosen_and_alternatives() -> None:
    content = [
        {
            "token": "yes",
            "logprob": -0.5,
            "top_logprobs": [
                {"token": "yes", "logprob": -0.5},
                {"token": "no", "logprob": -1.2},
            ],
        }
    ]
    toks = parse_openai_logprobs(content)
    assert len(toks) == 1
    assert toks[0].logprob == -0.5
    assert toks[0].top_logprobs == (-0.5, -1.2)


def test_parse_openai_logprobs_missing_top_logprobs_is_empty_tuple() -> None:
    toks = parse_openai_logprobs([{"token": "x", "logprob": -0.1}])
    assert toks[0].top_logprobs == ()


def test_parse_openai_logprobs_none_or_empty_content_is_empty_list() -> None:
    assert parse_openai_logprobs(None) == []
    assert parse_openai_logprobs([]) == []


def test_parse_then_summarize_pure_tool_call_turn_is_zero() -> None:
    # A tool-call turn with no scored tokens -> empty parse -> n_tokens == 0.
    assert summarize(parse_openai_logprobs(None)).n_tokens == 0


def test_parse_openai_logprobs_ignores_extra_provider_fields() -> None:
    # Real Fireworks/vLLM payloads carry extra keys per token (bytes, token_id,
    # sampling_logprob, text_offset, ...). The parser must read only logprob +
    # top_logprobs[].logprob and ignore the rest. Shape captured from a live
    # Fireworks kimi-k2p6 response (2026-07-28 probe).
    content = [
        {
            "token": "Yes",
            "bytes": [89, 101, 115],
            "logprob": -0.011,
            "token_id": 2742,
            "sampling_logprob": 0.0,
            "text_offset": 3,
            "top_logprobs": [
                {"token": "Yes", "bytes": [89, 101, 115], "logprob": -0.011, "token_id": 2742},
                {"token": " weather", "bytes": [], "logprob": -13.31, "token_id": 10666},
            ],
        }
    ]
    toks = parse_openai_logprobs(content)
    assert len(toks) == 1
    assert toks[0].logprob == -0.011
    assert toks[0].top_logprobs == (-0.011, -13.31)
