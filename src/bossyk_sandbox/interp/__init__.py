"""Interpretability over the agent/eval chain.

Research & learning track (understand the full model->agent->eval flow); NOT a
bossyk product-moat feature. Scope + framing:
`coding-tasks/bossyk-sandbox/phase-interp-chain-agent-internals.md` in the vault.

`logprob_metrics` is the pure, dependency-free heart of the behavioral layer:
raw per-token logprobs (from the agent's OpenAI-compatible endpoint) -> a
per-step uncertainty summary that later correlates against the Gate's policy
verdict. No torch/nnsight here -- the white-box (activation) layer lives in
separate modules gated behind the `interp` extra.
"""
