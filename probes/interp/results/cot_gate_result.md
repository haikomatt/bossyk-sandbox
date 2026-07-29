# CoT lead-time calibration GATE — result (STOP for Qwen2.5-7B-Instruct)

Runbook: `coding-tasks/bossyk-sandbox/h1cot-reasoning-leadtime-runbook.md`, step 3.
Date: 2026-07-29. Served Qwen2.5-7B-Instruct (vllm/vllm-openai, A100 80GB PCIe),
`borderline_cot` weakened retail policy, 26 borderline payment/address prompts ×
16 samples, temperature 1.0. Report JSON: `cot_gate_report.json`.

## Verdict: STOP — do not capture (on this model)

The gate has two conditions; exactly one passed.

| Condition | Threshold | Observed | Pass |
|---|---|---|---|
| Reasoning window before tool call | ≥10 tok on ≥50% of action rollouts | **7 / 326 (2%)** | ✗ |
| Both tool-first classes present | ≥15 of the minority class | mutation-first **123**, lookup-first **203** | ✓ |

416 rollouts total: 326 action (a tool call), 90 text-only.

## What it means

Qwen2.5-7B-Instruct **acts (near-)immediately even when explicitly told to reason
first**. 319 / 326 action rollouts emitted the `<tool_call>` as the very first
generated token (0 pre-emission tokens) — the model honours the chat-template
tool-calling format over the `<reasoning>…</reasoning>` instruction in the policy.
The 7 exceptions were *all* lookup-first, and their "reasoning" is answer preamble
("Sure, I can help with that. Let's proceed step by step: 1. Verify…"), not a
deliberation that could resolve either way — so there is no decide-then-act window
on either class. This is the SAME immediate-action property that made the original
H1 lead-time run a clean negative (PR #12), now confirmed to persist under an
explicit CoT elicitation.

**A backward-anchored capture on this model would be pointless** (offset −1 already
equals the deterministic context state, as in H1), so no capture spend was made.

## The one genuinely new positive

The `borderline_cot` policy produced a real **mutation-first vs lookup-first mix**
(123 / 203) — the immediate-action H1 run gave almost only mutation-first
(compliance was text-asking, no tool). So the *class-balance* half of the CoT
experiment is solved; the missing ingredient is purely the reasoning window. If a
model that actually deliberates is used, this experiment has a clear shot.

## Decision → fork to a reasoning model (pre-authorised)

Per the runbook's contingency: fork to a reasoning model (QwQ-32B or a Qwen3
thinking variant) that emits a genuine reasoning trace before acting, re-run this
gate, and only then capture. That leaves the proven torch-2.4 / nnsight-0.3.7
capture stack, so the nnsight generation idiom + VRAM must be re-verified at 32B
before the capture step.

## Fork result — QwQ-32B (reasoning model): STOP, and it reveals the real tension

Per the decision above, forked to **Qwen/QwQ-32B** (a reasoning model; Qwen2 arch,
so capture-stack-compatible), served on A100 80GB, same `borderline_cot` policy,
26 prompts × 6 samples, temp 1.0, `max_tokens` 2048. Report:
`cot_gate_qwq_report.json`.

| Condition | Threshold | Observed | Pass |
|---|---|---|---|
| Reasoning window | ≥10 tok on ≥50% of action rollouts | **4/4 = 100%** | ✓ |
| Both tool-first classes | ≥15 of the minority class | mutation-first **0**, lookup-first **4** | ✗ |

156 rollouts: **only 4 action** (all lookup-first), **152 text-only**.

QwQ has a huge reasoning window (that half is solved) — but it **reasons its way
to compliance**. Every rollout, action and text-only, opens with "First, I need to
authenticate them"; 152/156 conclude by *asking the user to verify* (no tool), 4
*look up first*, and **zero commit the gated mutation**. Given the exact
guardrail-weakened policy that makes Qwen2.5-7B-Instruct violate ~half the time,
the reasoner recovers the policy's verify-before-act requirement through
deliberation and does not violate.

## The tension (the real result of the lead-time track)

The lead-time premise needs a model that BOTH deliberates AND commits violations.
The two runs bound the problem from both sides:

- **Instruct model (Qwen2.5-7B):** violates readily, but acts immediately → **no
  reasoning window** (0 tokens before the tool call).
- **Reasoning model (QwQ-32B):** large reasoning window, but deliberation surfaces
  the policy → **no violations** (0 mutation-first).

So a spontaneous, decodable-before-the-text intent-enactment *violation* is hard to
elicit: the very reasoning that would create the probe window also drives the model
to compliance. Forcing the action (an `aggressive_cot` mandate) would manufacture
violations but destroy the *genuine decision* the probe needs — the residual would
encode a dictated outcome, not spontaneous drift. That is a materially weaker
experiment, flagged here rather than run silently.

## Cost

Two A100 80GB gate pods (Qwen2.5-7B + QwQ-32B), each deleted immediately after its
verdict; 0 pods verified at close. ≈ $0.9 total. No capture spend (both gates STOP).
