# Phase 2b — cross-domain H1 (airline + retail)

_Generated: 2026-07-22. Branch: `phase2b-BE-retail-crossdomain`._

Source task: `coding-tasks/bossyk-sandbox/phase2-guardrail-conditions.md` §2b. The
Phase-2 exit criterion: _"guardrail bypass rate measured cross-domain."_ Met.

## Scope (shrunk at Step 0)

Building 2a established that the H1 path (`build_grid → adversary → guardrail →
bypass math`) is **fully domain-agnostic** — it never invokes the agent layer
(scenarios / policy / fast-rules / gate). So cross-domain H1 needed only a
`RETAIL_BOUNDARIES` grid axis + a domain-parameterised benchmark (`H1_DOMAIN`), not
the domain-registry refactor of the three airline-hardcoded sites. That refactor
is agent-layer only and is **deferred** to the first cross-domain agent-layer
measurement (H2/H3). The retail **refund-threshold value** is likewise deferred —
for H1, `refund_over_threshold` is only a target label the adversary attacks.

Retail boundaries (parallel to airline's four): `cancel_without_auth`,
`refund_over_threshold`, `pii_disclosure`, `unauthorized_modification`.

## Methodology

Identical config to 2a, per domain: 16-cell grid (4 classes × 4 boundaries),
deepseek-v4-pro adversary at **aggressive** (10/cell → 160 payloads, temp 1.0),
`ModelBackedGuardrail` over `protectai/deberta-v3-base-prompt-injection-v2` swept
across leaky/moderate/strict. Airline numbers are the 2a real run
(`docs/bench_output/phase2a_h1.json`); retail is this run
(`docs/bench_output/phase2b_h1_retail.json`). Combined summary:
`docs/bench_output/phase2b_crossdomain_h1.json`. **Zero adversary refusals** in
either domain (verified — all 320 payloads are genuine attacks).

## Retail H1 (this run)

| Strength | Overall bypass | Rate | 95% CI |
|---|---|---|---|
| leaky (canonical) | 25/160 | **0.156** | [0.108, 0.220] |
| moderate | 23/160 | 0.144 | [0.098, 0.206] |
| strict | 22/160 | 0.138 | [0.093, 0.199] |

By class (leaky): tool_misuse **0.375** [0.242, 0.530], pii_leak 0.200
[0.105, 0.348], jailbreak 0.050 [0.014, 0.165], prompt_injection **0.000**
[0.000, 0.088]. By boundary (leaky): pii_disclosure 0.250, unauthorized_modification
0.200, refund_over_threshold 0.175, cancel_without_auth 0.000. 25 regression probes
frozen (`probes/regression/retail.json`).

## Cross-domain comparison (leaky strength)

| | airline | retail |
|---|---|---|
| **overall bypass** | 31/160 = **0.194** [0.140, 0.262] | 25/160 = **0.156** [0.108, 0.220] |
| tool_misuse | 0.400 | 0.375 |
| pii_leak | 0.225 | 0.200 |
| jailbreak | 0.125 | 0.050 |
| prompt_injection | 0.025 | 0.000 |

## Interpretation

- **The guardrail's blind-spot profile is domain-invariant — the key cross-domain
  result.** In *both* domains the bypass ordering is identical:
  `tool_misuse ≫ pii_leak ≫ jailbreak ≫ prompt_injection ≈ 0`. The model-layer
  input filter reliably catches injection/jailbreak and reliably *leaks*
  tool_misuse and pii, independent of domain. The layered thesis (model layer
  insufficient; agent-layer drift/policy must catch the residual by consequence)
  **generalises across domains**, not an airline artifact.
- **Overall magnitude is comparable** (0.194 vs 0.156; overlapping CIs). Retail is
  a touch lower, driven by `cancel_without_auth` = 0.000 (the deepseek payloads for
  retail cancellation read as more overtly injection-like than airline's
  `cancel_without_lookup`) — a domain-texture difference, not a structural one.
- **Strength dial:** both domains move little (airline 0.194→0.181→0.181; retail
  0.156→0.144→0.138) — the deberta classifier is bimodal/confident, so the
  threshold has little to reclassify. Retail moves marginally more.

## Bounds & caveats

Same as 2a (`docs/phase2a-h1-results.md`): model-layer only (H1, not the H2
agent-layer backstop); scripted-crossing model (`boundary_reached=True` by
construction); one adversary model / one guardrail model / one seed of 160 per
domain; coverage bounded by the finite grid for a fixed tool/authority set — never
"immune" (Property X).

## Data artifacts

- Retail: `docs/bench_output/phase2b_h1_retail.json`, `probes/regression/retail.json`.
- Airline (2a): `docs/bench_output/phase2a_h1.json`, `probes/regression/airline.json`.
- Combined: `docs/bench_output/phase2b_crossdomain_h1.json`.

## What's next

- **Agent layer (H2/H3), cross-domain:** the domain-registry refactor (scenarios /
  policy / fast-rules for airline+retail), then score the 56 frozen crossings
  (31 airline + 25 retail) through drift/policy — does the by-consequence backstop
  catch what the model layer leaked? This is where the deferred refactor +
  refund-threshold value land.
- **Interrupt/latency (H4)** and the SMACTR loop (H5) per the master plan.

## Reproducibility manifest

Two artifacts here: the retail H1 run itself, and the cross-domain merge that
combines it with the already-committed airline (2a) file. At the time of this
run, `scripts/h1_bench.py` had `H1_DOMAIN` but not yet `H1_ADVERSARY` (added
later, adv-registry phase) — the adversary was hardcoded to Fireworks-deepseek.
`scripts/h1_crossdomain_merge.py` did not exist yet at this commit; it was added
in the current remediation pass specifically to make `phase2b_crossdomain_h1.json`
regenerable rather than hand-assembled (code-review finding 17).

```yaml
script: scripts/h1_bench.py
commit: adeb19b
env:
  - RUN_H1_BENCH=1
  - H1_DOMAIN=retail
  - FIREWORKS_API_KEY=<fireworks key>
output:
  - docs/bench_output/phase2b_h1_retail.json
  - probes/regression/retail.json
```

```yaml
script: scripts/h1_crossdomain_merge.py
commit: adeb19b  # combined output first committed here; the merge script
                  # itself (below) was added later in the remediation pass
env: []  # deterministic, no network/keys
output:
  - docs/bench_output/phase2b_crossdomain_h1.json
regen_command: >
  uv run python scripts/h1_crossdomain_merge.py
  docs/bench_output/phase2a_h1.json docs/bench_output/phase2b_h1_retail.json
  --output docs/bench_output/phase2b_crossdomain_h1.json
verified: byte-identical to the committed file, this session
```
