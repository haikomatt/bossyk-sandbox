# The gate proxy, the audit, and the scorecard — how the pieces fit

Two tools, two moments, one story:

- **bossyk gate** decides **before execution**: it sits between your agent
  harness and your inference endpoint and blocks policy-violating tool
  calls before the harness ever sees them.
- **auditk** verifies **after the fact**: it reads the session file the
  harness wrote and scores what actually happened.
- The **scorecard** is one self-contained HTML per run that merges both,
  verifies every signature offline, and cross-checks that the two layers
  tell the same story.

## Where each piece runs

```mermaid
flowchart LR
    subgraph your_infra [your infrastructure]
        pi["pi coding agent\n(unchanged — one baseUrl edit\nin models.json)"]
        gate["bossyk gate proxy\nTCP host:port or Unix socket"]
        llm["your inference\n(llama.cpp / vLLM / any\nOpenAI-compatible endpoint)"]
        pol[/"policy-pack.yaml\n5 deterministic policies"/]
        ev[/"gate-events.jsonl\nEd25519-signed per decision"/]
        sess[/"session .jsonl\n(pi's own record)"/]
    end
    pi -- "chat/completions" --> gate
    gate -- "always non-streamed" --> llm
    llm --> gate
    gate -- "ALLOW: forwarded verbatim\nBLOCK: policy refusal instead" --> pi
    pol -.-> gate
    gate ==> ev
    pi ==> sess
```

The gate buffers each model response, evaluates every proposed tool call
against the policy pack, then either forwards the response untouched or
replaces it with a refusal that names the violated policies. **The refusal
goes back through pi, so pi records it in its own session file** — that is
deliberate, and it is what makes the next diagram possible.

## How the evidence fits together

```mermaid
flowchart LR
    sess[/"session .jsonl"/] --> ingest["auditk ingest\n(pi adapter)"]
    ingest --> trace[/"trace.json"/]
    trace --> report["auditk report"] --> md[/"report.md / .html"/]
    trace --> attest["auditk attest\n(drift scoring)"] --> pack[/"evidence-pack.json\nEd25519-signed"/]
    ev[/"gate-events.jsonl\nEd25519-signed"/] --> card
    trace --> card["scorecard\n(one HTML per run)"]
    pack --> card
    md --> card
    card -- "verifies BOTH signatures offline\n(public keys only)" --> v{{"verified?"}}
    card -- "every gate BLOCK must appear\nas a refusal in the trace" --> c{{"corroborated?"}}
```

The corroboration check is the load-bearing property: the gate's log
*claims* it blocked something; the session file — written by pi, not by
the gate — must independently show the refusal. Two records, produced by
two different components, that have to agree. A gate log that claims
blocks the trace doesn't show is surfaced as **NOT corroborated**, in
red, on the scorecard.

## Which layer owns which risk

| Trap | What it is | Policy | Layer |
|------|------------|--------|-------|
| T2 | editing protected test files | `protected-test-paths` | gate, pre-execution |
| T3 | network egress (`curl`, `wget`, `ssh`…) | `no-network-egress` | gate, pre-execution |
| T4 | writing outside the workspace | `workspace-write-boundary` | gate, pre-execution |
| T5 | reading credentials (`.env`, keys) | `no-credential-reads` | gate, pre-execution |
| T6 | installing non-allowlisted packages | `package-allowlist` | gate, pre-execution |
| T1 | goals smuggled in via data | — | audit, post-hoc |
| T7 | feature creep beyond the task | — | audit, post-hoc |

The gate enforces only what a deterministic predicate can decide before
execution (the enforceable class); everything needing judgement stays
audit-side. That split is a design position, stated on every scorecard,
not a limitation discovered later.

## Telemetry: the same decisions in your Grafana, Loki and Tempo

The signed `gate-events.jsonl` is the system of record. Optionally the
gate also **projects** every decision into an existing observability
stack, alongside the signed log and never instead of it:

- **OTLP/HTTP (JSON)**: one `gate.decision` span and one linked log
  record per decision, attributes `bossyk.verdict`, `bossyk.policy_id`,
  `bossyk.tool_name`, `bossyk.kind`, `bossyk.pack_sha256`,
  `bossyk.run_label`, `bossyk.gate_version` (plus `bossyk.resolution` /
  `bossyk.resolved_verdict` for held decisions). Tool arguments are never
  exported: telemetry carries the decision, the signed log carries the
  evidence.
- **Prometheus**: `bossyk_gate_decisions_total{verdict,policy_id}`,
  `bossyk_gate_blocks_total{policy_id}`, `bossyk_gate_holds_total{resolution}`,
  served on `GET /gate/metrics` for scraping and, if configured, pushed
  to a push gateway grouped by `job="bossyk-gate"` and `run_label`.

```bash
python -m bossyk_sandbox.gateproxy ... \
  --otlp-endpoint http://otel-collector:4318 \
  --pushgateway-url http://pushgateway:9091
```

Both are **off by default** (air-gap friendly); `/gate/metrics` is always
served since it needs no egress. An exporter that is unreachable or slow
is logged and ignored: a telemetry outage never becomes a gate outage,
and the signed log is written first either way.

The wire formats were validated against a real `otel/opentelemetry-collector`
(0.160.0) and `prom/pushgateway` (1.11.3) rather than assumed; the request
bodies and what the receivers decoded are kept under `tests/fixtures/otel/`,
and `tests/e2e/test_gate_telemetry_e2e.py` (gated on
`RUN_GATE_TELEMETRY_E2E=1`) replays the check against live receivers.
Your own collector config was not available, so this emits standard OTLP
and says so; a receiver that wants different attribute names gets a
collector-side transform, not a gate change.
## HOLD: pause for a decision instead of ending the turn

ALLOW and BLOCK are the two verdicts the gate reaches on its own. A policy
line can instead ask for a **HOLD**: the proposed action is paused and
handed out for a decision, then resumed on approve or refused on deny.
Precedence inside one response is BLOCK over HOLD over ALLOW, and as with
a block, one refused call refuses the whole response.

```yaml
policies:
  - id: destructive-shell
    type: network-egress        # any predicate type can be held
    on_match: hold              # default: block
    on_hold: block              # what a hold nobody answers resolves to; default: block
    tools: [bash]
    commands: [rm]
```

A `pi --print` batch run has no human at a console, so the decision comes
from an **approver hook** configured on the gate, and a hold falls back to
the policy's own `on_hold` default when no approver is configured or the
approver does not answer in time:

```bash
python -m bossyk_sandbox.gateproxy ... \
  --approver-cmd "./approve.sh"          # held action as JSON on stdin; exit 0 approves
# or
  --approver-url http://approvals.internal/hold   # JSON POST; reply {"decision": "allow"|"block"}
  --approver-timeout 30                  # seconds; then on_hold applies
```

The approver sees only what the signed event already records about the
action (tool name, arguments, policy id, reason, run label). Every held
decision is logged with `verdict: hold`, a `resolution`
(`held_then_allowed`, `held_then_blocked`, or `hold_timed_out`) and the
`resolved_verdict` it took effect with; the scorecard counts and shows
it, and the incident report treats a held-then-blocked decision exactly
like a block, corroboration check included. Interactive console approval
is a later increment; nothing here claims a human always answers.

This is the mechanism behind an AI Act Art. 14
(human oversight) line: the gate can stop and ask.

## Latency, honestly

The gate adds single-digit milliseconds per response (JSON parse,
microsecond predicates, one Ed25519 signature) against responses that
take tens of seconds to generate. The real cost is the **buffer-then-
decide** design: nothing is forwarded until the model finishes, because
the gate must see complete tool calls before ruling. At 10–20 tok/s
batch-style use this is invisible; interactive streaming gating is
round-2 scope.

## Run it

```bash
# 1. the gate, in front of your endpoint
python -m bossyk_sandbox.gateproxy \
  --upstream http://127.0.0.1:8080/v1 \
  --policy-pack policy-pack.yaml \
  --workspace /path/to/task/workspace \
  --key gate-key.ed25519 --events gate-events.jsonl \
  --run-label leg-b-qwen --listen 127.0.0.1:8200
# (or --uds /run/bossyk/gate.sock instead of --listen)

# 2. point pi at it: models.json baseUrl -> http://127.0.0.1:8200/v1

# 3. after the run: audit the session, then render the scorecard
python -m bossyk_sandbox.gateproxy.scorecard \
  --run-label leg-b-qwen --task-name "logsum leg B" \
  --trace trace.json --evidence-pack evidence-pack.json \
  --gate-events gate-events.jsonl --report-md report.md \
  --pack-public-key key.pub --gate-public-key gate-key.pub \
  --out scorecard.html
```
