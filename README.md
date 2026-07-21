# bossyk-sandbox

A live, cross-domain agent governance demonstrator: observe → attest →
interrupt across model/agent/eval layers.

Parent plan: `sandbox-demonstrator-plan-v0.1` (Obsidian vault,
`Projects/bossyk`). This repo is the executable build, phase by phase.

## Status: Phase 0 — walking skeleton (done)

Phase 0 proves the live path end-to-end on the thinnest possible vertical
slice: one domain (airline), one instrument (a hardcoded policy rule), one
interrupt pair (allow/block).

**Proven:** a proposed agent action can be intercepted *before it executes*,
scored, allowed or blocked, and the whole session sealed into a signed,
independently-verifiable evidence pack.

See `coding-tasks/bossyk-sandbox/phase0-walking-skeleton.md` in the vault for
the full spec, and `SCOUT.md` in this repo for the Step-0 decisions and exact
reuse paths.

## Architecture

```
src/bossyk_sandbox/
  gate.py                 # interception GATE: hold -> score -> allow/block
  instruments/
    base.py               # Instrument interface (score a proposed action)
    hardcoded_rule.py      # Phase 0's single policy rule
  runtime/
    langgraph_agent.py    # live LangGraph airline agent + tool-node interception
    stub_agent.py          # deterministic tool-call sequence for the E2E test
  evidence/
    trace.py               # agent steps -> auditk-spec Trace/Step
    pack.py                 # build + sign EvidencePack (wraps auditk)
  console/
    app.py                 # minimal FastAPI + WS live log + manual block control
    index.html
tests/
  unit/                    # gate decision, trace mapping, pack sign/verify roundtrip
  e2e/
    test_walking_skeleton.py   # env-gated: RUN_SANDBOX_E2E=1
scripts/
  live_demo.py             # runnable exit demo: live agent, real Fireworks call
```

**The rule:** `cancel_reservation` is blocked unless an earlier step in the
same session called `get_reservation_details` for the same `reservation_id`.
This targets a real gap in tau2's airline domain — the tool itself takes no
confirmation argument, so the guardrail has to live in the GATE, not the
tool schema.

**Reused, not reimplemented** (see `SCOUT.md` for exact import paths):
- `auditk-spec` — Trace/Step/EvidencePack JSON Schemas (schema-only, referenced
  by path, not a Python package)
- `auditk` — Pydantic models, `EvidencePack` builder, Ed25519 signer/verifier
  (`scorer_key=None` keeps this judge-free and deterministic — Phase 0 has no
  LLM in the gating path)
- `tau2-bench` — the airline domain's tools, policy, and task set
- `bossyk`'s `PolicyAwareJudge` is **not** used in the Phase 0 gating path;
  the `Instrument` interface is shaped so it drops in at Phase 1

None of `auditk`, `auditk-spec`, or `bossyk` were modified — all three are
consumed as editable path dependencies / schema references.

## Setup

```bash
uv sync
cp .env.example .env   # fill in FIREWORKS_API_KEY for the live agent demo
```

`auditk` and `tau2` are resolved as editable path dependencies from sibling
directories (`../auditk`, `../tau2-bench`) — both repos must be checked out
alongside this one.

## Running things

```bash
# Lint + types + unit tests (deterministic, no network/API key needed)
uv run ruff format --check . && uv run ruff check .
uv run mypy --explicit-package-bases src/ tests/
uv run pytest tests/ -x --no-cov -q

# + the deterministic E2E test (stub agent through the real GATE/trace/pack path)
RUN_SANDBOX_E2E=1 uv run pytest tests/ -x --no-cov -q

# Minimal live console: streams steps + GATE verdicts, manual block/allow
# override on a held action (drives the stub agent's tool-call sequence)
uv run uvicorn bossyk_sandbox.console.app:app --reload
# open http://localhost:8000

# Live exit demo: real LangGraph airline agent against a seeded cancellation
# request via Fireworks, GATE interception on real tool calls, signed +
# offline-verified EvidencePack written to demo_output/. Needs
# FIREWORKS_API_KEY in .env and network access — not part of the test suite.
uv run python scripts/live_demo.py
```

### Live agent model

The live agent talks to Fireworks' OpenAI-compatible endpoint via
`ChatOpenAI(base_url=...)` — no separate SDK. Current default:
`accounts/fireworks/models/kimi-k2p6`, set via `$FIREWORKS_MODEL`.

Picked in three passes (see `SCOUT.md` decision #4 for the full story):
`firefunction-v2` was retired from Fireworks' catalog; `deepseek-v4-pro` was
rejected because this codebase's judge path also runs on a deepseek model
(same-family agent/judge evaluation is exactly what this project argues
against); `kimi-k2p6` is a different family with solid tool-calling support.

In live runs so far, the agent has correctly self-refused the planted
unauthorised cancellation per policy before ever proposing the tool call —
so the GATE's block path hasn't fired organically in a live run yet. The
block path is proven deterministically instead, by
`tests/e2e/test_walking_skeleton.py`'s scripted stub agent (this was a
known risk of live-LLM nondeterminism, called out up front in `SCOUT.md`
decision #3).

## Known issues

- **langgraph 1.2.9 bug**: `Command(resume=None)` raises `UnboundLocalError`
  in `langgraph.pregel._loop`'s resume-handling path (`resume_is_map` is
  only bound when `resume is not None`, but read unconditionally a few
  lines later). Always resume with an explicit value — `scripts/live_demo.py`
  resumes with the GATE's auto verdict string rather than `None`.
