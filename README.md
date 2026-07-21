# bossyk-sandbox

A live, cross-domain agent governance demonstrator: observe → attest →
interrupt across model/agent/eval layers.

Parent plan: `sandbox-demonstrator-plan-v0.1` (Obsidian vault,
`Projects/bossyk`). This repo is the executable build, phase by phase.

## Status: Phase 1 — multi-instrument agent layer (machinery green; benchmark run pending)

Phase 0 proved the live path end-to-end on the thinnest possible vertical
slice: one domain (airline), one instrument (a hardcoded policy rule), one
interrupt pair (allow/block).

Phase 1 puts the real instruments on that seam: **drift** (wraps auditk's
drift scorer), **policy** (wraps bossyk's `PolicyAwareJudge`), and
**outcome** (the A1 scenario key — ground truth, not a live scorer). The
fast-path rule still gates synchronously; drift + policy run concurrently
and annotate without blocking (the two-speed split). A 10-scenario scripted
airline set spans `cancel_reservation` + `update_reservation_flights` across
benign/policy-violation/deviation A1 keys, feeding a 3-way orthogonality
table (Wilson 95% CIs) and a gate-vs-ground-truth confusion read (B2
safety-weighted + B3 bind/no-bind).

**Proven deterministically:** per-step verdicts from all three instruments
land in the signed EvidencePack; the gate decision never waits on the slow
judges; the orthogonality + confusion tables compute correctly on fixtures
and over a stubbed-judge run of the full scenario set (41 tests, `ruff` +
`mypy --strict` clean).

**Not yet run:** the real-judge benchmark (`scripts/benchmark_run.py`) that
produces the actual orthogonality/confusion numbers — gated on
`FIREWORKS_API_KEY` + `RUN_SANDBOX_BENCH=1` (and `auditk[nli]`'s
torch/transformers, `uv sync --extra bench`), same shape as Phase 0's live
demo. Run it explicitly when ready; Phase 1's exit criteria aren't met until
it produces the tables.

See `coding-tasks/bossyk-sandbox/phase0-walking-skeleton.md` and
`phase1-multi-instrument.md` in the vault for the full specs, and
`SCOUT.md` in this repo for the Step-0 decisions and exact reuse paths
(both phases).

## Architecture

```
src/bossyk_sandbox/
  gate.py                 # Gate (fast allow/block) + TwoSpeedGate (Phase 1)
  instruments/
    base.py               # Instrument / SlowInstrument protocols, InstrumentVerdict
    hardcoded_rule.py      # fast-path rule (cancel + update_reservation_flights)
    drift.py               # Phase 1: wraps auditk's drift scorer
    policy.py               # Phase 1: wraps bossyk's PolicyAwareJudge
    outcome_key.py           # Phase 1: A1 scenario-key ground-truth lookup
  runtime/
    langgraph_agent.py    # live LangGraph airline agent + tool-node interception
    stub_agent.py          # deterministic tool-call sequence for the E2E test
  evidence/
    trace.py               # agent steps -> auditk-spec Trace/Step
    pack.py                 # build + sign EvidencePack (wraps auditk)
  scenarios/               # Phase 1: scripted airline scenario set + A1 keys
    loader.py
    runner.py
    airline/scenarios.json
  scoring/                 # Phase 1: 3-way orthogonality + confusion math
    orthogonality.py
    confusion.py
  console/
    app.py                 # minimal FastAPI + WS live log + manual block control
    index.html
tests/
  unit/                    # gate decision, trace mapping, pack sign/verify roundtrip,
                           # Phase 1: instruments, two-speed ordering, scoring math
  e2e/
    test_walking_skeleton.py       # env-gated: RUN_SANDBOX_E2E=1
    test_multi_instrument_e2e.py   # env-gated: RUN_SANDBOX_E2E=1 (stubbed judges)
scripts/
  live_demo.py             # runnable exit demo: live agent, real Fireworks call
  benchmark_run.py         # Phase 1: real judges over the scenario set (key-gated)
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

# Phase 1 benchmark run: real drift (gpt-oss-120b) + policy (deepseek-v4-pro)
# judges over the 10-scenario airline set -> 3-way orthogonality table +
# B2/B3 confusion read. Needs FIREWORKS_API_KEY, network access, and the
# `bench` extra (torch/transformers) — not part of the test suite.
uv sync --extra bench
RUN_SANDBOX_BENCH=1 RUN_JUDGE_MODEL=1 RUN_NLI_MODEL=1 \
    uv run python scripts/benchmark_run.py
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
