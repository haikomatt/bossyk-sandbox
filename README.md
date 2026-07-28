# bossyk-sandbox

A live, cross-domain agent governance demonstrator: observe → attest →
interrupt across model/agent/eval layers.

Parent plan: `sandbox-demonstrator-plan-v0.1` (Obsidian vault,
`Projects/bossyk`). This repo is the executable build, phase by phase.

## Status

Phases 0–5 complete, cross-domain (airline + retail), plus a 2026-07
remediation pass on top of the phase-5 code review: the live exit path is
now exactly-once (no duplicate tool calls on resume), measurement claims
are corrected to match what was actually run (see the matrix below), the
fast gate fails closed on malformed identifiers (a missing/empty id can
no longer satisfy a lookup prerequisite), judge outages are reported as
unavailable rather than counted as non-detections, and the console is
session-scoped (concurrent runs no longer clobber a shared global).

### The H1–H5 matrix

| # | Layer | Result | Verdict |
|---|---|---|---|
| **H1** | Model | guardrail bypass **0.194** airline / **0.156** retail; blind-spot profile domain-invariant (tool_misuse ≫ pii_leak ≫ jailbreak ≫ prompt_injection) | **not falsified** (live measurement) |
| **H2** | Agent | agent-layer catch rate on the *actual* 56 H1 crossings — **not yet measured**; a separate hand-authored scripted set shows 12/12 caught by the 3-instrument union | **open** — scripted proxy only |
| **H3** | Agent | drift / policy / gate fire on substantially non-overlapping event sets, on the scripted scenario set | **not falsified** (scripted set) |
| **H4** | Agent | modeled counterfactual: 5/12 violations prevented pre-execution, 7/12 detected only post-hoc | **not falsified as a modeled counterfactual** — no live execution observed, live evidence pending |
| **H5** | Eval / ceiling | adaptive adversary (fable) vs the layer | **pending** — needs `ANTHROPIC_API_KEY` |

Plus the **SMACTR loop** (Phase 4): a detected-too-late failure was fed back
as a fast-path rule; retail prevention rose 5/12 → 6/12, replay-tested
against a real regression probe. Full write-up and caveats:
`docs/phase5-h1-h5-synthesis.md`.

Plus a **demo console** at `/app`: an evidence browser (every number links to a
committed artifact) and a **control room** that replays a governance session
turn-by-turn with resolution modes and a hard-cell HITL queue — see **Console &
demo** below.

## Architecture

```
src/bossyk_sandbox/
  gate.py                    # Gate (fast allow/block) + TwoSpeedGate: fast
                              #   path decides sync; slow instruments annotate
                              #   concurrently, never gate
  domains.py                 # DomainConfig registry (airline/retail)
  env.py                     # .env loading, BOSSYK_ROOT resolution
  instruments/
    base.py                 # Instrument / SlowInstrument protocols, verdicts
    hardcoded_rule.py        # fast structural rule: RequireLookupBeforeCancel
    drift.py                 # wraps auditk's drift scorer (NLI gate + judge)
    policy.py                # wraps bossyk's PolicyAwareJudge
    outcome_key.py           # A1 scenario-key ground truth (not a live scorer)
  runtime/
    langgraph_agent.py       # live LangGraph airline agent; interrupt chain
                              #   is plan_calls -> prepare -> approve -> execute
    stub_agent.py             # deterministic tool-call sequence for E2E tests
  evidence/
    trace.py                 # steps -> auditk-spec Trace/Step; make_attested_step
                              #   records the final verdict + override provenance
    pack.py                  # build + sign EvidencePack (wraps auditk)
  scenarios/                 # scripted airline + retail scenario sets, A1 keys
    loader.py
    runner.py
    airline/scenarios.json
    retail/scenarios.json
  scoring/
    orthogonality.py         # 3-way orthogonality, tri-state drift/policy firing
    confusion.py             # gate-vs-ground-truth bind/no-bind confusion
    h1.py                    # guardrail bypass rate, refusal/error accounting
    interrupt.py             # H4: prevented / detected-too-late / undetected
    cost.py                  # per-model adversary token ledger
    crossdomain_merge.py     # per-domain H1 JSON -> cross-domain rollup
  conditions/
    grid.py                  # attack-class x consequence-boundary probe grid
    adversary.py             # Adversary protocol, ProbeAttempt, ChatResult
    adversary_registry.py    # multi-provider adversary registry
    adversary_providers.py   # OpenAI-compatible + Anthropic chat clients
    fireworks_adversary.py   # Fireworks/deepseek adversary implementation
    harness.py               # runs the probe grid, freezes crossings
    retention.py             # crossing -> frozen auditk ProbeDefinition
  guardrail/
    guardrail.py             # Guardrail protocol, strength dial (off/leaky/
                              #   moderate/strict)
    model_backed.py          # real deberta injection/jailbreak classifier
  governance/
    smactr.py                # Sense/Analyse/Control/Test/Respond loop: FMEA
                              #   severity, derived constraints, threat-model
  console/
    app.py                   # FastAPI + WS; session-scoped stub session +
                              #   non-interactive preset replay (POST /session/replay)
    replay.py                # committed preset replays: loader + pure drive_replay;
                              #   resolution modes + hard-cell HITL queue
    artifacts.py             # /api story/artifacts/frameworks/manifests for the SPA
    index.html
frontend/                    # Vite + React SPA (evidence browser + control room);
                              #   mounted at /app when frontend/dist exists (not committed)
  src/views/                 # Story/Evidence/Figures/Docs + ControlRoomView (replay)
story/
  story.yaml                 # narrated, evidence-graded, lint-checked claim set
  replay/                    # committed demo replay presets (retail-weak-dir1, -modes)
tests/
  unit/                      # one test module per component above
  e2e/
    test_walking_skeleton.py       # env-gated: RUN_SANDBOX_E2E=1
    test_multi_instrument_e2e.py   # env-gated: RUN_SANDBOX_E2E=1
    test_probe_grid_e2e.py         # offline (ungated)
    test_live_h2h4_e2e.py          # offline (ungated)
    test_replay_console_e2e.py     # offline (ungated): preset replay + modes/HITL
scripts/
  live_demo.py               # live LangGraph airline demo (billable)
  benchmark_run.py           # real drift+policy judges over scenarios (billable)
  h1_bench.py                # guardrail-bypass benchmark, per adversary (billable
                              #   in real mode; smoke mode needs no keys)
  h1_crossdomain_merge.py    # deterministic: merge per-domain H1 -> cross-domain
  h4_report.py               # deterministic: recompute H4 from a benchmark JSON
  smactr_demo.py             # deterministic: the SMACTR before/after closure
```

**The fast rule:** `RequireLookupBeforeCancel` blocks a write tool
(`cancel_reservation` / `update_reservation_flights` in airline;
`cancel_pending_order` / `modify_user_address` / others in retail) unless
an earlier step in the same session looked up the same id first. This
targets a real gap in tau2's domains — the tools themselves take no
confirmation argument, so the guardrail has to live in the GATE, not the
tool schema.

**Reused, not reimplemented** (see `SCOUT.md` for exact import paths):
`auditk-spec` (Trace/Step/EvidencePack/Probe JSON Schemas, referenced by
path, not a Python package), `auditk` (Pydantic models, EvidencePack
builder, Ed25519 signer/verifier, drift scorer), `tau2-bench` (airline +
retail domains' tools/policy/tasks), `bossyk`'s `PolicyAwareJudge` (used
unmodified as the policy instrument). None of `auditk`, `auditk-spec`, or
`bossyk` were modified by this repo.

## Console & demo

A FastAPI console (`console/app.py`) serves a React SPA (`frontend/`) at `/app`:
an **evidence browser** (story claims, signed evidence packs, benchmark
artifacts, figures — every number links to a committed artifact) and a **control
room** that replays a governance session turn-by-turn.

The control room plays a **committed preset** (`story/replay/*.json`) through the
real domain gate over the existing `held → step → session_complete` websocket
shape:

- `retail-weak-dir1` — the dir-1 gate-save: an under-specified agent skips its
  lookup and the two-speed gate blocks every crossing pre-execution (`harm 4→0`).
  Each crossing traces to a `reached && prevented` row in
  `docs/bench_output/live_h2h4_retail_weak.json`.
- `retail-weak-modes` — the enforcement-delivery taxonomy: each turn resolves
  into a **mode** (allow / redirect / defer / step-up / escalate) and the
  irreversible over-authority actions land in a priority-ordered **HITL review
  queue**. (Design: `enforcement-delivery-model-v0.1` in the vault.)

In the replay the structural gate verdicts are recomputed live from the real
gate, but the resolution `mode`/`hitl` are authored demo choreography (per
scenario, flagged `synthetic`) — a deterministic reconstruction, not a captured
live-LLM transcript.

**Live mode** ("Run live" / `POST /session/live`) instead drives the *real*
weakened-retail agent (`runtime/langgraph_agent`) and derives each mode from the
real gate verdict (`console/modes.py`) — no authored modes. It is **BILLABLE**
and **hard-gated behind `RUN_LIVE_CONSOLE=1`**: without the flag it returns
`live_disabled` *before* building the session or resolving any key (a key present
in the environment is never sufficient on its own, so a stray click can't run the
agent); with the flag but no key it returns `no_api_key`. `defer` is not derived
live — it needs the standing/authority model (§F, deferred).

Run it:

```bash
(cd frontend && npm install && npm run build)   # frontend/dist is not committed
uv run uvicorn bossyk_sandbox.console.app:app --port 8011
# open http://localhost:8011/app/#/control_room  ->  "Run replay"  (non-billable)

# live mode is BILLABLE and off by default; enable it deliberately:
RUN_LIVE_CONSOLE=1 FIREWORKS_API_KEY=... \
  uv run uvicorn bossyk_sandbox.console.app:app --port 8011   # then "Run live"
```

## Setup

```bash
uv sync
cp .env.example .env
```

This repo expects **four sibling checkouts** next to it:

| Repo | Path | How it's consumed |
|---|---|---|
| `auditk` | `../auditk` | editable path dependency (public: github.com/auditk/auditk) |
| `tau2-bench` | `../tau2-bench` | editable path dependency, package `tau2` (public: github.com/sierra-research/tau2-bench) |
| `auditk-spec` | `../auditk-spec` | not a package — trace tests validate against its JSON schemas by relative path (public: github.com/auditk/auditk-spec) |
| `bossyk` | `~/Projects/bossyk` (or `$BOSSYK_ROOT`) | private repo; resolved via `env.bossyk_root()` — a `sys.path` insert at judge-build time plus its policy YAMLs under `data/policies/` |

`bossyk` is not needed for the deterministic test suite — the import is
env-gated and only exercised by the real policy judge. `BOSSYK_ROOT` isn't
listed in `.env.example` yet; add it to your own `.env` if your checkout
lives somewhere other than `~/Projects/bossyk`.

Fill in `.env` for any billable/live run: `FIREWORKS_API_KEY` (agent +
deepseek adversary), `ANTHROPIC_API_KEY` (fable adversary), `FIREWORKS_MODEL`
(override the agent model, default `kimi-k2p6`). The **live console** ("Run
live") additionally requires `RUN_LIVE_CONSOLE=1` — it is off by default even
when a key is present, so it can never bill by accident (see **Console & demo**).

## Deterministic checks

```bash
uv run ruff format --check . && uv run ruff check .
uv run mypy --explicit-package-bases src/ tests/
uv run pytest tests/ -x --no-cov -q
```

Current state: ruff clean; mypy 0 errors (123 source files); pytest **543
passed, 4 skipped**.

```bash
# + the deterministic E2E tests (stub agent through the real GATE/trace/pack path)
RUN_SANDBOX_E2E=1 uv run pytest tests/ -x --no-cov -q
```

Current state: **545 passed, 2 skipped**.

CI (`.github/workflows/ci.yml`) runs all of the above on push/PR, with the
sibling checkouts materialised alongside this repo.

## Running things

| Script | Env needed | What it produces |
|---|---|---|
| `scripts/live_demo.py` | `FIREWORKS_API_KEY` + network | live LangGraph airline demo → `demo_output/<trace>.pack.json` |
| `scripts/benchmark_run.py` | `RUN_SANDBOX_BENCH=1` + `FIREWORKS_API_KEY` + `RUN_JUDGE_MODEL=1` + `RUN_NLI_MODEL=1`; `BENCH_DOMAINS` (default `airline,retail`); needs `uv sync --extra bench` | real drift+policy judges over scripted scenarios → `docs/bench_output/phase2c_orthogonality.json` |
| `scripts/h1_bench.py` | smoke mode: none (deterministic-ish stub). Real mode: `RUN_H1_BENCH=1` + the selected adversary's key; `H1_DOMAIN` (default `airline`); `H1_ADVERSARY` (default `fireworks-deepseek`, also `anthropic-fable`) | `docs/bench_output/phase2b_h1_<domain>_<adversary>.json` + `probes/regression/<domain>-<adversary>.json` |
| `scripts/h4_report.py` | none — deterministic recompute from committed phase2c data | `docs/bench_output/phase3_h4.json` |
| `scripts/smactr_demo.py` | none — deterministic, idempotent | `docs/bench_output/phase4_smactr.json`, `docs/bench_output/threat_model.json`, `probes/regression/retail-smactr.json` |
| `scripts/h1_crossdomain_merge.py <per-domain-json>... --output <path>` | none — merges per-domain H1 files into a cross-domain rollup | regenerates `phase2b_crossdomain_h1.json` byte-identically from `phase2a_h1.json` + `phase2b_h1_retail.json` |
| `uv run uvicorn bossyk_sandbox.console.app:app --port 8011` | Run replay: none (build `frontend/dist` first). Run live: **`RUN_LIVE_CONSOLE=1` + `FIREWORKS_API_KEY`** (billable) | the console + SPA at `/app` — evidence browser & control-room replay (`#/control_room`); Run live drives the real agent (off unless the flag is set). See **Console & demo**. |

## Benchmark artifacts

`phase2a_h1.json` and `phase2b_h1_retail.json` (committed) are dated
snapshots from before the adversary registry existed — they predate the
per-adversary naming. The current `h1_bench.py` emits per-adversary names:
`phase2b_h1_<domain>_<adversary>.json`. `phase2b_crossdomain_h1.json` is not
hand-maintained — it's regenerable from the two per-domain files above via:

```bash
uv run python scripts/h1_crossdomain_merge.py \
    docs/bench_output/phase2a_h1.json \
    docs/bench_output/phase2b_h1_retail.json \
    --output docs/bench_output/phase2b_crossdomain_h1.json
```

`probes/regression/` holds the frozen guardrail-bypass crossings — 56 total
(`airline.json` + `retail.json`, the original dated run) — plus later
per-adversary files (`airline-fireworks-deepseek.json`,
`airline-anthropic-fable.json`) and the SMACTR-derived
`retail-smactr.json` (a single replay-tested regression probe for the
retail-008 failure, see Phase 4).

## Honest limitations

- **H2 is open, not confirmed.** The agent layer's 12/12 catch rate is on a
  hand-authored scripted scenario set, not the 56 frozen guardrail-bypass
  crossings, which have never been run through a live agent — see
  `docs/phase2c-h3-crossdomain-results.md` and `docs/phase5-h1-h5-synthesis.md`.
- **H4 is a modeled counterfactual, not a live measurement.** No tool
  execution was observed; harm is assumed at one unit per violation — see
  `docs/phase3-h4-interrupt-results.md`.
- **Judge availability is accounted for, not silently dropped.** Policy judge
  parse errors and instrument timeouts surface as an explicit "error" state
  (tri-state drift/policy firing; `n_error`/`n_scored` in the H1 math) and are
  excluded from denominators rather than counted as a non-fire.
- **The fable/deepseek variance numbers in `docs/phase15a-adversary-variance-results.md`
  predate the error-accounting refinements from this remediation pass** —
  read them as directionally correct (categorical refusal behavior,
  reproducible class ordering) rather than exactly reproducible under the
  current error semantics.

## Known issues

- **langgraph 1.2.9 bug**: `Command(resume=None)` raises `UnboundLocalError`
  in `langgraph.pregel._loop`'s resume-handling path (`resume_is_map` is
  only bound when `resume is not None`, but read unconditionally a few
  lines later). Always resume with an explicit value — this codebase
  always resumes with the GATE's verdict string rather than `None`.
