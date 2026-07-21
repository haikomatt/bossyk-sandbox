# bossyk-sandbox Scout Results — Phase 0

_Generated: 2026-07-21_

Source task: `coding-tasks/bossyk-sandbox/phase0-walking-skeleton.md` (Obsidian vault).

---

## Step 0 decisions (confirmed with user)

1. **Repo location**: new repo at `~/Projects/bossyk-sandbox` (this repo). Clean
   separation from `bossyk`; grows into the product demo.
2. **LangGraph version**: `langgraph>=1.0` (matches `a-prior-project`'s pin). No repo
   in `~/Projects` currently uses the `interrupt()`/HITL pattern — built fresh
   here. API: `langgraph.types.interrupt(value)` inside a node pauses execution
   and surfaces `value` to the caller; resumed via `Command(resume=...)`.
   Requires a checkpointer (e.g. `MemorySaver`) to persist state across the
   pause/resume boundary. `interrupt_before`/`interrupt_after` are compile-time
   `graph.compile(checkpointer=..., interrupt_before=[...])` node-name lists
   for static breakpoints — the GATE uses the manual `interrupt()` form so it
   can hold on a specific tool call rather than every call to a node.
3. **tau2 airline tool to gate**: `cancel_reservation(reservation_id: str)`
   (`tau2-bench/src/tau2/domains/airline/tools.py:339`) — state-changing
   (reverses payments, sets `status="cancelled"`). Its tool signature carries
   no confirmation field, so the block condition is **trace-based**: the GATE
   blocks a `cancel_reservation` call unless an earlier Step in the same trace
   contains a `get_reservation_details` tool call for the same
   `reservation_id`. Benign action for the allow path: `get_reservation_details`
   itself, or a `cancel_reservation` preceded by the required lookup.
   Task `id="0"` in `tau2-bench/data/tau2/domains/airline/tasks.json` is a
   cancellation scenario, useful as a live-agent reference task.
4. **Live agent LLM provider/model**: Fireworks' OpenAI-compatible endpoint
   (`https://api.fireworks.ai/inference/v1`), via `langchain_openai.ChatOpenAI`
   pointed at that `base_url` — no separate SDK dependency. Model:
   `accounts/fireworks/models/firefunction-v2` (purpose-built for
   OpenAI-style function calling; good fit for the airline domain's ~14
   tools). Env var convention matches `auditk-constellaration-experiment`'s
   `.env`: `FIREWORKS_API_KEY` (required) + `FIREWORKS_MODEL` (optional
   override). `auditk`'s own `FireworksJudge` uses the same base URL but a
   different model (`gpt-oss-120b`, a reasoning model tuned for judging, not
   tool-calling) — not reused here since the live agent's job is different.
   `.env` is loaded automatically on import via `tau2.utils`'s
   `load_dotenv()` call (searches upward from cwd) — no extra dependency
   needed. See `.env.example`.

---

## Reuse inventory — exact import paths

### auditk-spec (schema-only, no Python package — reference by file path)

`~/Projects/auditk-spec/spec/v0.1/`:
- `trace.schema.json` — `Step` required: `step_id, trace_id, timestamp, actor
  (user|agent|tool|environment), action`. `action.type` enum: `utterance,
  tool_call, state_transition, env_effect`. `Trace` required: `trace_id,
  flow_type, agent_config_ref, steps, source_adapter`. `flow_type` includes
  `generic` (use as the flow_type for Phase 0).
- `evidence-pack.schema.json` — `spec_version` const `"v0.1"`; required
  `pack_id, spec_version, issued_at, issuer, subject, trace_summary`;
  `signatures[].algorithm` enum includes `ed25519`.
- `agent-config.schema.json` — required `config_id, version, flow_type,
  system_prompt`.

Not installed as a dependency (no `pyproject.toml` in that repo); validate
against these files directly with `jsonschema`, referencing the path.

### auditk (installed editable via `uv.sources`)

- Pydantic models: `from auditk.schema import Trace, Step, Action, ActionType,
  Actor, FlowType, EvidencePack, Issuer, Subject, TraceSummary, ProbeResult,
  Signature, AgentConfig, RiskTier` — 1:1 mirror of the JSON Schemas
  (`auditk/src/auditk/schema.py`). Importing `Trace` pulls in
  `auditk.analysis.taxonomy.TaxonomyLabel` at module load — harmless, no judge
  machinery invoked unless `compute_drift` is called.
- Pack builder: `auditk.attestation.pack.build(traces: list[Trace],
  probe_results: list[ProbeResult], jurisdiction: list[str], risk_tier:
  RiskTier, issuer: Issuer, subject: Subject, signer: Signer, scorer_key:
  str | None = None) -> EvidencePack`. Pass `scorer_key=None` to skip
  drift/judge analysis entirely (`drift_metrics` is optional in the schema).
- Signing: `auditk.attestation.signer.LocalEd25519Signer(key_path)` (reads a
  PEM private key file), `LocalEd25519Verifier(public_key_pem:
  str).verify(payload, signature_b64) -> None` (raises
  `cryptography.exceptions.InvalidSignature` on failure).
  `generate_keypair(path) -> (priv_path, pub_path)` writes
  `<path>.ed25519` / `<path>.ed25519.pub`. File-path based, no env-var key
  loading.
- Canonicalization: `auditk.attestation.canonical.canonicalize(obj) -> bytes`
  (sorted-key JSON) — must match `pack.build`'s manifest construction
  (`pack.model_dump(mode="json", exclude={"signatures"})`) before signing/
  verifying.
- Reference: `auditk/src/auditk/cli.py` `verify()` command (~line 188) shows
  the canonical verify pattern end-to-end.
- `auditk.adapters.langgraph.LangGraphTraceAdapter` (registered as
  `"langgraph"` in `auditk.adapters.registry`) already converts a serialized
  LangGraph checkpoint dict into `Trace`/`Step` objects. Phase 0's
  `evidence/trace.py` may reuse this directly instead of hand-rolling a
  mapper — confirm its expected checkpoint shape when implementing Step 2.
- **No auditk internals need modification.** `pack.build` + signer/verifier
  operate on generic `Trace`/`EvidencePack` objects; `probe_results=[]`,
  `scorer_key=None` is a legal, judge-free call.

### tau2-bench (installed editable via `uv.sources`, package name `tau2`)

- Airline tools: `tau2-bench/src/tau2/domains/airline/tools.py` —
  `book_reservation, calculate, cancel_reservation, get_reservation_details,
  get_user_details, list_all_airports, search_direct_flight,
  search_onestop_flight, send_certificate, transfer_to_human_agents,
  update_reservation_baggages, update_reservation_flights,
  update_reservation_passengers, get_flight_status`.
- Loading a task: `from tau2.run import get_tasks, run_single_task`;
  `tasks = get_tasks("airline", task_ids=["0"])`. Task IDs are string
  integers (`"0"`..`"49"`) in
  `tau2-bench/data/tau2/domains/airline/tasks.json`. Higher-level:
  `tau2.run.run_domain(TextRunConfig(domain="airline", agent="llm_agent",
  ...))`. CLI also available: `tau2-bench/src/tau2/cli.py`.

### bossyk

- `PolicyAwareJudge` (see `~/Projects/bossyk/SCOUT.md`) — **not used in the
  Phase 0 gating path** per the plan's determinism decision. The
  `Instrument` interface in `src/bossyk_sandbox/instruments/base.py` is
  shaped so `PolicyAwareJudge` can drop in at Phase 1 without changing the
  GATE's call site.

---

## Environment

- `uv sync` resolves cleanly: `auditk` (path dep, editable) and `tau2` (path
  dep, editable) both installed from sibling repos; `langgraph==1.x` resolved
  alongside them (`langgraph-prebuilt`, `langgraph-sdk` pulled in
  transitively). `pytest`, `ruff`, `mypy` in the `dev` dependency group.
- Lint: `ruff format --check . && ruff check .`. Types: `mypy
  --explicit-package-bases` (also set in `[tool.mypy]`). Tests: `pytest
  tests/ -x --no-cov -q`.
