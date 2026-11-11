# Phase: fail-closed audit + decision provenance (results)

Scope: `coding-tasks/bossyk-sandbox/phase-fail-closed-audit-and-decision-provenance.md`
(vault). ARTEX item 4 (audit-record hygiene, fail-closed defaults) plus item 2
(shell tools are not class A, docs only). Base: `origin/main` `8ebc5b8`.

Decisions D1 to D4 were answered as recommended (2026-11-10). Phase 0's S1 to
S4 were approved (2026-11-11, "Approve all four").

## Phase 0: evidence (summary)

- **F1 confirmed end to end, in both live loops.** In the refund-budget
  scenario, the escalated refund (#R2) reaches `use_tool` in
  `console/live.py::run_live_session` and in `console/app.py::_run_live_session`.
  Its signed step records `gate_verdict=allow`, and no step carries a `mode`.
  The escalated action is also appended to standing history, so it consumes
  authority.
- **F3 confirmed.** `_resolve_hold` with `on_hold: allow` and an approver that
  returns no answer gives `("hold_timed_out", ALLOW)`.
- **F4 confirmed.** History pins ref `C1` inside the band. Ref `"C1"` is then
  BLOCKed, but `["C1"]` and `1` are ALLOWed.
- **Frontend.** The React control room reads `mode` from `held` events and
  `verdict` / `mode` / `mode_reason` / `hitl` from `step` events. It never
  reads `auto_verdict`. All of these fields are typed `string`, so Phase A
  needs no frontend change.

## Issues & Fixes

1. **S1: Phase A needs `runtime/langgraph_agent.py` (approved).** The scope
   did not list this file. The signed step is built inside `execute_node`
   from `state["final_verdict"]` alone, and the resume value was a bare
   verdict string. A live loop therefore had no way to put the mode or the
   source of the final verdict into the signed record. The resume value may
   now be a dict `{verdict, mode, source}`. The bare-string resume (`"allow"`
   / `"block"`) stays the human-override path for existing callers:
   `conditions/live_replay.py`, `live_multiturn.py`,
   `experiments/free_threshold_arms.py` and the e2e and outreach tests.
2. **S2: a standing-driven block is not a manual override (approved).**
   `make_attested_step` sets `overridden=True`, prefixes the reason with
   "manual override of automatic ...", and tags `verdict:overridden` whenever
   the final verdict differs from the automatic one. A block that standing
   authority produced keeps `overridden=False`. It records
   `metadata["mode"]` and `metadata["resolution_source"]` instead, and the
   reason does not claim a human. (Phase C generalises this into
   `decision_source` / `override_source`.)
3. **S3: F5 is added to Phase B (approved).** All five gate-proxy rules ALLOW
   a governed argument (`command`, `path`) that is present but not a string.
   A list-valued `command` (`["curl", "evil.com"]`) also evades
   `NetworkEgressRule` and `PackageInstallRule`, because
   `str(list)` tokenises to `"['curl',"` and matches no listed command. In
   Phase B, a governed argument that is present but not a string BLOCKs in
   every proxy rule. It is pinned Red-first, like F3 and F4.
4. **S4: `advice_fast_rules_observe_only` is a named exemption (approved).**
   This factory is not registered in `domains.py` and it ALLOWs everything by
   design (arms A/B measure the unmitigated leak). The Phase B registry test
   lists it as an asserted exemption instead of skipping it silently.
5. **Correction to the scope's count of `Decision(` sites in `src/`: 29, not
   35.** The scope's figure included substring matches on other types:
   `ModeDecision(` (`console/modes.py`, 5), `detector/pod_budget.py` (3) and
   `gateproxy/incident.py` (3). Phase C touches only
   `instruments.base.Decision`, which has 29 sites in `src/` and 40 in
   `tests/` across 14 files.
6. **Local environment (not a code issue).** `pyproject.toml` resolves
   `auditk` from `../auditk`, and conftest finds `../tau2-bench` and
   `../auditk-spec`. Inside `.claude/worktrees/` those paths don't resolve,
   so three untracked symlinks sit beside the worktree. They are not part of
   any commit.

## Phase A: escalation is enforced in the live console

### Red

New tests. Each fails on `main` for the expected reason:

- `tests/unit/test_live_session.py`
  - `test_an_escalated_refund_never_reaches_the_toolkit`: fails because
    `#R2` is in the recorded `use_tool` calls.
  - `test_an_escalated_refund_is_signed_as_a_standing_block_not_a_manual_override`:
    fails because `gate_verdict` is `allow`.
  - `test_an_escalated_refund_does_not_consume_standing_authority`: fails
    because the third refund escalates (60 + 60 + 30 = 150 was counted).
- `tests/unit/test_console_live.py`: the same three contracts for the async
  loop, plus the step event's `verdict` being `block`.
- `tests/unit/test_langgraph_agent.py`
  - `test_a_standing_block_resume_does_not_execute_and_is_not_a_manual_override`
    and `test_a_structured_allow_resume_executes_and_records_its_mode`: fail
    with `ValueError: {...} is not a valid Verdict`, because `approve_node`
    accepts only a bare string.

Pins (these pass on `main`; they guard against over-blocking):

- `test_a_deferred_cancel_still_executes` (sync loop) and
  `test_live_console_still_executes_a_deferred_cancel` (async loop). Both fail
  against a deliberately broken variant: a `resume_after` that resumes
  `block` for every held write, loaded as a scratch pytest plugin. Under that
  variant, `[] == ['#A', '#B', '#C']`.

Modified test helper (mechanical, no assertion changed): in
`test_langgraph_agent.py`, the `_run_to_completion` parameter
`resume_values` was widened from `dict[str, str]` to `dict[str, Any]` so a
test can pass a structured resume.

Full suite at Red: 8 failed (exactly the Red tests), 1886 passed, 6 skipped.

### Green

- `runtime/langgraph_agent.py`: `approve_node` accepts a dict resume
  `{verdict, mode, source}` and stores `final_mode` / `final_source` in
  state. A bare-string resume behaves as before (human override, or
  acceptance of the automatic verdict). `execute_node` passes mode and source
  to `make_attested_step`.
- `evidence/trace.py::make_attested_step`: takes optional `mode` and
  `resolution_source` and records both in step metadata. When a
  `resolution_source` is given, the step is not an override: `overridden`
  stays False, the reason says which component decided it
  (`"escalate by standing: final block over automatic allow: ..."`), and no
  `verdict:overridden` control is tagged.
- `console/live.py::resume_after(payload, decision=None)`: when the mode is
  `escalate`, it resumes with `{"verdict": "block", "mode": "escalate",
  "source": "standing"}`. Otherwise it resumes with the gate's own verdict,
  as before. `run_live_session` stops appending escalated actions to
  standing history.
- `console/app.py::_run_live_session`: computes the final verdict (BLOCK on
  `escalate`), appends to history only on a final ALLOW, passes the decision
  to `resume_after`, and sends the final verdict in the step event. The held
  event's `auto_verdict` is still the gate's own verdict.

No test changed in Green.

Known limit (Phase C's job): a dict resume with `source: "human"` would also
record `overridden=False`. No caller does this today. Human overrides still
use the bare-string resume, and Phase C introduces `override_source`.

### Green gate

- `ruff format --check .`: 355 files already formatted. `ruff check --no-cache .`:
  all checks passed. `mypy --explicit-package-bases src/ tests/`: no issues in
  315 source files.
- `pytest tests/ --no-cov -q` with `RUN_SANDBOX_E2E=1` and the API-key variables
  exported empty: 1896 passed, 4 skipped.
- Same run with `BOSSYK_ROOT` set to a path that doesn't exist (what CI sees):
  1890 passed, 10 skipped.

7. **Surprise: a nested worktree loads the main checkout's `.env`, and the
   gate made two real model-API calls.** `tau2/utils/utils.py` calls
   `load_dotenv()` with no arguments at import. That searches upward from
   `.venv/…/tau2` and reaches `~/Projects/bossyk-sandbox/.env`, because this
   worktree lives inside the main checkout at `.claude/worktrees/`. The first
   gate run copied CI's `RUN_SANDBOX_E2E=1`, so
   `tests/e2e/test_interp_behavioral_e2e.py` found a `FIREWORKS_API_KEY` and
   called Fireworks, once in each of the two pytest runs. Both calls returned
   `404 Model not found`, so no completion was served. Whether Fireworks bills
   for that wasn't checked. This broke the scope's no-network rule. The fix,
   from then on: run the gate with the key variables (`FIREWORKS_API_KEY`,
   `AGENT_API_KEY`, `AGENT_BASE_URL`, `ANTHROPIC_API_KEY`, `HF_TOKEN`,
   `RUNPOD_API_KEY`, `FIREWORKS_MODEL`) exported empty. `load_dotenv(override=False)`
   then leaves them empty, the e2e test skips, and the run matches CI.
   Follow-up, outside this scope: the test's own docstring says its gate is a
   billable one, but `RUN_SANDBOX_E2E=1` (which CI sets) is the only thing
   standing between it and a live call. A separate opt-in flag would close
   that.
8. **`BOSSYK_ROOT` pointing at an empty directory is not CI-faithful.**
   `tests/unit/_bossyk.py::requires_bossyk_checkout` skips only when
   `bossyk_root()` does not exist. An empty directory counts as present, so 6
   private-checkout tests run and fail: the five
   `test_domains.py::test_*_domain_policy_path_exists_on_disk` tests and
   `test_policy_provenance.py::test_policy_instrument_with_the_real_bossyk_loader`.
   Nothing in this phase touches those tests or the code they cover. CI has no
   checkout at all, so the CI-faithful run uses a path that doesn't exist
   (1890 passed, 10 skipped), as Phase B of the previous scope did.

### Refactor

- `console/live.py`: `resume_after` is replaced by
  `resolve_held(payload, gate_verdict, decision) -> HeldResolution(final, resume)`.
  This is now the one place a gate verdict and a derived mode become the
  final verdict and the graph resume. Both loops call it, and both use
  `resolution.final` for standing history (and, in the async loop, for the
  step event). That closes the drift Green left behind: one loop tested
  `verdict is ALLOW and mode != "escalate"`, the other computed its own
  `final`. `resume_after` had no other callers.
- Docstrings in `live.py` (module) and `app.py::_run_live_session` no longer
  say a held call is "resumed with its automatic verdict".
- No behaviour or test change. The gate result is identical to Green:
  1896 passed / 4 skipped with keys blanked and `RUN_SANDBOX_E2E=1`;
  1890 passed / 10 skipped with `BOSSYK_ROOT` set to a missing path. ruff
  and mypy are clean.

9. **Bug found in Refactor (Green regression; not fixed yet, awaiting a
   decision).** `derive_mode` also yields `escalate` for *gate BLOCK +
   irreversible write*, such as a refund without a prior lookup. Green's
   resume (kept unchanged by the refactor) sends every `escalate` as
   `{"verdict": "block", "mode": "escalate", "source": "standing"}`, so those
   gate-blocked steps are signed with `resolution_source: "standing"`. The
   gate decided them, not standing: wrong provenance in the signed record
   (design rule 2). On `main` these steps carried no source at all. The
   verdict is right (block) and nothing executes, so enforcement is
   unaffected.
   - **Decision (Matt, "Yes, source gate").** A gate-blocked escalation
     resumes with `{"verdict": "block", "mode": "escalate", "source": "gate"}`.
     `"standing"` is used only when the gate ALLOWed and standing turned the
     action into an escalation.
   - **Red** (`b7c4cb7`): `test_a_gate_blocked_escalation_is_not_attributed_to_standing`
     failed with `assert 'standing' != 'standing'`. It was written to fit
     either candidate fix, so it asserted only "not standing".
   - **Fix**: `resolve_held` picks the source from the gate's own verdict.
   - **Test tightened after Green (documented reason).** Once the decision
     fixed the expected value, the assertion changed from `!= "standing"` to
     `== "gate"`, and it now also asserts `mode == "escalate"`. The loose form
     would have passed for an absent or arbitrary source, so it pinned less
     than the agreed behaviour.

### Phase A final gate (before the PR)

Run with the API keys exported empty and `RUN_SANDBOX_E2E=1`. ruff format:
355 files already formatted. ruff check: clean. mypy over `src/` and
`tests/`: no issues in 315 files. pytest: 1897 passed, 4 skipped. pytest
with `BOSSYK_ROOT` set to a missing path: 1891 passed, 10 skipped. Nothing
under `frontend/` changed, so the frontend build isn't part of this gate.
