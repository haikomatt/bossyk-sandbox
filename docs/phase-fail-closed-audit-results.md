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

## Phase B: fail-closed defaults pinned

### Red / pin

The new `tests/unit/test_fail_closed.py` follows the standing-integrity
pattern: each guard is a function that returns its violations. Every guard
that pins behaviour which is already fail-closed has a `*_catches_*` test
that runs it against a deliberately broken variant and expects a violation.

What the file covers:

- **Registry sweep (design rule 5).** Every fast rule a `domains.py` factory
  builds, both with no scenario and for each shipped scenario. That is 13
  distinct rules: 9 `RequireLookupBeforeCancel`, 2 `RequirePassedCheck` and
  2 `MinimisationInstrument`.
  - Each rule is tried with these hostile inputs: missing, null, int, bool,
    list and dict key; and `arguments=None`.
  - Each input runs twice: with an empty history, and with a history where
    the prerequisite tool was called with the same hostile arguments.
  - The rule must never ALLOW the tool it gates. A raise is not counted
    here; it is pinned as "no effect" separately.
  - A rule class with no declared surface fails
    `test_every_registered_fast_rule_has_a_declared_hostile_input_surface`.
    The guard is shown catching an undeclared class and two broken rules: one
    that allows a missing key, and one that compares raw non-string values.
- **S4.** `advice_fast_rules_observe_only` is a named, asserted exemption.
  It has a stated reason, no registered domain uses it, and the test asserts
  it really does fail open.
- **`RequirePassedCheck`.** A raising predicate never ALLOWs, and neither do
  12 malformed observed results. Broken variants: a rule that treats an error
  as a pass, and a lenient predicate.
- **An instrument that raises has no effect.**
  - `Gate.score` and `TwoSpeedGate.process` raise, and the TwoSpeedGate
    history stays empty.
  - The LangGraph session stops before the toolkit is called.
  - The gate proxy returns 500 and forwards no tool call.
  - Broken variant: a `Gate.score` that swallows the exception and ALLOWs.
    It is caught at all three layers.
- **Slow instruments.** A raise and a timeout both become `error` labels, and
  the gate decision is identical to the fast gate's alone (one BLOCK case,
  one ALLOW case). Broken variant: a `TwoSpeedGate` whose `_gather` lets the
  failure escape.
- **Holds.**
  - A hold on a policy the pack can't name BLOCKs (broken variant: a resolver
    that ALLOWs).
  - An unresolved hold event (no `resolved_verdict` key) counts as `block`
    (broken variant: a reader that counts it as allow).
  - D2 boundary, pinned so F3's fix stays narrow: with no approver
    configured, `on_hold: allow` still means allow and log.
- **Unparseable proxy arguments** take the "unparseable (fail closed)" path,
  both at the parser and end to end. Broken variant: a parser that defaults
  to `{}`.
- **Agent graph.** It executes only on a final ALLOW: a resumed `hold` or
  `block` never reaches the toolkit. Broken variant: an agent-module
  `Verdict` that parses `hold` as ALLOW.

Red on `main`, as intended:

- **F4**: `test_no_registered_fast_rule_allows_its_gated_tool_on_hostile_input`
  fails for both advice `MinimisationInstrument` instances. Six hostile cases
  are ALLOWed: missing, null, int, bool, list and dict ref.
- **F5**: `test_every_proxy_rule_refuses_a_present_non_string_governed_argument`
  fails for all five compiler types. Examples: `command=["curl", …]` on
  `no-network-egress`, and `path=123` on `workspace-write-boundary`.
- **F3**: `test_an_approver_that_never_answers_resolves_to_block[allow]` fails
  with `('hold_timed_out', ALLOW)`. The end-to-end
  `test_the_proxy_forwards_nothing_when_the_approver_times_out_under_on_hold_allow`
  fails because the proxy forwarded `['bash']`.
- The existing
  `gateproxy/test_proxy.py::TestHoldPath::test_approver_timeout_resolves_by_policy_default[allow-…]`
  also fails. See item 11.

Gate at Red (API keys exported empty, `RUN_SANDBOX_E2E=1`, `BOSSYK_ROOT` set
to a missing path):

- ruff format and check: clean.
- mypy over `src/` and `tests/`: no issues in 316 files.
- pytest: 10 failed (exactly the cases above), 1933 passed, 10 skipped.

### Issues & Fixes (Phase B)

10. **Already-fail-closed claim that does not hold, reported and not fixed
    here.** `gateproxy/events.py::effective_verdict` returns `"block"` for a
    hold event with no `resolved_verdict` key. For a hold carrying an
    explicit `"resolved_verdict": null`, it returns the string `"None"`:
    `event.get(key, "block")` only covers the missing key. The proxy itself
    never writes such an event, because every HOLD it logs has a resolved
    verdict. But a log from another writer, or one edited by hand, would
    produce `"None"` in the scorecard and incident report, which is neither
    blocked nor allowed. Following the scope's instruction, only the
    missing-key case is pinned. The null case needs Matt's call: fix it in
    Phase B Green, or log it as a follow-up.
11. **Existing test changed in Red (documented, D2).**
    `test_approver_timeout_resolves_by_policy_default` parametrised
    `("allow", "allow")`: it encoded F3 itself. It is now `("allow", "block")`,
    with a comment citing D2. Its name ("resolves by policy default") no
    longer describes the timeout case. Renaming it is a Refactor item.
12. **F5's contract applies to present arguments only.** The proxy rules are
    deny-list predicates, so a *missing* governed argument still means "not
    applicable" and is ALLOWed. A tool call with no `command` or `path` gives
    the rule nothing to match. F5 covers only an argument that is present and
    not a string, including JSON `null`. `CredentialReadRule` governs both
    `path` and `command` on each of its tools.
13. **F4's scope.** Following D3 ("matching the other rules"),
    `MinimisationInstrument` is expected to BLOCK a missing ref as well as a
    non-string one, as the precondition rules do. An empty-string ref is not
    in the hostile set for any rule. The precondition rules already block it;
    minimisation treats it as a string.
14. **Audit-record note (from reading the code, not tested).** In the proxy,
    each tool call's event is signed and appended inside the loop. If the
    instrument for a *later* call in the same response raises, the request
    returns 500 and nothing is forwarded. The earlier call's signed
    `tool_call` event still says `allow`, so the log records an allow for an
    action the client never received. This still fails closed. It is the
    "record before effect" ordering seen from the other side. Noted for the
    claim boundary.

### Green (Phase B)

Item 10 joined Phase B on Matt's "Approve, fix item 10", through the
regression ratchet. The Red commit is `a22ad43`: 11 unrecognised
resolutions, including null, failed with `str(value)`.

- **F3** (`gateproxy/proxy.py::_resolve_hold`): a configured approver that
  gives no answer resolves to `("hold_timed_out", BLOCK)`, whatever `on_hold`
  says (D2). `on_hold` still decides when no approver is configured.
- **F4 and item 13** (`instruments/minimisation.py::score`): after the
  observe-only branch, a ref that is missing or not a string BLOCKs (D3), so
  the observe-only exemption is unchanged. The now-redundant
  `if isinstance(ref, str):` that follows is left for Refactor.
- **F5** (`gateproxy/policies.py`): a new `_non_string_governed(policy,
  proposed, keys)` BLOCKs when a governed argument is present and not a
  string. The reason starts with `"{policy.id}: "`, so the proxy's policy-id
  lookup and the `on_match: hold` wrapper behave as for any other match. It
  is called right after `_applies` in all five rules:
  - `command` for egress and install;
  - `path` for workspace-write and protected-path;
  - `path` and `command` for credential-read.

  A missing governed argument is unchanged ("not applicable", item 12).
- **Item 10** (`gateproxy/events.py::effective_verdict`): a hold resolves to
  `resolved_verdict` only when that is exactly `"allow"` or `"block"`.
  Anything else, including a present null, is `"block"`.

No test changed in Green. Items 12 and 14 stay unfixed by decision; they go
into the claim-boundary paragraph in the docs step.

Gate (API keys exported empty, `RUN_SANDBOX_E2E=1`): ruff format reports
356 files already formatted; ruff check is clean; mypy over `src/` and
`tests/` finds no issues in 316 files. pytest with `BOSSYK_ROOT` set to a
missing path: 1956 passed, 10 skipped. That is 1933 from before, plus the 10
Red cases now passing, plus the 13 item-10 cases.
