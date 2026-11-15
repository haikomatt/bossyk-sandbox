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

### Refactor (Phase B)

No behaviour change and no change in test results. The gate is identical to
Green: 1956 passed, 10 skipped; ruff and mypy clean.

- `instruments/minimisation.py::score`: removed the now-redundant
  `if isinstance(ref, str):` and dedented the width rule. The empty-string
  ref is unchanged, as Matt asked: it is still treated as a string.
- `tests/unit/test_fail_closed.py`: registry ids are now stable and
  readable, `domain:Class[gated_tool]`, and minimisation ids add the required
  band (e.g. `advice:MinimisationInstrument[is_income_above,band=50270]`).
  The old ids used discovery-order suffixes (`#8` / `#9`). The sweep test now
  asserts that ids are unique. No assertion was weakened.
- `tests/unit/gateproxy/test_proxy.py`: renamed
  `test_approver_timeout_resolves_by_policy_default` to
  `test_approver_timeout_blocks_whatever_the_policy_default`. Only the name
  changed (item 11).
- Brought wording in line with D2 wherever it still said a timeout falls back
  to `on_hold`: the `policies.py` comment, the `hold.py` module docstring and
  `Approver` comment, the proxy's `--approver-timeout` help text, the
  `test_hold.py` module docstring, and `docs/gateproxy.md`.

### Docs (Phase B)

- `probes/attestation/tcb_claim_boundary.md` has a new section, "Fail-closed
  defaults (what is pinned, what is fail-soft, and durability)". It covers:
  - what is pinned, from Phase A and B;
  - what is fail-soft by design: slow instruments, empty grants, the
    observe-only advice rules, a missing proxy argument (item 12), and
    `on_hold: allow` with no approver;
  - item 14: a signed `allow` event for a call never forwarded, because a
    later call's instrument raised;
  - durability: the proxy writes and closes each event before it responds,
    without `fsync`; the live-agent trace is signed only at session end.
- `docs/threat-model.md`:
  - The Expressiveness section gains ARTEX item 2's line: a raw-shell or
    generic-HTTP tool is not class A at the tool-call layer, and enforcement
    for one must move to the effect layer. The section explains why, and
    says the packs' `class: A` label on a bash or HTTP predicate classifies
    the predicate, not the policy (class C as built).
  - The T7 residual now points at the pinned defaults and the claim-boundary
    section.

## Phase C: decision source in signed steps

Branch `feature/decision-source`, cut from origin/main 0795006 (Phase B merge).

### Red

New file `tests/unit/test_decision_source.py`. 41 tests: 36 fail on `main`,
5 pass by design. Golden fixture: `tests/fixtures/gateproxy/scorecard-example-pack.html`.

| Area | Test(s) | Pins | Red failure on `main` |
|---|---|---|---|
| D4 | `test_decision_declares_a_required_source_field`, `test_a_decision_without_a_source_is_refused` | `Decision.source` exists with no default; omitting it is a `TypeError` | "Decision has no source field"; "DID NOT RAISE TypeError" |
| Fast rules | `test_every_registered_fast_rule_names_itself_as_the_source` (13, from the registry), `test_observe_only_advice_rules_name_themselves_too` | every decision a rule returns (an ungated ALLOW plus every hostile input) has source `"<RuleClass>:<id>"` with a non-empty id, the same for all its decisions | source `None` |
| Fast rules | `test_rules_in_one_gate_have_distinct_sources` (5 domains, every scenario) | no two rules in one gate share a source: the id must say WHICH rule (two `RequireLookupBeforeCancel` gating different tools, or two minimisation bands) | sources `[None, None, ...]` |
| Utterance | `test_the_utterance_rule_names_itself_as_the_source` | `ProhibitedPhraseRule` decisions name `ProhibitedPhraseRule:<id>` | source `None` |
| Gate | `test_the_gate_fall_through_allow_has_source_gate_default`, `test_the_gate_returns_the_blocking_rules_own_source` | the fall-through ALLOW is `gate:default` (distinguishable from a rule's ALLOW); a BLOCK carries the deciding rule's source | `None == 'gate:default'`; "the gate's decision has no source" |
| Proxy rules | `test_every_proxy_rule_names_its_policy_as_the_source` | every proxy rule (all five types: not-applicable, match, non-string) has source exactly `"<RuleClass>:<policy.id>"`; `HoldOnMatch` keeps it | source `None` |
| Proxy | `test_the_proxy_policy_id_survives_a_reason_without_the_policy_prefix`, `..._ignores_a_reason_that_names_another_policy`, `test_a_held_call_is_attributed_to_its_policy_by_source` | the logged `policy_id` comes from `source`: the egress rule decides as before but with its reason reworded | `policy_id` `None`; **`'package-allowlist'`** (a reason that starts with another policy's id misattributes the block); `None` for a held call |
| Invariant | `test_no_source_module_parses_provenance_out_of_a_reason` | no `src/` module calls `.reason.startswith/split/partition/find/...`, slices `.reason[...]`, or tests `... in x.reason` (an AST walk, not a grep) | `src/bossyk_sandbox/gateproxy/proxy.py:303` |
| Steps | `test_make_step_records_the_decision_source`, `..._gate_default_for_the_fall_through`, `test_an_unoverridden_attested_step_records_the_automatic_source`, `test_a_human_override_records_override_source_human`, `test_a_standing_resolution_is_not_a_human_override` | step metadata has `decision_source` (the automatic decision's source); a human override adds `override_source: "human"`; a standing resolution (Phase A) adds no `override_source` | `decision_source` missing |
| Signing | `test_decision_source_is_covered_by_the_trace_signature` | `decision_source` is in the step, and editing it breaks the trace signature | `'decision_source' in {'gate_verdict': 'block'}` |
| Live agent | `test_the_live_agent_signs_the_gates_source_on_every_step` | the LangGraph agent's attested steps carry the gate's source: `gate:default` for an allowed lookup, the cancel rule's source for a blocked cancel | `[None, None]` |

Passing by design (5):
- **Pin:** `test_the_proxy_scorecard_is_byte_identical_on_the_example_pack`.
  It drives the proxy over the shipped `examples/gateproxy/policy-pack.yaml`
  with a fixed response. The response holds one call per policy, a benign
  call, a non-string governed argument and unparseable arguments, so there are
  8 rows. The test renders the scorecard and compares it byte for byte with the
  golden HTML. The golden was rendered on 0795006, and two renders were
  checked to be byte-identical (no tmp paths, keys or timestamps in the HTML).
- **Broken variant for the pin:** `test_scorecard_pin_catches_a_reworded_reason`
  rewords one reason and shows the HTML no longer matches.
- **Broken variants for the guards:** `test_reason_parsing_guard_catches_each_parsing_shape`,
  `test_distinct_source_guard_catches_two_rules_sharing_a_source` and
  `test_source_guard_catches_a_rule_whose_sources_disagree`.

`Decision(` construction sites Green must touch (`grep -rnE "\bDecision\("`):

- **`src/`: 31.** That is 29 at 8ebc5b8 plus 2 added in Phase B: the F5
  `_non_string_governed` BLOCK and the F4 minimisation BLOCK. By file:
  - `gateproxy/policies.py` 8, including the shared module-level `_ALLOW`,
    which has to become per-rule because each rule needs its own source;
  - `hardcoded_rule.py` 8;
  - `minimisation.py` 6;
  - `utterance_rule.py` 2;
  - `evidence/sample.py` 2;
  - `gate.py`, `gateproxy/proxy.py`, `evidence/trace.py`,
    `runtime/langgraph_agent.py` and `conditions/live_replay.py` 1 each.
- **`tests/`: 48.** That is 47 before this Red plus 1 in the new file, which
  is built from an untyped tuple and needs no change.

Gate (Phase C Red: every API key blank, `BOSSYK_ROOT=/nonexistent/bossyk`,
`RUN_SANDBOX_E2E` unset):

| Check | Result |
|---|---|
| `uv run ruff format --check .` | 357 files already formatted |
| `uv run ruff check --no-cache .` | All checks passed! |
| `uv run mypy --explicit-package-bases src/ tests/` | Success: no issues found in 317 source files |
| `uv run pytest tests/ --no-cov -q` | 36 failed, 1959 passed, 12 skipped |

All 36 failures are in `test_decision_source.py`. Nothing outside that file
fails.

### Issues & Fixes (Phase C)

15. **Interpretation of design rule 2 for `decision_source`.** The scope names
    two keys, `decision_source` and `override_source: "human"`. Red reads
    them as follows:
    - `decision_source` is the source of the AUTOMATIC decision, the one
      `make_step` is handed.
    - `override_source` says who replaced it.
    - Phase A's `resolution_source` keeps naming a non-human resolver.

    The final verdict's source is therefore `override_source`, else
    `resolution_source`, else `decision_source`. The alternative is
    `decision_source` = the final verdict's source, which would make
    `override_source` redundant. Needs Matt's confirmation.
16. **Source format.**
    - Proxy rules are pinned exactly as `"<RuleClass>:<policy.id>"`, so the
      proxy can match a source to a policy by equality.
    - Instrument and utterance rules are pinned as `"<RuleClass>:<id>"` with
      a non-empty id that is distinct within a gate. Green chooses the id: a
      rule's configuration has no single name field, and
      `ProhibitedPhraseRule` has no id at all today.
17. **Green must regenerate a committed evidence artefact.** The sample trace
    (`evidence/sample.py`) is built with `make_attested_step`. Once
    `decision_source` is in step metadata,
    `test_committed_sample_trace_content_regenerates` will fail until
    `docs/bench_output/evidence_pack_sample.{json,trace.json,pub}` are
    regenerated with `scripts/export_evidence_sample.py`, which is offline
    and mints a fresh key. Its two hand-written decisions also need an honest
    source, since neither comes from a real rule. Needs Matt's OK, because
    these are committed evidence files.
18. **Proxy events.** The scope does not ask for the proxy's signed events to
    carry `decision_source`, and Red does not pin it. The events already
    carry `policy_id`, which will now come from `source`. Adding the field
    would be additive, and the scorecard ignores it, so the scorecard pin
    would hold. Matt's call whether Phase C includes it.
19. **Reason-prefix misattribution is a live shape on `main`.** A reason that
    begins with another policy's id credits the decision to that policy
    (`'package-allowlist'` above). No shipped rule writes such a reason today,
    so this is a test-constructed case. It is still the F2 failure mode that
    structured `source` removes.

**Matt's decisions at the Phase C Red gate** ("Approve all four"):

- **Item 15:** `decision_source` is always the source of the automatic
  decision, and is never overwritten. `override_source: "human"` or the
  existing `resolution_source` records who changed the final verdict.
- **Item 16:** every fast rule's source is `"<RuleClass>:<id>"`, the same
  form as the proxy rules. `ProhibitedPhraseRule` gets an id.
- **Item 17:** approved: regenerate `docs/bench_output/evidence_pack_sample.*`
  with the offline export script only. The diff must be only the added
  `decision_source` (and `override_source` where relevant) plus the
  signatures and hashes that change with them. Anything else is reported.
- **Item 18:** out of scope for Phase C. Proxy events do not carry
  `decision_source`. Follow-up below.

### Green (Phase C)

Minimum code to pass the 36 Red tests. The Red tests are unchanged.

**`src/` `Decision(` sites (31, all now pass `source=`):**

| File | Sites | Source |
|---|---|---|
| `instruments/hardcoded_rule.py` | 8 | `self.source`: a new property on `RequireLookupBeforeCancel` and `RequirePassedCheck`, `"<Class>:<gated_tool>"` |
| `gateproxy/policies.py` | 8 | per rule `self.source`, `"<Class>:<policy.id>"` (a property on each of the five rules); `HoldOnMatch` passes the inner decision's source; `_non_string_governed` takes the rule's source |
| `instruments/minimisation.py` | 6 | `self.source`, `"MinimisationInstrument:<predicate_tool>,band=<width>"`, e.g. `MinimisationInstrument:is_income_above,band=50270` |
| `instruments/utterance_rule.py` | 2 | `self.source`; `ProhibitedPhraseRule` gains `rule_id = "prohibited-financial-promotion"` (item 16) |
| `evidence/sample.py` | 2 | `sample:hand-authored`: neither sample decision comes from a real rule, so neither names one |
| `gate.py` | 1 | `GATE_DEFAULT_SOURCE = "gate:default"` |
| `gateproxy/proxy.py` | 1 | `gateproxy:unparseable-arguments` |
| `evidence/trace.py` | 1 | `make_attested_step` builds its final `Decision` with `auto_decision.source` (item 15) |
| `runtime/langgraph_agent.py` | 1 | the new `AgentState.auto_source`, set by `prepare_node` from the gate decision |
| `conditions/live_replay.py` | 1 | `live-replay:post-hoc` (the placeholder ALLOW on post-hoc replay steps) |

**`_ALLOW` per rule.** The shared module constant
`_ALLOW = Decision(Verdict.ALLOW, "not applicable")` is replaced by
`_not_applicable(source)`. Each rule returns
`_not_applicable(self.source)`, so a "not applicable" ALLOW names the rule
that said it. The reason text is unchanged.

**Proxy `policy_id` from source.** `policies.py` gains `_RULE_CLASSES` (policy
type → rule class) and `policy_source(policy)`. The proxy picks the policy
whose `policy_source(p) == decision.source`, takes its `policy_id` from it,
and passes the same `GatePolicy` to `_resolve_hold`, replacing the
second by-id lookup. The reason-prefix `startswith` is gone, so the AST
guard passes.

**Steps.**
- `make_step` puts `decision_source` in step metadata (signed).
- `make_attested_step` keeps the automatic source and adds
  `override_source: "human"` only when `overridden` is true.
- A Phase A standing or gate resolution keeps its `resolution_source`, with
  no `override_source`.

**Existing tests changed (D4 sweep), with the reason.** `Decision.source` is
now required, so every existing test that built a `Decision` without one
raised `TypeError`. 47 sites in 15 files gained `source="test:stub"` and
nothing else; no assertion changed:
- `test_fail_closed.py` 7, `test_utterance_crossing_score.py` 7;
- `test_trace.py` 5, `test_utterance_decision_threading.py` 5;
- `test_attribution_utterance.py` 4;
- `test_coverage.py` 3, `test_coverage_deterministic.py` 3;
- `test_gate.py` 2, `test_oracle_binding.py` 2, `test_oracle_reverification.py` 2,
  `test_policy_provenance.py` 2, `test_utterance_gate_wiring.py` 2;
- `test_attestation_tcb_forgery.py` 1, `test_live_h2h4_bench_script.py` 1,
  `test_scenario_runner.py` 1.

Before the sweep, six Phase B broken-variant tests in `test_fail_closed.py`
"failed". Their stand-ins (an instrument, a predicate and the fail-open
gate) built a source-less `Decision`, raised `TypeError`, and so never got
to show the guard catching them. The sweep restores them. The guards
themselves did not change.

**Evidence sample (item 17).** Regenerated with
`uv run python scripts/export_evidence_sample.py`, offline, with a fresh key
and every API key blank. Diff:
- `evidence_pack_sample.trace.json`: `"decision_source": "sample:hand-authored"`
  added to both steps. There is no `override_source`, because neither sample
  step is overridden. The public key, signature and signature timestamp
  also change.
- `evidence_pack_sample.json`: new public key, signature and signature
  timestamp. **Also `pack_id` and `issued_at`**, which auditk stamps fresh
  on every build with no seam to fix them, as the export script's docstring
  says. They are the only changes beyond signatures and keys, and
  `stable_manifest` excludes them.
- `evidence_pack_sample.pub`: the new public key.

Gate (Phase C Green: every API key blank, `BOSSYK_ROOT=/nonexistent/bossyk`,
`RUN_SANDBOX_E2E` unset):

| Check | Result |
|---|---|
| `uv run ruff format --check .` | 357 files already formatted |
| `uv run ruff check --no-cache .` | All checks passed! |
| `uv run mypy --explicit-package-bases src/ tests/` | Success: no issues found in 317 source files |
| `uv run pytest tests/ --no-cov -q` | 1995 passed, 12 skipped |

**Refactor candidates (not done in Green):**
- `_RULE_CLASSES` repeats the type → class mapping that `_COMPILERS`
  already encodes in its lambdas. One table should derive the other.
- The rule `source` properties repeat the same one-line f-string across seven
  classes.

**Follow-up (item 18).** The proxy's signed `tool_call` events carry
`policy_id` (now read from `source`) but not `decision_source` itself. A
`gateproxy:unparseable-arguments` block or a `gate:default` allow shows up
only as `policy_id: null`. Adding the field would be additive, and the
scorecard ignores it. Out of scope for Phase C per Matt.

The work is in two commits, Green and then the test sweep, as the scope's
Risks section asks. The tree between them is not gated: the Green commit
alone fails the 47 swept sites with `TypeError`. The gate above ran on the
combined tree.

### Refactor (Phase C)

No behaviour change. Every source string is byte-identical, and no test
changed: the Red pins, the scorecard golden and the D4 sweep are untouched.

- **One table.** `gateproxy/policies.py`'s `_COMPILERS` now maps each
  policy type to the rule class it compiles to. `_RULE_CLASSES` is gone.
  Compilation is `_COMPILERS[policy.type].compile(policy, root)`. Tests use
  only the table's keys, which did not change.
- **One source formatter.** The five proxy rules now derive from a new
  frozen-dataclass base, `_PolicyRule`, which holds:
  - the `policy` field;
  - the single `source` property;
  - a `compile` classmethod;
  - an abstract `score`.

  `WorkspaceWriteBoundaryRule` and `ProtectedPathEditRule` derive from
  `_WorkspaceRule`, which adds `workspace_root` and its `compile`. The
  `"<RuleClass>:<policy.id>"` string is spelled once, in `_rule_source`,
  which both the rule's `source` and `policy_source(policy)` use.
- **Small tidy.**
  - `_non_string_governed` now takes the rule, which supplies both its
    policy id and its source, instead of both separately.
  - The `HoldOnMatch` docstring no longer says the policy-id reason prefix
    is what attributes a hold; it is the source now.
- Left alone: `RequireLookupBeforeCancel` and `RequirePassedCheck` still each
  have their own one-line `source` property. They share no base class, and
  adding one is beyond Phase C.

### Docs (Phase C)

- `probes/attestation/tcb_claim_boundary.md` has a new section, "Decision
  provenance (who decided, in the signed record)". It covers:
  - the source of every decision kind;
  - `decision_source` (the automatic decision, never overwritten) versus
    `override_source: "human"` and `resolution_source`;
  - the proxy's `policy_id` now coming from source, and the item-18 gap;
  - the limit: a source is a label the gate's own code writes, inside the
    TCB.
- `docs/gateproxy.md`, "How the evidence fits together", has a new paragraph,
  "Which policy decided".

### Issues & Fixes (Phase C, continued)

20. **The evidence sample's public key changed.** The export script mints a
    fresh signing key on every run and commits only the public half. So
    regenerating `docs/bench_output/evidence_pack_sample.*` (item 17)
    replaced `evidence_pack_sample.pub`. The committed pack and trace verify
    against the new key (`test_committed_sample_trace_verifies_offline`).
    Anything outside this repo that pinned the old key, or a signature made
    with it, will no longer match. Nothing inside the repo cites the old key.
21. **Squash-merge the Phase C PR.** The Green commit (e803bb9) does not pass
    the suite on its own. It makes `Decision.source` required, so the 47
    existing test sites raise `TypeError` until the sweep commit (a190121)
    lands. The gate passes only on the pair (and after Refactor). Merging
    commit by commit would leave a red commit on `main`.
