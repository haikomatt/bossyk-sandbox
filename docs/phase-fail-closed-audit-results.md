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
