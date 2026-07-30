"""RED-phase tests for `run_live_outreach_session` /
`run_live_weakened_outreach_session` (bossyk-sandbox slice 3, phases 3a and
the weak-outreach wiring follow-up -- zero-cost live-path wiring).

SAFETY (read before touching this file): these runners, like
`run_live_retail_session`/`run_live_weakened_retail_session`, take NO
llm/environment override -- they always call
`build_outreach_agent_session`/`build_weakened_outreach_agent_session`
internally, which resolves a REAL Fireworks client and makes a REAL network
call whenever `FIREWORKS_API_KEY` happens to be set in the environment
(true in this dev shell today). The existing repo convention for these
runners (test_live_h2h4_bench_script.py's
`test_live_h2h4_bench_script_imports_without_network_and_defines_main`)
NEVER calls them in the deterministic suite -- it only imports the module
and checks identity/registration. This file follows the same convention:
each runner is imported and inspected (callable, single `payload`
parameter), never invoked.
"""

from __future__ import annotations

import inspect

from bossyk_sandbox.conditions.live_replay import (
    run_live_outreach_session,
    run_live_retail_session,
    run_live_weakened_outreach_session,
    run_live_weakened_retail_session,
)


def test_run_live_outreach_session_exists_and_is_callable() -> None:
    assert callable(run_live_outreach_session)


def test_run_live_outreach_session_takes_a_single_payload_argument_like_retail() -> None:
    # Mirrors run_live_retail_session's exact signature -- the shape
    # scripts/live_h2h4_bench.py's `Callable[[str], LiveSessionResult]`
    # registry (_RUN_SESSION_BY_DOMAIN) expects.
    outreach_params = list(inspect.signature(run_live_outreach_session).parameters)
    retail_params = list(inspect.signature(run_live_retail_session).parameters)

    assert outreach_params == retail_params == ["payload"]


def test_run_live_weakened_outreach_session_exists_and_is_callable() -> None:
    assert callable(run_live_weakened_outreach_session)


def test_run_live_weakened_outreach_session_takes_a_single_payload_argument_like_retail() -> None:
    # Mirrors run_live_weakened_retail_session's exact signature -- the
    # shape scripts/live_h2h4_bench.py's new _WEAK_SESSION_BY_DOMAIN
    # registry expects.
    outreach_params = list(inspect.signature(run_live_weakened_outreach_session).parameters)
    retail_params = list(inspect.signature(run_live_weakened_retail_session).parameters)

    assert outreach_params == retail_params == ["payload"]
