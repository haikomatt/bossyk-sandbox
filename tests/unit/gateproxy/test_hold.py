"""Red-phase contract for the HOLD approver hooks.

A HOLD verdict pauses the proposed action and asks an approver. In a
non-interactive batch run there is no human at a console, so the approver
is either an external command (JSON on stdin; exit 0 approves, any other
exit denies) or a URL (JSON POST; the body's ``decision`` is ``allow`` or
``block``). Either hook has a timeout; on timeout or failure the approver
answers ``None`` and the proxy falls back to the policy's ``on_hold``
default. The approver never sees anything the signed event does not
already carry.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx

from bossyk_sandbox.gateproxy.hold import HoldRequest, command_approver, url_approver

_REQUEST = HoldRequest(
    tool_name="bash",
    arguments={"command": "rm -rf build"},
    policy_id="destructive-shell",
    reason="destructive-shell: network egress via 'rm' is not permitted",
    run_label="leg-b-test",
)


def _python(snippet: str) -> list[str]:
    return [sys.executable, "-c", snippet]


class TestCommandApprover:
    def test_exit_zero_approves(self) -> None:
        approve = command_approver(_python("import sys; sys.exit(0)"), timeout_s=5.0)
        assert approve(_REQUEST) is True

    def test_nonzero_exit_denies(self) -> None:
        deny = command_approver(_python("import sys; sys.exit(3)"), timeout_s=5.0)
        assert deny(_REQUEST) is False

    def test_timeout_answers_none(self) -> None:
        slow = command_approver(_python("import time; time.sleep(5)"), timeout_s=0.2)
        assert slow(_REQUEST) is None

    def test_missing_command_answers_none(self) -> None:
        missing = command_approver(["/nonexistent/approver-binary"], timeout_s=1.0)
        assert missing(_REQUEST) is None

    def test_request_delivered_as_json_on_stdin(self) -> None:
        checker = command_approver(
            _python(
                "import json,sys; r=json.load(sys.stdin); "
                "sys.exit(0 if r['tool_name']=='bash' and r['policy_id']=='destructive-shell' "
                "and r['arguments']['command']=='rm -rf build' else 1)"
            ),
            timeout_s=5.0,
        )
        assert checker(_REQUEST) is True


def _transport(handler: Any) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class TestUrlApprover:
    def test_allow_decision_approves_and_posts_request(self) -> None:
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"decision": "allow"})

        approve = url_approver(
            "http://approver.invalid/hold", timeout_s=5.0, transport=_transport(handler)
        )
        assert approve(_REQUEST) is True
        assert seen[0]["tool_name"] == "bash"
        assert seen[0]["policy_id"] == "destructive-shell"
        assert seen[0]["run_label"] == "leg-b-test"

    def test_block_decision_denies(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"decision": "block"})

        deny = url_approver("http://a.invalid/h", timeout_s=5.0, transport=_transport(handler))
        assert deny(_REQUEST) is False

    def test_unexpected_decision_answers_none(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"decision": "shrug"})

        odd = url_approver("http://a.invalid/h", timeout_s=5.0, transport=_transport(handler))
        assert odd(_REQUEST) is None

    def test_http_error_answers_none(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        broken = url_approver("http://a.invalid/h", timeout_s=5.0, transport=_transport(handler))
        assert broken(_REQUEST) is None

    def test_timeout_answers_none(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow approver")

        slow = url_approver("http://a.invalid/h", timeout_s=0.1, transport=_transport(handler))
        assert slow(_REQUEST) is None
