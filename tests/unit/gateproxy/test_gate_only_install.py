"""INVARIANT: the gate proxy is installable and importable without the
`experiments` extra. tau2-bench is what the sandbox's agent experiments run
inside; an operator deploying only the gate must never need it. Checked in
a subprocess so this test's own interpreter (which has the extra) cannot
mask a transitive import."""

from __future__ import annotations

import subprocess
import sys


def test_importing_the_gate_proxy_never_imports_tau2() -> None:
    code = (
        "import sys; "
        "import bossyk_sandbox.gateproxy.proxy, bossyk_sandbox.gateproxy.scorecard, "
        "bossyk_sandbox.gateproxy.incident; "
        "leaked = sorted(m for m in sys.modules if m == 'tau2' or m.startswith('tau2.')); "
        "print(leaked); sys.exit(1 if leaked else 0)"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
