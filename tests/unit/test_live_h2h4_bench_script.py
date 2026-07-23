from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "live_h2h4_bench.py"

# scripts/ isn't a package (no __init__.py, not in pyproject's packages),
# so this loads the module by file path -- mirroring how `uv run python
# scripts/live_h2h4_bench.py` would import it, without going through
# `import scripts.live_h2h4_bench`. Importing it must be side-effect-free
# (no network, no billable call): only module-level defs/constants run at
# import time, main() is never called here.


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("live_h2h4_bench", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_live_h2h4_bench_script_imports_without_network_and_defines_main() -> None:
    module = _import_script()

    assert callable(module.main)
    # retail-primary: the default domain is retail unless LIVE_H2_DOMAIN
    # overrides it.
    assert module.DOMAIN in {"airline", "retail"}
    assert set(module._RUN_SESSION_BY_DOMAIN) == {"airline", "retail"}
