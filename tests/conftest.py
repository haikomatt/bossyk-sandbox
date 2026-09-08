"""Repo-wide test setup.

tau2 (the sandbox experiments' airline/retail environments) is installed
from a pinned git source, which ships the package but not its `data/`
tree; tau2 reads that tree from `TAU2_DATA_DIR`. Point it at the sibling
tau2-bench checkout (at the same pinned rev) unless the caller already
set it. Must run before any test module imports tau2, which is why it
lives in the root conftest and not the gateproxy one. Gate-only tests
never import tau2 and are unaffected when the sibling is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

_TAU2_DATA = Path(__file__).resolve().parents[2] / "tau2-bench" / "data"
if _TAU2_DATA.is_dir():
    os.environ.setdefault("TAU2_DATA_DIR", str(_TAU2_DATA))
