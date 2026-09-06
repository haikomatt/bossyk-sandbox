"""Pre-registered-hypothesis experiment harnesses (bossyk-sandbox).

Each module here runs (or reports on) ONE pre-registered hypothesis dossier
(kept outside this repo) -- it never re-derives the metric,
thresholds, or arm definitions the dossier fixes; it only builds/executes
the fixture the dossier's own build-order scope doc specifies. See
`free_threshold_arms`'s module docstring for the first one.
"""

from __future__ import annotations
