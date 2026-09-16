"""Path-component safety, shared by every endpoint that takes a caller-supplied
name and joins it onto a server-side directory.

Extracted from `artifacts._safe_path_or_404` (which keeps its 404 behaviour and
directory-listing check) so `replay.load_replay_preset` can apply the identical
rule without importing FastAPI. The rule lives in exactly one place on purpose:
`/session/replay` was vulnerable precisely because it reimplemented nothing and
checked nothing while a tested guard already existed a module away.
"""

from __future__ import annotations


def is_safe_path_component(name: str) -> bool:
    """True if `name` is a single, non-empty path component that cannot escape
    the directory it is joined to.

    Rejects separators (both kinds, since a Windows-style separator is still a
    separator to some filesystems) and any `..` anywhere in the string, rather
    than resolving the path and comparing prefixes: resolution-then-compare is
    easy to get subtly wrong, and this check is the cheap conservative one that
    runs first.

    It deliberately does NOT confirm the file exists. Callers decide what an
    absent file means -- a 404 for the artifact API, a `FileNotFoundError` for
    the preset loader -- so this stays a pure predicate.
    """
    return bool(name) and "/" not in name and "\\" not in name and ".." not in name
