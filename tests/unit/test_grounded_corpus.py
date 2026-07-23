from __future__ import annotations

from dataclasses import dataclass

from bossyk_sandbox.conditions.adversary import ProbeAttempt
from bossyk_sandbox.conditions.grid import AttackClass, ProbeCell, boundaries_for, build_grid
from bossyk_sandbox.conditions.grounded_corpus import (
    generate_grounded_attempts,
    generate_grounded_corpus,
)
from bossyk_sandbox.conditions.live_boundary import parse_family


@dataclass
class _FakeAdversary:
    """Deterministic fake `Adversary`: returns `budget` attempts per cell,
    cycling `statuses` across attempt indices so a test can control exactly
    which attempts come back `ok` vs refused/error (no model call)."""

    statuses: list[str]

    def generate_attempts(self, cell: ProbeCell, budget: int) -> list[ProbeAttempt]:
        return [
            ProbeAttempt(
                cell=cell,
                payload=f"payload-{cell.boundary}-{i}",
                attempt_index=i,
                status=self.statuses[i % len(self.statuses)],  # type: ignore[arg-type]
            )
            for i in range(budget)
        ]


def _two_distinct_retail_cells() -> list[ProbeCell]:
    return [
        ProbeCell("retail", AttackClass.TOOL_MISUSE, "cancel_without_auth"),
        ProbeCell("retail", AttackClass.PII_LEAK, "pii_disclosure"),
    ]


def test_generate_grounded_corpus_freezes_one_probe_per_ok_attempt() -> None:
    adversary = _FakeAdversary(statuses=["ok"])
    cells = _two_distinct_retail_cells()

    probes = generate_grounded_corpus(adversary, cells, budget=2)

    # 2 cells x budget 2, all ok -> 4 frozen probes.
    assert len(probes) == 4


def test_generate_grounded_corpus_skips_refused_and_error_attempts() -> None:
    # A refused/error attempt carries no payload, so it must never become a
    # live-corpus probe (mirrors the harness never freezing a non-ok attempt).
    adversary = _FakeAdversary(statuses=["ok", "refused"])
    cells = [ProbeCell("retail", AttackClass.TOOL_MISUSE, "cancel_without_auth")]

    probes = generate_grounded_corpus(adversary, cells, budget=2)

    assert len(probes) == 1
    assert probes[0].stimulus.payload["text"] == "payload-cancel_without_auth-0"


def test_generate_grounded_corpus_grounds_family_in_domain_and_boundary() -> None:
    # Every frozen probe's family must parse back into (domain, boundary) so
    # replay_crossing / the live boundary oracle can route it.
    adversary = _FakeAdversary(statuses=["ok"])
    cells = build_grid("retail", list(AttackClass), boundaries_for("retail"))

    probes = generate_grounded_corpus(adversary, cells, budget=1)

    assert len(probes) == len(cells)  # 4 classes x 4 boundaries
    for probe in probes:
        domain, boundary = parse_family(probe.family)
        assert domain == "retail"
        assert boundary in boundaries_for("retail")


def test_generate_grounded_corpus_probe_ids_are_unique() -> None:
    adversary = _FakeAdversary(statuses=["ok"])
    cells = build_grid("retail", list(AttackClass), boundaries_for("retail"))

    probes = generate_grounded_corpus(adversary, cells, budget=2)

    ids = [probe.probe_id for probe in probes]
    assert len(ids) == len(set(ids))


def test_generate_grounded_attempts_returns_every_attempt_including_non_ok() -> None:
    # The primitive the corpus generator wraps: it calls the adversary once
    # per cell and returns ALL attempts (ok + refused/error), so the caller
    # (the billable script) can build a token ledger over the full set --
    # refusals cost tokens too -- before freezing only the ok ones.
    adversary = _FakeAdversary(statuses=["ok", "refused"])
    cells = _two_distinct_retail_cells()

    attempts = generate_grounded_attempts(adversary, cells, budget=2)

    assert len(attempts) == 4  # 2 cells x budget 2, nothing dropped
    assert [a.status for a in attempts] == ["ok", "refused", "ok", "refused"]
