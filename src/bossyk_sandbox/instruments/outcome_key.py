from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BoundaryLabel(StrEnum):
    """The A1 scenario key's known-correct boundary outcome for a
    gate-worthy step. Ground truth, not a live scorer."""

    BENIGN = "benign"
    POLICY_VIOLATION = "policy_violation"
    DEVIATION = "deviation"


@dataclass(frozen=True)
class OutcomeKey:
    scenario_id: str
    step_index: int
    boundary_label: BoundaryLabel


@dataclass
class OutcomeKeyLookup:
    """Ground-truth lookup for scored steps, keyed by (scenario_id,
    step_index). Not a `SlowInstrument` — the outcome axis is authored fact,
    joined into the orthogonality/confusion tables after the fact."""

    keys: list[OutcomeKey]

    def __post_init__(self) -> None:
        self._index: dict[tuple[str, int], BoundaryLabel] = {
            (key.scenario_id, key.step_index): key.boundary_label for key in self.keys
        }

    def label_for(self, scenario_id: str, step_index: int) -> BoundaryLabel | None:
        return self._index.get((scenario_id, step_index))

    def is_violation(self, scenario_id: str, step_index: int) -> bool | None:
        label = self.label_for(scenario_id, step_index)
        if label is None:
            return None
        return label is not BoundaryLabel.BENIGN
