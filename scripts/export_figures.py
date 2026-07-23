#!/usr/bin/env python
"""Regenerates the 3 committed demo figures (`docs/figures/*.{svg,png}`)
from the committed benchmark JSONs. Deterministic: no network, no keys,
no wall-clock dependence -- re-running should reproduce the committed
files byte-for-byte (see `tests/unit/test_figures.py`).

Usage:
    uv run python scripts/export_figures.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bossyk_sandbox.figures import (
    DIR1_STRUCTURAL_ARTIFACT,
    DIR1_WEAK_ARTIFACT,
    H1_ARTIFACT,
    H4_ARTIFACT,
    SMACTR_ARTIFACT,
    render_dir1_gate_save,
    render_h1_blindspot_by_class,
    render_h4_prevention_waterfall,
    render_smactr_before_after,
    save_figure,
)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures"


def _load(relative_path: str) -> dict[str, object]:
    data: dict[str, object] = json.loads((REPO_ROOT / relative_path).read_text())
    return data


def export_all(output_dir: Path) -> list[Path]:
    written: list[Path] = []

    h1_data = _load(H1_ARTIFACT)
    fig1 = render_h1_blindspot_by_class(h1_data)
    written.extend(save_figure(fig1, output_dir, "h1-blindspot-by-class"))

    h4_data = _load(H4_ARTIFACT)
    fig2 = render_h4_prevention_waterfall(h4_data)
    written.extend(save_figure(fig2, output_dir, "h4-prevention-waterfall"))

    smactr_data = _load(SMACTR_ARTIFACT)
    combined = h4_data["combined"]
    assert isinstance(combined, dict)
    fig3 = render_smactr_before_after(
        smactr_data,
        combined_before_prevented=combined["prevented"],
        combined_n_violations=combined["n_violations"],
    )
    written.extend(save_figure(fig3, output_dir, "smactr-before-after"))

    dir1_weak = _load(DIR1_WEAK_ARTIFACT)
    dir1_structural = _load(DIR1_STRUCTURAL_ARTIFACT)
    fig4 = render_dir1_gate_save(dir1_weak, dir1_structural)
    written.extend(save_figure(fig4, output_dir, "dir1-gate-save"))

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write the 6 figure files into (default: docs/figures).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    written = export_all(output_dir)
    for path in written:
        print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
