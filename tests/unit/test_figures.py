from __future__ import annotations

import json
from pathlib import Path

import pytest

from bossyk_sandbox.figures import (
    render_h1_blindspot_by_class,
    render_h4_prevention_waterfall,
    render_smactr_before_after,
    save_figure,
)
from scripts.export_figures import DEFAULT_OUTPUT_DIR, export_all

REPO_ROOT = Path(__file__).parent.parent.parent
BENCH_OUTPUT = REPO_ROOT / "docs" / "bench_output"

_H1_DATA = json.loads((BENCH_OUTPUT / "phase2b_crossdomain_h1.json").read_text())
_H4_DATA = json.loads((BENCH_OUTPUT / "phase3_h4.json").read_text())
_SMACTR_DATA = json.loads((BENCH_OUTPUT / "phase4_smactr.json").read_text())

_ALL_FIGURE_IDS = ["h1-blindspot-by-class", "h4-prevention-waterfall", "smactr-before-after"]


def _all_output_files(out_dir: Path) -> list[str]:
    names = []
    for figure_id in _ALL_FIGURE_IDS:
        names.append(f"{figure_id}.svg")
        names.append(f"{figure_id}.png")
    return names


# --- (a) determinism: two independent runs produce byte-identical files ----


def test_exporter_is_byte_identical_across_two_runs(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    export_all(out_a)
    export_all(out_b)

    for name in _all_output_files(out_a):
        content_a = (out_a / name).read_bytes()
        content_b = (out_b / name).read_bytes()
        assert content_a == content_b, f"{name} differs between two runs"


# --- (b) fail loud on missing keys ------------------------------------------


def test_h1_figure_raises_on_missing_key() -> None:
    broken = json.loads(json.dumps(_H1_DATA))
    del broken["domains"]["airline"]["h1_by_class"]["tool_misuse"]

    with pytest.raises(KeyError):
        render_h1_blindspot_by_class(broken)


def test_h4_figure_raises_on_missing_key() -> None:
    broken = json.loads(json.dumps(_H4_DATA))
    del broken["per_domain"]["retail"]

    with pytest.raises(KeyError):
        render_h4_prevention_waterfall(broken)


def test_smactr_figure_raises_on_missing_key() -> None:
    broken = json.loads(json.dumps(_SMACTR_DATA))
    del broken["after"]

    with pytest.raises(KeyError):
        render_smactr_before_after(
            broken,
            combined_before_prevented=_H4_DATA["combined"]["prevented"],
            combined_n_violations=_H4_DATA["combined"]["n_violations"],
        )


def test_render_functions_do_not_silently_produce_empty_figures() -> None:
    # A sanity check alongside (b): the happy path actually draws
    # something (axes with data), not just an empty canvas, for all 3.
    fig1 = render_h1_blindspot_by_class(_H1_DATA)
    assert len(fig1.axes[0].patches) > 0

    fig2 = render_h4_prevention_waterfall(_H4_DATA)
    assert len(fig2.axes[0].patches) > 0

    fig3 = render_smactr_before_after(
        _SMACTR_DATA,
        combined_before_prevented=_H4_DATA["combined"]["prevented"],
        combined_n_violations=_H4_DATA["combined"]["n_violations"],
    )
    assert len(fig3.axes[0].patches) > 0


# --- (c) committed-files-current check --------------------------------------


def test_committed_figures_are_current(tmp_path: Path) -> None:
    """Regenerating into a tmp dir must match docs/figures/ byte-for-byte.
    This is also the CI-facing check: regenerate + commit figures whenever
    the underlying benchmark data changes."""
    fresh_dir = tmp_path / "fresh"
    export_all(fresh_dir)

    committed_names = sorted(p.name for p in DEFAULT_OUTPUT_DIR.iterdir())
    expected_names = sorted(_all_output_files(DEFAULT_OUTPUT_DIR))
    assert committed_names == expected_names, (
        f"docs/figures/ should contain exactly {expected_names}, found {committed_names}"
    )

    for name in expected_names:
        committed = (DEFAULT_OUTPUT_DIR / name).read_bytes()
        fresh = (fresh_dir / name).read_bytes()
        assert committed == fresh, (
            f"docs/figures/{name} is stale -- regenerate with scripts/export_figures.py"
        )


def test_save_figure_writes_both_svg_and_png(tmp_path: Path) -> None:
    fig = render_h1_blindspot_by_class(_H1_DATA)
    svg_path, png_path = save_figure(fig, tmp_path, "some-figure")
    assert svg_path.exists()
    assert png_path.exists()
    assert svg_path.suffix == ".svg"
    assert png_path.suffix == ".png"
