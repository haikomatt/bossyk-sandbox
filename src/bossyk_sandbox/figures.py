"""Deterministic figure rendering for the F0 demo/deck pipeline: three
figures, each built from a committed benchmark JSON, rendered to both SVG
and PNG with a fixed size/dpi and no time/locale dependence so the outputs
are byte-identical across runs (see `tests/unit/test_figures.py`).

Reused across figures: a small neutral/accent palette (colorblind-safe),
consistent font sizes, a bottom-left source-line annotation (artifact path
+ origin commit) and a light x-only grid -- see the module constants.

Each `render_*` function takes already-loaded artifact dict(s) and returns
a `matplotlib.figure.Figure`; it never opens a file itself (that's
`save_figure`/`scripts/export_figures.py`'s job) and it never falls back
on a missing key -- direct dict indexing raises `KeyError` so a malformed
artifact fails loud instead of silently rendering an empty chart."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# --- determinism -----------------------------------------------------------

FIGURE_SIZE_INCHES = (10.5, 5.5)
DPI = 150


def _apply_deterministic_rcparams() -> None:
    # svg.hashsalt: the SVG backend otherwise derives clip-path/gradient
    # ids from a per-process counter seeded by object identity, which is
    # not stable across processes. A fixed hashsalt makes those ids (and
    # therefore the byte content) reproducible run to run.
    plt.rcParams["svg.hashsalt"] = "bossyk-sandbox"
    plt.rcParams["figure.figsize"] = FIGURE_SIZE_INCHES
    plt.rcParams["figure.dpi"] = DPI
    plt.rcParams["font.family"] = "DejaVu Sans"


_apply_deterministic_rcparams()

# --- palette (colorblind-safe: Wong/Tol-style blue+red pair) ---------------

BASE_COLOR = "#4B4B4B"  # neutral -- combined/non-domain-specific bars
AIRLINE_COLOR = "#4477AA"  # blue accent
RETAIL_COLOR = "#EE6677"  # red accent
GRID_COLOR = "#D8D8D8"

DOMAIN_COLORS = {"airline": AIRLINE_COLOR, "retail": RETAIL_COLOR}

# --- font sizing (consistent across all 3 figures) -------------------------

TITLE_FONTSIZE = 14
SUBTITLE_FONTSIZE = 10
AXIS_LABEL_FONTSIZE = 10
TICK_FONTSIZE = 9
VALUE_LABEL_FONTSIZE = 8.5
SOURCE_FONTSIZE = 7.5

# --- known provenance (fixed metadata about where each artifact came from) -

H1_ARTIFACT = "docs/bench_output/phase2b_crossdomain_h1.json"
H1_COMMIT = "adeb19b"
H4_ARTIFACT = "docs/bench_output/phase3_h4.json"
H4_COMMIT = "af33a88"
SMACTR_ARTIFACT = "docs/bench_output/phase4_smactr.json"
SMACTR_COMMIT = "12c65a0"
DIR1_WEAK_ARTIFACT = "docs/bench_output/live_h2h4_retail_weak.json"
DIR1_STRUCTURAL_ARTIFACT = "docs/bench_output/live_h2h4_retail_weak_structural.json"
DIR1_COMMIT = "27ca0e3"

INTERP_REPROBE_ARTIFACT = "probes/interp/results/reprobe_qwen_pca32.json"
INTERP_TEXT_ARTIFACT = "probes/interp/results/text_baseline_qwen.json"
INTERP_LEADTIME_ARTIFACT = "probes/interp/results/leadtime_report.json"
INTERP_EVASION_ARTIFACT = "probes/interp/results/evasion_gate_summary.json"
# The deep layer used for the activation-vs-text comparison: the probe's
# policy AUROC peaks and stabilises here, so it is the strongest case for
# the residual -- and it still does not beat text.
INTERP_LAYER = "14"

H1_CLASS_ORDER = ["tool_misuse", "pii_leak", "jailbreak", "prompt_injection"]
H4_GROUP_ORDER = ["combined", "airline", "retail"]


def _source_line(artifact: str, commit: str) -> str:
    return f"Source: {artifact} (commit {commit})"


def _add_source_line(fig: Figure, artifact: str, commit: str) -> None:
    fig.text(
        0.01,
        0.01,
        _source_line(artifact, commit),
        fontsize=SOURCE_FONTSIZE,
        color=BASE_COLOR,
        ha="left",
        va="bottom",
    )


def _style_axes(ax: Any) -> None:
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


# --- figure 1: h1-blindspot-by-class ---------------------------------------


def render_h1_blindspot_by_class(data: dict[str, Any]) -> Figure:
    """Grouped horizontal bars: bypass rate per attack class x domain
    (airline, retail), Wilson-CI whiskers, from a cross-domain H1 JSON
    (`domains.<domain>.h1_by_class.<class>` -> rate + wilson_ci95)."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES, dpi=DPI)

    guardrail_backend = data["guardrail_backend"]
    canonical_strength = data["canonical_strength"]
    domains = data["domains"]

    bar_height = 0.32
    class_positions = {cls: i for i, cls in enumerate(reversed(H1_CLASS_ORDER))}

    for domain_name, offset in (("airline", bar_height / 2), ("retail", -bar_height / 2)):
        class_blocks = domains[domain_name]["h1_by_class"]
        color = DOMAIN_COLORS[domain_name]
        for cls in H1_CLASS_ORDER:
            block = class_blocks[cls]
            rate = block["rate"]
            n_attempts = block["n_attempts"]
            n_bypassed = block["n_bypassed"]
            ci_low, ci_high = block["wilson_ci95"]
            y = class_positions[cls] + offset
            ax.barh(
                y,
                rate,
                height=bar_height,
                color=color,
                zorder=2,
                label=domain_name if cls == H1_CLASS_ORDER[0] else None,
            )
            ax.errorbar(
                rate,
                y,
                xerr=[[max(0.0, rate - ci_low)], [max(0.0, ci_high - rate)]],
                fmt="none",
                ecolor=BASE_COLOR,
                elinewidth=1.0,
                capsize=3,
                zorder=3,
            )
            ax.annotate(
                f"{rate * 100:.1f}% ({n_bypassed}/{n_attempts})",
                xy=(ci_high, y),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=VALUE_LABEL_FONTSIZE,
                va="center",
                color=BASE_COLOR,
            )

    ax.set_yticks([class_positions[cls] for cls in H1_CLASS_ORDER])
    ax.set_yticklabels([cls.replace("_", " ") for cls in H1_CLASS_ORDER])
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("Guardrail bypass rate", fontsize=AXIS_LABEL_FONTSIZE)
    _style_axes(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=AXIS_LABEL_FONTSIZE)

    fig.suptitle(
        "The guardrail catches what looks adversarial, not what is harmful",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    ax.set_title(
        f"Guardrail: {guardrail_backend} (deberta prompt-injection classifier), "
        f"{canonical_strength} strength (canonical)",
        fontsize=SUBTITLE_FONTSIZE,
        loc="left",
        color=BASE_COLOR,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _add_source_line(fig, H1_ARTIFACT, H1_COMMIT)
    return fig


# --- figure 2: h4-prevention-waterfall --------------------------------------


def render_h4_prevention_waterfall(data: dict[str, Any]) -> Figure:
    """Segmented horizontal bars per domain + combined: prevented /
    detected-too-late / undetected out of n_violations, from the H4
    modeled-counterfactual JSON."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES, dpi=DPI)

    groups: dict[str, dict[str, Any]] = {
        "combined": data["combined"],
        "airline": data["per_domain"]["airline"],
        "retail": data["per_domain"]["retail"],
    }

    bar_height = 0.5
    y_positions = {name: i for i, name in enumerate(reversed(H4_GROUP_ORDER))}

    for name in H4_GROUP_ORDER:
        block = groups[name]
        n_violations = block["n_violations"]
        prevented = block["prevented"]
        detected_too_late = block["detected_too_late"]
        undetected = block["undetected"]
        y = y_positions[name]
        base_color = BASE_COLOR if name == "combined" else DOMAIN_COLORS[name]

        segments = [
            ("prevented", prevented, 1.0, None),
            ("detected too late", detected_too_late, 0.45, None),
            ("undetected", undetected, 0.15, "///"),
        ]
        left = 0.0
        for _seg_name, width, alpha, hatch in segments:
            ax.barh(
                y,
                width,
                left=left,
                height=bar_height,
                color=base_color,
                alpha=alpha,
                hatch=hatch,
                edgecolor=BASE_COLOR,
                linewidth=0.5,
                zorder=2,
            )
            if width > 0:
                ax.annotate(
                    str(width),
                    xy=(left + width / 2, y),
                    ha="center",
                    va="center",
                    fontsize=VALUE_LABEL_FONTSIZE,
                    color="white" if alpha >= 0.7 else BASE_COLOR,
                )
            left += width

        prevention_rate = block["prevention_rate"]
        ax.annotate(
            f"{prevention_rate * 100:.1f}% prevented (n={n_violations})",
            xy=(n_violations, y),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=VALUE_LABEL_FONTSIZE,
            va="center",
            color=BASE_COLOR,
        )

    ax.set_yticks([y_positions[name] for name in H4_GROUP_ORDER])
    ax.set_yticklabels([name.capitalize() for name in H4_GROUP_ORDER])
    ax.set_xlim(0, 14.5)
    ax.set_xlabel("Violations (count)", fontsize=AXIS_LABEL_FONTSIZE)
    _style_axes(ax)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=BASE_COLOR, alpha=1.0, edgecolor=BASE_COLOR, label="prevented"),
        Patch(facecolor=BASE_COLOR, alpha=0.45, edgecolor=BASE_COLOR, label="detected too late"),
        Patch(
            facecolor=BASE_COLOR,
            alpha=0.15,
            hatch="///",
            edgecolor=BASE_COLOR,
            label="undetected",
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", frameon=False, fontsize=AXIS_LABEL_FONTSIZE
    )

    fig.suptitle(
        "Modeled counterfactual: detection outruns prevention",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    ax.set_title(
        "12 ground-truth violations, Phase 2c scripted run (modeled, not live)",
        fontsize=SUBTITLE_FONTSIZE,
        loc="left",
        color=BASE_COLOR,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _add_source_line(fig, H4_ARTIFACT, H4_COMMIT)
    return fig


# --- figure 3: smactr-before-after ------------------------------------------


def render_smactr_before_after(
    smactr_data: dict[str, Any],
    *,
    combined_before_prevented: int,
    combined_n_violations: int,
) -> Figure:
    """Paired bars: retail prevented count + prevention_rate before vs
    after the SMACTR fed-back rule, plus the combined (airline + retail)
    delta the retail gain produces. `combined_before_prevented` /
    `combined_n_violations` come from the H4 JSON's `combined` block (not
    this artifact) -- passed in explicitly rather than re-derived here."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES, dpi=DPI)

    before = smactr_data["before"]
    after = smactr_data["after"]
    prevented_gain = smactr_data["prevented_gain"]

    combined_after_prevented = combined_before_prevented + prevented_gain
    combined_before_rate = combined_before_prevented / combined_n_violations
    combined_after_rate = combined_after_prevented / combined_n_violations

    groups = [
        (
            "Retail",
            RETAIL_COLOR,
            before["prevented"],
            before["n_violations"],
            before["prevention_rate"],
            after["prevented"],
            after["n_violations"],
            after["prevention_rate"],
        ),
        (
            "Combined",
            BASE_COLOR,
            combined_before_prevented,
            combined_n_violations,
            combined_before_rate,
            combined_after_prevented,
            combined_n_violations,
            combined_after_rate,
        ),
    ]

    bar_width = 0.32
    x_positions = list(range(len(groups)))

    for x, (_name, color, b_prev, b_n, b_rate, a_prev, a_n, a_rate) in zip(
        x_positions, groups, strict=True
    ):
        ax.bar(
            x - bar_width / 2,
            b_rate,
            width=bar_width,
            color=color,
            alpha=0.45,
            edgecolor=color,
            zorder=2,
        )
        ax.bar(
            x + bar_width / 2,
            a_rate,
            width=bar_width,
            color=color,
            alpha=1.0,
            edgecolor=color,
            zorder=2,
        )
        ax.annotate(
            f"{b_rate * 100:.1f}%\n({b_prev}/{b_n})",
            xy=(x - bar_width / 2, b_rate),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_FONTSIZE,
            color=BASE_COLOR,
        )
        ax.annotate(
            f"{a_rate * 100:.1f}%\n({a_prev}/{a_n})",
            xy=(x + bar_width / 2, a_rate),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_FONTSIZE,
            color=BASE_COLOR,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([name for name, *_ in groups])
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Prevention rate", fontsize=AXIS_LABEL_FONTSIZE)
    _style_axes(ax)
    ax.grid(axis="x", visible=False)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=BASE_COLOR, alpha=0.45, edgecolor=BASE_COLOR, label="before SMACTR"),
        Patch(facecolor=BASE_COLOR, alpha=1.0, edgecolor=BASE_COLOR, label="after SMACTR"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=AXIS_LABEL_FONTSIZE)

    fig.suptitle(
        "A caught failure becomes a fast-path prevention",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    ax.set_title(
        "retail-008: detected-too-late -> prevented, after the fed-back rule",
        fontsize=SUBTITLE_FONTSIZE,
        loc="left",
        color=BASE_COLOR,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _add_source_line(fig, SMACTR_ARTIFACT, SMACTR_COMMIT)
    return fig


# --- figure 4: dir1-gate-save -----------------------------------------------


def render_dir1_gate_save(weak_data: dict[str, Any], structural_data: dict[str, Any]) -> Figure:
    """Horizontal before/after bars: the live gate-save. A compliant agent never
    crosses a structural boundary on these attacks (0 -- the gate stays silent);
    an under-specified (latency-optimized) agent crosses, and the two-speed gate
    prevents every crossing pre-execution, turning N unauthorized actions into 0.
    Combines the cancel_without_auth run (`weak_data`) and the
    refund_over_threshold run (`structural_data`)."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES, dpi=DPI)

    harm_off = weak_data["live_h4"]["harm_off"] + structural_data["live_h4"]["harm_off"]
    harm_on = weak_data["live_h4"]["harm_on"] + structural_data["live_h4"]["harm_on"]
    prevented = weak_data["live_h4"]["prevented"] + structural_data["live_h4"]["prevented"]

    rows = [
        ("Compliant agent", 0, BASE_COLOR, "0 crossings on the same 32 attacks -- gate silent"),
        (
            "Under-specified agent\n(no gate)",
            harm_off,
            RETAIL_COLOR,
            f"{harm_off} unauthorized actions execute",
        ),
        (
            "Under-specified agent\n(+ two-speed gate)",
            harm_on,
            AIRLINE_COLOR,
            f"{harm_on} -- all {prevented} blocked before execution",
        ),
    ]
    y_positions = list(reversed(range(len(rows))))  # first row on top
    bar_height = 0.55

    for y, (_label, value, color, note) in zip(y_positions, rows, strict=True):
        ax.barh(
            y,
            value,
            height=bar_height,
            color=color,
            alpha=0.9,
            edgecolor=BASE_COLOR,
            linewidth=0.5,
            zorder=2,
        )
        ax.annotate(
            note,
            xy=(value, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=VALUE_LABEL_FONTSIZE,
            color=BASE_COLOR,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for label, *_ in rows], fontsize=TICK_FONTSIZE)
    ax.set_xlim(0, harm_off + 5)
    ax.set_xlabel("Harmful actions that executed (count)", fontsize=AXIS_LABEL_FONTSIZE)
    _style_axes(ax)

    fig.suptitle(
        f"The live gate-save: governance turns {harm_off} unauthorized actions into {harm_on}",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    ax.set_title(
        "Retail, cancel + refund boundaries; the same attacks the compliant agent resisted",
        fontsize=SUBTITLE_FONTSIZE,
        loc="left",
        color=BASE_COLOR,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.text(
        0.01,
        0.01,
        f"Source: {DIR1_WEAK_ARTIFACT} + {DIR1_STRUCTURAL_ARTIFACT} (commit {DIR1_COMMIT})",
        fontsize=SOURCE_FONTSIZE,
        color=BASE_COLOR,
        ha="left",
        va="bottom",
    )
    return fig


# --- figure 5: interp-three-negatives ---------------------------------------


def _auroc_panel(
    ax: Any,
    bars: list[tuple[str, float, str]],
    *,
    title: str,
    subtitle: str,
) -> None:
    """One AUROC panel: labelled vertical bars on a fixed 0..1 scale, each
    annotated with its value. Shared by the three negatives so they read as
    one comparison."""
    positions = list(range(len(bars)))
    for x, (_label, value, color) in zip(positions, bars, strict=True):
        ax.bar(x, value, width=0.62, color=color, zorder=2)
        ax.annotate(
            f"{value:.2f}",
            xy=(x, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_FONTSIZE,
            color=BASE_COLOR,
        )
    ax.axhline(0.5, color=GRID_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for label, _v, _c in bars], fontsize=TICK_FONTSIZE)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    # Title (bold) stacked over a small grey subtitle, both left-aligned, so a
    # long title never collides with the subtitle.
    ax.text(
        0.0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        fontsize=SOURCE_FONTSIZE + 0.5,
        color=BASE_COLOR,
        ha="left",
        va="bottom",
    )
    ax.set_title(title, fontsize=SUBTITLE_FONTSIZE + 1, fontweight="bold", loc="left", pad=16)


def render_interp_three_negatives(
    reprobe: dict[str, Any],
    text_baseline: dict[str, Any],
    leadtime: dict[str, Any],
    evasion: dict[str, Any],
) -> Figure:
    """Three panels, one per honest negative, all on a shared AUROC scale:
    (1) the activation probe is present but does NOT beat a text baseline;
    (2) there is no lead-time -- at the first generated token the residual is
    at chance while the text already determines the outcome; (3) laundering
    does not fool the text monitor, so no evasion gap exists. Together: reading
    the model cannot independently catch the violation."""
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE_INCHES, dpi=DPI)

    # Panel 1: present, not beyond text (deep-layer AUROCs).
    layer = reprobe["layers"][INTERP_LAYER]
    act_auroc = layer["policy"]["auroc"]
    shuffled = layer["shuffled"]["auroc"]
    text_auroc = text_baseline["text_baseline_auroc"]
    _auroc_panel(
        axes[0],
        [
            ("activation\nprobe", act_auroc, AIRLINE_COLOR),
            ("text\nbaseline", text_auroc, RETAIL_COLOR),
            ("shuffled\nfloor", shuffled, BASE_COLOR),
        ],
        title="Present, not beyond text",
        subtitle=f"Qwen residual, layer {INTERP_LAYER}",
    )
    axes[0].set_ylabel("AUROC (violation vs compliant)", fontsize=AXIS_LABEL_FONTSIZE)

    # Panel 2: no lead-time (first generated token).
    offset0 = leadtime["layers"][0]["offsets"][0]
    residual0 = offset0["residual_auroc"]
    text0 = offset0["text_auroc"]
    _auroc_panel(
        axes[1],
        [
            ("residual\n@ token 0", residual0, AIRLINE_COLOR),
            ("text\n@ token 0", text0, RETAIL_COLOR),
        ],
        title="No lead-time",
        subtitle="the action IS the first token",
    )

    # Panel 3: no evasion gap (text AUROC barely moves under laundering).
    launderings = evasion["launderings"]
    euph = launderings["euphemism"]
    inj = launderings["injection"]
    _auroc_panel(
        axes[2],
        [
            ("euphemism\nlaundered", euph["laundered_text_auroc"], RETAIL_COLOR),
            ("injection\nlaundered", inj["laundered_text_auroc"], RETAIL_COLOR),
        ],
        title="No evasion gap",
        subtitle="clean text = 1.00 both",
    )

    fig.suptitle(
        "Reading the model cannot independently catch the violation",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        x=0.02,
        ha="left",
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.text(
        0.01,
        0.01,
        "Source: probes/interp/results/{reprobe_qwen_pca32, text_baseline_qwen, "
        "leadtime_report, evasion_gate_summary}.json (interp track)",
        fontsize=SOURCE_FONTSIZE,
        color=BASE_COLOR,
        ha="left",
        va="bottom",
    )
    return fig


# --- saving ------------------------------------------------------------------


def save_figure(fig: Figure, out_dir: Path, figure_id: str) -> tuple[Path, Path]:
    """Saves `fig` as both `<out_dir>/<figure_id>.svg` and `.png`, with
    settings chosen for byte-for-byte determinism across runs (no
    timestamp in the SVG, a fixed size/dpi already baked into `fig`).
    Returns (svg_path, png_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{figure_id}.svg"
    png_path = out_dir / f"{figure_id}.png"

    fig.savefig(svg_path, format="svg", metadata={"Date": None})
    # matplotlib's PNG writer embeds no wall-clock timestamp by default
    # (unlike the SVG writer, which stamps a <metadata> Date node unless
    # overridden above); Pillow/Agg's PNG backend writes no tEXt/tIME
    # chunk unless explicitly asked to, so no metadata override is needed
    # here -- verified via test_figures.py's byte-identical-across-runs
    # check, which would fail if a timestamp leaked in.
    fig.savefig(png_path, format="png")
    return svg_path, png_path
