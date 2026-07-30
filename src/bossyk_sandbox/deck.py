"""Exec-facing HTML slide deck rendered from `story/story.yaml` -- the same
single source of narrative truth the SPA and story_lint read. One slide per
act: the short `exec_copy` line of each claim (never the technical paragraph),
its evidence grade, and the act's committed figure inlined so the file is
self-contained. Deterministic: no wall-clock, no network -- re-rendering the
same story + figures reproduces the same bytes (see `tests/unit/test_deck.py`),
so the deck can never drift from the data the way a hand-built deck would.

This is the deck half of the demo/deck pipeline; it reuses `story.load_story`
and the committed `docs/figures/*.svg` rather than restating any of them."""

from __future__ import annotations

import html
from pathlib import Path

from bossyk_sandbox.story import EvidenceGrade, Story, StoryClaim

DECK_TITLE = "bossyk -- agent governance"
DECK_TAGLINE = (
    "Govern the effect an action would have. Attest the record. Do not try to read intent."
)

# Short, exec-legible label per grade (the story's own honesty axis, compressed
# for a slide) plus a stable CSS class so the stylesheet colours them.
_GRADE_LABEL: dict[EvidenceGrade, str] = {
    EvidenceGrade.LIVE_MEASUREMENT: "live measurement",
    EvidenceGrade.SCRIPTED_PROXY: "scripted proxy",
    EvidenceGrade.MODELED_COUNTERFACTUAL: "modeled counterfactual",
    EvidenceGrade.DETERMINISTIC_RECOMPUTE: "deterministic recompute",
    EvidenceGrade.OPEN: "open",
}


def _inline_svg(figures_dir: Path, figure_id: str) -> str:
    """Returns the `<svg>...</svg>` body of a committed figure, stripped of the
    XML declaration and DOCTYPE so it embeds directly in HTML5. Raises if the
    figure is missing -- a deck must never reference a figure that is not
    there."""
    text = (figures_dir / f"{figure_id}.svg").read_text()
    start = text.index("<svg")
    return text[start:]


def _figure_ids_for(claims: list[StoryClaim]) -> list[str]:
    """Every figure the act's claims reference, de-duplicated, in first-seen
    order (so slide order is stable, never set-iteration order)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for claim in claims:
        for figure_id in claim.figure_ids:
            if figure_id not in seen:
                seen.add(figure_id)
                ordered.append(figure_id)
    return ordered


def _claim_item(claim: StoryClaim) -> str:
    grade_label = _GRADE_LABEL[claim.evidence_grade]
    grade_class = claim.evidence_grade.value
    return (
        '<li class="claim">'
        f'<span class="chip chip--{grade_class}">{html.escape(grade_label)}</span>'
        f'<span class="claim__copy">{html.escape(claim.exec_copy)}</span>'
        "</li>"
    )


def _title_slide() -> str:
    return (
        '<section class="slide slide--title">'
        f'<h1 class="deck-title">{html.escape(DECK_TITLE)}</h1>'
        f'<p class="deck-tagline">{html.escape(DECK_TAGLINE)}</p>'
        "</section>"
    )


def _act_slide(
    act_title: str, act_number: int, question: str, claims: list[StoryClaim], figures_dir: Path
) -> str:
    items = "".join(_claim_item(claim) for claim in claims)
    figures = "".join(
        f'<figure class="slide__figure">{_inline_svg(figures_dir, figure_id)}</figure>'
        for figure_id in _figure_ids_for(claims)
    )
    return (
        '<section class="slide slide--act">'
        f'<span class="slide__kicker">Act {act_number}</span>'
        f'<h2 class="slide__title">{html.escape(act_title)}</h2>'
        f'<p class="slide__question">{html.escape(question)}</p>'
        f'<ul class="claims">{items}</ul>'
        f"{figures}"
        "</section>"
    )


def render_deck_html(story: Story, figures_dir: Path) -> str:
    """Renders the whole deck to a single self-contained HTML string: a title
    slide followed by one slide per act (in act order), each carrying its
    claims' exec copy, evidence chips, and inlined figures. `figures_dir` is
    where the committed `<figure_id>.svg` files live (normally docs/figures)."""
    acts = sorted(story.acts, key=lambda a: a.act)
    slides = [_title_slide()]
    for act in acts:
        act_claims = [claim for claim in story.claims if claim.act == act.act]
        slides.append(_act_slide(act.title, act.act, act.question, act_claims, figures_dir))
    slides_html = "".join(slides)

    return _PAGE_TEMPLATE.format(
        title=html.escape(DECK_TITLE),
        slides=slides_html,
        style=_STYLE,
        script=_SCRIPT,
    )


_STYLE = """
:root {
  --bg: #f7f7f9; --fg: #1d1d21; --muted: #5a5a63; --card: #ffffff;
  --line: #e2e2e8; --accent: #4477AA; --kicker: #4477AA;
  --live: #2f7d4f; --scripted: #a86b1f; --modeled: #7a5ea8;
  --recompute: #3a6ea5; --open: #99323b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1117; --fg: #e6e6ea; --muted: #9a9aa6; --card: #171a22;
    --line: #262a35; --accent: #6f9fd8; --kicker: #6f9fd8;
    --live: #5fbf85; --scripted: #d69a4c; --modeled: #b295d8;
    --recompute: #7aa8dd; --open: #e0737d;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body {
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}
.stage {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 3vh 4vw;
}
.slide { display: none; width: 100%; max-width: 960px; }
.slide.active { display: flex; flex-direction: column; gap: 14px; }
.slide--title { align-items: flex-start; }
.deck-title {
  font-size: clamp(28px, 5vw, 52px);
  margin: 0;
  font-weight: 800;
  letter-spacing: -0.02em;
}
.deck-tagline {
  font-size: clamp(16px, 2.4vw, 24px);
  color: var(--muted);
  margin: 8px 0 0;
  max-width: 40ch;
}
.slide__kicker {
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--kicker);
}
.slide__title {
  font-size: clamp(24px, 4vw, 40px);
  margin: 2px 0 0;
  font-weight: 800;
  letter-spacing: -0.02em;
}
.slide__question {
  font-size: clamp(15px, 2vw, 20px);
  color: var(--muted);
  font-style: italic;
  margin: 0 0 6px;
}
.claims {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.claim {
  display: flex;
  gap: 12px;
  align-items: baseline;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 12px 14px;
}
.claim__copy { font-size: clamp(14px, 1.9vw, 18px); line-height: 1.45; }
.chip {
  flex: none;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 999px;
  color: #fff;
  white-space: nowrap;
}
.chip--live-measurement { background: var(--live); }
.chip--scripted-proxy { background: var(--scripted); }
.chip--modeled-counterfactual { background: var(--modeled); }
.chip--deterministic-recompute { background: var(--recompute); }
.chip--open { background: var(--open); }
.slide__figure {
  margin: 6px 0 0;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 10px;
}
.slide__figure svg { width: 100%; height: auto; display: block; }
.bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 4vw;
  border-top: 1px solid var(--line);
  background: var(--bg);
}
.counter { color: var(--muted); font-size: 14px; font-variant-numeric: tabular-nums; }
.nav { display: flex; gap: 10px; }
.nav button {
  font: inherit;
  font-size: 15px;
  padding: 8px 18px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: var(--card);
  color: var(--fg);
  cursor: pointer;
}
.nav button:hover { border-color: var(--accent); }
.nav button:disabled { opacity: 0.4; cursor: default; }
"""

_SCRIPT = """
(function () {
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var counter = document.getElementById('counter');
  var prev = document.getElementById('prev');
  var next = document.getElementById('next');
  var i = 0;
  function show(n) {
    i = Math.max(0, Math.min(slides.length - 1, n));
    slides.forEach(function (s, k) { s.classList.toggle('active', k === i); });
    counter.textContent = (i + 1) + ' / ' + slides.length;
    prev.disabled = i === 0;
    next.disabled = i === slides.length - 1;
    window.scrollTo(0, 0);
  }
  prev.addEventListener('click', function () { show(i - 1); });
  next.addEventListener('click', function () { show(i + 1); });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight' || e.key === 'PageDown') show(i + 1);
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') show(i - 1);
    else if (e.key === 'Home') show(0);
    else if (e.key === 'End') show(slides.length - 1);
  });
  show(0);
})();
"""

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{style}</style>
</head>
<body>
<main class="stage">{slides}</main>
<div class="bar">
  <span class="counter" id="counter"></span>
  <div class="nav">
    <button id="prev" type="button">Back</button>
    <button id="next" type="button">Next</button>
  </div>
</div>
<script>{script}</script>
</body>
</html>
"""
