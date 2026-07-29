#!/usr/bin/env python
"""H2 text-baseline control for the pre-statement (T2) probe claim.

The activation probe reads the residual at the LAST TOKEN OF THE PRECEDING
CONTEXT (before any intent/action tokens) -- which is drift-probe timepoint **T2**
(`drift-probe-experiment.md` §3), the "pre-statement detectability" position. For
that to mean the residual encodes the impending violation *beyond what the text
reveals*, the activation probe must BEAT a text classifier reading the same
context (drift-probe success criterion: +0.10 AUROC over the text baseline).

This builds a deterministic bag-of-words text probe over the decision CONTEXT
prompts (no action) and compares it, via the same CV + paired bootstrap the
activation probe uses, against the persisted activations. Pure numpy + stdlib
hashing -- no sklearn, no network, off-pod.

Finding on the 2026-07-29 sets: the text baseline is ~0.99 (the label is nearly
determined by the request text under aggressive weakening), so the activation
probe does NOT beat it -- the linear-decodability result stands, but
"pre-statement BEYOND text" is NOT established on this dataset. See RESULTS.md.

Usage:
    uv run python scripts/text_baseline.py --acts reportC.npz --decisions decisions.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bossyk_sandbox.interp.correlate import auroc
from bossyk_sandbox.interp.probe import crossval_oof_scores

Array = NDArray[np.float64]


def hashing_vectorize(texts: list[str], *, n_features: int = 2048) -> Array:
    """Deterministic bag-of-words hashing vectorizer (md5 -> bucket, so it is
    stable across processes, unlike Python's salted hash). Word tokens only. A
    standard, dependency-free text baseline -- not maximal (a fine-tuned encoder
    would be stronger), but a fair floor for 'is the label in the text'."""
    x = np.zeros((len(texts), n_features), dtype=np.float64)
    for i, text in enumerate(texts):
        for tok in re.findall(r"[a-z0-9#]+", text.lower()):
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % n_features
            x[i, bucket] += 1.0
    return x


def _cv_scores(x: Array, y: NDArray[np.bool_], *, n_components: int, l2: float, seed: int) -> Array:
    scores, _ = crossval_oof_scores(x, y, n_splits=5, l2=l2, n_components=n_components, seed=seed)
    return scores


def _auroc(scores: Array, y: NDArray[np.bool_]) -> float:
    m = ~np.isnan(scores)
    return auroc(scores[m].tolist(), y[m].tolist()) if m.sum() else float("nan")


def paired_diff_ci(
    scores_a: Array,
    scores_b: Array,
    y: NDArray[np.bool_],
    *,
    n_boot: int = 3000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap of AUROC(a) - AUROC(b) on the shared valid rows: returns
    (ci_lo, ci_hi, P(a>b)). Paired because both probes score the SAME items."""
    m = ~(np.isnan(scores_a) | np.isnan(scores_b))
    a, b, yv = scores_a[m], scores_b[m], y[m]
    n = int(m.sum())
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        da, db = auroc(a[idx].tolist(), yv[idx].tolist()), auroc(b[idx].tolist(), yv[idx].tolist())
        if not (np.isnan(da) or np.isnan(db)):
            diffs.append(da - db)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(np.asarray(diffs) > 0))


def run(
    acts_path: str,
    decisions_path: str,
    *,
    layers: list[int],
    n_components: int = 32,
    l2: float = 5.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare the activation probe against the text baseline at each layer."""
    z = np.load(acts_path, allow_pickle=False)
    rows = json.loads(Path(decisions_path).read_text())
    y = z["is_violation"].astype(bool)
    prompts = [str(r["prompt"]) for r in rows]
    if len(prompts) != len(y):
        raise ValueError(f"decisions ({len(prompts)}) != activations ({len(y)})")

    x_text = hashing_vectorize(prompts)
    s_text = _cv_scores(x_text, y, n_components=n_components, l2=l2, seed=seed)
    text_auroc = _auroc(s_text, y)
    ys = np.random.default_rng(seed + 7).permutation(y)
    text_shuffled = _auroc(_cv_scores(x_text, ys, n_components=n_components, l2=l2, seed=seed), ys)

    per_layer: dict[str, Any] = {}
    for layer in layers:
        s_act = _cv_scores(
            z[f"X_{layer}"].astype(np.float64), y, n_components=n_components, l2=l2, seed=seed
        )
        lo, hi, p = paired_diff_ci(s_act, s_text, y, seed=seed)
        per_layer[str(layer)] = {
            "activation_auroc": _auroc(s_act, y),
            "act_minus_text_ci": [lo, hi],
            "p_act_gt_text": p,
            "beats_text": bool(lo > 0.0),
        }
    return {
        "n_items": int(len(y)),
        "n_violation": int(y.sum()),
        "text_baseline_auroc": text_auroc,
        "text_shuffled_auroc": text_shuffled,
        "layers": per_layer,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H2 text-baseline control for the T2 probe")
    parser.add_argument("--acts", required=True, help="activations npz")
    parser.add_argument("--decisions", required=True, help="decisions JSON (for context prompts)")
    parser.add_argument("--layers", default="7,14,27")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    layers = [int(p) for p in args.layers.split(",") if p.strip()]
    report = run(args.acts, args.decisions, layers=layers)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(
        f"text baseline AUROC {report['text_baseline_auroc']:.3f} "
        f"(shuffled {report['text_shuffled_auroc']:.3f}) over n={report['n_items']}"
    )
    for layer, cell in report["layers"].items():
        lo, hi = cell["act_minus_text_ci"]
        beats = "BEATS text" if cell["beats_text"] else "does NOT beat text"
        print(
            f"  layer {layer}: activation {cell['activation_auroc']:.3f} | "
            f"act-text [{lo:+.3f},{hi:+.3f}] P={cell['p_act_gt_text']:.0%} -> {beats}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
