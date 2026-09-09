#!/usr/bin/env python
"""H1 lead-time analysis: how early, before the action tool-call, is the outcome
decodable from the residual -- and does it BEAT the text-so-far baseline?

Off-pod, from the during-generation npz (`interp_capture_gen.py`). Per backward
offset and layer: probe residual@offset -> outcome and text-so-far@offset ->
outcome, with GROUP-CV BY PROMPT (all rollouts of a prompt share a fold, so the
probe cannot memorise a prompt's identity and leak its base rate -- the single
most important guard). Reports the per-offset AUROC curve for both, their paired
difference, and the earliest offset where the residual clears BOTH chance AND the
text-so-far baseline (the detection lead time).

Pure numpy; reuses the probe machinery. Group-k-fold is implemented here because
the base `crossval_oof_scores` stratifies by label, not group.
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
from bossyk_sandbox.interp.probe import _pca_apply, _pca_fit, probe_scores, train_probe

Array = NDArray[np.float64]


def hashing_vectorize(texts: list[str], *, n_features: int = 1024) -> Array:
    """Deterministic bag-of-words (md5 buckets), matching scripts/text_baseline."""
    x = np.zeros((len(texts), n_features), dtype=np.float64)
    for i, text in enumerate(texts):
        for tok in re.findall(r"[a-z0-9#]+", text.lower()):
            x[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % n_features] += 1.0
    return x


def group_oof_scores(
    x: Array,
    y: NDArray[np.bool_],
    groups: NDArray[np.intp],
    *,
    n_splits: int = 5,
    l2: float = 5.0,
    n_components: int | None = 32,
    seed: int = 0,
) -> Array:
    """Out-of-fold scores with GROUP k-fold: every row sharing a `groups` value is
    in the same test fold, so a probe never trains and tests on rollouts of the
    same prompt. Returns per-row OOF scores (NaN where a fold's train set is
    single-class or a row was never scored)."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = {g: i % n_splits for i, g in enumerate(rng.permutation(uniq))}
    assign = np.asarray([fold_of[g] for g in groups])
    oof = np.full(len(x), np.nan, dtype=np.float64)
    for f in range(n_splits):
        test = assign == f
        train = ~test
        if len(set(y[train].tolist())) < 2 or test.sum() == 0:
            continue
        xt, xe = x[train], x[test]
        if n_components is not None:
            mean, comps = _pca_fit(xt, n_components)
            xt, xe = _pca_apply(xt, mean, comps), _pca_apply(xe, mean, comps)
        probe = train_probe(xt, y[train], l2=l2)
        oof[test] = probe_scores(probe, xe)
    return oof


def _auroc(scores: Array, y: NDArray[np.bool_]) -> float:
    m = ~np.isnan(scores)
    return auroc(scores[m].tolist(), y[m].tolist()) if m.sum() else float("nan")


LABELS = ("is_violation", "called_tool")


def analyse(
    payload: Any,
    *,
    layer: int,
    n_splits: int = 5,
    seed: int = 0,
    label: str = "is_violation",
) -> dict[str, Any]:
    """Per-offset residual vs text-so-far AUROC (group-CV by prompt) at one layer.

    `label` selects the binary target column: `is_violation` (ours: will this
    rollout cross the policy) or `called_tool` (the pre-generation tool-call
    decoding literature's: will any tool be called; emitted since #16)."""
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}; use one of {LABELS}")
    offsets = [int(o) for o in payload["offsets"]]
    y = np.asarray(payload[label], dtype=bool)
    groups = np.asarray(payload["prompt_id"], dtype=np.intp)
    text = np.asarray(payload["text_so_far"])  # (n, n_offsets) str
    xall = np.asarray(payload[f"X_{layer}"], dtype=np.float64)  # (n, n_offsets, d)

    rows = []
    for oi, off in enumerate(offsets):
        xr = xall[:, oi, :]
        valid = ~np.isnan(xr).any(axis=1)  # rollouts long enough to have this offset
        if valid.sum() < 2 * n_splits or len(set(y[valid].tolist())) < 2:
            rows.append(
                {
                    "offset": off,
                    "n": int(valid.sum()),
                    "residual_auroc": float("nan"),
                    "text_auroc": float("nan"),
                    "residual_beats_text": False,
                }
            )
            continue
        s_res = group_oof_scores(xr[valid], y[valid], groups[valid], n_splits=n_splits, seed=seed)
        x_txt = hashing_vectorize([str(t) for t in text[valid, oi]])
        s_txt = group_oof_scores(x_txt, y[valid], groups[valid], n_splits=n_splits, seed=seed)
        a_res, a_txt = _auroc(s_res, y[valid]), _auroc(s_txt, y[valid])
        # paired bootstrap of the residual-text AUROC gap
        m = ~(np.isnan(s_res) | np.isnan(s_txt))
        yv = y[valid][m]
        rng = np.random.default_rng(seed)
        diffs = []
        sr, st = s_res[m], s_txt[m]
        for _ in range(2000):
            idx = rng.integers(0, m.sum(), m.sum())
            d = auroc(sr[idx].tolist(), yv[idx].tolist()) - auroc(
                st[idx].tolist(), yv[idx].tolist()
            )
            if not np.isnan(d):
                diffs.append(d)
        lo = float(np.percentile(diffs, 2.5)) if diffs else float("nan")
        rows.append(
            {
                "offset": off,
                "n": int(valid.sum()),
                "residual_auroc": a_res,
                "text_auroc": a_txt,
                "residual_minus_text_ci_lo": lo,
                "residual_beats_text": bool(lo > 0.0),
            }
        )
    return {"layer": layer, "label": label, "offsets": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H1 lead-time probe (group-CV vs text-so-far)")
    parser.add_argument("--acts", required=True, help="during-generation npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="7,14,27")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--label",
        choices=LABELS,
        default="is_violation",
        help="target column to probe (called_tool = the tool-call-decoding label, #16)",
    )
    args = parser.parse_args(argv)

    with np.load(args.acts, allow_pickle=False) as npz:
        payload = {k: npz[k] for k in npz.files}
    n = int(len(payload[args.label]))
    nv = int(np.asarray(payload[args.label]).sum())
    report: dict[str, Any] = {
        "n_rollouts": n,
        "label": args.label,
        "n_positive": nv,
        "n_violation": int(np.asarray(payload["is_violation"]).sum()),
        "n_prompts": int(len(set(np.asarray(payload["prompt_id"]).tolist()))),
        "layers": [
            analyse(payload, layer=int(x), seed=args.seed, label=args.label)
            for x in args.layers.split(",")
        ],
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(
        f"wrote {args.out}: {n} rollouts ({nv} positive on {args.label}) "
        f"over {report['n_prompts']} prompts"
    )
    for lay in report["layers"]:
        print(f"layer {lay['layer']}:  offset  residual  text   res>text")
        for r in lay["offsets"]:
            beats = " *" if r["residual_beats_text"] else ""
            print(f"   {r['offset']:>4}  {r['residual_auroc']:.3f}   {r['text_auroc']:.3f}{beats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
