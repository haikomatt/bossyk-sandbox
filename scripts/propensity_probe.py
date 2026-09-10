#!/usr/bin/env python
"""H1a propensity probe (the caveated secondary in the H1 lead-time runbook).

The offset-0 (pre-generation) residual is byte-identical across a prompt's
rollouts, so under group-CV it can carry at most the model's PROPENSITY to
violate on that prompt, never the sampled outcome (RESULTS.md, "H1 re-run
2026-09-09"). This script asks the propensity question directly, which is the
regime the pre-generation tool-call decoding papers (arXiv 2605.09252,
2604.01202) work in: collapse each prompt to its one offset-0 vector and its
empirical violation RATE over K rollouts, then leave-one-prompt-out ridge
regression from residual -> rate, scored by Spearman on the held-out
predictions, against the same procedure on a bag-of-words of the PROMPT text.

n = number of prompts (26 in the 2026-09-09 capture), so only a large effect is
detectable and the paired bootstrap CI is wide by construction. Report the CI;
do not headline the point estimate. Off-pod, numpy only.

Usage:
    uv run python scripts/propensity_probe.py --acts <gen npz> \\
        --prompts probes/interp/results/borderline_prompts.json --out rep.json
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

Array = NDArray[np.float64]
LABELS = ("is_violation", "called_tool")


def hashing_vectorize(texts: list[str], *, n_features: int = 1024) -> Array:
    """Bag-of-words via feature hashing (same idiom as leadtime_probe)."""
    out = np.zeros((len(texts), n_features), dtype=np.float64)
    for i, t in enumerate(texts):
        for tok in re.findall(r"[a-z0-9_<>/]+", t.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % n_features
            out[i, h] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1.0, norms)


def collapse_to_prompts(
    payload: Any, *, layer: int, label: str = "is_violation"
) -> tuple[Array, Array, list[int]]:
    """One row per prompt: the offset-0 residual (asserted identical across the
    prompt's rollouts) and the prompt's empirical positive rate on `label`."""
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}; use one of {LABELS}")
    offsets = [int(o) for o in payload["offsets"]]
    oi = offsets.index(0)
    y = np.asarray(payload[label], dtype=bool)
    pid = np.asarray(payload["prompt_id"], dtype=np.intp)
    x0 = np.asarray(payload[f"X_{layer}"], dtype=np.float64)[:, oi, :]
    ids = sorted(set(pid.tolist()))
    rows, rates = [], []
    for p in ids:
        m = pid == p
        xs = x0[m]
        if np.nanmax(xs.std(axis=0)) > 1e-4:
            raise ValueError(f"offset-0 residual varies within prompt {p}; not the pre-gen state")
        rows.append(xs[0])
        rates.append(float(y[m].mean()))
    return np.stack(rows), np.asarray(rates), ids


def _spearman(a: Array, b: Array) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def loo_predict(x: Array, target: Array, *, l2: float = 10.0) -> Array:
    """Leave-one-out ridge (features standardised on the training fold)."""
    n = len(target)
    preds = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        mu, sd = x[tr].mean(axis=0), x[tr].std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        xt = (x[tr] - mu) / sd
        xe = (x[i] - mu) / sd
        yt = target[tr]
        ym = yt.mean()
        # dual form: w = X^T (X X^T + l2 I)^-1 (y - ym), cheap when d >> n
        k = xt @ xt.T
        alpha = np.linalg.solve(k + l2 * np.eye(len(yt)), yt - ym)
        preds[i] = float(xe @ (xt.T @ alpha)) + ym
    return preds


def loo_spearman(x: Array, target: Array, *, l2: float = 10.0) -> float:
    return _spearman(loo_predict(x, target, l2=l2), target)


def bootstrap_ci(
    pred_a: Array, target: Array, *, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(target)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = _spearman(pred_a[idx], target[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_diff_ci(
    pred_a: Array, pred_b: Array, target: Array, *, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap CI of Spearman(pred_a, target) - Spearman(pred_b, target) over
    the SAME resampled prompts, so the residual-vs-text comparison is paired."""
    rng = np.random.default_rng(seed)
    vals = []
    n = len(target)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = _spearman(pred_a[idx], target[idx]) - _spearman(pred_b[idx], target[idx])
        if not np.isnan(d):
            vals.append(d)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H1a propensity probe (offset-0 residual -> rate)")
    parser.add_argument("--acts", required=True, help="during-generation npz (anchor=start)")
    parser.add_argument("--prompts", required=True, help="prompts JSON, indexed by prompt_id")
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="7,14,27")
    parser.add_argument("--label", choices=LABELS, default="is_violation")
    parser.add_argument("--l2", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    with np.load(args.acts, allow_pickle=False) as npz:
        payload = {k: npz[k] for k in npz.files}
    prompts = [str(p) for p in json.loads(Path(args.prompts).read_text())]

    layers_out: list[dict[str, Any]] = []
    text_pred: Array | None = None
    for layer in (int(x) for x in args.layers.split(",")):
        xp, rates, ids = collapse_to_prompts(payload, layer=layer, label=args.label)
        if text_pred is None:
            xt = hashing_vectorize([prompts[i] for i in ids])
            text_pred = loo_predict(xt, rates, l2=args.l2)
            text_rho, text_ci = _spearman(text_pred, rates), bootstrap_ci(text_pred, rates)
        res_pred = loo_predict(xp, rates, l2=args.l2)
        layers_out.append(
            {
                "layer": layer,
                "residual_spearman": _spearman(res_pred, rates),
                "residual_ci": bootstrap_ci(res_pred, rates, seed=args.seed),
                "text_spearman": text_rho,
                "text_ci": text_ci,
                "residual_minus_text_ci": paired_diff_ci(
                    res_pred, text_pred, rates, seed=args.seed
                ),
            }
        )
    report = {
        "n_prompts": int(len(rates)),
        "label": args.label,
        "rates": [float(r) for r in rates],
        "l2": args.l2,
        "layers": layers_out,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}: {report['n_prompts']} prompts, label={args.label}")
    for row in layers_out:
        rc, tc, dc = row["residual_ci"], row["text_ci"], row["residual_minus_text_ci"]
        print(
            f"layer {row['layer']:>2}: residual rho={row['residual_spearman']:.3f} "
            f"CI[{rc[0]:.2f},{rc[1]:.2f}]  text rho={row['text_spearman']:.3f} "
            f"CI[{tc[0]:.2f},{tc[1]:.2f}]  paired diff CI[{dc[0]:.2f},{dc[1]:.2f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
