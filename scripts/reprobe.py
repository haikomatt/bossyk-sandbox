#!/usr/bin/env python
"""Off-pod re-probe from persisted activations -- the authoritative analysis.

Loads the activations npz `scripts/interp_capture.py` saved on the pod and runs
the split-robust CV probe (stratified k-fold + optional PCA + bootstrap CI) for
policy vs shuffled vs the coherence-error confound, per layer. No GPU, no
network: this is the whole reason the capture persists activations -- a
methodology change (PCA comps, L2, folds) re-runs here for free, never re-bills a
pod.

A policy result is real only if its CI clears BOTH controls' CIs at some layer.

Usage:
    uv run python scripts/reprobe.py --acts report.npz --out reprobe.json \\
        --n-components 64 --l2 5 --n-splits 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bossyk_sandbox.interp.probe import probe_cv_report


def prepare_labels(
    is_violation: NDArray[np.bool_],
    is_error: NDArray[np.int8],
    *,
    seed: int = 0,
) -> tuple[NDArray[np.bool_], dict[str, list[bool]]]:
    """Build the label vectors to probe and the row mask they share.

    `is_error` uses -1 for 'unlabelled'. To keep policy / shuffled / error on the
    IDENTICAL rows (so their AUROCs are comparable), when only some items carry a
    coherence label we restrict every probe to the labelled rows. When none do,
    the error confound is simply omitted. The shuffled floor is a permutation of
    the policy label on the same rows."""
    known = is_error != -1
    mask = known if (known.any() and not known.all()) else np.ones(len(is_violation), dtype=bool)

    policy = is_violation[mask].astype(bool)
    shuffled = np.random.default_rng(seed).permutation(policy)
    labels: dict[str, list[bool]] = {
        "policy": policy.tolist(),
        "shuffled": shuffled.tolist(),
    }
    if known.any():
        labels["error"] = (is_error[mask] == 1).tolist()
    return mask, labels


def reprobe_from_arrays(
    payload: Any,
    *,
    n_splits: int = 5,
    l2: float = 1.0,
    n_components: int | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the CV probe report per layer from a loaded npz payload (any mapping
    with `layers`, `is_violation`, `is_error`, and `X_<layer>` keys). Pure numpy;
    the unit test drives it with synthetic arrays."""
    layers = [int(v) for v in payload["layers"]]
    is_violation = np.asarray(payload["is_violation"], dtype=bool)
    is_error = np.asarray(payload["is_error"], dtype=np.int8)
    mask, labels = prepare_labels(is_violation, is_error, seed=seed)

    per_layer: dict[str, Any] = {}
    for layer in layers:
        x = np.asarray(payload[f"X_{layer}"], dtype=np.float64)[mask]
        per_layer[str(layer)] = probe_cv_report(
            x,
            labels,
            n_splits=n_splits,
            l2=l2,
            n_components=n_components,
            n_boot=n_boot,
            seed=seed,
        )
    return {
        "n_items": int(len(is_violation)),
        "n_used": int(mask.sum()),
        "n_violation": int(is_violation.sum()),
        "has_error_label": "error" in labels,
        "params": {
            "n_splits": n_splits,
            "l2": l2,
            "n_components": n_components,
            "n_boot": n_boot,
            "seed": seed,
        },
        "layers": per_layer,
    }


def _fmt(cell: dict[str, float]) -> str:
    return f"{cell['auroc']:.3f} [{cell['ci_lo']:.3f},{cell['ci_hi']:.3f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="off-pod CV re-probe from persisted activations")
    parser.add_argument("--acts", required=True, help="activations npz from interp_capture.py")
    parser.add_argument("--out", required=True, help="path to write the re-probe report JSON")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument(
        "--n-components", type=int, default=None, help="PCA components (default: no PCA)"
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    with np.load(args.acts, allow_pickle=False) as npz:
        payload = {key: npz[key] for key in npz.files}

    report = reprobe_from_arrays(
        payload,
        n_splits=args.n_splits,
        l2=args.l2,
        n_components=args.n_components,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(
        f"wrote {args.out}: n_used={report['n_used']}/{report['n_items']} "
        f"({report['n_violation']} violation), params={report['params']}"
    )
    names = list(next(iter(report["layers"].values())).keys())
    print("layer  " + "  ".join(f"{n:>22}" for n in names))
    for layer, cells in report["layers"].items():
        print(f"{layer:>5}  " + "  ".join(f"{_fmt(cells[n]):>22}" for n in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
