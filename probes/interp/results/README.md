# Rigorous activation-probe result — artifacts

See RESULTS.md for the finding and controls.

- `decisions_judged.json` — the 211-item labelled dataset (is_violation + is_error).
- `reprobe_pca64.json`, `reprobe_sweep.json` — CV+CI probe reports (scripts/reprobe.py).
- Activations `reportB.npz` (211 × 3584 × 5 layers, ~15MB) are NOT in git; kept in
  the developer's `.interp_data/`. Regenerate via scripts/interp_capture.py on a
  torch-2.4 + nnsight-0.3.7 pod (see the vault runbook), or re-analyse the npz
  off-pod with scripts/reprobe.py.

Repro (off-pod, from the npz):
    uv run python scripts/reprobe.py --acts reportB.npz --out out.json \
      --n-components 32 --l2 5 --n-splits 5 --n-boot 2000
