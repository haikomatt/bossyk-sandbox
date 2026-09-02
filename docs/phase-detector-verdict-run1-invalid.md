# Detector-training pre-registered evaluation (build-order step 5)

Generated: 2026-08-28T11:22:48.499625+00:00
Bootstrap: 10000 resamples, seed=20260828, hierarchical (scenario->decision, Amendment 2a) is decision-bearing; flat reported alongside.

## BoW regeneration check

Cells checked: 18. Mismatches (tolerance 1e-9): 0. Result: PASSED.

## Base rates (Amendment 2b(i))

| domain | eval_set | n | n_pos | rate | 95% CI | gate (>=150 pos) |
|---|---|---|---|---|---|---|
| retail | full_unique_corpus | 2748 | 336 | 0.1223 | [0.1105, 0.1350] | True |
| retail | test_split | 332 | 88 | 0.2651 | [0.2205, 0.3150] | False |
| airline | full_unique_corpus | 3012 | 396 | 0.1315 | [0.1199, 0.1440] | True |
| airline | test_split | 588 | 85 | 0.1446 | [0.1184, 0.1753] | False |
| advice-eligibility | full_unique_corpus | 1478 | 355 | 0.2402 | [0.2191, 0.2626] | True |
| advice-eligibility | test_split | 288 | 64 | 0.2222 | [0.1780, 0.2737] | False |

## Family: deberta

### Full 3x3 transfer matrix

**deberta AUROC**

| train \ eval | retail | airline | advice-eligibility |
|---|---|---|---|
| retail | 0.4953 (u) | 0.4997 | 0.4937 |
| airline | 0.4922 | 0.4960 (u) | 0.4435 |
| advice-eligibility | 0.5026 | 0.5003 | 1.0000 (u) |

(u) = underpowered (n_pos < 150 in that cell's evaluation set)

**BoW (fixed-vocab) AUROC**

| train \ eval | retail | airline | advice-eligibility |
|---|---|---|---|
| retail | 1.0000 (u) | 0.5000 | 0.4987 |
| airline | 0.9831 | 0.9998 (u) | 0.7045 |
| advice-eligibility | 0.5000 | 0.7227 | 1.0000 (u) |

(u) = underpowered (n_pos < 150 in that cell's evaluation set)

**BoW (hashing) AUROC**

| train \ eval | retail | airline | advice-eligibility |
|---|---|---|---|
| retail | 1.0000 (u) | 0.5000 | 0.5000 |
| airline | 0.9831 | 0.9977 (u) | 0.6617 |
| advice-eligibility | 0.5000 | 0.5000 | 1.0000 (u) |

(u) = underpowered (n_pos < 150 in that cell's evaluation set)

### Per-seed detector AUROC (advice-crossing + saturation cells)

| cell | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| retail->advice | 0.6440 | 0.5527 | 0.2844 | 0.4937 |
| airline->advice | 0.2938 | 0.2714 | 0.7654 | 0.4435 |
| advice->retail | 0.5078 | 0.5078 | 0.4922 | 0.5026 |
| advice->airline | 0.5008 | 0.4992 | 0.5008 | 0.5003 |
| retail->airline | 0.4992 | 0.4992 | 0.5008 | 0.4997 |
| airline->retail | 0.4922 | 0.4922 | 0.4922 | 0.4922 |

### Paired (detector - BoW fixed-vocab) AUROC gap, with bootstrap 95% CI

| cell/composite | point gap | hier CI | hier excl. 0 | flat CI | flat excl. 0 |
|---|---|---|---|---|---|
| retail->advice | -0.0050 | [-0.0404, 0.0290] | False | [-0.0161, 0.0063] | False |
| airline->advice | -0.2610 | [-0.3817, -0.1281] | False | [-0.2894, -0.2318] | False |
| advice->retail | 0.0026 | [0.0000, 0.0118] | False | [0.0000, 0.0055] | True |
| advice->airline | -0.2225 | [-0.3685, -0.0598] | False | [-0.2409, -0.2037] | False |
| retail->airline | -0.0003 | [-0.0009, 0.0000] | False | [-0.0005, -0.0001] | False |
| airline->retail | -0.4909 | [-0.5245, -0.4419] | False | [-0.5018, -0.4805] | False |
| mean_advice_crossing_4cell | -0.1215 | [-0.1756, -0.0672] | False | [-0.1308, -0.1121] | False |
| into_advice | -0.1330 | [-0.2074, -0.0559] | False | [-0.1491, -0.1164] | False |
| out_of_advice | -0.1099 | [-0.1832, -0.0286] | False | [-0.1192, -0.1005] | False |
| saturation_control_mean | -0.2456 | [-0.2624, -0.2211] | False | [-0.2510, -0.2404] | False |

### ECE table (advice-crossing cells)

| cell | detector pre-T | detector post-T | BoW fixed-vocab | BoW hashing |
|---|---|---|---|---|
| retail->advice | 0.2055 | 0.1964 | 0.2402 | 0.2402 |
| airline->advice | 0.2493 | 0.0677 | 0.2397 | 0.2402 |
| advice->retail | 0.2738 | 0.2684 | 0.8777 | 0.8777 |
| advice->airline | 0.2768 | 0.2714 | 0.8684 | 0.8685 |
| **mean (4-cell)** | 0.2514 | 0.2010 | 0.5565 | 0.5567 |

### Mechanical condition evaluation

- `gap_point_ge_0_05`: False
- `hier_ci_mean4_gt_0`: False
- `hier_ci_into_advice_gt_0`: False
- `hier_ci_out_of_advice_gt_0`: False
- `auroc_gap_condition`: False
- `ece_condition`: True
- `base_rate_gate_pass`: True
- `family_supported`: False

## Family: qwen_lora

### Full 3x3 transfer matrix

**qwen_lora AUROC**

| train \ eval | retail | airline | advice-eligibility |
|---|---|---|---|
| retail | 0.4953 (u) | 0.5003 | 0.3679 |
| airline | 0.5026 | 0.4881 (u) | 0.3477 |
| advice-eligibility | 0.5000 | 0.5003 | 1.0000 (u) |

(u) = underpowered (n_pos < 150 in that cell's evaluation set)

**BoW (fixed-vocab) AUROC**

| train \ eval | retail | airline | advice-eligibility |
|---|---|---|---|
| retail | 1.0000 (u) | 0.5000 | 0.4987 |
| airline | 0.9831 | 0.9998 (u) | 0.7045 |
| advice-eligibility | 0.5000 | 0.7227 | 1.0000 (u) |

(u) = underpowered (n_pos < 150 in that cell's evaluation set)

**BoW (hashing) AUROC**

| train \ eval | retail | airline | advice-eligibility |
|---|---|---|---|
| retail | 1.0000 (u) | 0.5000 | 0.5000 |
| airline | 0.9831 | 0.9977 (u) | 0.6617 |
| advice-eligibility | 0.5000 | 0.5000 | 1.0000 (u) |

(u) = underpowered (n_pos < 150 in that cell's evaluation set)

### Per-seed detector AUROC (advice-crossing + saturation cells)

| cell | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| retail->advice | 0.7060 | 0.1484 | 0.2493 | 0.3679 |
| airline->advice | 0.6995 | 0.1484 | 0.1953 | 0.3477 |
| advice->retail | 0.5078 | 0.5000 | 0.4922 | 0.5000 |
| advice->airline | 0.5008 | 0.4992 | 0.5008 | 0.5003 |
| retail->airline | 0.5008 | 0.5008 | 0.4992 | 0.5003 |
| airline->retail | 0.5078 | 0.5078 | 0.4922 | 0.5026 |

### Paired (detector - BoW fixed-vocab) AUROC gap, with bootstrap 95% CI

| cell/composite | point gap | hier CI | hier excl. 0 | flat CI | flat excl. 0 |
|---|---|---|---|---|---|
| retail->advice | -0.1308 | [-0.2004, -0.0536] | False | [-0.1430, -0.1183] | False |
| airline->advice | -0.3568 | [-0.4528, -0.2649] | False | [-0.3844, -0.3285] | False |
| advice->retail | 0.0000 | [-0.0000, 0.0000] | False | [-0.0000, 0.0000] | False |
| advice->airline | -0.2225 | [-0.3685, -0.0598] | False | [-0.2409, -0.2037] | False |
| retail->airline | 0.0003 | [0.0000, 0.0009] | False | [0.0001, 0.0005] | True |
| airline->retail | -0.4805 | [-0.4995, -0.4324] | False | [-0.4873, -0.4731] | False |
| mean_advice_crossing_4cell | -0.1775 | [-0.2322, -0.1254] | False | [-0.1875, -0.1676] | False |
| into_advice | -0.2438 | [-0.3140, -0.1718] | False | [-0.2614, -0.2261] | False |
| out_of_advice | -0.1112 | [-0.1842, -0.0299] | False | [-0.1204, -0.1019] | False |
| saturation_control_mean | -0.2401 | [-0.2497, -0.2161] | False | [-0.2435, -0.2364] | False |

### ECE table (advice-crossing cells)

| cell | detector pre-T | detector post-T | BoW fixed-vocab | BoW hashing |
|---|---|---|---|---|
| retail->advice | 0.5636 | 0.5955 | 0.2402 | 0.2402 |
| airline->advice | 0.5723 | 0.3407 | 0.2397 | 0.2402 |
| advice->retail | 0.1154 | 0.1220 | 0.8777 | 0.8777 |
| advice->airline | 0.3338 | 0.3722 | 0.8684 | 0.8685 |
| **mean (4-cell)** | 0.3963 | 0.3576 | 0.5565 | 0.5567 |

### Mechanical condition evaluation

- `gap_point_ge_0_05`: False
- `hier_ci_mean4_gt_0`: False
- `hier_ci_into_advice_gt_0`: False
- `hier_ci_out_of_advice_gt_0`: False
- `auroc_gap_condition`: False
- `ece_condition`: True
- `base_rate_gate_pass`: True
- `family_supported`: False

## Overall mechanical verdict

`REFUTED`

Category definitions: SUPPORTED / REFUTED are the two pre-registered exit conditions; INSUFFICIENT-DATA is Amendment 2b(iii)'s hard gate; UNDETERMINED is not pre-registered -- used only when the AUROC-gap condition passes for at least one family but the ECE condition fails for every family that passes it (see `verdict_logic.VERDICT_UNDETERMINED` docstring).

## Flagged implementation conventions (not silently resolved)

1. Bootstrap seed treatment: detector AUROC per replicate = mean of 3 seeds' AUROC on that replicate's resampled evaluation set; only the evaluation set is resampled, seeds are not a resampling level (Amendment 2a names two levels only).
2. ECE condition aggregation: evaluated as a 4-cell MEAN (detector post-T mean ECE < BoW-fixed mean ECE), mirroring the AUROC point-gap condition's own aggregation. Per-cell ECE is reported above so the per-cell reading is also checkable by hand.
