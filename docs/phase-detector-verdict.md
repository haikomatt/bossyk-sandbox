# Detector-training pre-registered evaluation (build-order step 5)

Generated: 2026-08-30T00:47:14.562191+00:00
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
| retail | 1.0000 (u) | 0.9737 | 0.9170 |
| airline | 1.0000 | 1.0000 (u) | 0.8266 |
| advice-eligibility | 0.9839 | 0.8980 | 1.0000 (u) |

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
| retail->advice | 0.9916 | 0.8008 | 0.9585 | 0.9170 |
| airline->advice | 0.7404 | 0.9000 | 0.8395 | 0.8266 |
| advice->retail | 0.9549 | 0.9995 | 0.9973 | 0.9839 |
| advice->airline | 0.8906 | 0.9427 | 0.8605 | 0.8980 |
| retail->airline | 0.9831 | 0.9502 | 0.9880 | 0.9737 |
| airline->retail | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Paired (detector - BoW fixed-vocab) AUROC gap, with bootstrap 95% CI

| cell/composite | point gap | hier CI | hier excl. 0 | flat CI | flat excl. 0 |
|---|---|---|---|---|---|
| retail->advice | 0.4183 | [0.3940, 0.4468] | True | [0.4093, 0.4271] | True |
| airline->advice | 0.1221 | [0.0442, 0.1952] | True | [0.0984, 0.1464] | True |
| advice->retail | 0.4839 | [0.4711, 0.4990] | True | [0.4759, 0.4912] | True |
| advice->airline | 0.1752 | [0.0492, 0.3148] | True | [0.1573, 0.1936] | True |
| retail->airline | 0.4737 | [0.4601, 0.4883] | True | [0.4697, 0.4777] | True |
| airline->retail | 0.0169 | [0.0001, 0.0642] | True | [0.0108, 0.0238] | True |
| mean_advice_crossing_4cell | 0.2999 | [0.2613, 0.3407] | True | [0.2919, 0.3079] | True |
| into_advice | 0.2702 | [0.2300, 0.3096] | True | [0.2577, 0.2827] | True |
| out_of_advice | 0.3296 | [0.2666, 0.4006] | True | [0.3197, 0.3395] | True |
| saturation_control_mean | 0.2453 | [0.2327, 0.2703] | True | [0.2416, 0.2492] | True |

### ECE table (advice-crossing cells)

| cell | detector pre-T | detector post-T | BoW fixed-vocab | BoW hashing |
|---|---|---|---|---|
| retail->advice | 0.1570 | 0.1602 | 0.2402 | 0.2402 |
| airline->advice | 0.2427 | 0.2500 | 0.2397 | 0.2402 |
| advice->retail | 0.1247 | 0.1248 | 0.8777 | 0.8777 |
| advice->airline | 0.2177 | 0.2175 | 0.8684 | 0.8685 |
| **mean (4-cell)** | 0.1855 | 0.1881 | 0.5565 | 0.5567 |

### Mechanical condition evaluation

- `gap_point_ge_0_05`: True
- `hier_ci_mean4_gt_0`: True
- `hier_ci_into_advice_gt_0`: True
- `hier_ci_out_of_advice_gt_0`: True
- `auroc_gap_condition`: True
- `ece_condition`: True
- `base_rate_gate_pass`: True
- `family_supported`: True

## Family: qwen_lora

### Full 3x3 transfer matrix

**qwen_lora AUROC**

| train \ eval | retail | airline | advice-eligibility |
|---|---|---|---|
| retail | 1.0000 (u) | 0.9989 | 0.9181 |
| airline | 0.9990 | 1.0000 (u) | 0.9577 |
| advice-eligibility | 0.8598 | 0.7385 | 1.0000 (u) |

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
| retail->advice | 0.9958 | 0.9776 | 0.7809 | 0.9181 |
| airline->advice | 0.9757 | 0.9556 | 0.9418 | 0.9577 |
| advice->retail | 0.9952 | 0.8760 | 0.7083 | 0.8598 |
| advice->airline | 0.9901 | 0.7286 | 0.4967 | 0.7385 |
| retail->airline | 0.9992 | 0.9985 | 0.9990 | 0.9989 |
| airline->retail | 1.0000 | 0.9999 | 0.9972 | 0.9990 |

### Paired (detector - BoW fixed-vocab) AUROC gap, with bootstrap 95% CI

| cell/composite | point gap | hier CI | hier excl. 0 | flat CI | flat excl. 0 |
|---|---|---|---|---|---|
| retail->advice | 0.4194 | [0.3942, 0.4493] | True | [0.4108, 0.4283] | True |
| airline->advice | 0.2532 | [0.1653, 0.3423] | True | [0.2283, 0.2776] | True |
| advice->retail | 0.3598 | [0.3195, 0.3905] | True | [0.3481, 0.3714] | True |
| advice->airline | 0.0158 | [-0.1846, 0.1948] | False | [-0.0058, 0.0373] | False |
| retail->airline | 0.4989 | [0.4966, 0.4996] | True | [0.4984, 0.4993] | True |
| airline->retail | 0.0159 | [-0.0015, 0.0631] | False | [0.0098, 0.0229] | True |
| mean_advice_crossing_4cell | 0.2621 | [0.2060, 0.3117] | True | [0.2533, 0.2708] | True |
| into_advice | 0.3363 | [0.2966, 0.3766] | True | [0.3243, 0.3485] | True |
| out_of_advice | 0.1878 | [0.0852, 0.2791] | True | [0.1755, 0.2003] | True |
| saturation_control_mean | 0.2574 | [0.2484, 0.2811] | True | [0.2543, 0.2609] | True |

### ECE table (advice-crossing cells)

| cell | detector pre-T | detector post-T | BoW fixed-vocab | BoW hashing |
|---|---|---|---|---|
| retail->advice | 0.1913 | 0.2007 | 0.2402 | 0.2402 |
| airline->advice | 0.1509 | 0.1603 | 0.2397 | 0.2402 |
| advice->retail | 0.1186 | 0.1341 | 0.8777 | 0.8777 |
| advice->airline | 0.1447 | 0.1673 | 0.8684 | 0.8685 |
| **mean (4-cell)** | 0.1514 | 0.1656 | 0.5565 | 0.5567 |

### Mechanical condition evaluation

- `gap_point_ge_0_05`: True
- `hier_ci_mean4_gt_0`: True
- `hier_ci_into_advice_gt_0`: True
- `hier_ci_out_of_advice_gt_0`: True
- `auroc_gap_condition`: True
- `ece_condition`: True
- `base_rate_gate_pass`: True
- `family_supported`: True

## Overall mechanical verdict

`SUPPORTED`

Category definitions: SUPPORTED / REFUTED are the two pre-registered exit conditions; INSUFFICIENT-DATA is Amendment 2b(iii)'s hard gate; UNDETERMINED is not pre-registered -- used only when the AUROC-gap condition passes for at least one family but the ECE condition fails for every family that passes it (see `verdict_logic.VERDICT_UNDETERMINED` docstring).

## Flagged implementation conventions (not silently resolved)

1. Bootstrap seed treatment: detector AUROC per replicate = mean of 3 seeds' AUROC on that replicate's resampled evaluation set; only the evaluation set is resampled, seeds are not a resampling level (Amendment 2a names two levels only).
2. ECE condition aggregation: evaluated as a 4-cell MEAN (detector post-T mean ECE < BoW-fixed mean ECE), mirroring the AUROC point-gap condition's own aggregation. Per-cell ECE is reported above so the per-cell reading is also checkable by hand.
