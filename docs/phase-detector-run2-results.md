# Detector datagen — run-2 scaled generation results (corpus_version=v2)

Authorized by Matt 2026-08-27: target 300 positives per domain, **$15 hard
cap total**, Fireworks `gpt-oss-20b`, `--temperature 0.7`,
`--corpus-version v2`. Completes step 2 of
`phase-detector-training-step2-datagen.md` (fix-plan item 3, following
run-1's collapse and the calib-t07 diversity-fix calibration).

## Sizing arithmetic

Inputs (measured, calib-t07, `docs/phase-detector-calib-sample-results.md`):
decisions-per-task 1.33 (retail) / 1.30 (airline) / 2.12 (advice); unique-rate
0.99248 / 1.0 / 0.99057; violation-rate 0.12121 / 0.11538 / 0.24762. Target:
>=300 violations among unique decisions per domain, with ~15% margin, 10
scenarios/domain.

For each domain: `unique_needed = 300 / violation_rate`, inflated 15% for
margin; `raw_decisions_needed = unique_needed / unique_rate`;
`tasks_needed = raw_decisions_needed / decisions_per_task`;
`variants_per_scenario = tasks_needed / 10`. Cross-checked against
`scripts/size_run.py --targets 345` (345 = 300*1.15), which returned
calls-needed 2869/2991/1408 — consistent with the manual arithmetic below.

| domain | unique needed (+15%) | raw decisions needed | tasks needed | variants/scenario (chosen) | expected decisions at chosen variants | vs sizing-table baseline (2495/2601/1224) |
|---|---|---|---|---|---|---|
| retail | 2,852 | 2,874 | 2,162 | **216** | 2,873 | 1.151x (within 20%) |
| airline | 3,000 | 3,000 | 2,308 | **230** | 2,990 | 1.150x (within 20%) |
| advice-eligibility | 1,392 | 1,406 | 664 | **67** | 1,420 | 1.160x (within 20%) |

Airline's variant count was trimmed from the arithmetic's raw 231 to 230 to
keep the expected decision volume (~2,990 calls x $0.002 = $5.98) safely
under the $6.00/3,000-call domain caps rather than landing right on the
boundary; this still delivered ~15% margin on the target.

Caps used: `--max-usd 6.00 --max-calls 3000` (retail, airline), `--max-usd
3.00 --max-calls 1500` (advice-eligibility), `--max-wall-min 120`
(per-domain ceiling; actual runs used the browser's 10-minute foreground-call
limit, chunked below), `--cost-per-call-usd 0.002` throughout. Sum of
per-domain `--max-usd` = $15.00, matching the total authorization.

## Foreground supervision note

`generate_corpus.py`'s cost/call cap accounting (`BudgetGuard`) is
per-invocation, not cumulative across resumed checkpoint runs, and each
Bash foreground call is capped at 10 minutes. Every domain was therefore run
as a sequence of foreground, blocking invocations of the same
`--variants-per-scenario`/`--corpus-version v2` command, each with
`--max-wall-min 8` so the process exits on its own before the tool timeout,
resuming from the on-disk checkpoint each time. Before each subsequent
invocation the remaining `--max-usd`/`--max-calls` were manually reduced by
the cumulative calls/spend already made, so the sum across all chunks for a
domain never exceeded that domain's authorized cap (airline's final chunk
overshot its own per-chunk sub-budget by $0.03 due to the worker pool
finishing an in-flight batch after the cap fired — see Issues below). No
command was ever backgrounded/detached; each call blocked the turn until it
returned.

## Per-domain generation summaries (verbatim)

```
# retail — chunk 1 (max-usd=6.00 max-calls=3000 max-wall-min=8)
domain=retail total=2160 already_done=0 completed=679 failed=0 timed_out=0 calls=984 est_usd=1.9680 elapsed_min=8.29 cap_hit=wall_min decisions=984 violations=183 corpus_version=v2

# retail — chunk 2 (max-usd=4.03 max-calls=2016 max-wall-min=8)
domain=retail total=2160 already_done=679 completed=1121 failed=0 timed_out=0 calls=1271 est_usd=2.5420 elapsed_min=8.62 cap_hit=wall_min decisions=2255 violations=246 corpus_version=v2

# retail — chunk 3 (max-usd=1.49 max-calls=745 max-wall-min=8) — FINISHED (all tasks done, no cap hit)
domain=retail total=2160 already_done=1800 completed=360 failed=0 timed_out=0 calls=498 est_usd=0.9960 elapsed_min=3.74 cap_hit=None decisions=2753 violations=338 corpus_version=v2

# airline — chunk 1 (max-usd=6.00 max-calls=3000 max-wall-min=8)
domain=airline total=2300 already_done=0 completed=972 failed=0 timed_out=0 calls=1445 est_usd=2.8900 elapsed_min=8.46 cap_hit=wall_min decisions=1445 violations=286 corpus_version=v2

# airline — chunk 2 (max-usd=3.11 max-calls=1555 max-wall-min=8)
domain=airline total=2300 already_done=972 completed=885 failed=0 timed_out=0 calls=1246 est_usd=2.4920 elapsed_min=8.35 cap_hit=wall_min decisions=2691 violations=387 corpus_version=v2

# airline — chunk 3 (max-usd=0.618 max-calls=309 max-wall-min=8) — CAP FIRED (usd)
domain=airline total=2300 already_done=1857 completed=306 failed=0 timed_out=0 calls=324 est_usd=0.6480 elapsed_min=2.08 cap_hit=usd decisions=3015 violations=396 corpus_version=v2

# advice-eligibility — chunk 1 (max-usd=3.00 max-calls=1500 max-wall-min=8) — FINISHED (all tasks done, no cap hit)
domain=advice-eligibility total=670 already_done=0 completed=670 failed=0 timed_out=0 calls=1495 est_usd=2.9900 elapsed_min=6.24 cap_hit=None decisions=1495 violations=356 corpus_version=v2
```

No auth failures. No failed/timed-out tasks in any chunk (0 across all
seven invocations).

**Domain outcomes:**
- **retail**: finished naturally, 2160/2160 tasks, no cap hit. Cumulative
  calls=2753, est spend=$5.506, violations=338.
- **airline**: `usd` cap fired, 2163/2300 tasks (94.0%). Cumulative
  calls=3015, est spend=$6.030 (target $6.00; $0.03 / 15 calls over, a
  concurrency-batch overshoot — see Issues), violations=396.
- **advice-eligibility**: finished naturally, 670/670 tasks, no cap hit.
  Cumulative calls=1495, est spend=$2.990, violations=356.

Per the brief: airline's cap fired before variant exhaustion; accepted as-is
(396 violations already well above the 300 floor), no budget raided from
another domain.

## Assembly summaries (verbatim)

```
domain=retail n_input=2753 exact_dropped=0 near_dropped=5 n_output=2748 group_fields=('scenario_id', 'persona_id') n_groups=10
  train: n=1872 violations=248
  val: n=544 violations=0
  test: n=332 violations=88
domain=airline n_input=3015 exact_dropped=0 near_dropped=3 n_output=3012 group_fields=('scenario_id',) n_groups=10
  train: n=2063 violations=226
  val: n=361 violations=85
  test: n=588 violations=85
domain=advice-eligibility n_input=1495 exact_dropped=0 near_dropped=17 n_output=1478 group_fields=('scenario_id', 'persona_id') n_groups=9
  train: n=1000 violations=244
  val: n=190 violations=47
  test: n=288 violations=64
```

## QC table

| domain | decisions (raw) | unique after dedup | unique-rate | violations (unique, total) | train n / viol | val n / viol | test n / viol | near-dup rate (within-domain) |
|---|---|---|---|---|---|---|---|---|
| retail | 2,753 | 2,748 | 99.82% | 336 | 1,872 / 248 | 544 / 0 | 332 / 88 | 0.1707 |
| airline | 3,015 | 3,012 | 99.90% | 396 | 2,063 / 226 | 361 / 85 | 588 / 85 | 0.0153 |
| advice-eligibility | 1,495 | 1,478 | 98.86% | 355 | 1,000 / 244 | 190 / 47 | 288 / 64 | 0.1570 |

All three domains clear the >=300-unique-violations target after dedup
(336 / 396 / 355).

**Leakage scan:** CLEAN for all three domains (grouped by
`scenario_id[, persona_id]` — same group never split across train/val/test).

**Cross-domain near-dup scan** (all three domains, one pass): 0/16,790,256
pairs, rate=0.0000.

## Confirmation overlap re-run (v2 GENERATED corpora)

`lexical_overlap_audit.py --decisions-root <symlinked v2 decisions.jsonl>`
(symlinks used only because the script's `--decisions-root` expects a flat
`<root>/<domain>/decisions.jsonl` layout and v2 data lives at
`<domain>/v2/decisions.jsonl`; no repo files touched).

| pair | tfidf cosine | top-k jaccard |
|---|---|---|
| retail <-> airline (control) | 0.7309 | 0.3636 |
| retail <-> advice-eligibility | 0.6034 | 0.1765 |
| airline <-> advice-eligibility | 0.5485 | 0.1765 |

**GATE PASSED.** Advice-crossing max (cosine 0.6034, jaccard 0.1765) stays
materially below the retail<->airline control (cosine 0.7309, jaccard
0.3636) on both metrics, margin=0.05. The pre-registered gate that passed on
run-1 sample text (2026-08-05/26) is confirmed on the actual v2 training
corpus. Matrix written to
`probes/detector/results/lexical_overlap_matrix_generated.json`.

## Power check — test-split positives vs 150/300

| domain | test-split violations | vs 150 | vs 300 |
|---|---|---|---|
| retail | 88 | below (-62) | below (-212) |
| airline | 85 | below (-65) | below (-215) |
| advice-eligibility | 64 | below (-86) | below (-236) |

As requested, this is the test-split-only count (test is deliberately the
smallest of the three splits by construction, ~12-20% of each domain's
unique decisions) — none of the three clear either floor on the test split
alone. The dossier's 150/300-positives-per-OOD-cell power floor is specified
against per-domain totals, and all three domains clear it there (336 / 396 /
355 total unique violations, see QC table). Flagging the distinction rather
than picking one reading silently.

## Gates

```
pytest tests/ --no-cov -q        -> 1221 passed, 5 skipped
ruff format --check .            -> 259 files already formatted
ruff check .                     -> All checks passed!
mypy --explicit-package-bases src/ tests/   -> Success: no issues found in 228 source files
```

## Spend

| domain | calls | est. spend @ $0.002/call | cap | outcome |
|---|---|---|---|---|
| retail | 2,753 | $5.506 | $6.00 | finished naturally |
| airline | 3,015 | $6.030 | $6.00 | usd cap fired (+$0.03 concurrency overshoot) |
| advice-eligibility | 1,495 | $2.990 | $3.00 | finished naturally |
| **total** | **7,263** | **$14.526** | **$15.00** | under cap |

Real metered Fireworks spend has not been checked against the dashboard
(same caveat as calib-t07 and run-1); $0.002/call is the placeholder used
throughout sizing and cap accounting for this task, per instruction.

## Data committed

Per-domain `v2/manifest.json`, `train.jsonl`, `val.jsonl`, `test.jsonl`;
`audit/v2/<domain>_sample50.jsonl` hand-audit exports; the updated
`probes/detector/results/lexical_overlap_matrix_generated.json`; this
report. Combined train/val/test size ~53.6MB, committed in full (the
~50MB guideline in the brief applies to the raw `decisions.jsonl`).

**Raw `v2/decisions.jsonl` left UNCOMMITTED**: retail 22M + airline 27M +
advice-eligibility 3.1M = ~52MB total, over the ~50MB guideline (mirrors
run-1's precedent, whose 53MB raw file also stayed uncommitted). Left in the
working tree, untracked.

## Issues & Fixes

1. **QC `--cap-hit` CLI misuse, self-caught and fixed.** First QC pass ran
   all three domains together with a single combined `--cap-hit
   "retail=none,airline=usd,advice-eligibility=none"` string; the CLI applies
   this value verbatim to every domain's manifest, so all three manifests
   incorrectly recorded that literal string and `cap_was_hit=True`. Re-ran
   QC once per domain with the correct individual value. Second mistake in
   the same fix: passing the literal string `"none"` is still truthy
   (`cap_was_hit = cap_hit is not None`), so retail/advice-eligibility still
   showed `cap_was_hit=True`. Final fix: omit `--cap-hit` entirely for the
   two domains that finished naturally (default `None` -> `cap_was_hit:
   false`), pass `--cap-hit usd` only for airline. Manifests now correct
   (verified: retail/advice `cap_hit: null, cap_was_hit: false`; airline
   `cap_hit: "usd", cap_was_hit: true`). The cross-domain near-dup number
   from the first combined pass is unaffected by this (dedup math doesn't
   depend on `--cap-hit`) and is the number reported above.
2. **retail val split has 0 violations** (train 248, test 88, val 0 of 544).
   Group-stratified splitting by `(scenario_id, persona_id)` does not
   guarantee every split gets positives when the group count is only 10;
   this run's random group assignment put none of retail's violating groups
   in val. Flagging for step 3/4 (baselines/training): val-based threshold
   tuning for retail will have no positives to calibrate against under this
   split. Not fixed here — reassembling with a different seed/stratification
   was out of scope for this task and reassembly is cheap/offline if wanted.
3. **Orphaned v1 checkpoint data confirmed pre-existing, untouched.** Before
   any spend, found large uncommitted `probes/detector/data/<domain>/
   decisions.jsonl` (root, no version folder — the v1 default path) already
   present: retail 23M/3082 lines, airline 26M/2871 lines, advice-eligibility
   6M/2954 lines, all newer than the committed run-1 manifests (`9ff9100`).
   This matches the documented "run-1's 53MB raw file, deliberately
   uncommitted" from the plan's Issues & Fixes and the calib doc. `--corpus-
   version v2` writes to a separate `<domain>/v2/` checkpoint path, so this
   run never read or appended to those files. Left untouched (no delete
   without confirmation); still sitting uncommitted in the working tree, not
   part of this task's deliverable.
4. **Airline's final chunk overshot its per-chunk sub-budget by $0.03/15
   calls** ($0.648 actual vs $0.618 requested for that chunk, 324 calls vs
   309 requested) — the cap check runs after a batch of up to `--workers 13`
   concurrent calls lands, so it can overshoot by a small in-flight margin
   before `cap_hit` is set. Cumulative domain total ($6.030 vs $6.00
   nominal) is 0.5% over its individual cap but the three-domain total
   ($14.526) stayed under the $15 authorization.
5. No stop conditions were triggered: no domain had zero violations in its
   first 500 decisions (first-chunk violations were 183/286/... all
   comfortably nonzero early), no tool-call parse failures, no auth
   failures, overlap confirmation passed, and cumulative spend stayed under
   the $15 cap.
