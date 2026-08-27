# Detector datagen — calibration sample results (corpus_version=calib-t07)

Authorized by Matt 2026-08-27, ~$1-2 cap, calibration ONLY (measurement, not
run-2). Live Fireworks calls (`gpt-oss-20b`), `--temperature 0.7`,
`--variants-per-scenario 10`, `--workers 13`, per-domain caps `--max-usd 0.50
--max-calls 250 --max-wall-min 20 --cost-per-call-usd 0.002`. Purpose: measure
what the hermetic diversity fix (compositional phrasings + temperature
plumbing + corpus versioning, `detector-datagen-BE-partb-generation`
`d889cb4`..`b5c9f30`) actually buys on the live model, per
`phase-detector-training-step2-datagen.md` fix-plan item 2.

## Per-domain generation summaries (verbatim)

```
domain=retail total=100 already_done=0 completed=100 failed=0 timed_out=0 calls=133 est_usd=0.2660 elapsed_min=0.92 cap_hit=None decisions=133 violations=16 corpus_version=calib-t07
domain=airline total=100 already_done=0 completed=100 failed=0 timed_out=0 calls=130 est_usd=0.2600 elapsed_min=0.86 cap_hit=None decisions=130 violations=15 corpus_version=calib-t07
domain=advice-eligibility total=100 already_done=0 completed=100 failed=0 timed_out=0 calls=212 est_usd=0.4240 elapsed_min=1.12 cap_hit=None decisions=212 violations=52 corpus_version=calib-t07
```

No domain hit its cap. No auth failures, no failed/timed-out tasks, no
degenerate output observed (spot-checked completions: coherent tool calls and
refusals, varied compositional urgency phrasing, not the 4 fixed run-1
phrases).

## Assembly summaries (verbatim)

```
domain=retail n_input=133 exact_dropped=0 near_dropped=1 n_output=132 group_fields=('scenario_id', 'persona_id') n_groups=10
  train: n=90 violations=11
  val: n=27 violations=0
  test: n=15 violations=5
domain=airline n_input=130 exact_dropped=0 near_dropped=0 n_output=130 group_fields=('scenario_id',) n_groups=10
  train: n=92 violations=11
  val: n=15 violations=2
  test: n=23 violations=2
domain=advice-eligibility n_input=212 exact_dropped=0 near_dropped=2 n_output=210 group_fields=('scenario_id', 'persona_id') n_groups=9
  train: n=138 violations=29
  val: n=39 violations=13
  test: n=33 violations=10
```

## Measured rates vs run-1 (v1)

| domain | run-1 calls | run-1 unique | run-1 unique-rate | run-1 violation-rate | calib-t07 calls | calib-t07 unique | calib-t07 unique-rate | calib-t07 violation-rate |
|---|---|---|---|---|---|---|---|---|
| retail | ~2,820 | 50 | ~1.8% | ~4% | 133 | 132 | 99.2% | 12.1% (16/132) |
| airline | ~2,839 | 72 | ~2.5% | ~8% | 130 | 130 | 100.0% | 11.5% (15/130) |
| advice-eligibility | ~2,889 | 121 | ~4.2% | ~20% | 212 | 210 | 99.1% | 24.8% (52/210) |

Run-1 figures are as recorded in `phase-detector-training-step2-datagen.md`
Issues & Fixes (2026-08-26/27 entry): unique 50/72/121, violation rates
"~4%/8%/20%"; run-1 totals of 2820/2839/2889 are the pre-dedup call counts
implied by that entry (8,548 calls total, all three domains near-evenly
split) and are reproduced here only to derive the unique-rate deltas, not
re-measured in this task.

**The delta is enormous on unique-rate** (fix takes retail/airline/advice
from ~1.8%/2.5%/4.2% unique to ~99.2%/100.0%/99.1% unique — the temp-0 +
4-phrase collapse is gone) and **directionally consistent but higher on
violation-rate** for all three domains (12.1% vs ~4% retail, 11.5% vs ~8%
airline, 24.8% vs ~20% advice-eligibility). The calibration sample (130-212
calls/domain) is small, so these violation-rate deltas carry real sampling
noise — they are inputs to sizing below, not a re-confirmed base rate.

Cross-domain near-duplicate scan: 0/72,180 pairs (rate=0.0000). Leakage scan:
CLEAN (group_fields=('scenario_id', 'persona_id')).

## Sizing table (`scripts/size_run.py`, MEASURED calib-t07 rates)

Two cost assumptions: the `$0.002/call` placeholder (used throughout
generation/sizing so far) and `$0.0004/call` (realistic-band estimate per
`phase-detector-training-step2-datagen.md`'s 2026-08-27 entry; Matt to supply
the true Fireworks dashboard figure separately — this run's real metered
spend has not yet been checked against the dashboard).

| domain | target positives | unique decisions needed | calls needed | cost @ $0.002/call | cost @ $0.0004/call |
|---|---|---|---|---|---|
| retail | 150 | 1,238 | 1,248 | $2.50 | $0.50 |
| retail | 300 | 2,476 | 2,495 | $4.99 | $1.00 |
| airline | 150 | 1,301 | 1,301 | $2.60 | $0.52 |
| airline | 300 | 2,601 | 2,601 | $5.20 | $1.04 |
| advice-eligibility | 150 | 606 | 612 | $1.22 | $0.24 |
| advice-eligibility | 300 | 1,212 | 1,224 | $2.45 | $0.49 |

Inputs used: retail unique-rate=0.99248 (132/133), violation-rate=0.12121
(16/132); airline unique-rate=1.0 (130/130), violation-rate=0.11538
(15/130); advice-eligibility unique-rate=0.99057 (210/212), violation-rate=
0.24762 (52/210).

**Total run-2 cost across all three domains, target=150/cell:** ~$6.31 at
$0.002/call, ~$1.26 at $0.0004/call. **Target=300/cell:** ~$12.64 at
$0.002/call, ~$2.53 at $0.0004/call. All figures pending the real Fireworks
per-call rate and fresh spend authorization — run-2 is NOT authorized by this
task.

## Sanity checks

- **(a) corpus_version on every row:** confirmed — 133/133 retail, 130/130
  airline, 212/212 advice-eligibility rows carry `corpus_version=calib-t07`.
- **(b) mix-refusal fires against v1 data:** confirmed —
  `uv run python scripts/assemble_corpus.py --domain retail
  --decisions-path probes/detector/data/retail/decisions.jsonl
  --corpus-version calib-t07` exits 1 with `error: ... is corpus_version='v1'
  but --corpus-version='calib-t07' was requested; refusing to silently mix
  corpus versions`, and nothing is written. v1's `decisions.jsonl` mtime/size
  confirmed unchanged by this check (read-only).
- **(c) leakage scan on calibration splits:** CLEAN (see QC output above);
  cross-domain near-dup 0/72,180.

## Gates

`pytest tests/ --no-cov -q`: 1221 passed, 5 skipped.
`ruff format --check`: 259 files already formatted.
`ruff check`: all checks passed.
`mypy src/ scripts/`: 2 pre-existing errors, both in files untouched by this
task (`scripts/text_baseline.py`, `scripts/smoke_tool_call.py`) — confirmed
via `git diff --stat -- '*.py'` showing zero `.py` files changed by this
task; not a regression introduced here.

## Spend

475 live calls total (133 retail + 130 airline + 212 advice-eligibility) at
the $0.002/call placeholder: **$0.95 estimated**, against the ~$1-2
authorization and the $1.50 sum of the three domain caps. Real metered
Fireworks spend not yet checked against the dashboard (see sizing note
above).

## Data committed

Calibration manifests (`probes/detector/data/<domain>/calib-t07/
manifest.json`), assembled splits (`train.jsonl`/`val.jsonl`/`test.jsonl`),
QC hand-audit samples (`probes/detector/data/audit/calib-t07/
<domain>_sample50.jsonl`), and the raw `decisions.jsonl` per domain (2.7MB
total across all three domains — small enough per the ~5MB calibration-data
guideline, unlike run-1's 53MB raw file which stayed uncommitted). Total
committed data: ~6.3MB.
