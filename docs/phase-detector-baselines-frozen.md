# Detector-training baselines -- FROZEN (build-order step 3)

Dated 2026-08-27. This is the pre-training baseline freeze required by
`phase-detector-training.md` build-order step 3: "Baselines, frozen first:
bag-of-words + logreg (existing), zero-shot LLM judge, majority class. **No
training until baselines are locked in a results table.**" No detector of
any kind has been trained; this document and its machine-readable twin
(`probes/detector/results/baselines_frozen.json`) are that lock.

Evaluated on the v2 corpus (`docs/phase-detector-run2-results.md`), against
the pre-registered dossier (vault path
`Hypotheses/a-fine-tuned-violation-detector-transfers-cross-domain.md`),
**including Amendment 2's evaluation-set definition** (Amendment 2 is
recorded there as DRAFT/awaiting Matt's ruling, but its evaluation-set
correction is the one this freeze uses -- see "Design decisions" below):

- **In-domain** cell = the tested domain's TEST split only.
- **OOD** cell (train-A / test-B, A != B) = domain B's **FULL unique
  corpus** (train+val+test) -- not B's test split. Rationale (quoted from the
  dossier's 2026-08-27 wording revision): "in a train-on-A/test-on-B cell, B
  is never trained on by that detector, so restricting to B's test split
  discards evidence for no leakage benefit."
- **Power gate** (Amendment 2biii): a cell needs >= 150 positives to count
  toward a supported/refuted verdict. Formally the hard gate applies to OOD
  cells (all three domains' full corpora clear it: 336/396/355 positives).
  In-domain test-split positives are also reported against the same 150
  floor for transparency even though the amendment does not gate them -- all
  three fall short by construction (test is deliberately the smallest
  split), which is expected, not a failure of this freeze.

## Design decisions the brief left open

1. **Judge baseline has no train-domain axis.** The BoW+logreg baseline is
   fit per train-domain and evaluated on every eval-domain (a genuine 3x3
   transfer matrix). The zero-shot judge has no fitted parameters -- it is
   always given the CORRECT domain's own policy YAML when judging that
   domain's decisions, so there is no "trained on A, tested on B" question
   for it to answer. The judge is therefore reported **per domain (3
   cells)**, not as a 3x3 matrix. The brief's "$6 cap ... ~2.2k calls total"
   sizing (ALL positives + equal negatives per domain, drawn from each
   domain's full unique corpus) confirms this reading: 336+396+355 = 1,087
   positives, doubled = 2,174 ~= 2.2k, which only reconciles under a
   per-domain (not per-cell-of-a-matrix) sampling frame.
2. **Decision text = context (`prompt`) + the action taken.** Both the BoW
   vectorisers and the judge are given the agent's action, not just the
   customer's request -- the policy label is about the ACTION (skip
   verification vs. verify first), so a baseline reading only the request
   would be handicapped in a way that doesn't test "is the label in the
   text", it tests "is the label in half the text". `scripts/text_baseline.py`
   (a different, pre-existing probe control at drift-probe timepoint T2,
   deliberately BEFORE the action) reads prompt-only by design; that is a
   different question from this freeze's and is unaffected.
3. **Amendment 2's hierarchical bootstrap (2a) is NOT applied here.** This
   freeze reports point AUROC/ECE/FPR/FNR only, no confidence intervals -- CIs are a build-order step 5 (detector transfer evaluation) requirement,
   compared against a trained detector's paired gap. The baseline table
   itself does not need a bootstrap to be "frozen"; the numbers below are
   exactly reproducible from the fixed seed and are what step 5 must beat.

## 1. Domain summary (v2 corpus)

| domain | train n / pos | test n / pos | full n / pos |
|---|---|---|---|
| retail | 1,872 / 248 | 332 / 88 | 2,748 / 336 |
| airline | 2,063 / 226 | 588 / 85 | 3,012 / 396 |
| advice-eligibility | 1,000 / 244 | 288 / 64 | 1,478 / 355 |

Matches `docs/phase-detector-run2-results.md`'s QC table exactly (cross-check,
not a new generation).

## 2. BoW + logistic regression (free, local)

Config: seed=0, L2 grid `[0.3, 1.0, 3.0, 10.0, 30.0]` chosen by 5-fold
stratified CV **within the train split only** (never touches val/test/OOD);
fixed-vocab vectoriser `max_features=4096, min_df=2` (fit on train text
only, OOV dropped elsewhere); hashing vectoriser `n_features=4096` (md5
bucket, unbounded vocabulary). No temperature scaling -- raw probabilities
throughout, per pre-registration. Chosen L2 per train-domain (identical for
both vectorisers): retail=0.3, airline=3.0, advice-eligibility=0.3.

### 2a. Fixed-vocabulary vectoriser

| train \\ eval | retail | airline | advice-eligibility |
|---|---|---|---|
| **retail** | AUROC 1.000, ECE 0.012, n_pos 88/332 **(<150)** | AUROC 0.500, ECE 0.131, n_pos 396/3,012 | AUROC 0.499, ECE 0.240, n_pos 355/1,478 |
| **airline** | AUROC 0.983, ECE 0.122, n_pos 336/2,748 | AUROC 1.000, ECE 0.070, n_pos 85/588 **(<150)** | AUROC 0.705, ECE 0.240, n_pos 355/1,478 |
| **advice-eligibility** | AUROC 0.500, ECE 0.878, n_pos 336/2,748 | AUROC 0.723, ECE 0.868, n_pos 396/3,012 | AUROC 1.000, ECE 0.058, n_pos 64/288 **(<150)** |

Rows = train domain, columns = eval domain. Diagonal = in-domain (test
split); off-diagonal = OOD (full unique corpus). All OOD cells clear the
150-positive gate; all in-domain (diagonal) cells fall short by
construction (flagged, not gated per Amendment 2's OOD-only wording).

### 2b. Hashing vectoriser

| train \\ eval | retail | airline | advice-eligibility |
|---|---|---|---|
| **retail** | AUROC 1.000, ECE 0.014, n_pos 88/332 **(<150)** | AUROC 0.500, ECE 0.131, n_pos 396/3,012 | AUROC 0.500, ECE 0.240, n_pos 355/1,478 |
| **airline** | AUROC 0.983, ECE 0.122, n_pos 336/2,748 | AUROC 0.998, ECE 0.059, n_pos 85/588 **(<150)** | AUROC 0.662, ECE 0.240, n_pos 355/1,478 |
| **advice-eligibility** | AUROC 0.500, ECE 0.878, n_pos 336/2,748 | AUROC 0.500, ECE 0.869, n_pos 396/3,012 | AUROC 1.000, ECE 0.044, n_pos 64/288 **(<150)** |

**Reading it.** In-domain saturates for all three domains and both
vectorisers (AUROC ~1.0, consistent with `policy-violation-linearly-
decodable-in-qwen-residual.md`'s prior finding that the label is textually
determined in-domain). Transfer is asymmetric and NOT uniformly weak: airline
-> retail is strong (0.983) under both vectorisers; retail -> {airline,
advice-eligibility} and advice-eligibility -> retail sit at chance (~0.50)
under both; advice-eligibility -> airline and airline -> advice-eligibility
sit in between (0.66-0.72) and the two vectorisers disagree noticeably there
(fixed-vocab 0.705/0.723 vs. hashing 0.662/0.500) -- an OOV-fairness
divergence exactly of the kind confound 2 exists to surface: the fixed
vocabulary (fit on the train domain only) is carrying some cross-domain
signal the unbounded hashing features are not, on the advice-crossing
cells specifically. This is reported, not resolved, here -- it is exactly
the comparison a trained detector's transfer numbers need to beat.

## 3. Majority-class baseline (free, for completeness)

Majority label is **compliant (False) in every domain's train split** -- no
domain crosses 50% violations. AUROC is undefined for a constant predictor
(reported `NaN`, not a spurious 0.5); accuracy only:

| eval domain | eval set | n | n_pos | accuracy |
|---|---|---|---|---|
| retail | test split (in-domain) | 332 | 88 | 0.735 |
| retail | full corpus (OOD target) | 2,748 | 336 | 0.878 |
| airline | test split (in-domain) | 588 | 85 | 0.855 |
| airline | full corpus (OOD target) | 3,012 | 396 | 0.868 |
| advice-eligibility | test split (in-domain) | 288 | 64 | 0.778 |
| advice-eligibility | full corpus (OOD target) | 1,478 | 355 | 0.760 |

(Accuracy depends only on the eval set, not the train domain, since every
train domain picks the same majority label -- collapsed to one row per eval
set above; the full 3x3 is in `baselines_frozen.json`.)

## 4. Zero-shot LLM judge (billable, $6-capped)

Model `accounts/fireworks/models/gpt-oss-20b` via Fireworks (same endpoint/
credential convention as prior runs -- `FIREWORKS_API_KEY` from
`/home/matt/Projects/bossyk-sandbox/.env`, this checkout has no `.env` of
its own, key never printed/logged/committed), temperature 0. Prompt: the
tested domain's own policy YAML (`/home/matt/Projects/bossyk/data/policies/`)
plus the decision context (agent's system+conversation prompt) and the
action taken, asking for `p_violation` in `[0, 1]`
(`scripts/detector_judge_baseline.py::JUDGE_SYSTEM_PROMPT`, sha256
`2dbe2e823a1a020c1424e0a739b523dabda4fdca11a37ad9b1bd7cd6ad861476`).
Sample: seed 0, per domain ALL positives + an equal-sized random sample of
negatives drawn from that domain's full v2 unique corpus (no train/eval
split distinction -- see "Design decisions" #1). Decision threshold for
FPR/FNR: `p_violation >= 0.5`.

| domain | n scored / sample | n_pos | AUROC | FPR@0.5 | FNR@0.5 | TP / FP / FN / TN |
|---|---|---|---|---|---|---|
| retail | 672 / 672 | 336 | 0.737 | 0.247 | 0.342 | 221 / 83 / 115 / 253 |
| airline | 792 / 792 | 396 | 0.464 | 0.429 | 0.538 | 183 / 170 / 213 / 226 |
| advice-eligibility | 710 / 710 | 355 | 0.986 | 0.028 | 0.000 | 355 / 10 / 0 / 345 |

All 2,174 sampled items were judged (0 unjudged -- no persistent parse
failures or exhausted retries in any domain).

**Reading it, honestly.** The judge is NOT a uniformly strong baseline:
advice-eligibility is near-perfect (AUROC 0.986, FNR 0.000 -- it never misses
a real violation there, at the cost of a small 2.8% false-positive rate),
retail is middling (0.737), and **airline is BELOW chance (0.464)**, with a
FNR of 0.538 -- the judge misses more than half of airline's actual
violations, and its FPR (0.429) is also high, so on airline it is closer to
a coin flip that leans toward under-flagging than a working detector. This
is a systematic, domain-specific failure, not noise (n=792, well-powered) -- consistent with `detector-training-prior-work.md` §2's citation of SWiRL's
finding that judge errors are systematic rather than random, and worth
carrying forward as the judge baseline's headline caveat: **this zero-shot
judge should not be treated as a floor a trained detector trivially clears
on every domain.**

**Caveats (per the brief, stated up front rather than buried):**
- **Self-consistency.** The judge (`gpt-oss-20b`) is the SAME model family
  as the corpus generator (`gpt-oss-20b`, per `docs/phase-detector-run2-
  results.md`'s manifest). A shared blind spot between generator and judge
  cannot be ruled out from this baseline alone.
- **Judge capability ceiling.** This is a 20b-class judge. A stronger judge
  (e.g. a frontier model) would likely cost meaningfully more per call and
  was not authorized under this step's $6 cap; upgrading the judge is a
  noted future option, not taken here.
- **No CI.** Per "Design decisions" #3, these are point estimates.

## 5. Cost / spend

Judge run executed as **3 chunked foreground invocations** (the 10-minute
Bash tool limit vs. observed throughput of ~1.3-1.5 calls/sec at 12
workers), each resuming from the per-domain on-disk checkpoint
(`probes/detector/results/judge_baseline/<domain>_judged.jsonl`) -- same
convention as `docs/phase-detector-run2-results.md`'s chunked generation.
No command was backgrounded/detached; each chunk blocked the turn until it
returned or the shell `timeout` ended it, and the check "how much has been
judged so far" (`wc -l` on the checkpoints) gated whether another chunk was
started -- **no chunk was started once all three checkpoints reached their
target sample sizes.**

| chunk | wall-clock cap | retail done | airline done | advice-eligibility done | cumulative calls |
|---|---|---|---|---|---|
| 1 | 520s (hit cap, killed) | 672/672 | 90/792 | 0/710 | 762 |
| 2 | 520s (hit cap, killed) | 672/672 (skip) | 792/792 | 444/710 | 1,908 |
| 3 | 350s (finished naturally) | 672/672 (skip) | 792/792 (skip) | 710/710 | 2,174 |

| | calls | est. spend @ $0.002/call | cap | outcome |
|---|---|---|---|---|
| **total** | **2,174** | **$4.348** | **$6.00** | under cap, all domains fully judged |

The in-script `BudgetGuard` never fired (`cap_reason=None` at completion) -- the three domains' full authorized sample (2,174 calls) finished naturally
under the $6/3,000-call caps. Each individual chunk's own `BudgetGuard`
instance only sees calls made in that process (a fresh guard per
invocation, same limitation the resumable-checkpoint design implies); the
totals above are the TRUE cumulative sums over the per-domain checkpoint
files, not any single chunk's self-reported count.
`probes/detector/results/baselines_judge.json`'s top-level `budget` field
has been corrected to carry this same cumulative total (with a note
explaining why), rather than the last chunk's own undercount.

Placeholder accounting at $0.002/call throughout, same convention as prior
runs (real metered Fireworks spend not checked against the dashboard).

## 6. Gates

```
pytest tests/ --no-cov -q                    -> 1279 passed, 5 skipped
ruff format --check .                        -> 272 files already formatted
ruff check .                                 -> All checks passed!
mypy --explicit-package-bases src/ tests/    -> Success: no issues found in 239 source files
```

## 7. Data / artefacts committed

- This document.
- `probes/detector/results/baselines_bow_majority.json` -- raw BoW+logreg +
  majority-class output (`scripts/detector_baselines.py`).
- `probes/detector/results/baselines_judge.json` -- raw judge output
  (`scripts/detector_judge_baseline.py`), `budget` field corrected per §5.
- `probes/detector/results/judge_baseline/{retail,airline,advice-
  eligibility}_judged.jsonl` -- the per-item judge checkpoint (row_id,
  is_violation, p_violation, reason) for every sampled item, the audit
  trail behind §4's numbers.
- `probes/detector/results/baselines_frozen.json` -- machine-readable
  combination of the above three, plus config (seed, vectoriser params,
  judge model/prompt hash), for step 5's evaluation code to load directly.

## 8. What this unblocks / does not unblock

This freeze satisfies build-order step 3's gate: exit criterion "Baseline
table frozen before any training run" is now met. It does **not** start
step 4 (training) -- no detector of any kind exists after this commit, and
none was trained to produce these numbers. The transfer-matrix asymmetry in
§2 (some OOD cells near-saturating, others at chance) and the judge's
domain-specific failure on airline in §4 are exactly what a trained
detector's headline transfer-AUROC gap (dossier's pre-registered primary
metric, advice-crossing cells) must be compared against in step 5.
