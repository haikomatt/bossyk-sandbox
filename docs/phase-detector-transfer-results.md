# Detector training: cross-domain transfer results (step 6 write-up)

**Verdict: SUPPORTED, both detector families.** A fine-tuned small-LM
violation detector transfers across policy domains better than a
bag-of-words baseline, by a margin roughly six times the pre-registered
threshold, with the calibration criterion also met. Computed mechanically
by `scripts/detector_verdict.py` against conditions locked before any
training run; see `probes/detector/results/verdict.json` for every named
condition boolean.

## The question

In-domain, detecting a structural policy violation
(mutation-without-lookup) from an agent's decision text is trivial: a
bag-of-words classifier scores ~1.0 because the label is textually
determined. The open question, and the pre-registered hypothesis
([[a-fine-tuned-violation-detector-transfers-cross-domain]]): does a
*trained* detector generalise to a lexically distant domain where the
vocabulary gives the baseline nothing to memorise?

Three domains, identical structural label by construction (the same
generic gate rule instantiated per domain): retail (orders/refunds),
airline (bookings/flights), and advice-eligibility (benefits/enrolment),
the third built deliberately lexically distant and levelled up to match
retail's structural surface diversity (3 gated tools, 2 key_args, 10
scenario seeds). Pre-registered confound gate: cross-domain lexical
overlap must sit materially below the retail-airline control. Passed on
seed text, passed again on generated decisions, passed a third time on
the final v2 corpora.

## Headline numbers

Transfer AUROC, mean over 3 seeds, on the full unique corpus of the
held-out domain (eval sets and gates per Amendment 2). Baseline shown is
fixed-vocab BoW / hashing BoW.

**DeBERTa-v3-small (full fine-tune):**

| train \ eval | retail | airline | advice-elig |
|---|---|---|---|
| retail | 1.000 (1.000/1.000) | 0.974 (0.500/0.500) | 0.917 (0.499/0.500) |
| airline | 1.000 (0.983/0.983) | 1.000 (1.000/0.998) | 0.827 (0.705/0.662) |
| advice-elig | 0.984 (0.500/0.500) | 0.898 (0.723/0.500) | 1.000 (1.000/1.000) |

**Qwen2.5-0.5B-Instruct (LoRA):**

| train \ eval | retail | airline | advice-elig |
|---|---|---|---|
| retail | 1.000 (1.000/1.000) | 0.999 (0.500/0.500) | 0.918 (0.499/0.500) |
| airline | 0.999 (0.983/0.983) | 1.000 (1.000/0.998) | 0.958 (0.705/0.662) |
| advice-elig | 0.860 (0.500/0.500) | 0.738 (0.723/0.500) | 1.000 (1.000/1.000) |

Composite paired gaps (detector minus fixed-vocab BoW), hierarchical
bootstrap 95% CIs (scenario then decision, 10,000 resamples, seed
20260828; flat bootstrap reported alongside in verdict.json):

| composite | DeBERTa | Qwen-LoRA |
|---|---|---|
| 4-cell advice-crossing mean | +0.42 [+0.39, +0.45] | +0.42 [+0.39, +0.45] |
| into-advice | +0.27 [+0.23, +0.31] | +0.34 [+0.30, +0.38] |
| out-of-advice | +0.33 [+0.27, +0.40] | +0.19 [+0.09, +0.28] |

Pre-registered bar: point gap >= 0.05 on the mean, hierarchical CI lower
bound > 0 on the mean and in EACH direction (Amendment 1), ECE condition,
base-rate gates (all three domains 336/396/355 positives, >= 150). Every
condition true for both families.

## Reading the numbers honestly

- **The baseline collapses cross-domain, as theory predicts.** Fixed-vocab
  BoW lands at ~0.5 on lexically distant pairs. This is the designed
  contrast, not a strawman: the hashing vectoriser (OOV-fair, confound 2)
  is reported in every cell and behaves the same. Where vocabulary
  partially carries (airline-advice fixed-vocab 0.705), the detector still
  clears it.
- **Per-direction reporting earned its keep.** Qwen's out-of-advice
  direction passes with a lower bound of +0.09, and its advice-to-airline
  cell (0.738) nearly ties fixed-vocab BoW (0.723). Under the original
  mean-only registration this weakness would have been invisible inside a
  +0.42 average. Amendment 1 (adopted pre-data) is why it is visible.
- **Asymmetry is real.** Training on the two similar domains and testing
  on the distant one (into-advice) transfers better for Qwen; DeBERTa is
  the stronger out-of-advice model. Advice-trained models see the least
  diverse gated-surface vocabulary mix and transfer worst, consistent
  with surface diversity mattering for generalisation.
- **Calibration transfers worse than discrimination.** The ECE condition
  passed (detector beats BoW's badly miscalibrated OOD probabilities),
  but temperature scaling fitted on the train domain made OOD calibration
  slightly WORSE in most cells (e.g. DeBERTa retail-to-advice 0.157 pre
  vs 0.160 post). A monitor deployed cross-domain cannot assume its
  in-domain temperature; thresholds need per-deployment recalibration.
  This negative rides along for free and is worth its own line in any
  deployment guidance.
- **Seed variance is not negligible.** DeBERTa retail-to-advice per-seed
  AUROC spans 0.80 to 0.99. Three seeds and paired CIs are the minimum
  honest design here, not a formality.

## Scope and limitations

1. **Vocabulary transfer is proven; structural transfer is not.** All
   three domains share trace format, gate mechanics, and the identical
   structural label by construction. That is precisely what makes the
   comparison valid, and precisely what bounds it: a detector recognising
   "mutation without its required lookup" across vocabularies has not
   been shown to recognise a *different kind* of violation, or the same
   kind in a different trace architecture. The SWE-bench
   contamination-by-familiarity critique applies at the structural level
   and is accepted rather than rebutted.
2. **One generator.** All decision text was generated by gpt-oss-20b
   (Fireworks) driving the weakened agent. Detector transfer across
   generator models is untested (Math-Shepherd's verifier-generator
   mismatch results suggest caution).
3. **Sandbox domains.** Three domains, sharing a scenario-seed authoring
   style, persona machinery, and the compositional phrasing pool.
   Real-deployment text is messier in ways this corpus does not model.
4. **In-domain diagonal saturates at 1.0** for detector and baseline
   alike, as expected (the label is textually determined in-domain); the
   diagonal carries no evidence and the claim rests entirely on the
   off-diagonal cells.
5. **Shared weakening boilerplate carries retail vocabulary into every
   domain** (found by the hand-audit pass, 2026-08-31, before any human
   audit began). `POLICY_WEAKENING_OVERRIDE` in
   `runtime/langgraph_agent.py` names retail tools
   (`cancel_pending_order` etc.) and is appended verbatim to the weakened
   agent's prompt in all three domains: present in 100% of items, both
   classes, all splits, at the text tail (surviving left-truncation, so
   every detector saw it). Because it is class-constant it carries no
   label signal and does not affect the verdict, and since it *inflates*
   measured cross-domain lexical overlap, the overlap gate passed despite
   it, meaning true content-vocabulary distance is larger than the gate
   numbers suggest. It does, however, soften the "lexically distant"
   narrative at the margin and should be parameterised per domain in any
   future corpus. Disclosed here rather than discovered by a reviewer.

## Provenance, amendments, and the invalid first run

The experiment was pre-registered with the threshold, metric, refutation
conditions, confound gates and power floor fixed before any training.
Four amendments, each dated, each adopted before the evidence it governs,
and each a strict tightening or a validity precondition:

1. Per-direction CIs co-primary (pre-data).
2. Hierarchical bootstrap decision-bearing; base-rate hard gate; OOD
   evaluation-set definition (adopted post-generation, pre-evaluation;
   the 2b wording defect was caught and corrected before adoption).
3. Calibration-slice protocol after retail's val split drew 0 positives
   (pre-training).
4. Training-validity gate (in-domain AUROC >= 0.95 per checkpoint) plus
   the left-truncation fix, adopted after an invalid first training run
   was caught and discarded.

The first training run produced a mechanically REFUTED verdict that was
**invalid, not negative**: right-truncation at 512 tokens deleted the
label-bearing tail of retail and airline texts, so 12 of 18 checkpoints
trained on near-identical policy boilerplate and output constants
(in-domain AUROC ~0.5 while BoW scored 1.0). The per-item score
inspection mandated by the evaluation caught it; the invalid run is
preserved in history (`4e4fa5b`) and both runs are reported per Amendment
4d. Under the fixed pipeline all 18 checkpoints pass the validity gate
(in-domain AUROC = 1.0, full score variance).

## Cost and reproduction

Whole phase, metered: ~$5.26 GPU (RunPod A40, including every failed and
zombie pod of the 2026-08-29 incident) plus roughly $22 of Fireworks
inference at the conservative internal estimate (real metered cost
likely materially lower). Reproduce: `scripts/generate_corpus.py`
(corpus v2, temperature 0.7), `scripts/detector_baselines.py` (frozen
first), `scripts/detector_pod_train_all.py` via
`scripts/detector_pod_runner.py` (18 cells), `scripts/detector_verdict.py`
(mechanical verdict). The held-out domain is regenerable from
[[advice-eligibility-domain-spec]]; a sceptic can mint a fresh test set.

## What this buys downstream

- **bossyk:** a deployable monitoring instrument with evidence it
  survives a vocabulary change, to sit beside (never replace) the
  structural gate; prevention-only positioning unchanged.
- **auditk:** the hierarchical bootstrap module, the per-item
  score-variance validity gate, and the truncation lesson all transfer
  directly to clustered-transcript scoring.
- **Careers:** gap #1 of the safety-RE landscape closed with a
  supported, pre-registered, honestly-bounded result and two first-hand
  war stories (the boilerplate detector; the blinded pod monitoring)
  that are themselves oversight material.
