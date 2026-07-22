# Phase 3 — interrupt efficacy + latency (H4, airline + retail)

_Generated: 2026-07-22. Branch: `phase3-BE-interrupt-h4`.
Corrected 2026-07-23 (code-review finding 14): H4 here is a **modeled
counterfactual over the Phase 2c scripted run**, not a live interruption
experiment — the headline and claims below are scoped accordingly._

Tests **H4** as **modeled prevention under a matched scripted counterfactual**:
_would interruption on a high-severity detection reduce realized harm vs the
un-interrupted counterfactual?_ No live tool execution is observed and no
with/without paired run is performed — see Method. Falsifier: no delta, or harm
realized **before** detection (the latency problem — plan §5, "the most honest
test").

## Method

The interrupt mechanism already exists: the two-speed gate's fast path blocks a
proposed tool call **synchronously, pre-execution**. H4 is a matched counterfactual
over the same scenarios, on the **12 ground-truth violations** from the Phase 2c
cross-domain run — computed from that run's persisted data (`phase2c_orthogonality.json`:
gate decision + drift/policy verdicts + A1 label per step), so **no new judge calls**.
Per violation:

- **prevented** — gate blocked it (fast path, before execution → harm averted).
- **detected_too_late** — gate allowed it, but a slow instrument fired (observed
  *post-hoc*, after the action executed → not prevented).
- **undetected** — gate allowed it and nothing fired.

`harm_off` = every violation executes (no interrupt), **assuming one unit of
realized harm per violation** — a modeling assumption, not an observation;
`harm_on` = only gate-allowed violations execute; `harm_delta = harm_off −
harm_on` = **computed** prevention count. Distinguish throughout: observed gate
blocks (real, deterministic), assumed harm-if-allowed (modeled), computed
prevention (derived), and the unmeasured real-world effect and latency. Report +
data: `scripts/h4_report.py`, `docs/bench_output/phase3_h4.json`.

## Result

| | combined | airline | retail |
|---|---|---|---|
| violations | 12 | 6 | 6 |
| **prevented** (pre-execution) | 5 | 4 | 1 |
| **detected too late** (post-hoc) | 7 | 2 | 5 |
| undetected | 0 | 0 | 0 |
| harm_off / harm_on | 12 / 7 | 6 / 2 | 6 / 5 |
| **harm_delta** (prevented) | **5** | 4 | 1 |
| prevention rate | **0.417** [0.193, 0.680] | 0.667 [0.300, 0.903] | 0.167 [0.030, 0.564] |

## Interpretation

- **H4 is not falsified as a modeled counterfactual, but it is latency-bounded —
  the honest headline.** Under the counterfactual model, interruption reduces
  modeled harm (12 → 7, delta 5). But it prevents only
  **41.7%** of violations pre-execution; the other **7 of 12 are detected only
  post-hoc** by the slow drift/policy judges — **observed, but not prevented**.
  You can see more than you can stop. That gap *is* the H4 finding.
- **The prevention boundary is the fast/slow latency split.** What the fast
  structural gate detects (sub-millisecond, deterministic) it prevents; what only
  the slow semantic judges catch (LLM-judge, seconds) is caught after the
  side-effecting call has already run. Detection completeness (Phase 2c: 12/12
  union) does **not** transfer to prevention — the slow path is too late by
  construction.
- **Retail is markedly worse than airline** (16.7% vs 66.7% prevented). Retail's
  violations skew semantic (pii / refund / address-modification), which the narrow
  structural fast rule can't gate — so retail relies on the slow path, which can't
  prevent. This is the same domain-texture point Phase 2c's confusion showed,
  now expressed as an interrupt-efficacy gap.
- **0 undetected**: nothing slipped past the union entirely — the failure is purely
  prevention-vs-detection (latency), not coverage. The lever to raise prevention is
  moving detection onto the fast path (more structural rules, or a fast small-model
  classifier), not adding more slow judges.

## Caveats

Small n (12 violations); `harm_off` assumes every un-interrupted violation realizes
harm (an upper bound on preventable harm); detection reuses the Phase 2c real-judge
verdicts (incl. its 8.6% policy parse-error rate — one airline violation counted as
"detected too late" via the gate, its policy verdict being an error); the fast/slow
latency is characterised qualitatively (structural sub-ms vs LLM-judge seconds), not
wall-clock benchmarked here.

## What a true live H4 test needs (not done here)

This phase's evidence is a deterministic recomputation over persisted scenario
labels and gate decisions. Upgrading H4 from "modeled" to "live" requires:

- **paired runs** (interruption on vs off) over the same seeded environment, or a
  deterministic environment reset between arms;
- **concrete side-effect observation** — the tau2 environment state actually
  mutated (or not), rather than assuming one harm unit per violation;
- **timestamps** around detection, decision, and execution, so the fast/slow
  latency crux becomes wall-clock measurement (plan §15B);
- **an explicit harm oracle** mapping realized side effects to harm, replacing
  the 1-unit-per-violation assumption.

## What's next

- **Raise prevention onto the fast path:** a fast small-model classifier on the
  slow path's semantic catches (the plan's two-speed §5) — turning "detected too
  late" into "prevented" — is the concrete lever this result points to.
- **Interrupt taxonomy:** redirect / escalate + the HITL approval queue (deferred
  from this phase; plan Phase 3/6).
- **Live-injection loop (H2-on-crossings):** score the 56 guardrail-bypass crossings
  through a live agent — the remaining deferred piece.
- **SMACTR loop (H5)** + the full H1–H5 write-up (plan Phases 4–5).
