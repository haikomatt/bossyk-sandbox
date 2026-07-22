# bossyk sandbox — H1–H5 synthesis (the research proof)

_Generated: 2026-07-22. Phases 0–4 complete; cross-domain (airline + retail).
Corrected 2026-07-23 (code-review findings 13/14): H2 re-marked **open** — its
12/12 result is from hand-authored scripted scenarios, not the 56 actual H1
guardrail crossings, which have not yet been run through a live agent; H4
re-labeled a **modeled counterfactual** (no live execution observed)._

The falsifiable claim under test (plan §1, after Cox / NIST-Vassilev): **prevention
is structurally insufficient at every layer — for any guardrail there exists a
bypass — so the only viable posture is threat-model-informed continuous monitoring +
human review + threat-model update.** Built to be able to *disprove* as well as
confirm. This doc pulls the five hypotheses' results into one honest matrix.

## The H1–H5 matrix

| # | Layer | Hypothesis | Result | Verdict |
|---|---|---|---|---|
| **H1** | Model | A representative guardrail has a non-trivial bypass rate | bypass **0.194** airline / **0.156** retail; blind-spot profile **domain-invariant** | **not falsified** |
| **H2** | Agent | Agent-layer monitoring detects a meaningful fraction of **actual H1 guardrail bypasses** | **Not yet measured**: the 56 frozen crossings have not been consumed by a live agent. On a separate hand-authored scripted scenario set, the three-leg union detected 12/12 violations (fast gate alone 5/12) | **open** (partially tested — scripted proxy only) |
| **H3** | Agent | drift / policy / (gate) fire on substantially non-overlapping event sets | three-legged division catches 12/12 **on the scripted scenario set**; **no single leg does**; each orthogonal in *what* it catches | **not falsified** (as a 3-legged division; strict drift⊥policy is weaker — see below) |
| **H4** | Agent | Interruption reduces realized harm vs the un-interrupted counterfactual | **modeled counterfactual over the scripted run** (no live execution observed): modeled harm 12→7 (**5/12 prevented**), **7/12 detected only post-hoc** | **not falsified as a modeled counterfactual, latency-bounded** — live paired-run evidence pending |
| **H5** | Eval / ceiling | An adversary optimised to pass all behavioural instruments evades the layer | **pending** — the adaptive adversary (fable) awaits an `ANTHROPIC_API_KEY` | open |

Plus the **SMACTR loop (Phase 4)**: a caught failure → fast-path rule → prevention
rose 5/12 → 6/12. The eval loop closes.

## The through-line (why the layers need each other)

1. **The model layer leaks — predictably (H1).** A prompt-injection classifier
   catches what *looks* adversarial (prompt_injection bypass ≈ 0–2.5%) and reliably
   *lets through* what reads as a plausible request (tool_misuse ≈ 40%, pii ≈ 20%).
   The profile is **identical across airline and retail** — a property of
   input-classification, not a domain artifact.
   ([[guardrails-catch-what-looks-adversarial-not-what-is-harmful]])
2. **The agent layer is built to catch the residual by consequence (H2/H3) — so
   far proven on a scripted proxy, not the actual residual.** Violations are
   caught downstream by *what the action does*: a fast structural gate (wrong-id /
   no-lookup), a semantic policy judge (out-of-policy), and a drift check
   (intent–action deviation). Each has a blind spot the others cover — the semantic
   judge calls a wrong-id cancel *faithful*; the structural rule can't see a semantic
   violation. **Union caught 12/12 on the hand-authored scenario set; no single
   instrument did.** Whether it catches what *actually* bypassed the H1 guardrail
   (the 56 frozen crossings) is untested until the payload→agent→action path runs
   (H2, open). ([[orthogonal-agent-monitors-together-catch-what-none-catches-alone]])
3. **But detection ≠ prevention (H4, modeled).** Only what the *fast* path detects
   can be stopped before the side-effecting call runs. Under the scripted
   counterfactual model, the gate prevented **5/12**; the other **7** were caught
   only post-hoc by the slow LLM judges — observed, not prevented. You can see more
   than you can stop. (Modeled: no live execution or paired run — see
   `docs/phase3-h4-interrupt-results.md`.)
   ([[you-can-detect-more-agent-violations-than-you-can-prevent]])
4. **The loop closes the gap, incrementally (SMACTR).** A caught, post-hoc failure
   (retail-008) was converted into a fast-path *prevention* by deriving one rule and
   feeding it back — prevention 5/12 → 6/12. Continuous monitoring, operationalised:
   each caught failure makes the system measurably safer.
   ([[a-caught-failure-should-become-a-fast-path-rule-not-just-an-alert]])
5. **The ceiling is still open (H5).** An adversary explicitly optimised to be
   self-consistent, in-policy-looking, and still harmful is the hardest test — and
   the motivation for the interpretability hook (Property X). Pending fable.

**Net:** Cox's hypothesis is *supported* on the evidence so far, with its
strength graded by evidence type — **no layer prevents alone** (H1, live
measurement), the layers are **orthogonal on the scripted set** (H3), detection
outruns prevention (**latency**, H4 — modeled counterfactual), and the
**monitoring loop** converts caught failures into durable controls (SMACTR,
deterministic recompute). The demonstrator was built to be able to falsify this;
H1, H3, and the modeled H4 did not falsify it. **H2 — the load-bearing claim that
the agent layer catches what actually bypasses the guardrail — remains open**
until the 56 frozen crossings run through a live agent.

## Cross-domain

Every result holds on **both** airline and retail — the blind-spot profile (H1),
the three-legged orthogonality (H3, scripted set), and the modeled
interrupt-latency gap (H4, retail
worse at 16.7% vs airline 66.7% because its violations skew semantic). The claims
are not an airline artifact.

## Honest negatives & caveats (these are results, not failures)

- **Detection ≠ prevention** is itself a partial negative on naive "interrupt =
  safety": 7/12 violations were unpreventable at detection time (H4).
- **Drift under-fires** (⊆ policy this run) — a calibration ceiling of the auditk NLI
  soft-paraphrase gate, not a non-orthogonality. Bounds H3's drift leg.
- **Strict drift⊥policy is not cleanly supported** at this n; the orthogonality that
  *is* supported is the three-legged gate/policy/drift division and instrument-vs-ground-truth.
- **Small n** (H1 160 payloads/domain; H2–H4 35 steps / 12 violations), single run,
  **single adversary model** (deepseek-v4-pro) — hence the §15A multi-provider plan.
- **One guardrail** (a deberta prompt-injection classifier); **8.6% policy judge
  parse errors**; the H4 fast/slow latency is characterised qualitatively, not yet
  wall-clock instrumented (§15B).
- Coverage is bounded by the finite `class × boundary` grid for a fixed
  tool/authority set — never "immune" (Property X).

## Artifacts

Per-phase results: `docs/phase2a-h1-results.md`, `docs/phase2b-h1-crossdomain-results.md`,
`docs/phase2c-h3-crossdomain-results.md`, `docs/phase3-h4-interrupt-results.md`,
`docs/phase4-smactr-results.md`. Data: `docs/bench_output/*.json`
(`phase2a_h1`, `phase2b_h1_retail`, `phase2b_crossdomain_h1`, `phase2c_orthogonality`,
`phase3_h4`, `phase4_smactr`, `threat_model`). Regression probes:
`probes/regression/{airline,retail}.json` (56 frozen crossings) plus
`probes/regression/retail-smactr.json` (the SMACTR-derived retail-008 replay
probe, Phase 4).

## Open frontier (what would complete the matrix)

- **H5:** the adaptive adversary — fable + multi-provider registry (plan §15A),
  blocked on `ANTHROPIC_API_KEY` (placeholder now in `.env`).
- **Latency instrumentation** (§15B) to make H4's crux quantitative.
- **Fast hot-path semantic detector** (§15C) — the strategic fix that would move H4's
  7 "detected too late" onto the prevented side at the class level, not one rule at a time.
- **Live-injection loop (H2-on-crossings):** score the 56 frozen crossings through a
  live agent — the deferred piece that closes H2 on the *actual* guardrail bypasses.
- The interpretability hook (Property X deception scope) that H5 motivates.
