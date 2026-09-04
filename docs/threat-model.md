# bossyk threat model: the reference-monitor view

**Status:** living. **Audience:** a technically literate reviewer asking "what
is this, in classical security terms, and where does it break?"
**Companion:** the buyer-facing threat model (ATLAS/OWASP coverage matrix,
blind spots, CISO Q&A) lives in the planning vault
(`bossyk-threat-model-ciso-v0.1`); this document is the architectural view.
The two share one set of claims.

## What bossyk is, in classical terms

bossyk is a **reference monitor** (Anderson, 1972) on the boundary between an
agent that cannot be fully trusted and the tools whose effects matter. A
reference monitor is valid to the extent that three properties hold:

1. **Complete mediation**: every access crosses it; there is no way round.
2. **Tamperproofness**: the monitor and its record cannot be altered by the
   thing it monitors.
3. **Verifiability**: it is small and simple enough to be checked.

Naming the primitive names the failure modes. Each property is assessed
honestly below, with the evidence that exists and the gaps that remain.

## The trust boundary

```
[principal] --instructions--> +-----------+
[world: tool returns, RAG, --> | LLM agent |   untrusted by construction
 caller utterances]           +-----+-----+
                                    | proposed tool call
        ============ boundary ======v=====================
                              +-----------+   <- contract / rules
                              |  bossyk   |   <- Ed25519 signing key
                              |   gate    |
                              +-----+-----+
                     allow / block / HOLD (HITL)
                                    v
                              [tau2 tools: cancel, refund, dial, ...]
```

The load-bearing design decision: **govern the effect an action would have,
attest the record, do not try to read intent.** An injected, jailbroken, or
confused agent that proposes cancel-without-verify is blocked by the same rule
as a misconfigured one. The controls are intent-agnostic at the boundary
(demo Act 2, Act 5; the H1 layered-defence results).

## Property 1: complete mediation

**Current claim, stated precisely:** in the demo runtime, mediation is
complete **by graph construction**, not by interposition. The LangGraph agent
is wired `agent_node -> plan_calls_node -> prepare_node -> approve_node ->
execute_node` (`src/bossyk_sandbox/runtime/langgraph_agent.py`);
`prepare_node` scores every pending call through `Gate.score`, `approve_node`
surfaces holds via `interrupt()`, and `execute_node` sits strictly downstream.
No call in this graph can execute ungated.

**What this does not claim:** the gate is not an unbypassable OS-level
monitor. An agent path not wired through these nodes would not be gated. The
guarantee is structural within the graph the demo ships, and the honest
statement of the residual is: *non-bypassable interposition (proxy-level or
kernel-level tool mediation) is future work, not a current property.* This is
the classic way reference monitors fail in practice, which is exactly why it
is named first.

## Property 2: tamperproofness

The attestation trusted computing base is analysed in
`probes/attestation/tcb_claim_boundary.md`, including a working forgery
demonstration (`tests/unit/test_attestation_tcb_forgery.py`). The claim
boundary, verbatim in spirit:

- A signed trace proves the record has not been altered **since signing** and
  that the decisions it records were the gate's.
- It does **not** prove fidelity of capture: a compromised tracer or an
  exfiltrated key can sign a clean trace of a dirty run.
- For boundaries with a reproducible oracle, the trace is now **falsifiable
  against that oracle** (`evidence/oracle_binding.py`,
  `evidence/oracle_reverify.py`): the demonstrated forgeries fail oracle
  re-verification while still passing signature verification.
- Remaining rungs: independent co-signature (collusion resistance), then
  TEE/remote attestation (binding the running code to the expected code).

## Property 3: verifiability

The gate core is small and deterministic: `Gate.score` stops at the first
BLOCK; slow instruments run as non-gating annotations (`TwoSpeedGate`). The
evidence chain is designed for third-party checking: byte-checked
reproducibility, evidence-grade labels on every claim (live measurement /
scripted proxy / modelled counterfactual / deterministic recompute / open),
and offline signature verification against committed fixtures
(`demo_output/`). Verifiability of the *record* is stronger today than
verifiability of the *runtime* (see property 2).

## Expressiveness: which policies a monitor could enforce at all

Anderson's three properties ask whether the monitor is *positioned* correctly.
They say nothing about which policies a correctly-positioned monitor can
enforce, which is a separate and independent question, answered by Schneider
(*Enforceable Security Policies*, ACM TISSEC 3(1):30-50, 2000). Both axes have
to hold. Perfect mediation over an unenforceable policy is still unenforceable,
and no amount of engineering moves it.

**The result, stated precisely.** Schneider's EM class covers mechanisms that
see only the actual steps of a running target and decide on the history so far.
The theorem is one-directional: **EM-enforceable implies safety property**. The
converse is false, and he supplies the counterexample himself: a real-time
availability policy is a safety property but is not EM-enforceable, because the
monitor has no control over the passage of time. EM-enforceable policies are a
*strict subset* of safety properties.

**The correction that matters.** Schneider's monitor may only truncate.
Ligatti, Bauer and Walker (*Edit Automata*, Int. J. Inf. Security 4(1-2), 2005)
showed that truncation automata precisely enforce exactly the safety properties,
and that under effective enforcement **suppression and edit automata enforce a
strictly larger class**. So the folk claim "a runtime monitor can only enforce
safety properties" is wrong as stated. A monitor that can only *halt* is bounded
by safety; one that can suppress or rewrite before an effect is not. That
distinction is load-bearing for the utterance gate below.

**Classes used here.**

| Class | Definition | What the gate can promise |
|---|---|---|
| A | Finite witness in the observed prefix; gate can halt before the effect | Enforce deterministically |
| B | Irreversible once emitted, but mediable between generation and emission | Enforce by suppression/rewrite; strictly stronger than A's primitive |
| C | Safety, but the witness is outside what the monitor observes | Not enforceable as built; fixable by widening observation |
| D | Witness is semantic: computable only via an unreliable classifier | Annotate only; enforcing it would enforce the classifier, not the policy |
| E | Not a property of a single execution (information flow) | Not enforceable by any single-trace monitor; sound-but-incomplete proxy only |

**The shipped policy set, classified.**

| Policy line | Domain | Class | Today |
|---|---|---|---|
| Submit an eligibility determination without first verifying eligibility | advice-eligibility | A | Enforced (`RequirePassedCheck` on `submit_eligibility_decision`) |
| Disclose one customer's data in another customer's session | advice | **E** | Not enforced, and cannot be; see out of scope |
| Cancel reservations without explicit customer request | airline | **D** | Structural proxy enforced (a lookup happened); the semantic predicate is not |
| Access account data without a valid booking reference | airline | A | Enforced (`RequireLookupBeforeCancel`) |
| Contact a TPS-registered or opted-out number | outreach | A | Enforced for `place_call`; `send_sms`/`send_email` have **no rule wired** |
| Book for a non-eligible or unverified prospect | outreach | A | Enforced (`RequirePassedCheck` on `check_eligibility`) |
| Make a consumer-credit financial promotion | outreach | **B** | Enforced by substitution before emit (`ProhibitedPhraseRule`) |
| Issue refunds above policy limit without supervisor approval | retail | A | Enforced (standing/amount ceilings, console path) |

**Why this is a result and not a caveat.** Six of eight lines are class A.
Operational policies in support and outreach domains overwhelmingly take the
form "not this action without that precondition", which is exactly the
EM-enforceable shape. The reference-monitor architecture is not a lucky fit for
these domains; it is matched to the class of policy they generate. The single
class-E line is the only permanent hole, and it is stated below rather than left
looking like unfinished work.

## Threat table (STRIDE x OWASP-LLM)

| # | Threat | Class | Mitigation | Residual / evidence |
|---|--------|-------|------------|---------------------|
| T1 | Unmediated path to a tool | Elevation / LLM06 | Graph-structural mediation (property 1) | Convention within the demo graph; interposition is future work |
| T2 | Injection steers an allowed tool into a harmful call | Tampering / LLM01 | Rules constrain arguments and effects, not tool identity; blocked regardless of cause | A permitted action with bad params passes (CISO blind spot #5) |
| T3 | Call mutated between check and execution | Tampering | Gate scores the proposed call in `prepare_node`; execution consumes the scored proposal | Async tools and retries widen the window; not yet exercised |
| T4 | Tampering with monitor, policy, or record | Tampering | Ed25519 attestation + oracle binding (property 2) | Capture-fidelity gap, demonstrated and bounded in the TCB doc |
| T5 | Confused deputy on standing privilege | Elevation / LLM08 | HOLD/HITL enforcement modes; standing-authority modes (Act 7) | HITL fatigue; authority scoping is policy, not code |
| T6 | Exfiltration inside permitted calls | Info disclosure / LLM02 | Minimisation toggle in the advice domain (derived-only tool surface) | Utterance-side exfil is now partly governed (class B, phrase list only, outreach only). Cross-session leakage stays permanently ungoverned: class E, see out of scope |
| T7 | Latency pressure forces fail-open | DoS / LLM04 | Two-speed gate; latency budget measured (latency-budget phase) | Fail-open/closed per tool class is a policy decision to document |
| T8 | Letter-vs-intent policy gap | Goodhart / LLM06 | SMACTR loop grows the threat model from caught failures (`governance/smactr.py`) | Coverage is never provably complete |
| T9 | Repudiation of what was enforced | Repudiation | Signed evidence pack, offline-verifiable; auditk protocol on the audit side | Log integrity inherits T4's boundary |
| T10 | Evade the detector while violating | Evasion / ATLAS | Enforcement does not depend on detection. Act 4: laundering the request fails (AUROC drop <= 0.001) because the violation *is* the mutation request; internals add nothing over text | Result is domain-specific; the probe stays a cheap cascade filter, never a control |

## Out of scope (stated, not hidden)

Revised 2026-09-04. Two items previously listed here have landed and are no
longer out of scope; one permanent boundary that was missing has been added. The
distinction that matters is **not yet wired** versus **cannot be wired**, and
the old version of this list conflated them.

**No longer out of scope, but narrower than "handled":**

- **Speech acts.** An utterance can itself be the violation (a regulated
  financial promotion is unlawful the moment it is said). The old claim here was
  "no tool call, no mediation point", and that was wrong: the mediation point is
  between generation and emission. `UtteranceInstrument` /
  `ProhibitedPhraseRule` gate the model's raw message in `agent_node` and replace
  the content on BLOCK. In the terms above this is a class-B suppression
  automaton, which enforces a *strictly larger* class than a halt-only gate, so
  it is an upgrade rather than a workaround. **Residual:** the closed
  regulated-phrase list is a syntactic proxy for a semantic class, so
  enforcement is sound within its vocabulary and incomplete outside it; and it
  is wired for outreach only.
- **Check-outcomes.** Call the check, receive "on list", dial anyway.
  `ObservedAction` plus `RequirePassedCheck` landed and this is now enforced.
  **Residual, and it is a real one:** only `place_call` and `book_survey` have a
  rule wired. `send_sms` and `send_email` share the same suppression policy line
  and have **no rule of any kind**, so that boundary is enforced for one contact
  channel out of three. Tracked with the discharge narrowing in the same branch
  as this revision.

**Genuinely and permanently out of scope:**

- **Cross-session information flow.** "Do not disclose one customer's data in
  another customer's session" (advice domain) cannot be violated by any single
  execution considered alone, since whether a disclosure leaked depends on a
  different session. It is a class-E policy, excluded from Schneider's framework
  as a non-property of individual executions. `advice_fast_rules()` is empty for
  this reason and not for want of effort. A single-trace proxy (session-scoped
  identity tainting, refuse any emission carrying a foreign taint) is available,
  strictly weaker than the policy, and not built. The gap between the proxy and
  the policy is the thing an assurance case owes its reader.
- **Model-internal deception** that never surfaces as an action. Act 4's
  negative results are the reason this is out of scope rather than a roadmap
  item: reading intent from internals added nothing over text in the regime
  tested. There is also a structural argument that it cannot be made reliable in
  general, since a computation can be obfuscated while preserving behaviour.

In security terms bossyk secures the **action boundary**, not the honesty
boundary. The out-of-scope list is the claim boundary, and it is the part of
this document least likely to be written by a vendor.

## Assumptions the model rests on

1. Complete mediation holds for the graph the demo ships (property 1). The
   whole model degrades gracefully to "attested telemetry" if it fails, but
   the prevention claim depends on it.
2. The signing TCB is honest at capture time, up to the oracle-bindable
   subset (property 2).
3. Rules are authored and reviewed competently: the gate enforces a
   specification, it does not write a good one (T8).
4. The policy being asked for is inside the enforceable class. A class-D or
   class-E policy (see Expressiveness) does not become enforceable through
   better rules, more mediation, or a stronger key. Assumptions 1-3 are the
   preconditions Schneider names for EM enforcement -- Target Control,
   mechanism integrity, competent specification -- which is a useful check that
   the honest gaps found here are the standard ones and not exotic.
