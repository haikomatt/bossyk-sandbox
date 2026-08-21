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

## Threat table (STRIDE x OWASP-LLM)

| # | Threat | Class | Mitigation | Residual / evidence |
|---|--------|-------|------------|---------------------|
| T1 | Unmediated path to a tool | Elevation / LLM06 | Graph-structural mediation (property 1) | Convention within the demo graph; interposition is future work |
| T2 | Injection steers an allowed tool into a harmful call | Tampering / LLM01 | Rules constrain arguments and effects, not tool identity; blocked regardless of cause | A permitted action with bad params passes (CISO blind spot #5) |
| T3 | Call mutated between check and execution | Tampering | Gate scores the proposed call in `prepare_node`; execution consumes the scored proposal | Async tools and retries widen the window; not yet exercised |
| T4 | Tampering with monitor, policy, or record | Tampering | Ed25519 attestation + oracle binding (property 2) | Capture-fidelity gap, demonstrated and bounded in the TCB doc |
| T5 | Confused deputy on standing privilege | Elevation / LLM08 | HOLD/HITL enforcement modes; standing-authority modes (Act 7) | HITL fatigue; authority scoping is policy, not code |
| T6 | Exfiltration inside permitted calls | Info disclosure / LLM02 | Minimisation toggle in the advice domain (derived-only tool surface) | Laundered exfil via permitted lookup + utterance is ungoverned (out of scope, below) |
| T7 | Latency pressure forces fail-open | DoS / LLM04 | Two-speed gate; latency budget measured (latency-budget phase) | Fail-open/closed per tool class is a policy decision to document |
| T8 | Letter-vs-intent policy gap | Goodhart / LLM06 | SMACTR loop grows the threat model from caught failures (`governance/smactr.py`) | Coverage is never provably complete |
| T9 | Repudiation of what was enforced | Repudiation | Signed evidence pack, offline-verifiable; auditk protocol on the audit side | Log integrity inherits T4's boundary |
| T10 | Evade the detector while violating | Evasion / ATLAS | Enforcement does not depend on detection. Act 4: laundering the request fails (AUROC drop <= 0.001) because the violation *is* the mutation request; internals add nothing over text | Result is domain-specific; the probe stays a cheap cascade filter, never a control |

## Out of scope (stated, not hidden)

bossyk governs **tool calls**. It does not govern:

- **Speech acts**: an utterance can itself be the violation (a regulated
  financial promotion is unlawful the moment it is said). No tool call, no
  mediation point.
- **Check-outcomes**: call the check, receive "on list", dial anyway; the
  outreach-domain work (`ObservedAction`, outcome-aware rules) is scoped but
  not landed.
- **Model-internal deception** that never surfaces as an action. Act 4's
  negative results are the reason this is out of scope rather than a roadmap
  item: reading intent from internals added nothing over text in the regime
  tested.

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
