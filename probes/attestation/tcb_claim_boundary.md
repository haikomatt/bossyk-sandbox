# Attestation TCB: the forgery demonstration, the threat model, and the honest claim boundary

**Status:** supported (demonstrated in code).
**Demo:** `tests/unit/test_attestation_tcb_forgery.py`.
**Signing path under analysis:** `evidence/trace.py` (the tracer),
`evidence/signed_trace.py` and `evidence/pack.py` (the sign/verify paths).

## The one-line result

A bossyk signature proves a signed trace has not been altered *since signing*.
It does not prove *fidelity of capture*: that the tracer which built the trace
told the truth about the run. Those are different guarantees, and the gap
between them is bossyk's honest attestation claim boundary.

## The demonstration (Part A)

`test_attestation_tcb_forgery.py` runs two arms against the *same* trusted
public key, both ending in `verify_signed_trace(...) is True`:

1. **Honest arm.** The sample gated trace, in which an unauthorised cancel is
   blocked, is signed and verifies. The blocked step is present and its
   incident-response control (`soc2:cc7-4`) shows up in the signed coverage
   roll-up. The record tells the truth.
2. **Compromised-tracer arm.** A test-only adversary, `forge_trace`, stands in
   for a doctored `build_trace`. From the same dirty run it emits a trace that
   either:
   - **omits** the blocked step (the violation simply never happened), or
   - **relabels** it as an allowed action, re-tagged through the legitimate
     tracer machinery so it carries none of the block-only control tags and
     reads as a genuinely permitted step.
   Both forgeries are signed with the same local key and both verify against
   the same public key.

The landing assertion: a verifier holding only the public key and a signed
bundle cannot distinguish the forged record from the honest one. Both verify;
the coverage roll-ups are internally self-consistent in every case; yet the
forged bundles misrepresent the run (`soc2:cc7-4` is present in the honest
coverage and absent from the forged coverage, and the forged bundles are
genuinely different records that only *verify* like the honest one).

`forge_trace` lives in `tests/`, not in the shippable package, on purpose: no
forgery tool ships. It exists to prove a property of signatures, not to add a
capability.

## The trusted computing base (Part B)

The **TCB** for a bossyk attestation is: the gate that decides
allow/block/HITL, the tracer (`make_attested_step` / `build_trace`) that
records those decisions as steps, and the process that holds the local Ed25519
private key. Everything inside it is trusted by construction today. How each
element is attacked:

| TCB element | Attack | Does the current design catch it? |
|---|---|---|
| Tracer code | Doctored `build_trace` omits or relabels steps (Part A) | No. Fidelity of capture is not attested. |
| Local signing key | Key exfiltrated from disk, attacker signs any trace offline | No. A valid signature over a fabricated trace verifies. |
| Operator / build | A modified build runs a tampered gate + tracer | No. Nothing binds the running code to expected code. |
| Selective omission | Drop the steps you dislike before signing | No. The roll-up is derived from whatever steps are present. |
| Post-hoc tampering | Alter a trace *after* it was signed | **Yes.** The signature does catch alteration after signing. |

The single honest summary: the signature detects tampering *after* signing and
nothing about fidelity *before or during* signing.

## The mitigation ladder (Part C)

Ranked by what each rung buys against what it costs. The product now sits on
**rung 4 for the bindable subset** (built, see below); rung 3 co-signature is
the remaining collusion-resistant gap.

1. **Status quo: trust the operator.** Honest claim only: integrity after
   signing, given an honest signer. Free. This is where we are.
2. **Transparency log** (append-only public ledger of trace hashes).
   Non-repudiation of *what was published* plus tamper-evidence over time.
   Still does not prove capture fidelity. Low/medium cost.
3. **Independent co-signer / second enforcement point.** A separately-operated
   gate co-signs, so a forgery requires collusion between two independently
   run enforcers. Medium cost.
4. **External-oracle binding (the bossyk-native rung). BUILT for the
   reproducible-oracle subset.** Bind the trace to independently-observable
   effects, the outreach check-outcome oracles, a real clock, the suppression
   list, so a forged trace contradicts checkable reality. A forged "clean" run
   that claims a suppressed number was never dialled is refuted against the
   oracle's own record. Effects are externally verifiable even when intent is
   not. This is the cheap, in-house rung and it ties directly to the outreach
   domain work.
   - **Slice 1 (`evidence/oracle_binding.py`):** the oracle result a
     `RequirePassedCheck` decision relied on is bound into the attested step's
     signed metadata (an `OracleObservation`: check, key, claimed result,
     snapshot ref). Signed and tamper-evident; check-free traces are
     byte-identical.
   - **Slice 2 (`evidence/oracle_reverify.py`):** `verify_against_oracle`
     re-checks each bound observation against an independent, snapshot-pinned
     oracle. The Part-A forgeries now FAIL re-verification while still passing
     signature verification: *relabel* is caught as a `contradict` (claimed
     result != oracle truth); *omit* is caught as an omission (a key the oracle
     logged as queried that no bound observation accounts for).
   - **Honest limits (enforced, not hidden):** (i) binding into the same
     key-signed payload buys nothing alone, the value is the *independent*
     re-check; (ii) only violations reducing to a checkable effect/precondition
     are bindable, so utterance boundaries stay `unbindable`; (iii) a check with
     no oracle or a mismatched snapshot is reported `unbindable`, never a false
     contradiction.
   - **Remaining gap:** the oracle here is verifier-reproducible, not
     independently *attested*. A forger who also controls the oracle's data is
     only caught once the oracle co-signs its observations with a separate key
     (rung 3 overlap, below). Not built here.
5. **TEE / remote attestation / measured boot.** Hardware-rooted proof that the
   code producing the trace is the expected, un-tampered code. Strongest,
   most expensive, the enterprise endgame. Named, not built here.

## The honest claim boundary (Part D)

> A signed bossyk trace proves, today, that the record verifies offline against
> a named public key and has not been altered since it was signed, and that the
> enforcement decisions it records were the decisions the gate made. It does
> not yet prove that the runtime which produced and signed the trace was the
> expected, un-tampered code: a compromised tracer or an exfiltrated signing
> key can produce a clean, correctly-signed trace of a dirty run that a
> verifier cannot distinguish from an honest one. Closing that gap is a
> capture-fidelity control (rung 4 external-oracle binding, or rung 5 hardware
> attestation), not a property of the signature itself.

This is the paragraph to publish. Own the boundary and sell the ladder above
it, rather than let a CISO or auditor discover the gap and break the pitch with
it.

## Recommendation

- **Publish the claim boundary now.** It is already slotted into the CISO
  threat-model doc (blind spot #2) and the assurance positioning note.
- **Rung 4 is built for the reproducible-oracle subset** (`oracle_binding.py` +
  `oracle_reverify.py`): the bound-and-signed observation is re-checkable
  against an independent oracle, so a relabelled or omitted violation is
  falsifiable by an auditor who holds the oracle. The updated claim: *for
  boundaries with a reproducible oracle, a signed bossyk trace is falsifiable
  against that oracle.*
- **Next: rung 3 co-signature** (independent oracle attests its own observations
  with a separate key) to close the remaining collusion case, and wire the three
  live runtime call sites to bind real histories.
- **Name TEE / remote attestation (rung 5) as the endgame**, for the enterprise
  tier, not now.
