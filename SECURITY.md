# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities privately via GitHub's security advisory
form for this repository ("Report a vulnerability" under the Security tab),
or by email to the address on the maintainer's GitHub profile. Please do not
open public issues for security reports.

Only the tip of `main` is supported.

## Scope worth knowing before you report

This repository is a research demonstrator, and its threat model is public
and unusually explicit: [`docs/threat-model.md`](docs/threat-model.md)
documents the trust boundary, a working forgery demonstration against the
attestation TCB, and a stated out-of-scope list (utterance-side coverage,
cross-session information flow, capture fidelity). A report that one of the
documented residuals exists is appreciated but expected; a report that a
*claimed* property fails — a path around the gate inside the shipped graph,
a signed pack that verifies after tampering, an oracle re-verification that
passes a forged trace — is exactly what we want to hear about.
