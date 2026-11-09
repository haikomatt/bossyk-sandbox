# Phase A: policy provenance binding

Every signed trace carries a manifest of the policy the gate ran (policy YAML
hash, instantiated fast rules, standing grants), and
`verify_policy_provenance` checks it against an approved manifest. See
`src/bossyk_sandbox/evidence/policy_provenance.py` and
`tests/unit/test_policy_provenance.py`.

## Issues & Fixes

1. **Design-rule-1 bypass in the call-site wiring (fixed, test-first).**
   The first Green wired every call site as
   `manifest_for_domain(domain, instruments, grants)` with a fallback to
   `cfg.policy_path.read_bytes()`. The YAML hash was therefore taken from the
   file at trace-build time, not from the bytes the `PolicyInstrument` loaded
   (a TOCTOU bypass), and a run in which no policy instrument existed still
   claimed a hash for a policy that never loaded.
   Fix: policy bytes come only from the `PolicyInstrument` among the gate's
   instruments; the disk fallback is removed; more than one instrument is
   refused; `policy_yaml_sha256` is nullable (`None` plus the
   `POLICY_NOT_LOADED` note when nothing loaded it); `MANIFEST_SCHEMA_VERSION`
   is 2.
   Guard: `tests/unit/test_policy_provenance_wiring.py` (Red commit `2b2f91d`,
   shown failing before the fix). It edits the policy file after load and
   builds the trace through `_live_run_result`, and asserts the file is never
   read at build time.
   Consequence: today no live call site has a `PolicyInstrument` (it is a slow
   instrument used in the benchmark), so live traces record
   `policy_yaml_sha256: null` and verify with `POLICY_NOT_LOADED`. The YAML
   hash is only bound where a policy instrument actually ran.

2. **`tests/unit/test_console.py`: `_fake_build_trace` widened (mechanical).**
   The two fakes had a fixed keyword signature and rejected the new optional
   `policy_manifest` argument of `build_trace`. They now accept
   `policy_manifest: object = None`. No assertion was changed or weakened.

## Notes for the PR description

- `docs/bench_output/evidence_pack_sample.pub` changed because
  `scripts/export_evidence_sample.py` generates a fresh key on every run. The
  sample trace and pack were regenerated because the sample trace now carries
  a manifest.
