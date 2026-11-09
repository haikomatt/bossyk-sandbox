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

3. **CI failed: provenance tests read the private bossyk policy files (fixed).**
   27 tests in `test_policy_provenance.py` and `test_policy_provenance_wiring.py`
   failed in CI with `FileNotFoundError` on `.../Projects/bossyk/data/policies/*.yaml`.
   The policy YAMLs, and the `judge` / `policy` modules that
   `build_default_policy_instrument` imports, live in the private bossyk
   checkout (`BOSSYK_ROOT`). CI does not have it; dev machines do, which is why
   every local gate passed. Reproduced locally with `BOSSYK_ROOT=/nonexistent/bossyk`
   (a missing root, as in CI; an empty directory is not equivalent because
   `requires_bossyk_checkout` only checks that the root exists). That run is
   now part of the gate.
   Fix: tests that need only some policy bytes use a committed synthetic
   placeholder (`tests/fixtures/policies/synthetic-support-v1.yaml`); the
   `build_default_policy_instrument` tests run in CI against a `fake_bossyk`
   fixture that stands in for the private `load_policy` and `judge` (the fake
   `load_policy` reads the path it is given, so the hash-of-loaded-bytes and
   TOCTOU properties are still exercised). Only one test needs the real loader
   and is gated by `requires_bossyk_checkout`:
   `test_policy_instrument_with_the_real_bossyk_loader` (skipped in CI).
   `requires_bossyk_checkout` moved to `tests/unit/_bossyk.py`.

## Notes for the PR description

- `docs/bench_output/evidence_pack_sample.pub` changed because
  `scripts/export_evidence_sample.py` generates a fresh key on every run. The
  sample trace and pack were regenerated because the sample trace now carries
  a manifest.

# Phase B: standing integrity guard

## Issues & Fixes

1. **Retail return reconciliation: the test's assumption was wrong (test changed, code not).**
   `test_retail_return_resolved_amount_equals_observed_refund` asserted that the
   amount `resolve_amount` prices before a return equals the `payment_history`
   delta after it. It failed on real data (order `#W4817420`: 47.76 vs 0).
   tau2's `return_delivered_order_items` only records a request: it sets
   `order.status = "return requested"`, `order.return_items` and
   `return_payment_method_id`, and moves no money at execution. A requested
   return still consumes refund authority, so the test now reconciles the
   resolved amount against the sum of the prices of `order.return_items` after
   execution (47.76 == 47.76). Cancellation and payment change still reconcile
   on the `payment_history` delta. Renamed to
   `test_retail_return_resolved_amount_equals_priced_return_items`.
   The Red commit already contains the corrected test.
2. **Finding 4 (live `apply_discount` amount resolved to `None`).** See the
   fix commit below.
