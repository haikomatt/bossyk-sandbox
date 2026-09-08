#!/usr/bin/env bash
# The audit leg: one pi session file in, a rendered report plus a signed,
# offline-verifiable evidence pack out. Pair with the gate's signed
# gate-events.jsonl and render the scorecard (see docs/gateproxy.md).
#
# Usage:  AUDITK_DIR=/path/to/auditk ISSUER="Your Name" ./audit.sh <pi-session.jsonl>
#   AUDITK_DIR  auditk checkout (default: ../../auditk, the sibling layout)
#   ISSUER      name written into the evidence pack's issuer field
#   SCORER      llm-judge (default; needs the judge's API key), nli, or jaccard
#   OUT_ROOT    where artefacts go (default: ./audit-out next to this script)
set -euo pipefail

SESSION="$(readlink -f "${1:?usage: ./audit.sh <pi-session.jsonl>}")"
HERE="$(cd "$(dirname "$0")" && pwd)"
AUDITK_DIR="${AUDITK_DIR:-$HERE/../../../auditk}"
ISSUER="${ISSUER:-operator}"
OUT_ROOT="${OUT_ROOT:-$HERE/audit-out}"

SESSION_ID="$(head -1 "$SESSION" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
OUT="$OUT_ROOT/$SESSION_ID"
mkdir -p "$OUT"
cd "$AUDITK_DIR"

KEY="$OUT_ROOT/audit-signing-key"
if [ ! -f "$KEY.ed25519" ]; then
  echo "==> one-time audit signing key"
  uv run auditk key-gen "$KEY"
fi

echo "==> ingest (pi session -> normalised trace)"
uv run auditk ingest --adapter pi --in "$SESSION" --out "$OUT/trace.json"

echo "==> report (deterministic structural post-mortem)"
uv run auditk report --adapter pi --in "$SESSION" --no-policy-context > "$OUT/report.md"

SCORER="${SCORER:-llm-judge}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export RUN_NLI_MODEL=1
[ "$SCORER" = "llm-judge" ] && export RUN_JUDGE_MODEL=1
echo "==> attest (signed evidence pack, scorer: $SCORER)"
uv run auditk attest --traces "$OUT/trace.json" \
  --signer "$KEY" \
  --issuer-name "$ISSUER" --agent-id pi --agent-version "${PI_VERSION:-unknown}" \
  --scorer "$SCORER" \
  --out "$OUT/evidence-pack.json"

echo "==> verify (offline, public key only)"
uv run auditk verify "$OUT/evidence-pack.json" --public-key "$KEY.ed25519.pub"

echo
echo "artefacts in $OUT (pass trace.json, evidence-pack.json, report.md and $KEY.ed25519.pub to the scorecard):"
ls -l "$OUT"
