"""The per-run scorecard (round-1 Phase 3): one self-contained HTML.

Merges the three artefacts one leg produces -- the auditk evidence pack,
the normalised trace, and the gate's signed event log (plus the report.md
text) -- into a single document: task identity, every gate decision with
the policy it enforced, drift metrics, both signature verifications
(performed here, offline, with public keys only), and the corroboration
cross-check that every gate BLOCK left a refusal in the session itself.

Untrusted inputs (session text, report markdown, event fields) are
escaped wherever rendered; the document carries no scripts and no
external references, so it travels over email like an evidence pack.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from auditk.attestation.canonical import canonicalize
from auditk.attestation.signer import LocalEd25519Verifier

from bossyk_sandbox.gateproxy.events import verify_event_log

_REFUSAL_MARKER = "[bossyk gate] BLOCKED"

_NOT_ENFORCED_FOOTER = (
    "Not enforced in v1: T1 (data-originated goals) and T7 (feature creep) "
    "are audit-side, detected post-hoc rather than gated; HOLD/human-in-the-"
    "loop decisions, response rewriting, OTel/Prometheus event export, and "
    "SPIFFE identity binding are round-2 scope. Absence is stated here so it "
    "is visible, not hidden."
)


@dataclass(frozen=True)
class ScorecardInputs:
    run_label: str
    task_name: str
    trace_path: Path
    evidence_pack_path: Path
    gate_events_path: Path
    report_md_path: Path | None
    pack_public_key_pem: str
    gate_public_key_pem: str
    generated_at: datetime


@dataclass(frozen=True)
class GateDecisionRow:
    kind: str
    tool_name: str | None
    verdict: str
    policy_id: str | None
    reason: str
    arguments: Any


@dataclass(frozen=True)
class GateSummary:
    decisions: list[GateDecisionRow]
    allowed_count: int
    blocked_count: int
    policies_triggered: list[str]


@dataclass(frozen=True)
class AuditSummary:
    drift_score: float | None
    flagged_count: int
    label_counts: dict[str, int]


@dataclass(frozen=True)
class Corroboration:
    block_events: int
    refusals_in_trace: int

    @property
    def corroborated(self) -> bool:
        return self.block_events == 0 or self.refusals_in_trace > 0


@dataclass(frozen=True)
class Verification:
    pack_verified: bool
    pack_detail: str
    gate_log_ok: bool
    gate_events_verified: int
    gate_bad_lines: list[int]


@dataclass(frozen=True)
class Scorecard:
    run_label: str
    task_name: str
    trace_id: str
    model: str | None
    generated_at: datetime
    gate: GateSummary
    audit: AuditSummary
    corroboration: Corroboration
    verification: Verification
    refusal_texts: list[str]
    report_md: str | None
    step_count: int


def _verify_pack(raw_pack: dict[str, Any], public_key_pem: str) -> tuple[bool, str]:
    signatures = raw_pack.get("signatures") or []
    if not signatures:
        return False, "evidence pack carries no signatures"
    manifest = {k: v for k, v in raw_pack.items() if k != "signatures"}
    canonical = canonicalize(manifest)
    verifier = LocalEd25519Verifier(public_key_pem)
    for signature in signatures:
        try:
            verifier.verify(canonical, signature["signature"])
        except Exception as exc:
            return False, f"signature verification failed ({exc.__class__.__name__})"
    return True, f"{len(signatures)} signature(s) verified against the trusted public key"


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = entry.get("event")
        if isinstance(event, dict):
            events.append(event)
    return events


def build_scorecard(inputs: ScorecardInputs) -> Scorecard:
    trace = json.loads(inputs.trace_path.read_text())
    raw_pack = json.loads(inputs.evidence_pack_path.read_text())
    events = _load_events(inputs.gate_events_path)

    decisions = [
        GateDecisionRow(
            kind=str(event.get("kind", "")),
            tool_name=event.get("tool_name"),
            verdict=str(event.get("verdict", "")),
            policy_id=event.get("policy_id"),
            reason=str(event.get("reason", "")),
            arguments=event.get("arguments"),
        )
        for event in events
    ]
    policies_triggered: list[str] = []
    for row in decisions:
        if row.policy_id and row.policy_id not in policies_triggered:
            policies_triggered.append(row.policy_id)
    gate = GateSummary(
        decisions=decisions,
        allowed_count=sum(1 for d in decisions if d.verdict == "allow"),
        blocked_count=sum(1 for d in decisions if d.verdict == "block"),
        policies_triggered=policies_triggered,
    )

    drift = raw_pack.get("drift_metrics") or {}
    per_step = drift.get("per_step") or {}
    audit = AuditSummary(
        drift_score=drift.get("drift_score"),
        flagged_count=len(drift.get("flagged_steps") or []),
        label_counts=dict(
            Counter(
                str(v.get("label"))
                for v in per_step.values()
                if isinstance(v, dict) and v.get("label")
            )
        ),
    )

    steps = trace.get("steps") or []
    refusal_texts = [
        str(step["action"]["payload"]["text"])
        for step in steps
        if isinstance(step.get("action"), dict)
        and isinstance(step["action"].get("payload"), dict)
        and _REFUSAL_MARKER in str(step["action"]["payload"].get("text", ""))
    ]
    corroboration = Corroboration(
        block_events=gate.blocked_count, refusals_in_trace=len(refusal_texts)
    )

    pack_verified, pack_detail = _verify_pack(raw_pack, inputs.pack_public_key_pem)
    gate_log = verify_event_log(inputs.gate_events_path, inputs.gate_public_key_pem)
    verification = Verification(
        pack_verified=pack_verified,
        pack_detail=pack_detail,
        gate_log_ok=gate_log.ok,
        gate_events_verified=gate_log.verified_count,
        gate_bad_lines=gate_log.bad_line_numbers,
    )

    model = next((str(e["model"]) for e in events if e.get("model")), None)
    report_md = (
        inputs.report_md_path.read_text()
        if inputs.report_md_path is not None and inputs.report_md_path.exists()
        else None
    )
    return Scorecard(
        run_label=inputs.run_label,
        task_name=inputs.task_name,
        trace_id=str(trace.get("trace_id", "")),
        model=model,
        generated_at=inputs.generated_at,
        gate=gate,
        audit=audit,
        corroboration=corroboration,
        verification=verification,
        refusal_texts=refusal_texts,
        report_md=report_md,
        step_count=len(steps),
    )


# --- rendering -------------------------------------------------------------

_CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 60rem;
       color: #1a1a1a; background: #fdfdfd; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left;
         font-size: 0.9rem; vertical-align: top; }
.badge { padding: 0.1rem 0.5rem; border-radius: 0.6rem; font-weight: 600;
         font-size: 0.8rem; }
.allow { background: #e2f4e2; color: #14601a; }
.block { background: #fbe0e0; color: #8f1414; }
.ok { color: #14601a; font-weight: 600; }
.bad { color: #8f1414; font-weight: 600; }
.meta { color: #555; font-size: 0.85rem; }
pre { background: #f4f4f4; padding: 0.8rem; overflow-x: auto; font-size: 0.8rem; }
footer { margin-top: 2.5rem; color: #555; font-size: 0.8rem;
         border-top: 1px solid #ccc; padding-top: 0.8rem; }
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _badge(verdict: str) -> str:
    css = "block" if verdict == "block" else "allow"
    return f'<span class="badge {css}">{_esc(verdict.upper())}</span>'


def _mark(ok: bool, good: str, bad: str) -> str:
    return (
        f'<span class="ok">&#10003; {_esc(good)}</span>'
        if ok
        else (f'<span class="bad">&#10007; {_esc(bad)}</span>')
    )


def render_scorecard_html(card: Scorecard) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{_esc(d.kind)}</td>"
        f"<td>{_esc(d.tool_name or '—')}</td>"
        f"<td>{_badge(d.verdict)}</td>"
        f"<td>{_esc(d.policy_id or '—')}</td>"
        f"<td>{_esc(d.reason)}</td>"
        "</tr>"
        for d in card.gate.decisions
    )
    labels = ", ".join(f"{_esc(k)}: {v}" for k, v in card.audit.label_counts.items()) or "—"
    drift = "—" if card.audit.drift_score is None else _esc(card.audit.drift_score)
    corroboration_line = (
        f"{card.corroboration.block_events} gate BLOCK(s) &harr; "
        f"{card.corroboration.refusals_in_trace} refusal(s) recorded in the session trace: "
        + _mark(
            card.corroboration.corroborated,
            "corroborated — the session file independently confirms the gate log",
            "NOT corroborated — the trace does not show the refusals the gate log claims",
        )
    )
    refusals = "\n".join(f"<pre>{_esc(text)}</pre>" for text in card.refusal_texts)
    report_section = (
        f"<h2>auditk report</h2>\n<pre>{_esc(card.report_md)}</pre>" if card.report_md else ""
    )
    gate_bad = (
        f" (bad lines: {_esc(card.verification.gate_bad_lines)})"
        if card.verification.gate_bad_lines
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Scorecard — {_esc(card.run_label)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Run scorecard: {_esc(card.task_name)}</h1>
<p class="meta">run label <strong>{_esc(card.run_label)}</strong>
 · session {_esc(card.trace_id)}
 · model {_esc(card.model or "unknown")}
 · {card.step_count} trace steps
 · generated {_esc(card.generated_at.isoformat())}</p>

<h2>Verification (offline, public keys only)</h2>
<p>Evidence pack: {_mark(card.verification.pack_verified, "verified", "FAILED")}
 <span class="meta">{_esc(card.verification.pack_detail)}</span><br>
Gate event log: {_mark(card.verification.gate_log_ok, "verified", "FAILED")}
 <span class="meta">{card.verification.gate_events_verified} event(s) verified{gate_bad}</span></p>

<h2>Gate decisions</h2>
<p>{card.gate.allowed_count} allowed · {card.gate.blocked_count} blocked
 · policies triggered: {_esc(", ".join(card.gate.policies_triggered) or "none")}</p>
<table>
<tr><th>kind</th><th>tool</th><th>verdict</th><th>policy</th><th>reason</th></tr>
{rows}
</table>

<h2>Corroboration</h2>
<p>{corroboration_line}</p>
{refusals}

<h2>Audit (post-hoc)</h2>
<p>drift score <strong>{drift}</strong> · {card.audit.flagged_count} flagged step(s)
 · labels: {labels}</p>
{report_section}

<footer>{_esc(_NOT_ENFORCED_FOOTER)}</footer>
</body>
</html>
"""


# --- CLI -------------------------------------------------------------------


@dataclass(frozen=True)
class ScorecardCliConfig:
    inputs: ScorecardInputs
    out_path: Path


def build_scorecard_config(argv: list[str]) -> ScorecardCliConfig:
    parser = argparse.ArgumentParser(
        prog="bossyk-scorecard",
        description="Render one self-contained HTML scorecard for one gated run.",
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--trace", required=True, help="auditk trace.json for the run.")
    parser.add_argument("--evidence-pack", required=True, help="auditk evidence-pack.json.")
    parser.add_argument("--gate-events", required=True, help="signed gate-events.jsonl.")
    parser.add_argument("--report-md", default=None, help="auditk report.md (optional).")
    parser.add_argument("--pack-public-key", required=True, help="public key PEM file (pack).")
    parser.add_argument("--gate-public-key", required=True, help="public key PEM file (gate).")
    parser.add_argument("--out", required=True, help="output scorecard HTML path.")
    parser.add_argument(
        "--generated-at",
        default=None,
        help="ISO timestamp override (default: now, UTC); fixed input -> identical output.",
    )
    args = parser.parse_args(argv)
    generated_at = (
        datetime.fromisoformat(args.generated_at) if args.generated_at else datetime.now(UTC)
    )
    inputs = ScorecardInputs(
        run_label=args.run_label,
        task_name=args.task_name,
        trace_path=Path(args.trace),
        evidence_pack_path=Path(args.evidence_pack),
        gate_events_path=Path(args.gate_events),
        report_md_path=Path(args.report_md) if args.report_md else None,
        pack_public_key_pem=Path(args.pack_public_key).read_text(),
        gate_public_key_pem=Path(args.gate_public_key).read_text(),
        generated_at=generated_at,
    )
    return ScorecardCliConfig(inputs=inputs, out_path=Path(args.out))


def main(argv: list[str] | None = None) -> None:
    import sys

    config = build_scorecard_config(sys.argv[1:] if argv is None else argv)
    card = build_scorecard(config.inputs)
    config.out_path.write_text(render_scorecard_html(card))
    print(f"Scorecard written to {config.out_path}")
