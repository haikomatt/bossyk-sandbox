"""Incident report builder (Build C): one self-contained incident.html,
built entirely by DETERMINISTIC TEMPLATING over the same three artefacts
the scorecard reads (trace, evidence pack, gate events) plus their
public keys -- no LLM anywhere in this module, by design: an incident
record has to say the same thing every time it is regenerated from the
same inputs, which an LLM cannot promise.

An incident exists only when at least one of three conditions holds:
 - any gate BLOCK event (something was actually stopped), or
 - a corroboration failure (a BLOCK with no matching refusal recorded in
   the session trace -- the gate says it acted, the session doesn't
   agree), or
 - a verification failure (the evidence pack or gate-log signature does
   not check out against the trusted public key).
When none of the three holds, nothing is written: the CLI prints
"no incident: <one-line reason>" and exits 0, per the brief -- a clean
run should leave no artefact behind to imply otherwise.

`determine_incident`/`build_incident` reuse
`bossyk_sandbox.gateproxy.scorecard.build_scorecard` for the
gate/corroboration/verification/provenance numbers rather than
recomputing them -- extend, don't duplicate. This module does its own
light parsing only for what the scorecard doesn't already expose:
per-event timestamps and step ordering, needed to write the prose
timeline.

Severity: "contained" when the trigger is a BLOCK that IS corroborated
and verified (the gate did its job and the story checks out);
"integrity" when a verification or corroboration failure is present at
all, since that means the record itself cannot be fully trusted --
outranks "contained" even if a BLOCK is also present.

Timeline construction is fully deterministic: each gate BLOCK event (in
gate-log file order) is paired by ORDINAL POSITION with the Nth
refusal-marked utterance step found in the trace (in trace file order),
not by comparing timestamps -- gate-event timestamps are wall-clock at
signing time while trace-step timestamps come from the session/adapter,
so the two clocks are not assumed to be comparable instant-for-instant.
When a run has fewer refusal steps than BLOCK events, the unmatched
BLOCK events say so explicitly ("no corresponding refusal was found in
the session trace") rather than guessing a pairing.

Handles legacy input gracefully: gate events that predate Build A carry
neither `pack_sha256` nor `gate_version`; both render as "not recorded"
rather than raising. This is proven both by a unit test and by the
brief's own field check against the real (pre-Build-A) smoke-run
gate-log artefacts.

Untrusted content -- tool names, arguments, reasons, policy ids, step
ids, trace text -- is escaped wherever rendered, exactly as scorecard.py
does. The document is self-contained and script-free.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bossyk_sandbox.gateproxy.events import effective_verdict, load_events
from bossyk_sandbox.gateproxy.scorecard import Scorecard, ScorecardInputs, build_scorecard

_REFUSAL_MARKER = "[bossyk gate] BLOCKED"

_FOOTER = (
    "This record is produced to support serious-incident documentation duties (in the style "
    "of EU AI Act Article 73) for a governed agent run. It is generated evidence, not legal "
    "advice; whether a given event meets any jurisdiction's reporting threshold is a "
    "determination for the deployer, not this tool."
)


@dataclass(frozen=True)
class IncidentDecision:
    triggered: bool
    severity: str | None
    reason: str


@dataclass(frozen=True)
class ArtefactRow:
    filename: str
    sha256: str
    status: str


@dataclass(frozen=True)
class Incident:
    run_label: str
    task_name: str
    trace_id: str
    generated_at: datetime
    severity: str
    reason: str
    timeline: list[str]
    artefacts: list[ArtefactRow]
    pack_sha256: str | None
    gate_version: str | None


@dataclass(frozen=True)
class IncidentCliConfig:
    inputs: ScorecardInputs
    out_path: Path


def _verification_reason(card: Scorecard) -> str:
    return (
        f"{card.gate.blocked_count} BLOCK event(s), corroboration "
        f"{'ok' if card.corroboration.corroborated else 'FAILED'}, "
        f"pack verification {'ok' if card.verification.pack_verified else 'FAILED'}, "
        f"gate log verification {'ok' if card.verification.gate_log_ok else 'FAILED'}"
    )


def determine_incident(card: Scorecard) -> IncidentDecision:
    """The three OR'd trigger conditions from the module docstring,
    evaluated over an already-built Scorecard's own summaries."""
    verification_failed = not card.verification.pack_verified or not card.verification.gate_log_ok
    corroboration_failed = not card.corroboration.corroborated
    has_block = card.gate.blocked_count > 0
    reason = _verification_reason(card)

    if verification_failed or corroboration_failed:
        return IncidentDecision(triggered=True, severity="integrity", reason=reason)
    if has_block:
        return IncidentDecision(triggered=True, severity="contained", reason=reason)
    return IncidentDecision(triggered=False, severity=None, reason=reason)


def _is_refusal_step(step: dict[str, Any]) -> bool:
    action = step.get("action")
    if not isinstance(action, dict):
        return False
    payload = action.get("payload")
    if not isinstance(payload, dict):
        return False
    return _REFUSAL_MARKER in str(payload.get("text", ""))


def _format_time(timestamp: Any) -> str:
    """HH:MM:SSZ, always normalised to UTC -- the compact form the
    brief's own example sentence uses ("At 22:41:03Z ...")."""
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return "an unrecorded time"
    return parsed.astimezone(UTC).strftime("%H:%M:%SZ")


def _compact_arguments(arguments: Any) -> str:
    if isinstance(arguments, dict):
        inner = ", ".join(f"{k}: {v}" for k, v in arguments.items())
        return f"{{{inner}}}"
    return str(arguments)


def _block_sentences(
    event: dict[str, Any], event_index: int, refusal_step: dict[str, Any] | None, steps_after: int
) -> str:
    timestamp = _format_time(event.get("timestamp"))
    tool_name = event.get("tool_name") or "an action"
    args_repr = _compact_arguments(event.get("arguments"))
    policy_id = event.get("policy_id")
    proposal = (
        f"At {timestamp} the model proposed {tool_name} {args_repr} (gate event {event_index})."
    )
    under = f"under {policy_id}" if policy_id else f"({event.get('reason', 'no reason recorded')})"
    if event.get("verdict") == "hold":
        block_clause = f"The gate held it {under} and it was refused ({event.get('resolution')})."
    else:
        block_clause = f"The gate blocked it {under}."
    if refusal_step is None:
        refusal_clause = "No corresponding refusal was found in the session trace."
        ending = ""
    else:
        step_id = refusal_step.get("step_id", "?")
        refusal_clause = f"The refusal was recorded in the session at step {step_id}."
        ending = (
            "This was the final step of the run."
            if steps_after == 0
            else f"The run ended {steps_after} step(s) later."
        )
    return " ".join(part for part in (proposal, block_clause, refusal_clause, ending) if part)


def _build_timeline(
    card: Scorecard, raw_events: list[dict[str, Any]], trace_steps: list[dict[str, Any]]
) -> list[str]:
    sentences: list[str] = []
    refusal_steps = [s for s in trace_steps if _is_refusal_step(s)]
    block_ordinal = 0
    for index, event in enumerate(raw_events, start=1):
        if effective_verdict(event) != "block":
            continue
        refusal_step = refusal_steps[block_ordinal] if block_ordinal < len(refusal_steps) else None
        steps_after = 0
        if refusal_step is not None:
            steps_after = len(trace_steps) - 1 - trace_steps.index(refusal_step)
        sentences.append(_block_sentences(event, index, refusal_step, steps_after))
        block_ordinal += 1

    if not card.verification.pack_verified:
        sentences.append(
            f"The evidence pack failed signature verification ({card.verification.pack_detail})."
        )
    if not card.verification.gate_log_ok:
        bad = ", ".join(str(n) for n in card.verification.gate_bad_lines)
        sentences.append(f"The gate event log failed signature verification at line(s) {bad}.")
    if not card.corroboration.corroborated:
        sentences.append(
            f"{card.corroboration.block_events} gate BLOCK event(s) were recorded but only "
            f"{card.corroboration.refusals_in_trace} refusal(s) were found in the session "
            "trace (corroboration failed)."
        )
    if not sentences:
        sentences.append("No notable events were recorded for this run.")
    return sentences


def _artefact_row(path: Path, status: str) -> ArtefactRow:
    return ArtefactRow(
        filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), status=status
    )


def build_incident(
    inputs: ScorecardInputs, card: Scorecard, decision: IncidentDecision
) -> Incident:
    raw_events = load_events(inputs.gate_events_path)
    trace = json.loads(inputs.trace_path.read_text())
    trace_steps = trace.get("steps") or []
    timeline = _build_timeline(card, raw_events, trace_steps)
    artefacts = [
        _artefact_row(inputs.trace_path, "not independently signed"),
        _artefact_row(
            inputs.evidence_pack_path, "verified" if card.verification.pack_verified else "FAILED"
        ),
        _artefact_row(
            inputs.gate_events_path, "verified" if card.verification.gate_log_ok else "FAILED"
        ),
    ]
    severity = decision.severity or "contained"
    return Incident(
        run_label=inputs.run_label,
        task_name=inputs.task_name,
        trace_id=str(trace.get("trace_id", "")),
        generated_at=inputs.generated_at,
        severity=severity,
        reason=decision.reason,
        timeline=timeline,
        artefacts=artefacts,
        pack_sha256=card.pack_sha256,
        gate_version=card.gate_version,
    )


# --- rendering -------------------------------------------------------------

_CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 60rem;
       color: #1a1a1a; background: #fdfdfd; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left;
         font-size: 0.9rem; vertical-align: top; word-break: break-all; }
.badge { padding: 0.1rem 0.5rem; border-radius: 0.6rem; font-weight: 600; font-size: 0.8rem; }
.contained { background: #fff3cd; color: #7a5b00; }
.integrity { background: #fbe0e0; color: #8f1414; }
.meta { color: #555; font-size: 0.85rem; }
footer { margin-top: 2.5rem; color: #555; font-size: 0.8rem;
         border-top: 1px solid #ccc; padding-top: 0.8rem; }
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _provenance_line(incident: Incident) -> str:
    pack = f"pack {incident.pack_sha256[:12]}…" if incident.pack_sha256 else "pack not recorded"
    gate = (
        f"gate v{incident.gate_version}" if incident.gate_version else "gate version not recorded"
    )
    return f"{pack} · {gate}"


def render_incident_html(incident: Incident) -> str:
    timeline_html = "\n".join(f"<p>{_esc(sentence)}</p>" for sentence in incident.timeline)
    artefact_rows = "\n".join(
        f"<tr><td>{_esc(row.filename)}</td><td>{_esc(row.sha256)}</td>"
        f"<td>{_esc(row.status)}</td></tr>"
        for row in incident.artefacts
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Incident report — {_esc(incident.run_label)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Incident report: {_esc(incident.task_name)}</h1>
<p class="meta">run label <strong>{_esc(incident.run_label)}</strong>
 · session {_esc(incident.trace_id)}
 · {_esc(_provenance_line(incident))}
 · generated {_esc(incident.generated_at.isoformat())}</p>

<p><span class="badge {_esc(incident.severity)}">{_esc(incident.severity.upper())}</span>
 <span class="meta">{_esc(incident.reason)}</span></p>

<h2>Timeline</h2>
{timeline_html}

<h2>Artefact inventory</h2>
<table>
<tr><th>filename</th><th>sha256</th><th>status</th></tr>
{artefact_rows}
</table>

<footer>{_esc(_FOOTER)}</footer>
</body>
</html>
"""


# --- CLI -------------------------------------------------------------------


def build_incident_config(argv: list[str]) -> IncidentCliConfig:
    parser = argparse.ArgumentParser(
        prog="bossyk-incident",
        description="Render one self-contained incident.html for a gated run, if warranted.",
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--trace", required=True, help="auditk trace.json for the run.")
    parser.add_argument("--evidence-pack", required=True, help="auditk evidence-pack.json.")
    parser.add_argument("--gate-events", required=True, help="signed gate-events.jsonl.")
    parser.add_argument("--pack-public-key", required=True, help="public key PEM file (pack).")
    parser.add_argument("--gate-public-key", required=True, help="public key PEM file (gate).")
    parser.add_argument("--out", required=True, help="output incident HTML path.")
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
        report_md_path=None,
        pack_public_key_pem=Path(args.pack_public_key).read_text(),
        gate_public_key_pem=Path(args.gate_public_key).read_text(),
        generated_at=generated_at,
    )
    return IncidentCliConfig(inputs=inputs, out_path=Path(args.out))


def main(argv: list[str] | None = None) -> None:
    import sys

    config = build_incident_config(sys.argv[1:] if argv is None else argv)
    card = build_scorecard(config.inputs)
    decision = determine_incident(card)
    if not decision.triggered:
        print(f"no incident: {decision.reason}")
        return
    incident = build_incident(config.inputs, card, decision)
    config.out_path.write_text(render_incident_html(incident))
    print(f"Incident report written to {config.out_path}")


if __name__ == "__main__":
    main()
