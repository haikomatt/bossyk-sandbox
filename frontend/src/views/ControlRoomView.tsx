import { useCallback, useEffect, useRef, useState } from "react";
import type { ReplayCompleteEvent, ReplayControl, ReplayEvent, ReplayHitlItem } from "../api";
import { replaySocketUrl, startLive, startReplay } from "../api";
import { navigate } from "../router";

type RunKind = "replay" | "live";

const PRESET_ID = "retail-weak-modes";
const PACING_S = 1.2;

const SEVERITY_RANK: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3 };

type RunStatus = "idle" | "running" | "done" | "error";

interface TurnState {
  label: string;
  role: string;
  probeId: string | null;
  toolName: string;
  args: Record<string, unknown>;
  autoReason: string;
  mode: string | null;
  modeReason: string | null;
  status: "held" | "done";
  verdict?: string;
  controls?: ReplayControl[];
}

function heldTurn(event: Extract<ReplayEvent, { type: "held" }>): TurnState {
  return {
    label: event.label,
    role: event.role,
    probeId: event.probe_id,
    toolName: event.tool_name,
    args: event.arguments,
    autoReason: event.auto_reason,
    mode: event.mode,
    modeReason: null,
    status: "held",
  };
}

/** Drives one preset replay over the console websocket. Held/step events
 * arrive strictly interleaved (held i, then step i, then held i+1 …), so each
 * "step" finalizes the most recent still-held turn. Escalate steps also push
 * onto the live HITL queue. */
function useReplaySocket() {
  const [turns, setTurns] = useState<TurnState[]>([]);
  const [hitlQueue, setHitlQueue] = useState<ReplayHitlItem[]>([]);
  const [summary, setSummary] = useState<ReplayCompleteEvent | null>(null);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const run = useCallback(
    (kind: RunKind) => {
      if (status === "running") return;
      setTurns([]);
      setHitlQueue([]);
      setSummary(null);
      setError(null);
      setStatus("running");

      const ws = new WebSocket(replaySocketUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        const starting = kind === "live" ? startLive(PACING_S) : startReplay(PRESET_ID, PACING_S);
        starting
          .then((result) => {
            if (result.status !== "started") {
              const messages: Record<string, string> = {
                live_disabled:
                  "Live runs are disabled — set RUN_LIVE_CONSOLE=1 (and an agent key) in the console environment to enable.",
                no_api_key:
                  "Live run needs an agent key — set FIREWORKS_API_KEY in the console environment, then Run live again.",
              };
              setError(messages[result.status] ?? `${kind} could not start: ${result.status}`);
              setStatus("error");
              ws.close();
            }
          })
          .catch((err: unknown) => {
            setError(err instanceof Error ? err.message : String(err));
            setStatus("error");
            ws.close();
          });
      };

    ws.onmessage = (message) => {
      const event = JSON.parse(message.data as string) as ReplayEvent;
      if (event.type === "held") {
        setTurns((prev) => [...prev, heldTurn(event)]);
      } else if (event.type === "step") {
        setTurns((prev) => {
          const next = [...prev];
          for (let i = next.length - 1; i >= 0; i--) {
            if (next[i].status === "held") {
              next[i] = {
                ...next[i],
                status: "done",
                verdict: event.verdict,
                mode: event.mode,
                modeReason: event.mode_reason,
                controls: event.controls,
              };
              break;
            }
          }
          return next;
        });
        if (event.mode === "escalate" && event.hitl) {
          const item = event.hitl;
          setHitlQueue((prev) =>
            [...prev, item].sort(
              (a, b) => (SEVERITY_RANK[a.severity] ?? 9) - (SEVERITY_RANK[b.severity] ?? 9),
            ),
          );
        }
      } else if (event.type === "session_complete") {
        setSummary(event);
        setHitlQueue(event.hitl_queue);
        setStatus("done");
        ws.close();
      }
    };

    ws.onerror = () => {
      setError("websocket error -- is the console backend running?");
      setStatus("error");
    };
  }, [status]);

  useEffect(() => () => wsRef.current?.close(), []);

  return { turns, hitlQueue, summary, status, error, run };
}

// --- resolution mode ---------------------------------------------------------

function modeLabel(mode: string): string {
  return mode.toUpperCase();
}

function ModeBadge({ mode }: { mode: string }) {
  return <span className={`cr-mode cr-mode--${mode}`}>{modeLabel(mode)}</span>;
}

// --- compliance controls (folded baseline + per-action) ----------------------

const BASELINE_BASES = new Set(["substrate", "verdict:gated"]);

function basisLabel(basis: string): string {
  switch (basis) {
    case "substrate":
      return "logged & signed";
    case "verdict:gated":
      return "passed risk gate";
    case "verdict:blocked":
      return "remediated · blocked";
    case "verdict:overridden":
      return "human oversight";
    case "boundary:cancellation":
      return "consumer-duty boundary";
    case "data-class:personal-data":
      return "personal-data access";
    default:
      return basis;
  }
}

function ControlChip({ control }: { control: ReplayControl }) {
  return (
    <span className="framework-tag" title={`${control.title} — ${control.ref}`}>
      <span className="framework-tag__name">{control.framework}</span>
      <span className="framework-tag__ref">{control.control}</span>
      <span className="ev-basis">{basisLabel(control.basis)}</span>
    </span>
  );
}

function ControlPills({ controls }: { controls: ReplayControl[] }) {
  if (controls.length === 0) return null;
  const baseline = controls.filter((control) => BASELINE_BASES.has(control.basis));
  const specific = controls.filter((control) => !BASELINE_BASES.has(control.basis));
  return (
    <div className="cr-controls">
      {baseline.length > 0 && (
        <details className="cr-baseline">
          <summary className="cr-baseline__summary">audit baseline ✓ ({baseline.length})</summary>
          <div className="ev-tags cr-baseline__tags">
            {baseline.map((control) => (
              <ControlChip key={`${control.ref}:${control.basis}`} control={control} />
            ))}
          </div>
        </details>
      )}
      {specific.length > 0 && (
        <div className="ev-tags">
          {specific.map((control) => (
            <ControlChip key={`${control.ref}:${control.basis}`} control={control} />
          ))}
        </div>
      )}
    </div>
  );
}

// --- the trace ---------------------------------------------------------------

function TurnRow({ turn }: { turn: TurnState }) {
  const args = JSON.stringify(turn.args);
  const mode = turn.mode;
  return (
    <div className={`cr-turn cr-turn--${mode ?? turn.role} cr-turn--${turn.status}`}>
      <div className="cr-turn__head">
        {turn.status === "held" ? (
          <span className="cr-held">HELD · scoring…</span>
        ) : (
          <>
            {mode && <ModeBadge mode={mode} />}
            <span className={`cr-verdict cr-verdict--${turn.verdict}`}>gate: {turn.verdict}</span>
          </>
        )}
        <span className="cr-turn__label">{turn.label}</span>
        {turn.probeId && <span className="cr-turn__probe">{turn.probeId}</span>}
      </div>
      <code className="ev-step__call">
        {turn.toolName}({args})
      </code>
      {turn.status === "done" && turn.modeReason && (
        <p className="cr-turn__reason">{turn.modeReason}</p>
      )}
      <p className="cr-turn__gate-reason">{turn.autoReason}</p>
      {turn.status === "done" && turn.controls && <ControlPills controls={turn.controls} />}
    </div>
  );
}

// --- HITL queue --------------------------------------------------------------

function HitlQueueItem({ item }: { item: ReplayHitlItem }) {
  const [outcome, setOutcome] = useState<"approved" | "denied" | null>(null);
  return (
    <div className={`hitl-item hitl-item--${item.severity}`}>
      <div className="hitl-item__head">
        <span className={`hitl-sev hitl-sev--${item.severity}`}>{item.severity}</span>
        <span className="hitl-item__label">{item.label}</span>
      </div>
      <p className="hitl-item__reason">{item.reason}</p>
      <code className="hitl-item__tool">{item.tool_name}</code>
      {outcome ? (
        <p className="hitl-item__resolved">
          {outcome} · {item.resolution} ({item.channel})
        </p>
      ) : (
        <div className="hitl-item__actions">
          <button type="button" className="hitl-btn hitl-btn--approve" onClick={() => setOutcome("approved")}>
            Approve
          </button>
          <button type="button" className="hitl-btn hitl-btn--deny" onClick={() => setOutcome("denied")}>
            Deny
          </button>
        </div>
      )}
    </div>
  );
}

function HitlQueuePanel({ queue }: { queue: ReplayHitlItem[] }) {
  return (
    <aside className="hitl-panel">
      <h3 className="hitl-panel__title">HITL review queue ({queue.length})</h3>
      <p className="hitl-panel__sub">
        Hard-cell escalations — irreversible or over standing authority. Resolved async (callback),
        never an in-call hold.
      </p>
      {queue.length === 0 ? (
        <p className="hitl-panel__empty">No escalations — everything resolved in-band.</p>
      ) : (
        <div className="hitl-panel__list">
          {queue.map((item, index) => (
            <HitlQueueItem key={`${item.tool_name}:${index}`} item={item} />
          ))}
        </div>
      )}
    </aside>
  );
}

// --- completion --------------------------------------------------------------

function CompletionBanner({ summary }: { summary: ReplayCompleteEvent }) {
  const escalated = summary.hitl_queue.length;

  if (summary.live) {
    return (
      <div className="cr-summary">
        <div className="cr-summary__headline">
          Live run · {summary.step_count} actions governed · {escalated} escalated to human review
        </div>
        <p className="cr-summary__body">
          A live under-specified agent was driven through the real gate. Each resolution mode was{" "}
          <strong>derived from the gate verdict</strong>, not authored: skip-lookup writes{" "}
          <strong>redirected</strong>, PII access <strong>stepped up</strong> to the customer, and only
          irreversible over-authority actions <strong>escalated</strong> to the queue. (<code>defer</code>{" "}
          awaits the standing-authority model.)
        </p>
      </div>
    );
  }

  const artifact = summary.source_artifact;
  return (
    <div className="cr-summary">
      <div className="cr-summary__headline">
        {summary.harm_prevented ?? 0} harmful actions stopped pre-execution · {escalated} escalated to
        human review
      </div>
      <p className="cr-summary__body">
        Skip-lookup writes were <strong>redirected</strong> to the compliant path, a PII disclosure
        bounced to customer <strong>step-up</strong>, a within-limit refund was <strong>deferred</strong>{" "}
        to async approval, and only the irreversible over-authority actions <strong>escalated</strong>{" "}
        to the queue — no in-call hold. The structural gate verdicts recompute from the same gate the
        live run used; the resolution modes are the demo scenario.
      </p>
      <div className="cr-summary__refs">
        {summary.story_claim && (
          <button type="button" className="artifact-ref artifact-ref--link" onClick={() => navigate("story")}>
            story claim: {summary.story_claim}
          </button>
        )}
        {artifact && (
          <button
            type="button"
            className="artifact-ref artifact-ref--link"
            onClick={() => navigate("bench_output", artifact.split("/").pop() ?? "")}
          >
            source: {artifact}
          </button>
        )}
      </div>
    </div>
  );
}

export function ControlRoomView() {
  const { turns, hitlQueue, summary, status, error, run } = useReplaySocket();
  const started = turns.length > 0 || status !== "idle";

  return (
    <div className="control-room">
      <header className="control-room__header">
        <div>
          <h2 className="control-room__title">Control room — resolution modes &amp; HITL queue</h2>
          <p className="control-room__sub">
            An under-specified retail agent, driven through the real gate. Each action resolves into a
            mode — <strong>allow · redirect · defer · step-up · escalate</strong> — and only the
            hard-cell escalations reach the review queue on the right. <em>Replay</em> plays a committed
            scenario; <em>Run live</em> drives the real agent (needs an agent key).
          </p>
        </div>
        <div className="control-room__actions">
          <button
            type="button"
            className="control-room__run"
            onClick={() => run("replay")}
            disabled={status === "running"}
          >
            {status === "running" ? "Running…" : "Run replay"}
          </button>
          <button
            type="button"
            className="control-room__run control-room__run--ghost"
            onClick={() => run("live")}
            disabled={status === "running"}
          >
            Run live
          </button>
        </div>
      </header>

      {error && <p className="control-room__error">{error}</p>}

      {status === "idle" && turns.length === 0 && (
        <p className="control-room__placeholder">
          Press “Run replay” to watch each proposed action be held, scored by the gate, and resolved
          into its enforcement mode — with irreversible over-authority actions escalating to the HITL
          queue.
        </p>
      )}

      {started && (
        <div className="cr-layout">
          <div className="cr-main">
            {turns.length > 0 && (
              <p className="cr-legend">
                Each row shows the action’s <strong>resolution mode</strong> (the enforcement
                decision) and the real <strong>gate</strong> verdict beneath it. Compliance controls
                fold into an “audit baseline” pill.
              </p>
            )}
            <div className="cr-timeline">
              {turns.map((turn, index) => (
                <TurnRow key={index} turn={turn} />
              ))}
            </div>
            {summary && <CompletionBanner summary={summary} />}
          </div>
          <HitlQueuePanel queue={hitlQueue} />
        </div>
      )}
    </div>
  );
}
