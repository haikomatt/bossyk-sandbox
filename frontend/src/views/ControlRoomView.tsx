import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ReplayCompleteEvent,
  ReplayControl,
  ReplayEvent,
} from "../api";
import { replaySocketUrl, startReplay } from "../api";
import { navigate } from "../router";

const PRESET_ID = "retail-weak-dir1";
const PACING_S = 1.2;

type RunStatus = "idle" | "running" | "done" | "error";

interface TurnState {
  label: string;
  role: string;
  probeId: string | null;
  toolName: string;
  args: Record<string, unknown>;
  autoReason: string;
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
    status: "held",
  };
}

/** Drives one preset replay over the console websocket. Held/step events
 * arrive strictly interleaved (held i, then step i, then held i+1 …), so each
 * "step" finalizes the most recent still-held turn. */
function useReplaySocket() {
  const [turns, setTurns] = useState<TurnState[]>([]);
  const [summary, setSummary] = useState<ReplayCompleteEvent | null>(null);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const run = useCallback(() => {
    if (status === "running") return;
    setTurns([]);
    setSummary(null);
    setError(null);
    setStatus("running");

    const ws = new WebSocket(replaySocketUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      startReplay(PRESET_ID, PACING_S)
        .then((result) => {
          if (result.status !== "started") {
            setError(`replay could not start: ${result.status}`);
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
                controls: event.controls,
              };
              break;
            }
          }
          return next;
        });
      } else if (event.type === "session_complete") {
        setSummary(event);
        setStatus("done");
        ws.close();
      }
    };

    ws.onerror = () => {
      setError("websocket error -- is the console backend running?");
      setStatus("error");
    };
  }, [status]);

  // Close the socket if the view unmounts mid-run.
  useEffect(() => () => wsRef.current?.close(), []);

  return { turns, summary, status, error, run };
}

function VerdictChip({ verdict }: { verdict: string }) {
  return <span className={`ev-verdict ev-verdict--${verdict}`}>{verdict.toUpperCase()}</span>;
}

// The substrate + gated controls fire on every attested action (it was
// logged, signed, and passed the risk gate), so they carry no per-action
// signal -- fold them into one "audit baseline" pill and surface inline only
// the controls that are specific to THIS action.
const BASELINE_BASES = new Set(["substrate", "verdict:gated"]);

/** Plain-English gloss for a control's `basis` (why the control applies). */
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

function TurnRow({ turn }: { turn: TurnState }) {
  const args = JSON.stringify(turn.args);
  return (
    <div className={`cr-turn cr-turn--${turn.role} cr-turn--${turn.status}`}>
      <div className="cr-turn__head">
        {turn.status === "held" ? (
          <span className="cr-held">HELD · scoring…</span>
        ) : (
          <VerdictChip verdict={turn.verdict ?? "allow"} />
        )}
        <span className="cr-turn__label">{turn.label}</span>
        {turn.probeId && <span className="cr-turn__probe">{turn.probeId}</span>}
      </div>
      <code className="ev-step__call">
        {turn.toolName}({args})
      </code>
      <p className="cr-turn__reason">{turn.autoReason}</p>
      {turn.status === "done" && turn.controls && <ControlPills controls={turn.controls} />}
    </div>
  );
}

function CompletionBanner({ summary }: { summary: ReplayCompleteEvent }) {
  return (
    <div className="cr-summary">
      <div className="cr-summary__headline">
        harm {summary.harm_delta} → 0 · {summary.harm_prevented}/{summary.harm_prevented} crossings
        prevented pre-execution
      </div>
      <p className="cr-summary__body">
        The under-specified agent skipped its lookup and proposed a destructive write on every
        crossing; the two-speed gate blocked all of them before execution. These verdicts and the
        harm delta recompute from the same gate the live run used.
      </p>
      <div className="cr-summary__refs">
        <button
          type="button"
          className="artifact-ref artifact-ref--link"
          onClick={() => navigate("story")}
        >
          story claim: {summary.story_claim}
        </button>
        <button
          type="button"
          className="artifact-ref artifact-ref--link"
          onClick={() => navigate("bench_output", summary.source_artifact.split("/").pop() ?? "")}
        >
          source: {summary.source_artifact}
        </button>
      </div>
    </div>
  );
}

export function ControlRoomView() {
  const { turns, summary, status, error, run } = useReplaySocket();

  return (
    <div className="control-room">
      <header className="control-room__header">
        <div>
          <h2 className="control-room__title">Control room — live gate-save replay</h2>
          <p className="control-room__sub">
            A deterministic, turn-by-turn replay of the dir-1 gate-save: a latency-optimized retail
            agent that skips its verification lookup, driven through the real two-speed gate. Each
            crossing traces to a measured probe in the committed run.
          </p>
        </div>
        <button
          type="button"
          className="control-room__run"
          onClick={run}
          disabled={status === "running"}
        >
          {status === "running" ? "Replaying…" : status === "idle" ? "Run replay" : "Replay again"}
        </button>
      </header>

      {error && <p className="control-room__error">{error}</p>}

      {status === "idle" && turns.length === 0 && (
        <p className="control-room__placeholder">
          Press “Run replay” to watch each proposed tool call be held, scored by the gate, and
          attested — allowed baseline first, then the four skip-lookup crossings the gate blocks.
        </p>
      )}

      {turns.length > 0 && (
        <p className="cr-legend">
          Each governed action is tagged with the compliance controls it discharges. The folded{" "}
          <strong>audit baseline</strong> (logging + risk gate) applies to every action; the chips
          beside it are specific to that action.
        </p>
      )}

      <div className="cr-timeline">
        {turns.map((turn, index) => (
          <TurnRow key={index} turn={turn} />
        ))}
      </div>

      {summary && <CompletionBanner summary={summary} />}
    </div>
  );
}
