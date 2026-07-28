// Typed fetch helpers for the evidence-browser API. Mirrors the shapes
// returned by `src/bossyk_sandbox/console/artifacts.py` exactly -- see that
// module's docstring for the on-disk artifact layout each endpoint reads.

export type EvidenceGrade =
  | "live-measurement"
  | "scripted-proxy"
  | "modeled-counterfactual"
  | "deterministic-recompute"
  | "open";

export interface StoryAct {
  act: number;
  title: string;
  question: string;
}

export interface NumericCheck {
  artifact: string;
  json_path: string;
  expected: number | string;
}

export interface StoryClaim {
  id: string;
  act: number;
  exec_copy: string;
  tech_copy: string;
  verdict: string | null;
  evidence_grade: EvidenceGrade;
  artifact_refs: string[];
  figure_ids: string[];
  numeric_checks: NumericCheck[];
  // Compliance controls this claim is evidence FOR, each "<framework>:<control>".
  // Orthogonal to evidence_grade (how good the number is) -- see the frameworks
  // catalogue below for the human-readable names.
  control_refs: string[];
}

export type FrameworkType =
  | "regulation"
  | "sector-regulation"
  | "certification"
  | "attestation";

export interface FrameworkControl {
  id: string;
  ref: string;
  title: string;
}

export interface Framework {
  id: string;
  name: string;
  type: FrameworkType;
  controls: FrameworkControl[];
}

export interface FrameworkRegistry {
  disclaimer: string;
  entries: Framework[];
}

export interface Story {
  acts: StoryAct[];
  claims: StoryClaim[];
  // Absent (null) on a story with no compliance mapping declared.
  frameworks: FrameworkRegistry | null;
}

// --- signed trace sidecar (the evidence pack's browsable steps) ---

export interface ControlTag {
  ref: string; // "<framework>:<control>"
  basis: string; // substrate | verdict:gated | verdict:blocked | verdict:overridden | ...
}

export interface TraceStep {
  step_id: string;
  declared_intent: string | null;
  action: {
    payload: {
      tool_name: string;
      arguments: Record<string, unknown>;
      gate_verdict: string;
    };
  };
  metadata: {
    bossyk_sandbox_controls?: ControlTag[];
    overridden?: boolean;
  };
}

export interface SignedTrace {
  trace: {
    trace_id: string;
    agent_config_ref: string;
    steps: TraceStep[];
  };
  // control ref -> ids of the steps that discharge it
  coverage: Record<string, string[]>;
  signatures: unknown[];
}

export type ArtifactCategory = "bench_output" | "probes" | "figures" | "docs" | "packs";

export type ArtifactsIndex = Record<ArtifactCategory, string[]>;

export interface DocArtifact {
  name: string;
  markdown: string;
}

export interface ManifestSet {
  doc: string;
  manifests: Record<string, unknown>[];
}

/** JSON categories are the ones `GET /api/artifacts/{category}/{name}`
 * returns as arbitrary parsed JSON rather than an image or a doc wrapper. */
export const JSON_ARTIFACT_CATEGORIES: readonly ArtifactCategory[] = [
  "bench_output",
  "probes",
  "packs",
];

class ApiError extends Error {}

// --- control-room replay (POST /session/replay + the /ws event stream) ---

export interface ReplayStartResult {
  status: string; // "started" | "already_running" | "unknown_preset"
  session_id?: string;
}

/** One resolved compliance control on a broadcast step, already carrying its
 * human framework/control names (see console `_display_controls`). Richer than
 * the story `ControlTag`, which is just ref + basis. */
export interface ReplayControl {
  ref: string;
  basis: string;
  framework: string;
  control: string;
  title: string;
}

/** A hard-cell item routed to the HITL review queue (an `escalate` turn). */
export interface ReplayHitlItem {
  severity: string; // critical | high | medium | low
  reason: string;
  resolution: string;
  channel: string;
  label: string;
  tool_name: string;
}

export interface ReplayHeldEvent {
  type: "held";
  replay: boolean;
  preset: string;
  session_id: string;
  action_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  auto_verdict: string;
  auto_reason: string;
  label: string;
  role: string;
  probe_id: string | null;
  mode: string | null;
}

export interface ReplayStepEvent {
  type: "step";
  tool_name: string;
  arguments: Record<string, unknown>;
  verdict: string;
  overridden: boolean;
  controls: ReplayControl[];
  label: string;
  role: string;
  probe_id: string | null;
  // Authored resolution mode (allow / redirect / defer / step-up / escalate)
  // layered on the real structural verdict; hitl present only on escalate.
  mode: string | null;
  mode_reason: string | null;
  hitl?: ReplayHitlItem;
}

export interface ReplayCompleteEvent {
  type: "session_complete";
  step_count: number;
  hitl_queue: ReplayHitlItem[];
  // true on a live agent run; absent on a preset replay.
  live?: boolean;
  // Present only on a preset replay (a live run has no committed artifact).
  harm_prevented?: number;
  harm_delta?: number;
  preset?: string;
  source_artifact?: string;
  story_claim?: string;
}

export type ReplayEvent = ReplayHeldEvent | ReplayStepEvent | ReplayCompleteEvent;

/** Same-origin websocket URL for the console broadcast stream. The SPA is
 * served by the console at /app, so /ws is same-origin in the demo. */
export function replaySocketUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws`;
}

async function postStart(path: string): Promise<ReplayStartResult> {
  let response: Response;
  try {
    response = await fetch(path, { method: "POST" });
  } catch {
    throw new ApiError(`could not reach the API at ${path} -- is the backend running?`);
  }
  if (!response.ok) {
    throw new ApiError(`${path} -> HTTP ${response.status}`);
  }
  return (await response.json()) as ReplayStartResult;
}

export function startReplay(presetId: string, pacingS: number): Promise<ReplayStartResult> {
  return postStart(`/session/replay?preset_id=${encodeURIComponent(presetId)}&pacing_s=${pacingS}`);
}

/** Start a LIVE agent session. status is "started" | "already_running" |
 * "no_api_key" (the last when FIREWORKS_API_KEY is unset -- non-billable). */
export function startLive(pacingS: number): Promise<ReplayStartResult> {
  return postStart(`/session/live?pacing_s=${pacingS}`);
}

async function getJson<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path);
  } catch {
    throw new ApiError(`could not reach the API at ${path} -- is the backend running?`);
  }
  if (!response.ok) {
    throw new ApiError(`${path} -> HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export function fetchStory(): Promise<Story> {
  return getJson<Story>("/api/story");
}

export function fetchArtifactsIndex(): Promise<ArtifactsIndex> {
  return getJson<ArtifactsIndex>("/api/artifacts");
}

export function fetchManifests(): Promise<ManifestSet[]> {
  return getJson<ManifestSet[]>("/api/manifests");
}

export function fetchFrameworks(): Promise<FrameworkRegistry> {
  return getJson<FrameworkRegistry>("/api/frameworks");
}

export function fetchSignedTrace(name: string): Promise<SignedTrace> {
  return getJson<SignedTrace>(`/api/artifacts/bench_output/${encodeURIComponent(name)}`);
}

/** For bench_output/probes/packs this is parsed JSON; for docs it is
 * {name, markdown}. Figures are images, fetched directly as `<img src>`
 * rather than through this helper. */
export function fetchArtifact(category: ArtifactCategory, name: string): Promise<unknown> {
  return getJson(`/api/artifacts/${category}/${encodeURIComponent(name)}`);
}

export function figureUrl(name: string): string {
  return `/api/artifacts/figures/${encodeURIComponent(name)}`;
}

export { ApiError };
