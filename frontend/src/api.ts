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
}

export interface Story {
  acts: StoryAct[];
  claims: StoryClaim[];
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
