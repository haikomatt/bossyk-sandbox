// Maps a claim's `artifact_refs` entry (a repo-relative path, e.g.
// "docs/bench_output/phase2b_crossdomain_h1.json") to the browser-view
// route that serves it, mirroring the directory layout
// `console/artifacts.py` reads from. Refs outside those directories (or
// docs mid-path, e.g. a nested doc) resolve to `null` and render as plain
// text rather than a broken link.

import type { ArtifactCategory } from "./api";

export interface ResolvedRef {
  category: ArtifactCategory;
  name: string;
}

const PREFIXES: Array<[string, ArtifactCategory]> = [
  ["docs/bench_output/", "bench_output"],
  ["docs/figures/", "figures"],
  ["probes/regression/", "probes"],
  ["demo_output/", "packs"],
];

export function resolveArtifactRef(ref: string): ResolvedRef | null {
  for (const [prefix, category] of PREFIXES) {
    if (ref.startsWith(prefix)) {
      const rest = ref.slice(prefix.length);
      if (rest && !rest.includes("/")) {
        return { category, name: rest };
      }
      return null;
    }
  }

  if (ref.startsWith("docs/") && ref.endsWith(".md")) {
    const rest = ref.slice("docs/".length);
    if (rest && !rest.includes("/")) {
      return { category: "docs", name: rest };
    }
  }

  return null;
}
