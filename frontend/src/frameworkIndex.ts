// Resolves a claim's `control_refs` ("<framework>:<control>") into the
// human-readable names the story's frameworks catalogue defines, so the
// story view can render "EU AI Act · Art. 14" tags without every card
// re-scanning the registry. Built once from the registry; an unresolved
// ref (should not happen -- load_story rejects those server-side) is left
// out so a stale ref never renders as a confident tag.

import type { FrameworkRegistry } from "./api";

export interface ControlInfo {
  frameworkId: string;
  frameworkName: string;
  controlId: string;
  ref: string;
  title: string;
}

export type ControlIndex = ReadonlyMap<string, ControlInfo>;

export function buildControlIndex(registry: FrameworkRegistry | null): ControlIndex {
  const index = new Map<string, ControlInfo>();
  if (!registry) return index;

  for (const framework of registry.entries) {
    for (const control of framework.controls) {
      index.set(`${framework.id}:${control.id}`, {
        frameworkId: framework.id,
        frameworkName: framework.name,
        controlId: control.id,
        ref: control.ref,
        title: control.title,
      });
    }
  }
  return index;
}
