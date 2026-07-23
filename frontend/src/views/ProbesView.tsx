import { ArtifactBrowserView } from "./ArtifactBrowserView";

export function ProbesView() {
  return <ArtifactBrowserView sections={[{ category: "probes", title: "Regression probes" }]} />;
}
