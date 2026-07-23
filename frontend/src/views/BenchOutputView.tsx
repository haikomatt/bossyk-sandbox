import { ArtifactBrowserView } from "./ArtifactBrowserView";

export function BenchOutputView() {
  return (
    <ArtifactBrowserView
      sections={[
        { category: "bench_output", title: "Benchmark output" },
        { category: "packs", title: "Demo packs" },
      ]}
    />
  );
}
