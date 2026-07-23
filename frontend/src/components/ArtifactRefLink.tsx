import { navigate } from "../router";
import { resolveArtifactRef } from "../resolveArtifactRef";

/** Renders a claim's artifact_ref as a link into the matching browser view
 * when it resolves to a browsable category (bench_output/probes/figures/
 * docs); otherwise as plain text, per the F1 brief. */
export function ArtifactRefLink({ refPath }: { refPath: string }) {
  const resolved = resolveArtifactRef(refPath);
  if (!resolved) {
    return <span className="artifact-ref artifact-ref--plain">{refPath}</span>;
  }
  return (
    <button
      type="button"
      className="artifact-ref artifact-ref--link"
      onClick={() => navigate(resolved.category, resolved.name)}
    >
      {refPath}
    </button>
  );
}
