import type { DocArtifact } from "../api";
import { fetchArtifact, fetchArtifactsIndex, fetchManifests } from "../api";
import { useAsync } from "../hooks";
import { LoadingPane, ErrorPane } from "../components/StatusPane";
import { navigate, useHashRoute } from "../router";

function ManifestTable({ manifest }: { manifest: Record<string, unknown> }) {
  return (
    <table className="manifest-table">
      <tbody>
        {Object.entries(manifest).map(([key, value]) => (
          <tr key={key}>
            <th>{key}</th>
            <td>{Array.isArray(value) ? value.join(", ") || "—" : String(value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function DocsView() {
  const indexState = useAsync(fetchArtifactsIndex, []);
  const manifestsState = useAsync(fetchManifests, []);
  const route = useHashRoute();
  const selected = route.category === "docs" ? route.name : null;

  const docState = useAsync(
    () => (selected ? (fetchArtifact("docs", selected) as Promise<DocArtifact>) : Promise.resolve(null)),
    [selected],
  );

  if (indexState.status === "loading" || manifestsState.status === "loading") {
    return <LoadingPane label="Loading docs…" />;
  }
  if (indexState.status === "error") return <ErrorPane error={indexState.error} />;
  if (manifestsState.status === "error") return <ErrorPane error={manifestsState.error} />;

  const names = indexState.data.docs;
  const manifestsByDoc = new Map(manifestsState.data.map((entry) => [entry.doc, entry.manifests]));

  return (
    <div className="artifact-browser">
      <div className="artifact-browser__list">
        <div className="artifact-browser__section">
          <h3 className="artifact-browser__section-title">Docs ({names.length})</h3>
          <ul>
            {names.map((name) => (
              <li key={name}>
                <button
                  type="button"
                  className={
                    name === selected ? "artifact-browser__item artifact-browser__item--active" : "artifact-browser__item"
                  }
                  onClick={() => navigate("docs", name)}
                >
                  {name}
                  {manifestsByDoc.has(name) && <span className="artifact-browser__badge">manifest</span>}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
      <div className="artifact-browser__detail">
        {!selected && <p className="artifact-browser__placeholder">Select a doc to view it.</p>}
        {selected && docState.status === "loading" && <LoadingPane label={`Loading ${selected}…`} />}
        {selected && docState.status === "error" && <ErrorPane error={docState.error} />}
        {selected && docState.status === "ready" && docState.data && (
          <>
            <h2 className="artifact-browser__detail-title">{selected}</h2>
            {(manifestsByDoc.get(selected) ?? []).map((manifest, index) => (
              <div key={index} className="manifest-block">
                <h3 className="manifest-block__title">reproducibility manifest {index + 1}</h3>
                <ManifestTable manifest={manifest} />
              </div>
            ))}
            <pre className="markdown-pane">{docState.data.markdown}</pre>
          </>
        )}
      </div>
    </div>
  );
}
