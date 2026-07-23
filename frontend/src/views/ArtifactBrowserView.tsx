import type { ArtifactCategory } from "../api";
import { fetchArtifact, fetchArtifactsIndex } from "../api";
import { useAsync } from "../hooks";
import { LoadingPane, ErrorPane } from "../components/StatusPane";
import { JsonTree } from "../components/JsonTree";
import { navigate, useHashRoute } from "../router";

interface Section {
  category: ArtifactCategory;
  title: string;
}

/** Shared list+detail layout for the JSON artifact categories
 * (bench_output, probes, packs): a list of names on the left, the selected
 * artifact's JSON rendered as a collapsible tree on the right. One or more
 * `sections` can be shown in the same view (bench_output + packs share a
 * "Benchmark artifacts" view; see BenchOutputView). */
export function ArtifactBrowserView({ sections }: { sections: Section[] }) {
  const indexState = useAsync(fetchArtifactsIndex, []);
  const route = useHashRoute();

  const activeCategory = sections.some((s) => s.category === route.category)
    ? (route.category as ArtifactCategory)
    : null;
  const activeName = activeCategory ? route.name : null;

  const detailState = useAsync(
    () => (activeCategory && activeName ? fetchArtifact(activeCategory, activeName) : Promise.resolve(null)),
    [activeCategory, activeName],
  );

  if (indexState.status === "loading") return <LoadingPane label="Loading artifact list…" />;
  if (indexState.status === "error") return <ErrorPane error={indexState.error} />;

  return (
    <div className="artifact-browser">
      <div className="artifact-browser__list">
        {sections.map((section) => {
          const names = indexState.data[section.category] ?? [];
          return (
            <div key={section.category} className="artifact-browser__section">
              <h3 className="artifact-browser__section-title">
                {section.title} ({names.length})
              </h3>
              {names.length === 0 && (
                <p className="artifact-browser__empty">none committed yet</p>
              )}
              <ul>
                {names.map((name) => (
                  <li key={name}>
                    <button
                      type="button"
                      className={
                        section.category === activeCategory && name === activeName
                          ? "artifact-browser__item artifact-browser__item--active"
                          : "artifact-browser__item"
                      }
                      onClick={() => navigate(section.category, name)}
                    >
                      {name}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>
      <div className="artifact-browser__detail">
        {!activeName && <p className="artifact-browser__placeholder">Select an artifact to view its JSON.</p>}
        {activeName && detailState.status === "loading" && <LoadingPane label={`Loading ${activeName}…`} />}
        {activeName && detailState.status === "error" && <ErrorPane error={detailState.error} />}
        {activeName && detailState.status === "ready" && detailState.data !== null && (
          <>
            <h2 className="artifact-browser__detail-title">{activeName}</h2>
            <JsonTree value={detailState.data} />
          </>
        )}
      </div>
    </div>
  );
}
