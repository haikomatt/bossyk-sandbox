import { fetchArtifactsIndex, figureUrl } from "../api";
import { useAsync } from "../hooks";
import { LoadingPane, ErrorPane } from "../components/StatusPane";
import { navigate, useHashRoute } from "../router";

function figureId(name: string): string {
  return name.replace(/\.(svg|png)$/i, "");
}

export function FiguresView() {
  const indexState = useAsync(fetchArtifactsIndex, []);
  const route = useHashRoute();
  const selected = route.category === "figures" ? route.name : null;

  if (indexState.status === "loading") return <LoadingPane label="Loading figures…" />;
  if (indexState.status === "error") return <ErrorPane error={indexState.error} />;

  const names = indexState.data.figures;

  if (selected) {
    if (!names.includes(selected)) {
      return <ErrorPane error={new Error(`no committed figure named "${selected}"`)} />;
    }
    return (
      <div className="figure-detail">
        <button type="button" className="figure-detail__back" onClick={() => navigate("figures")}>
          ← all figures
        </button>
        <h2 className="figure-detail__title">{figureId(selected)}</h2>
        <img className="figure-detail__image" src={figureUrl(selected)} alt={figureId(selected)} />
      </div>
    );
  }

  return (
    <div className="figure-grid">
      {names.length === 0 && <p className="artifact-browser__empty">no figures committed yet</p>}
      {names.map((name) => (
        <button key={name} type="button" className="figure-thumb" onClick={() => navigate("figures", name)}>
          <img src={figureUrl(name)} alt={figureId(name)} />
          <span className="figure-thumb__label">{name}</span>
        </button>
      ))}
    </div>
  );
}
