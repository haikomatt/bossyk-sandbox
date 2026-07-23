import { fetchStory } from "../api";
import type { StoryClaim } from "../api";
import { useAsync } from "../hooks";
import { LoadingPane, ErrorPane } from "../components/StatusPane";
import { EvidenceChip } from "../components/EvidenceChip";
import { ArtifactRefLink } from "../components/ArtifactRefLink";
import { navigate } from "../router";

function ClaimCard({ claim }: { claim: StoryClaim }) {
  return (
    <article className="claim-card" id={`claim-${claim.id}`}>
      <div className="claim-card__head">
        <EvidenceChip grade={claim.evidence_grade} />
      </div>
      <p className="claim-card__exec">{claim.exec_copy}</p>
      <p className="claim-card__tech">{claim.tech_copy}</p>
      {claim.verdict && <p className="claim-card__verdict">verdict: {claim.verdict}</p>}

      {claim.artifact_refs.length > 0 && (
        <div className="claim-card__refs">
          {claim.artifact_refs.map((ref) => (
            <ArtifactRefLink key={ref} refPath={ref} />
          ))}
        </div>
      )}

      {claim.figure_ids.length > 0 && (
        <div className="claim-card__figures">
          {claim.figure_ids.map((figureId) => (
            <button
              key={figureId}
              type="button"
              className="artifact-ref artifact-ref--link"
              onClick={() => navigate("figures", `${figureId}.svg`)}
            >
              figure: {figureId}
            </button>
          ))}
        </div>
      )}

      {claim.numeric_checks.length > 0 && (
        <details className="claim-card__checks">
          <summary>numeric checks ({claim.numeric_checks.length})</summary>
          <table>
            <thead>
              <tr>
                <th>artifact</th>
                <th>json_path</th>
                <th>expected</th>
              </tr>
            </thead>
            <tbody>
              {claim.numeric_checks.map((check, index) => (
                <tr key={index}>
                  <td>{check.artifact}</td>
                  <td>
                    <code>{check.json_path}</code>
                  </td>
                  <td>{String(check.expected)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </article>
  );
}

export function StoryView() {
  const state = useAsync(fetchStory, []);

  if (state.status === "loading") return <LoadingPane label="Loading story…" />;
  if (state.status === "error") return <ErrorPane error={state.error} />;

  const { acts, claims } = state.data;

  return (
    <div className="story-view">
      {acts.map((act) => {
        const actClaims = claims.filter((claim) => claim.act === act.act);
        return (
          <section key={act.act} className="act-section" id={`act-${act.act}`}>
            <header className="act-section__header">
              <span className="act-section__number">Act {act.act}</span>
              <h2 className="act-section__title">{act.title}</h2>
              <p className="act-section__question">{act.question}</p>
            </header>
            <div className="act-section__claims">
              {actClaims.map((claim) => (
                <ClaimCard key={claim.id} claim={claim} />
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}
