import type { ControlTag, SignedTrace, TraceStep } from "../api";
import { fetchArtifactsIndex, fetchFrameworks, fetchSignedTrace } from "../api";
import { useAsync } from "../hooks";
import { buildControlIndex } from "../frameworkIndex";
import type { ControlIndex } from "../frameworkIndex";
import { LoadingPane, ErrorPane } from "../components/StatusPane";
import { navigate, useHashRoute } from "../router";

const TRACE_SUFFIX = ".trace.json";

function ControlTagPills({ tags, index }: { tags: ControlTag[]; index: ControlIndex }) {
  const resolved = tags
    .map((tag) => ({ tag, info: index.get(tag.ref) }))
    .filter((x): x is { tag: ControlTag; info: NonNullable<ReturnType<ControlIndex["get"]>> } =>
      Boolean(x.info),
    );
  if (resolved.length === 0) return null;
  return (
    <div className="ev-tags">
      {resolved.map(({ tag, info }) => (
        <span
          key={`${tag.ref}:${tag.basis}`}
          className={`framework-tag framework-tag--${info.frameworkId}`}
          title={info.title}
        >
          <span className="framework-tag__name">{info.frameworkName}</span>
          <span className="framework-tag__ref">{info.ref}</span>
          <span className="ev-basis">{tag.basis}</span>
        </span>
      ))}
    </div>
  );
}

function StepRow({ step, index }: { step: TraceStep; index: ControlIndex }) {
  const verdict = step.action.payload.gate_verdict;
  const args = JSON.stringify(step.action.payload.arguments);
  const tags = step.metadata.bossyk_sandbox_controls ?? [];
  return (
    <div className="ev-step">
      <div className="ev-step__head">
        <span className={`ev-verdict ev-verdict--${verdict}`}>{verdict.toUpperCase()}</span>
        <code className="ev-step__call">
          {step.action.payload.tool_name}({args})
        </code>
        {step.metadata.overridden && <span className="ev-overridden">overridden</span>}
      </div>
      {step.declared_intent && <p className="ev-step__intent">{step.declared_intent}</p>}
      <ControlTagPills tags={tags} index={index} />
    </div>
  );
}

function TraceDetail({ trace, index }: { trace: SignedTrace; index: ControlIndex }) {
  const coverageRefs = Object.keys(trace.coverage);
  return (
    <div className="ev-detail">
      <div className="ev-detail__head">
        <h2 className="ev-detail__title">{trace.trace.trace_id}</h2>
        <span className="ev-signed" title="Ed25519-signed; verifies offline against the issuer key">
          signed · {trace.signatures.length} signature{trace.signatures.length === 1 ? "" : "s"}
        </span>
      </div>

      <section className="ev-coverage">
        <h3 className="ev-section-title">
          Controls this record discharges ({coverageRefs.length})
        </h3>
        <div className="ev-coverage__grid">
          {coverageRefs.map((ref) => {
            const info = index.get(ref);
            const stepCount = trace.coverage[ref].length;
            return (
              <div key={ref} className="ev-coverage__cell">
                <span className={`framework-tag framework-tag--${info?.frameworkId ?? "unknown"}`}>
                  <span className="framework-tag__name">{info?.frameworkName ?? ref}</span>
                  <span className="framework-tag__ref">{info?.ref ?? ""}</span>
                </span>
                <span className="ev-coverage__count">
                  {stepCount} step{stepCount === 1 ? "" : "s"}
                </span>
              </div>
            );
          })}
        </div>
      </section>

      <section className="ev-steps">
        <h3 className="ev-section-title">Attested actions ({trace.trace.steps.length})</h3>
        {trace.trace.steps.map((step) => (
          <StepRow key={step.step_id} step={step} index={index} />
        ))}
      </section>
    </div>
  );
}

export function EvidenceView() {
  const indexState = useAsync(fetchArtifactsIndex, []);
  const frameworksState = useAsync(fetchFrameworks, []);
  const route = useHashRoute();

  const activeName = route.category === "evidence" ? route.name : null;
  const traceState = useAsync(
    () => (activeName ? fetchSignedTrace(activeName) : Promise.resolve(null)),
    [activeName],
  );

  if (indexState.status === "loading" || frameworksState.status === "loading")
    return <LoadingPane label="Loading evidence…" />;
  if (indexState.status === "error") return <ErrorPane error={indexState.error} />;
  if (frameworksState.status === "error") return <ErrorPane error={frameworksState.error} />;

  const controlIndex = buildControlIndex(frameworksState.data);
  const traceNames = (indexState.data.bench_output ?? []).filter((name) =>
    name.endsWith(TRACE_SUFFIX),
  );

  return (
    <div className="artifact-browser">
      <div className="artifact-browser__list">
        <div className="artifact-browser__section">
          <h3 className="artifact-browser__section-title">Evidence packs ({traceNames.length})</h3>
          {traceNames.length === 0 && (
            <p className="artifact-browser__empty">no signed traces committed yet</p>
          )}
          <ul>
            {traceNames.map((name) => (
              <li key={name}>
                <button
                  type="button"
                  className={
                    name === activeName
                      ? "artifact-browser__item artifact-browser__item--active"
                      : "artifact-browser__item"
                  }
                  onClick={() => navigate("evidence", name)}
                >
                  {name}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
      <div className="artifact-browser__detail">
        {!activeName && (
          <p className="artifact-browser__placeholder">
            Select a signed evidence pack to see each attested action and the compliance controls it
            discharges.
          </p>
        )}
        {activeName && traceState.status === "loading" && (
          <LoadingPane label={`Loading ${activeName}…`} />
        )}
        {activeName && traceState.status === "error" && <ErrorPane error={traceState.error} />}
        {activeName && traceState.status === "ready" && traceState.data !== null && (
          <TraceDetail trace={traceState.data} index={controlIndex} />
        )}
      </div>
    </div>
  );
}
