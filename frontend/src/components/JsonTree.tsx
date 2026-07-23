// Recursive collapsible JSON viewer built entirely on <details>/<summary>
// -- deliberately no json-viewer dependency, per the F1 brief. Arrays with
// more than 50 items render collapsed (closed by default) with a count in
// the summary so a large artifact doesn't dump thousands of DOM nodes open
// on first paint; everything else defaults open for easy scanning.

const LARGE_ARRAY_THRESHOLD = 50;

function typeLabel(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return `array[${value.length}]`;
  return typeof value;
}

function Primitive({ value }: { value: string | number | boolean | null }) {
  const text = value === null ? "null" : JSON.stringify(value);
  return <span className={`json-primitive json-primitive--${typeLabel(value)}`}>{text}</span>;
}

export function JsonTree({ value, label }: { value: unknown; label?: string }) {
  if (value === null || typeof value !== "object") {
    return (
      <div className="json-row">
        {label !== undefined && <span className="json-key">{label}: </span>}
        <Primitive value={value as string | number | boolean | null} />
      </div>
    );
  }

  if (Array.isArray(value)) {
    const large = value.length > LARGE_ARRAY_THRESHOLD;
    return (
      <details className="json-node" open={!large}>
        <summary>
          {label !== undefined && <span className="json-key">{label}: </span>}
          <span className="json-type">array[{value.length}]</span>
        </summary>
        <div className="json-children">
          {value.map((item, index) => (
            <JsonTree key={index} value={item} label={String(index)} />
          ))}
        </div>
      </details>
    );
  }

  const entries = Object.entries(value as Record<string, unknown>);
  return (
    <details className="json-node" open>
      <summary>
        {label !== undefined && <span className="json-key">{label}: </span>}
        <span className="json-type">object{`{${entries.length}}`}</span>
      </summary>
      <div className="json-children">
        {entries.map(([key, item]) => (
          <JsonTree key={key} value={item} label={key} />
        ))}
      </div>
    </details>
  );
}
