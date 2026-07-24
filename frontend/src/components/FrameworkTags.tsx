import type { ControlIndex } from "../frameworkIndex";

// Renders a claim's compliance control tags -- "EU AI Act · Art. 14" --
// secondary to the evidence chip: the chip says how good the number is,
// these say which regulatory control it is evidence for. An unresolved ref
// is dropped (see buildControlIndex), so nothing renders a control the
// catalogue does not define.
export function FrameworkTags({
  controlRefs,
  index,
}: {
  controlRefs: string[];
  index: ControlIndex;
}) {
  const resolved = controlRefs
    .map((ref) => index.get(ref))
    .filter((info): info is NonNullable<typeof info> => info !== undefined);

  if (resolved.length === 0) return null;

  return (
    <div className="framework-tags" aria-label="compliance controls this claim supports">
      {resolved.map((info) => (
        <span
          key={`${info.frameworkId}:${info.controlId}`}
          className={`framework-tag framework-tag--${info.frameworkId}`}
          title={info.title}
        >
          <span className="framework-tag__name">{info.frameworkName}</span>
          <span className="framework-tag__ref">{info.ref}</span>
        </span>
      ))}
    </div>
  );
}
