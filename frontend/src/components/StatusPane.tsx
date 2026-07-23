// Shared loading/error presentation so every view fails visibly (per the
// F1 brief: an unreachable API must show a clear message, not a blank
// page) instead of each view rolling its own.

export function LoadingPane({ label }: { label?: string }) {
  return <div className="status-pane status-pane--loading">{label ?? "Loading…"}</div>;
}

export function ErrorPane({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="status-pane status-pane--error">
      <strong>Couldn't load this.</strong>
      <div className="status-pane__detail">{message}</div>
    </div>
  );
}
