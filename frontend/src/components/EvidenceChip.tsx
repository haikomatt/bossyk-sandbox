import type { EvidenceGrade } from "../api";

const LABELS: Record<EvidenceGrade, string> = {
  "live-measurement": "live measurement",
  "scripted-proxy": "scripted proxy",
  "modeled-counterfactual": "modeled counterfactual",
  "deterministic-recompute": "deterministic recompute",
  open: "open",
};

export function EvidenceChip({ grade }: { grade: EvidenceGrade }) {
  return <span className={`evidence-chip evidence-chip--${grade}`}>{LABELS[grade]}</span>;
}
