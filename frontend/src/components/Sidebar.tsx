import type { ViewCategory } from "../router";
import { navigate } from "../router";

const NAV_ITEMS: Array<{ category: ViewCategory; label: string }> = [
  { category: "story", label: "Story claims" },
  { category: "evidence", label: "Evidence packs" },
  { category: "bench_output", label: "Benchmark artifacts" },
  { category: "figures", label: "Figures" },
  { category: "probes", label: "Regression probes" },
  { category: "docs", label: "Docs & manifests" },
];

// `packs` has no dedicated nav entry -- demo_output/ is typically empty in
// a checkout, and packs are shown as a secondary section inside
// "Benchmark artifacts" (see BenchOutputView). It still highlights
// "Benchmark artifacts" as active so a deep link into a pack doesn't leave
// the sidebar looking unselected.
function activeCategory(current: ViewCategory): ViewCategory {
  return current === "packs" ? "bench_output" : current;
}

export function Sidebar({ current }: { current: ViewCategory }) {
  const active = activeCategory(current);
  return (
    <nav className="sidebar">
      <ul className="sidebar__nav">
        {NAV_ITEMS.map((item) => (
          <li key={item.category}>
            <button
              type="button"
              className={
                item.category === active ? "sidebar__link sidebar__link--active" : "sidebar__link"
              }
              onClick={() => navigate(item.category)}
            >
              {item.label}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
