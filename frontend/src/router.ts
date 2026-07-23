// Tiny hand-rolled hash router. No react-router: the app only ever needs
// "which of 5 views" + "which item is selected within it", and the URL
// shape `#/<category>/<name>` (name optionally absent) survives a refresh
// for free because it's just the hash.

import { useEffect, useState } from "react";

export type ViewCategory = "story" | "bench_output" | "probes" | "figures" | "docs" | "packs";

export interface Route {
  category: ViewCategory;
  name: string | null;
}

const DEFAULT_ROUTE: Route = { category: "story", name: null };

function parseHash(hash: string): Route {
  const trimmed = hash.replace(/^#\/?/, "");
  if (!trimmed) return DEFAULT_ROUTE;

  const [rawCategory, ...rest] = trimmed.split("/");
  const category = rawCategory as ViewCategory;
  const validCategories: ViewCategory[] = [
    "story",
    "bench_output",
    "probes",
    "figures",
    "docs",
    "packs",
  ];
  if (!validCategories.includes(category)) return DEFAULT_ROUTE;

  const name = rest.length > 0 ? decodeURIComponent(rest.join("/")) : null;
  return { category, name };
}

export function navigate(category: ViewCategory, name?: string | null): void {
  const hash = name ? `#/${category}/${encodeURIComponent(name)}` : `#/${category}`;
  window.location.hash = hash;
}

export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));

  useEffect(() => {
    const onHashChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  return route;
}
