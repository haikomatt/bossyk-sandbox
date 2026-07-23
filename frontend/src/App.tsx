import { Sidebar } from "./components/Sidebar";
import { StoryView } from "./views/StoryView";
import { BenchOutputView } from "./views/BenchOutputView";
import { ProbesView } from "./views/ProbesView";
import { FiguresView } from "./views/FiguresView";
import { DocsView } from "./views/DocsView";
import { useHashRoute } from "./router";

function CurrentView() {
  const route = useHashRoute();
  switch (route.category) {
    case "story":
      return <StoryView />;
    case "bench_output":
    case "packs":
      return <BenchOutputView />;
    case "probes":
      return <ProbesView />;
    case "figures":
      return <FiguresView />;
    case "docs":
      return <DocsView />;
  }
}

function App() {
  const route = useHashRoute();
  return (
    <div className="app">
      <header className="app-header">
        <span className="app-header__title">bossyk — evidence browser</span>
        <span className="app-header__tagline">every number traces to a committed artifact</span>
      </header>
      <div className="app-body">
        <Sidebar current={route.category} />
        <main className="app-main">
          <CurrentView />
        </main>
      </div>
    </div>
  );
}

export default App;
