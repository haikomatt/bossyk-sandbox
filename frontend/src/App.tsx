import { Sidebar } from "./components/Sidebar";
import { StoryView } from "./views/StoryView";
import { ControlRoomView } from "./views/ControlRoomView";
import { EvidenceView } from "./views/EvidenceView";
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
    case "control_room":
      return <ControlRoomView />;
    case "evidence":
      return <EvidenceView />;
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
        <span className="logo-mark">bk</span>
        <span className="logo-rule" />
        <span className="logo-name">bossyk</span>
        <span className="app-header__tagline">agent governance console · built on auditk</span>
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
