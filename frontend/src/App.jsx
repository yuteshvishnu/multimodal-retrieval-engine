import { useEffect, useState } from "react";
import { Icon } from "./components/ui.jsx";
import RagTab from "./tabs/RagTab.jsx";
import TextToSqlTab from "./tabs/TextToSqlTab.jsx";

// Rexie's two products
const TABS = [
  { id: "rag", label: "RAG", icon: "doc" },
  { id: "sql", label: "Text-to-SQL", icon: "database" },
];

const THEMES = [
  { id: "system", label: "System theme", icon: "monitor" },
  { id: "light", label: "Light theme", icon: "sun" },
  { id: "dark", label: "Dark theme", icon: "moon" },
];

function savedTheme() {
  try {
    return localStorage.getItem("theme") || "system";
  } catch {
    return "system";
  }
}

export default function App() {
  const [active, setActive] = useState("rag");
  const [theme, setTheme] = useState(savedTheme);

  // "system" follows the OS setting; light/dark override it (see styles.css)
  useEffect(() => {
    if (theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("theme", theme);
    } catch {
      // storage blocked: the choice just isn't remembered
    }
  }, [theme]);

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <span className="brand-mark">
              <Icon name="sparkle" size={16} />
            </span>
            <span className="brand-name">Rexie</span>
          </div>

          <div className="topbar-actions">
            <nav className="tabs">
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  className={tab.id === active ? "tab active" : "tab"}
                  onClick={() => setActive(tab.id)}
                >
                  <Icon name={tab.icon} size={14} />
                  {tab.label}
                </button>
              ))}
            </nav>

            <div className="tabs theme-toggle" role="group" aria-label="Theme">
              {THEMES.map((t) => (
                <button
                  key={t.id}
                  className={t.id === theme ? "tab active" : "tab"}
                  onClick={() => setTheme(t.id)}
                  title={t.label}
                  aria-label={t.label}
                  aria-pressed={t.id === theme}
                >
                  <Icon name={t.icon} size={14} />
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      <main className="page">
        {/* both stay mounted, so switching tabs keeps each tab's results */}
        <div hidden={active !== "rag"}>
          <RagTab />
        </div>
        <div hidden={active !== "sql"}>
          <TextToSqlTab />
        </div>
      </main>
    </>
  );
}
