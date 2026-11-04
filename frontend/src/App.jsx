import { useEffect, useState } from "react";
import { getJson } from "./api.js";
import { AccuracyRing, Icon } from "./components/ui.jsx";
import ScoutTab from "./tabs/ScoutTab.jsx";
import ScribeTab from "./tabs/ScribeTab.jsx";
import DashboardTab from "./tabs/DashboardTab.jsx";
import ArchitectureTab from "./tabs/ArchitectureTab.jsx";
import EvaluationTab from "./tabs/EvaluationTab.jsx";

// Scribe (RAG) is hidden for now; set to true to bring its tab back
const SHOW_SCRIBE = false;

// Rexie's two products
const TABS = [
  { id: "rag", label: "Scribe", hint: "Rexie Scribe: ask your documents (RAG)", icon: "doc", hidden: !SHOW_SCRIBE },
  { id: "sql", label: "Scout", hint: "Rexie Scout: ask your data (Text-to-SQL)", icon: "database" },
  { id: "dashboard", label: "Dashboard", hint: "LLM calls, tokens, time and cost of every question", icon: "chart" },
  { id: "architecture", label: "Architecture", hint: "How Scout is built, and what each version changed", icon: "arch" },
  { id: "evaluation", label: "Evaluation", hint: "How accurate Scout is on the BIRD benchmark", icon: "target" },
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
  const [active, setActive] = useState(SHOW_SCRIBE ? "rag" : "sql");
  const [theme, setTheme] = useState(savedTheme);
  const [score, setScore] = useState(null); // the latest benchmark run, for the ring by the name

  function loadScore() {
    getJson("/api/eval/summary").then(setScore).catch(() => setScore(null));
  }

  useEffect(loadScore, []);

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
            {score && (
              <button
                type="button"
                className="brand-score"
                onClick={() => setActive("evaluation")}
                title={`${score.benchmark}: ${score.correct} of ${score.total} right${score.sample ? " (sample data)" : ""}. Open Evaluation.`}
              >
                <AccuracyRing value={score.accuracy} size={24} stroke={3.5} />
                <span className="brand-score-text">
                  <small>{score.short || "BIRD"}</small>
                  <b>{score.accuracy == null ? "not run" : `${score.accuracy}%`}</b>
                </span>
                {score.sample && <span className="brand-score-sample">sample</span>}
              </button>
            )}
          </div>

          <div className="topbar-actions">
            <nav className="tabs">
              {TABS.filter((tab) => !tab.hidden).map((tab) => (
                <button
                  key={tab.id}
                  className={tab.id === active ? "tab active" : "tab"}
                  onClick={() => setActive(tab.id)}
                  title={tab.hint}
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
        {SHOW_SCRIBE && (
          <div hidden={active !== "rag"}>
            <ScribeTab />
          </div>
        )}
        <div hidden={active !== "sql"}>
          <ScoutTab />
        </div>
        <div hidden={active !== "dashboard"}>
          <DashboardTab active={active === "dashboard"} />
        </div>
        <div hidden={active !== "architecture"}>
          <ArchitectureTab />
        </div>
        <div hidden={active !== "evaluation"}>
          <EvaluationTab active={active === "evaluation"} onRunFinished={loadScore} />
        </div>
      </main>
    </>
  );
}
