import { useState } from "react";
import RagTab from "./tabs/RagTab.jsx";
import SqlTab from "./tabs/SqlTab.jsx";

const TABS = [
  { id: "rag", label: "RAG (Version-1)" },
  { id: "sql", label: "Text-to-SQL (Version-2)" },
];

export default function App() {
  const [active, setActive] = useState("rag");

  return (
    <main>
      <h1>Retrieval &amp; Reasoning Engine</h1>

      <nav className="tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={tab.id === active ? "tab active" : "tab"}
            onClick={() => setActive(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* both stay mounted, so switching tabs keeps each tab's results */}
      <div hidden={active !== "rag"}>
        <RagTab />
      </div>
      <div hidden={active !== "sql"}>
        <SqlTab />
      </div>
    </main>
  );
}
