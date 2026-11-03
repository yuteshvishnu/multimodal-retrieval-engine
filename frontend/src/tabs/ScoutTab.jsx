import { useState } from "react";
import { Icon } from "../components/ui.jsx";
import SqlTab from "./SqlTab.jsx";
import TablesTab from "./TablesTab.jsx";

const VIEWS = [
  { id: "ask", label: "Ask", icon: "search" },
  { id: "tables", label: "Tables", icon: "table" },
];

// Rexie Scout (Text-to-SQL): ask questions about the data, and see (and fix) the tables it queries.
export default function ScoutTab() {
  const [view, setView] = useState("ask");

  return (
    <>
      <nav className="subnav" aria-label="Rexie Scout">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            className={v.id === view ? "subnav-item active" : "subnav-item"}
            onClick={() => setView(v.id)}
            aria-current={v.id === view ? "page" : undefined}
          >
            <Icon name={v.icon} size={14} />
            {v.label}
          </button>
        ))}
      </nav>

      {/* both stay mounted, so switching keeps each view's state */}
      <div hidden={view !== "ask"}>
        <SqlTab />
      </div>
      <div hidden={view !== "tables"}>
        <TablesTab />
      </div>
    </>
  );
}
