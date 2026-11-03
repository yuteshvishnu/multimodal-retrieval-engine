import { useState } from "react";
import { Icon } from "../components/ui.jsx";
import RagTab from "./RagTab.jsx";
import SourcesTab from "./SourcesTab.jsx";

const VIEWS = [
  { id: "ask", label: "Ask", icon: "search" },
  { id: "sources", label: "Sources", icon: "doc" },
];

// Rexie Scribe (RAG): ask questions about your documents, and see and add the documents it answers from.
export default function ScribeTab() {
  const [view, setView] = useState("ask");
  const [refreshKey, setRefreshKey] = useState(0); // bumps when a document is added

  return (
    <>
      <nav className="subnav" aria-label="Rexie Scribe">
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
        <RagTab refreshKey={refreshKey} />
      </div>
      <div hidden={view !== "sources"}>
        <SourcesTab onAdded={() => setRefreshKey((k) => k + 1)} />
      </div>
    </>
  );
}
