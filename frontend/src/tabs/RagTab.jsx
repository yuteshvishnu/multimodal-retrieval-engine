import { useState } from "react";
import { postForm } from "../api.js";

// Version-1: ask a question over the text notes; get an answer with citations.
export default function RagTab() {
  const [question, setQuestion] = useState("");
  const [collections, setCollections] = useState("");
  const [sources, setSources] = useState("");
  const [status, setStatus] = useState("");
  const [result, setResult] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    setStatus("Loading...");
    setResult(null);

    try {
      const data = await postForm("/api/rag/query", {
        query_text: question,
        collections,
        sources,
      });
      setResult(data);
      setStatus("");
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  }

  return (
    <section>
      <form onSubmit={handleSubmit} className="stacked">
        <label>
          Question
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. What does the heart do?"
            required
          />
        </label>
        <label>
          Collections (optional, comma-separated)
          <input
            type="text"
            value={collections}
            onChange={(e) => setCollections(e.target.value)}
            placeholder="e.g. medical_notes, research_papers"
          />
        </label>
        <label>
          Sources (optional, comma-separated)
          <input
            type="text"
            value={sources}
            onChange={(e) => setSources(e.target.value)}
            placeholder="e.g. heart_notes, lung_notes"
          />
        </label>
        <button type="submit">Ask</button>
      </form>

      {status && <p className="status">{status}</p>}

      {result && (
        <>
          <h3>Answer</h3>
          <pre>{result.answer}</pre>

          <h3>Citations</h3>
          <ul className="citations">
            {result.citations.map((c) => (
              <li key={c.id}>
                <strong>[{c.id}]</strong> score={c.score.toFixed(3)} (source={c.source}) — {c.snippet}
              </li>
            ))}
          </ul>

          <h3>Reasoning Summary</h3>
          <pre>{result.reasoning_summary}</pre>
        </>
      )}
    </section>
  );
}
