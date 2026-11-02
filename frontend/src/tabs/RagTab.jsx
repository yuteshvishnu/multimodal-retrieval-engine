import { useState } from "react";
import { postForm } from "../api.js";
import { ErrorAlert, Icon, Loading } from "../components/ui.jsx";

const EXAMPLES = ["What does the heart do?", "How do the lungs exchange oxygen?"];

// RAG: ask a question over the text notes; get an answer with citations.
export default function RagTab() {
  const [question, setQuestion] = useState("");
  const [collections, setCollections] = useState("");
  const [sources, setSources] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  async function ask(text) {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await postForm("/api/rag/query", { query_text: text, collections, sources });
      setResult(data);
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    ask(question);
  }

  function tryExample(text) {
    setQuestion(text);
    ask(text);
  }

  const maxScore = Math.max(...(result?.citations || []).map((c) => c.score), 0) || 1;

  return (
    <section>
      <div className="page-header">
        <h1>Talk to your documents.</h1>
        <p>Ask anything about your notes. Rexie answers only from what's in them, and shows you exactly where.</p>
      </div>

      <form onSubmit={handleSubmit} className="card composer">
        <div className="composer-row">
          <label className="search-field">
            <Icon name="search" size={18} />
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. What does the heart do?"
              aria-label="Question"
              required
            />
          </label>
          <button type="submit" className="btn btn-primary" disabled={loading}>
            Ask
            <Icon name="arrow" />
          </button>
        </div>

        <button type="button" className="filters-toggle" onClick={() => setShowFilters(!showFilters)}>
          <Icon name="sliders" size={13} />
          {showFilters ? "Hide filters" : "Filters (optional)"}
        </button>

        {showFilters && (
          <div className="filters">
            <label className="field">
              Collections (comma-separated)
              <input
                type="text"
                value={collections}
                onChange={(e) => setCollections(e.target.value)}
                placeholder="e.g. medical_notes, research_papers"
              />
            </label>
            <label className="field">
              Sources (comma-separated)
              <input
                type="text"
                value={sources}
                onChange={(e) => setSources(e.target.value)}
                placeholder="e.g. heart_notes, lung_notes"
              />
            </label>
          </div>
        )}

        {!result && !loading && (
          <div className="examples">
            <span className="examples-label">Try:</span>
            {EXAMPLES.map((text) => (
              <button type="button" key={text} className="chip" onClick={() => tryExample(text)}>
                {text}
              </button>
            ))}
          </div>
        )}
      </form>

      {loading && <Loading text="Searching your documents and writing an answer..." />}
      {error && <ErrorAlert message={error} />}

      {!result && !loading && !error && (
        <div className="card empty">
          <div className="empty-icon">
            <Icon name="doc" size={22} />
          </div>
          <h3>Ask a question to get started</h3>
          <p>The answer cites the exact passages it used.</p>
        </div>
      )}

      {result && (
        <>
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">
                <Icon name="sparkle" />
                Answer
              </h3>
            </div>
            <div className="card-body">
              <p className="answer">{result.answer}</p>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h3 className="card-title">
                <Icon name="link" />
                Citations
              </h3>
              <span className="tag">{result.citations.length}</span>
            </div>
            <div className="citations">
              {result.citations.map((c, i) => (
                <div key={c.id} className="citation">
                  <span className="citation-id">{i + 1}</span>
                  <div>
                    <div className="citation-meta">
                      <span className="citation-source">{c.source}</span>
                      <span className="citation-chunk">{c.id}</span>
                      <span className="score-bar" title={`score ${c.score.toFixed(3)}`}>
                        <span style={{ width: `${Math.max(0, (c.score / maxScore) * 100)}%` }} />
                      </span>
                      <span>score {c.score.toFixed(3)}</span>
                    </div>
                    <p className="citation-snippet">{c.snippet}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {result.reasoning_summary && (
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">
                  <Icon name="bulb" />
                  Reasoning summary
                </h3>
              </div>
              <div className="card-body">
                <p className="reasoning">{result.reasoning_summary}</p>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
