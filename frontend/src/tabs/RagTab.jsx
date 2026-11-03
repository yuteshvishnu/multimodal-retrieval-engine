import { useEffect, useRef, useState } from "react";
import { getJson, postForm, postJson } from "../api.js";
import { ErrorAlert, Icon, Loading } from "../components/ui.jsx";

const EXAMPLES = ["What does the heart do?", "How do the lungs exchange oxygen?"];

// Rexie Scribe, Ask view: ask a question over the chosen documents (or all of them);
// get an answer with citations. refreshKey changes when a document is added.
export default function RagTab({ refreshKey = 0 }) {
  const [question, setQuestion] = useState("");
  const [documents, setDocuments] = useState([]); // [{name, type}]
  const [selected, setSelected] = useState(null); // null = all documents, else [names]
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const inputRef = useRef(null);

  async function ask(text) {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await postForm("/api/rag/query", {
        query_text: text,
        sources: selected ? selected.join(",") : "",
      });
      setResult(data);
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    getJson("/api/rag/documents", { all: true })
      .then((data) => setDocuments(data.items))
      .catch(() => setDocuments([]));
  }, [refreshKey]);

  // clear the answer and start a fresh question (the document choice is kept)
  function startOver() {
    setQuestion("");
    setResult(null);
    setError("");
    inputRef.current?.focus();
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
        <p>Ask anything about your documents. Scribe answers only from what's in them, and shows you exactly where.</p>
      </div>

      <form onSubmit={handleSubmit} className="card composer">
        <div className="composer-row">
          <label className="search-field">
            <Icon name="search" size={18} />
            <input
              ref={inputRef}
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
          {(result || error) && !loading && (
            <button type="button" className="btn btn-secondary" onClick={startOver}>
              <Icon name="refresh" />
              New question
            </button>
          )}
        </div>

        <DocumentPicker documents={documents} selected={selected} onChange={setSelected} />

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
              <Feedback key={result.answer} question={question} result={result} />
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

// "Search in: All documents" + a dropdown to pick any number of documents.
function DocumentPicker({ documents, selected, onChange }) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const box = useRef(null);

  // close when clicking anywhere else
  useEffect(() => {
    if (!open) return undefined;
    function onClick(event) {
      if (box.current && !box.current.contains(event.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [open]);

  function toggle(name) {
    const current = selected || [];
    const next = current.includes(name) ? current.filter((n) => n !== name) : [...current, name];
    onChange(next.length === 0 || next.length === documents.length ? null : next);
  }

  const label = !selected
    ? `All documents (${documents.length})`
    : selected.length === 1
      ? selected[0]
      : `${selected.length} documents`;
  const shown = documents.filter((d) => d.name.toLowerCase().includes(filter.toLowerCase()));

  return (
    <div className="picker" ref={box}>
      <span className="examples-label">Search in:</span>
      <button type="button" className="picker-button" onClick={() => setOpen(!open)} aria-expanded={open}>
        <Icon name="doc" size={13} />
        {label}
        <span className={`picker-caret ${open ? "open" : ""}`}>
          <Icon name="chevron" size={12} />
        </span>
      </button>

      {open && (
        <div className="picker-menu" role="dialog" aria-label="Choose documents">
          <label className="picker-option picker-all">
            <input type="checkbox" checked={!selected} onChange={() => onChange(null)} />
            <span>All documents</span>
          </label>
          {documents.length > 6 && (
            <input
              className="picker-search"
              type="text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Find a document"
              autoFocus
            />
          )}
          <div className="picker-list">
            {shown.map((d) => (
              <label key={d.name} className="picker-option">
                <input type="checkbox" checked={!!selected?.includes(d.name)} onChange={() => toggle(d.name)} />
                <span className="picker-name">{d.name}</span>
                <span className="tag">{d.type}</span>
              </label>
            ))}
            {shown.length === 0 && <p className="muted picker-empty">No documents match.</p>}
          </div>
          <div className="picker-footer">
            <span className="muted">{selected ? `${selected.length} selected` : "Searching everything"}</span>
            <button type="button" className="btn btn-primary btn-sm" onClick={() => setOpen(false)}>
              Done
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// Correct / Report issue on an answer, logged to data/rag/feedback/ to tune retrieval later (Version-1.7).
function Feedback({ question, result }) {
  const [sent, setSent] = useState(null); // "up" | "down"
  const [error, setError] = useState("");

  async function send(rating) {
    setError("");
    try {
      await postJson("/api/rag/feedback", {
        query_text: question,
        answer: result.answer,
        citations: result.citations.map(({ id, source, score }) => ({ id, source, score })),
        rating,
        question_id: result.question_id, // so it shows on the Dashboard
      });
      setSent(rating);
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="feedback">
      {sent ? (
        <span className="muted">Thanks for the feedback.</span>
      ) : (
        <>
          <span className="muted">Is this answer correct?</span>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => send("up")}>
            <Icon name="check" size={14} />
            Correct
          </button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => send("down")}>
            <Icon name="flag" size={14} />
            Report issue
          </button>
        </>
      )}
      {error && <span className="problems">{error}</span>}
    </div>
  );
}
