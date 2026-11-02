import { useState } from "react";
import { postForm } from "../api.js";
import { CopyButton, DataTable, ErrorAlert, Icon, Loading, Pagination, SqlCode } from "../components/ui.jsx";

const EXAMPLES = [
  "How many orders were placed in March 2026?",
  "Which product category generated the most revenue, and how much in dollars?",
  "For each store that has orders, which employee handled the most orders there?",
];

const STEPS = ["Ask", "Confirm meaning", "Answer"];

// The model's meaning plus its choices, as one text for the SQL step.
function meaningText(option) {
  return [option.meaning, ...option.choices.map((c) => `- ${c}`)].join("\n");
}

// Splits the meaning text back into its sentence and its choices, for display.
function splitMeaning(text) {
  const [first, ...rest] = text.split("\n");
  return { meaning: first, choices: rest.map((line) => line.replace(/^- /, "")) };
}

const PAGE_SIZE = 10;

// Text-to-SQL, Ask view: ask a question, pick what it means (or explain), then the LLM writes SQL,
// the backend runs it, and we show the table.
export default function SqlTab() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(""); // what we're waiting for, or ""
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [clarify, setClarify] = useState(null); // {options, round, max_rounds}
  const [history, setHistory] = useState([]); // rejected sets of options + the user's comment
  const [picked, setPicked] = useState(null); // option index, or "none"
  const [comment, setComment] = useState("");

  const step = result ? 2 : clarify ? 1 : loading === "query" ? 2 : 0;

  async function interpret(text, pastRounds) {
    setLoading("interpret");
    setError("");
    setClarify(null);
    setPicked(null);
    setComment("");
    try {
      const data = await postForm("/api/sql/interpret", {
        query_text: text,
        history: JSON.stringify(pastRounds),
      });
      if (data.error) setError(data.error);
      else setClarify(data);
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading("");
    }
  }

  function start(text) {
    setResult(null);
    setHistory([]);
    interpret(text, []);
  }

  function handleSubmit(event) {
    event.preventDefault();
    start(question);
  }

  function tryExample(text) {
    setQuestion(text);
    start(text);
  }

  async function handleConfirm(event) {
    event.preventDefault();
    if (picked === "none") {
      const pastRounds = [...history, { options: clarify.options, comment }];
      setHistory(pastRounds);
      interpret(question, pastRounds);
      return;
    }

    const meaning = meaningText(clarify.options[picked]);
    setClarify(null);
    setLoading("query");
    setError("");
    try {
      const data = await postForm("/api/sql/query", { query_text: question, meaning });
      if (data.error) setError(data.error);
      setResult(data);
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading("");
    }
  }

  const busy = loading !== "";
  const idle = !busy && !clarify && !result && !error;

  return (
    <section>
      <div className="page-header">
        <h1>Talk to your data.</h1>
        <p>Ask anything in plain English. Rexie checks what you mean, then writes and runs the SQL for you.</p>
      </div>

      <form onSubmit={handleSubmit} className="card composer">
        <div className="composer-row">
          <label className="search-field">
            <Icon name="search" size={18} />
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about your data"
              aria-label="Question"
              required
            />
          </label>
          <button type="submit" className="btn btn-primary" disabled={busy}>
            Ask
            <Icon name="arrow" />
          </button>
        </div>

        {idle && (
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

      {!idle && (
        <div className="steps" aria-label="Progress">
          {STEPS.map((label, i) => (
            <div key={label} className="step-group" style={{ display: "contents" }}>
              {i > 0 && <span className="step-line" />}
              <span className={`step ${i === step ? "active" : i < step ? "done" : ""}`}>
                <span className="step-dot">{i < step ? <Icon name="check" size={12} /> : i + 1}</span>
                {label}
              </span>
            </div>
          ))}
        </div>
      )}

      {loading === "interpret" && <Loading text="Working out what you mean..." />}
      {loading === "query" && <Loading text="Writing SQL, checking it and running it..." />}
      {error && <ErrorAlert message={error} />}

      {idle && (
        <div className="card empty">
          <div className="empty-icon">
            <Icon name="database" size={22} />
          </div>
          <h3>Ask a question to get started</h3>
          <p>You'll see the SQL that was run and every row it returned.</p>
        </div>
      )}

      {clarify && (
        <form onSubmit={handleConfirm} className="card">
          <div className="card-header">
            <h3 className="card-title">
              <Icon name="bulb" />
              Which did you mean?
            </h3>
            <span className="tag">
              Try {clarify.round} of {clarify.max_rounds}
            </span>
          </div>
          <div className="card-body">
            <div className="options">
              {clarify.options.map((option, i) => (
                <label key={i} className={`option ${picked === i ? "picked" : ""}`}>
                  <input type="radio" name="meaning" checked={picked === i} onChange={() => setPicked(i)} />
                  <div>
                    <div className="option-text">{option.meaning}</div>
                    {option.choices.length > 0 && (
                      <div className="option-choices">
                        {option.choices.map((c) => (
                          <span key={c} className="tag">
                            {c}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </label>
              ))}

              <label className={`option ${picked === "none" ? "picked" : ""}`}>
                <input type="radio" name="meaning" checked={picked === "none"} onChange={() => setPicked("none")} />
                <div style={{ flex: 1 }}>
                  <div className="option-text">None of these</div>
                  {picked === "none" && (
                    <textarea
                      value={comment}
                      onChange={(e) => setComment(e.target.value)}
                      placeholder="Explain what you meant, e.g. count cancelled orders too"
                      rows={3}
                      style={{ marginTop: 10 }}
                      autoFocus
                      required
                    />
                  )}
                </div>
              </label>
            </div>

            <div className="clarify-footer">
              <button type="submit" className="btn btn-primary btn-sm" disabled={picked === null}>
                {picked === "none" ? "Show new options" : "Confirm and run"}
                <Icon name="arrow" size={14} />
              </button>
            </div>
          </div>
        </form>
      )}

      {result?.sql && <SqlResult key={result.sql} result={result} />}
    </section>
  );
}

function SqlResult({ result }) {
  const meaning = result.meaning ? splitMeaning(result.meaning) : null;
  const [page, setPage] = useState(1);
  const pages = Math.max(1, Math.ceil(result.rows.length / PAGE_SIZE));
  const start = (page - 1) * PAGE_SIZE;

  return (
    <>
      <div className="card">
        {meaning && (
          <div className="meaning-box">
            <div className="label">Meaning you confirmed</div>
            <div className="option-text">{meaning.meaning}</div>
            {meaning.choices.length > 0 && (
              <div className="option-choices">
                {meaning.choices.map((c) => (
                  <span key={c} className="tag">
                    {c}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
        <div className="stats">
          <div className="stat">
            <div className="label">Rows</div>
            <div className="stat-value">{result.rows.length}</div>
          </div>
          <div className="stat">
            <div className="label">Confidence</div>
            <div className="stat-value">
              {result.confidence ? <span className={`badge ${result.confidence}`}>{result.confidence}</span> : "—"}
            </div>
          </div>
          <div className="stat">
            <div className="label">LLM calls</div>
            <div className="stat-value">{result.llm_calls}</div>
          </div>
          <div className="stat">
            <div className="label">Tables used</div>
            <div className="stat-tables">
              {result.tables.map((t) => (
                <span key={t} className="tag">
                  {t}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {result.warnings?.length > 0 && (
        <div className="alert alert-warning">
          <Icon name="alert" />
          <ul>
            {result.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {result.columns.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              <Icon name="table" />
              Answer
            </h3>
            <span className="tag">
              {result.rows.length} row{result.rows.length === 1 ? "" : "s"}
            </span>
          </div>
          <DataTable
            columns={result.columns}
            rows={result.rows.slice(start, start + PAGE_SIZE)}
            firstIndex={start + 1}
          />
          <Pagination page={page} pages={pages} total={result.rows.length} onPage={setPage} />
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <Icon name="code" />
            SQL
          </h3>
          <CopyButton text={result.sql} />
        </div>
        <SqlCode sql={result.sql} />
      </div>

      {result.attempts?.length > 0 && (
        <details className="card attempts">
          <summary>
            <span className="card-title">
              <Icon name="layers" />
              All attempts
              <span className="tag">{result.attempts.length}</span>
            </span>
            <span className="chevron">
              <Icon name="chevron" />
            </span>
          </summary>
          {result.attempts.map((a, i) => (
            <div key={i} className="attempt">
              <div className="attempt-head">
                <strong>Attempt {i + 1}</strong>
                <span>· Round {a.round} ·</span>
                <span className={`status-${a.status}`}>{a.status}</span>
              </div>
              {a.problems.length > 0 && <p className="problems">{a.problems.join(" ")}</p>}
              {a.sql && <SqlCode sql={a.sql} />}
            </div>
          ))}
        </details>
      )}
    </>
  );
}
