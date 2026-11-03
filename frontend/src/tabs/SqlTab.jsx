import { useRef, useState } from "react";
import { getJson, postForm } from "../api.js";
import { CopyButton, DataTable, ErrorAlert, Icon, LimitAlert, Loading, Pagination, SafetyAlert, SqlCode } from "../components/ui.jsx";

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

// Rexie Scout, Ask view: ask a question, pick what it means (or explain), then the LLM writes SQL,
// the backend runs it, and we show the table.
export default function SqlTab() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(""); // what we're waiting for, or ""
  const [error, setError] = useState("");
  const [safety, setSafety] = useState(""); // blocked: could change data (Version-7)
  const [limit, setLimit] = useState(""); // stopped: too big or too slow (Version-8)
  const [result, setResult] = useState(null);
  const [clarify, setClarify] = useState(null); // {options, round, max_rounds}
  const [history, setHistory] = useState([]); // rejected sets of options + the user's comment
  const [picked, setPicked] = useState(null); // option index, or "none"
  const [comment, setComment] = useState("");
  const inputRef = useRef(null);
  const [lastAnswer, setLastAnswer] = useState(null); // {question, meaning, sql} of the latest answer
  const [followUp, setFollowUp] = useState(false); // ask the next question as a follow-up to it

  // what a follow-up refers to, sent with both steps
  const previous = followUp && lastAnswer ? JSON.stringify(lastAnswer) : "";

  const step = result ? 2 : clarify ? 1 : loading === "query" ? 2 : 0;

  async function interpret(text, pastRounds) {
    setLoading("interpret");
    setError("");
    setSafety("");
    setLimit("");
    setClarify(null);
    setPicked(null);
    setComment("");
    try {
      const data = await postForm("/api/sql/interpret", {
        query_text: text,
        history: JSON.stringify(pastRounds),
        previous,
      });
      if (data.safety) setSafety(data.safety);
      else if (data.error) setError(data.error);
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
    setSafety("");
    setLimit("");
    try {
      const data = await postForm("/api/sql/query", { query_text: question, meaning, previous });
      if (data.safety) {
        setSafety(data.safety); // blocked: nothing was run, so there is no result to show
      } else if (data.limit) {
        setLimit(data.limit); // too big or too slow: nothing to show either
      } else {
        if (data.error) setError(data.error);
        setResult(data);
        if (data.sql && !data.error) {
          setLastAnswer({ question, meaning, sql: data.sql });
          setFollowUp(false);
        }
      }
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading("");
    }
  }

  // clear the answer and start a fresh question (the last answer stays available to follow up on)
  function startOver() {
    setQuestion("");
    setResult(null);
    setClarify(null);
    setHistory([]);
    setPicked(null);
    setComment("");
    setError("");
    setSafety("");
    setLimit("");
    setFollowUp(false);
    inputRef.current?.focus();
  }

  const busy = loading !== "";
  const idle = !busy && !clarify && !result && !error && !safety && !limit;

  return (
    <section>
      <div className="page-header">
        <h1>Talk to your data.</h1>
        <p>Ask anything in plain English. Scout checks what you mean, then writes and runs the SQL for you.</p>
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
              placeholder="Ask a question about your data"
              aria-label="Question"
              required
            />
          </label>
          <button type="submit" className="btn btn-primary" disabled={busy}>
            Ask
            <Icon name="arrow" />
          </button>
          {!idle && !busy && (
            <button type="button" className="btn btn-secondary" onClick={startOver}>
              <Icon name="refresh" />
              New question
            </button>
          )}
        </div>

        {lastAnswer && (
          <label className={`followup ${followUp ? "on" : ""}`}>
            <input type="checkbox" checked={followUp} onChange={(e) => setFollowUp(e.target.checked)} />
            <Icon name="link" size={13} />
            <span>
              Follow up on <strong>“{lastAnswer.question}”</strong>
            </span>
          </label>
        )}

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
      {safety && <SafetyAlert message={safety} />}
      {limit && <LimitAlert message={limit} />}
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
  // no query worked: the message is shown above; here only what was tried (Version-9)
  if (result.error) return <Attempts attempts={result.attempts} />;

  const meaning = result.meaning ? splitMeaning(result.meaning) : null;
  // page 1 comes with the answer; other pages are loaded from the server when clicked (Version-8)
  const [view, setView] = useState({ page: 1, rows: result.rows });
  const [pageError, setPageError] = useState("");
  const [loadingPage, setLoadingPage] = useState(false);
  const pageSize = result.page_size || result.rows.length || 1;
  const pages = Math.max(1, Math.ceil(result.row_count / pageSize));
  const rowsLabel = `${result.row_count}${result.more ? "+" : ""}`;

  async function loadPage(page) {
    setLoadingPage(true);
    setPageError("");
    try {
      const data = await getJson(`/api/sql/results/${result.result_id}`, { page });
      if (data.error) setPageError(data.error);
      else setView({ page: data.page, rows: data.rows });
    } catch (err) {
      setPageError(err.message);
    } finally {
      setLoadingPage(false);
    }
  }

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
            <div className="stat-value">{rowsLabel}</div>
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
              {rowsLabel} row{result.row_count === 1 ? "" : "s"}
            </span>
          </div>
          <div className={loadingPage ? "is-loading" : ""}>
            <DataTable
              columns={result.columns}
              rows={view.rows}
              firstIndex={(view.page - 1) * pageSize + 1}
            />
          </div>
          {pageError && <ErrorAlert message={pageError} />}
          <Pagination page={view.page} pages={pages} total={rowsLabel} onPage={loadPage} />
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

      <Attempts attempts={result.attempts} />
    </>
  );
}

// Every SQL the model wrote and what happened to it, collapsed by default.
function Attempts({ attempts, open = false }) {
  if (!attempts?.length) return null;
  return (
    <details className="card attempts" open={open}>
      <summary>
        <span className="card-title">
          <Icon name="layers" />
          All attempts
          <span className="tag">{attempts.length}</span>
        </span>
        <span className="chevron">
          <Icon name="chevron" />
        </span>
      </summary>
      {attempts.map((a, i) => (
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
  );
}
