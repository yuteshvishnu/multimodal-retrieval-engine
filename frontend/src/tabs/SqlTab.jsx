import { useRef, useState } from "react";
import { getJson, postForm } from "../api.js";
import {
  CopyButton, DataTable, ErrorAlert, Icon, LimitAlert, Loading, Pagination, SafetyAlert, SqlCode,
} from "../components/ui.jsx";

const EXAMPLES = [
  "How many orders were placed in March 2026?",
  "Which product category generated the most revenue, and how much in dollars?",
  "For each store that has orders, which employee handled the most orders there?",
];

const STEPS = ["Ask", "Confirm meaning", "Check steps", "Answer"];
// meanings in all when the checks send a question back to the meaning (MAX_MEANING_TRIES in config.py)
const MAX_MEANING_TRIES = 2;

// The model's meaning plus its choices, as one text for the SQL step.
function meaningText(option) {
  return [option.meaning, ...option.choices.map((c) => `- ${c}`)].join("\n");
}

// Splits the meaning text back into its sentence and its choices, for display.
function splitMeaning(text) {
  const [first, ...rest] = text.split("\n");
  return { meaning: first, choices: rest.map((line) => line.replace(/^- /, "")) };
}

// Rexie Scout, Ask view: ask a question, pick what it means (or explain), check the steps
// (or comment on wrong ones), then the LLM writes SQL that follows them, the backend runs it,
// and we show the table with the steps; the user can report an issue with it. When the question
// was answered before (in any words), its meaning is the first option, and picking it answers
// straight away (Version-12).
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
  const [questionId, setQuestionId] = useState(""); // ties this question's steps together on the Dashboard
  const inputRef = useRef(null);
  const [lastAnswer, setLastAnswer] = useState(null); // {question, meaning, sql} of the latest answer
  const [followUp, setFollowUp] = useState(false); // ask the next question as a follow-up to it
  const [meaning, setMeaning] = useState(""); // the meaning the user confirmed
  const [plan, setPlan] = useState(null); // {steps, round, max_rounds} to check (Version-10)
  const [planHistory, setPlanHistory] = useState([]); // earlier steps + the user's comments
  const [shownOptions, setShownOptions] = useState([]); // the meanings offered when the user picked one
  const [dataNote, setDataNote] = useState(""); // the checks found the meaning asks for values the data doesn't have

  // what a follow-up refers to, sent with both steps
  const previous = followUp && lastAnswer ? JSON.stringify(lastAnswer) : "";

  const step = result || loading === "query" ? 3 : plan || loading === "plan" ? 2 : clarify ? 1 : 0;

  async function interpret(text, pastRounds, id = questionId) {
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
        question_id: id,
      });
      if (data.question_id) setQuestionId(data.question_id);
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
    setPlan(null);
    setPlanHistory([]);
    setHistory([]);
    setDataNote("");
    setQuestionId(""); // a new question gets a new id
    interpret(text, [], "");
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

    const option = clarify.options[picked];
    const confirmed = meaningText(option);
    setShownOptions(clarify.options);
    setDataNote("");
    setMeaning(confirmed);
    setClarify(null);
    setPlanHistory([]);
    // a saved answer's meaning: run its SQL again; if that can't be done, write the steps as usual
    if (option.reuse_id) runQuery([], confirmed, option.reuse_id);
    else requestPlan(confirmed, []);
  }

  // the steps for the confirmed meaning; pastTries holds earlier steps with the user's comments
  async function requestPlan(confirmed, pastTries) {
    setLoading("plan");
    setError("");
    setSafety("");
    setPlan(null);
    try {
      const data = await postForm("/api/sql/plan", {
        query_text: question,
        meaning: confirmed,
        previous,
        question_id: questionId,
        history: JSON.stringify(pastTries),
      });
      if (data.safety) setSafety(data.safety);
      else if (data.error) setError(data.error);
      else setPlan(data);
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading("");
    }
  }

  // only the steps are written again, with the user's comments; the meaning stays
  function rerunSteps(comments) {
    const pastTries = [...planHistory, { steps: plan.steps, comments }];
    setPlanHistory(pastTries);
    requestPlan(meaning, pastTries);
  }

  async function runQuery(steps = plan.steps, confirmed = meaning, reuseId = "") {
    setPlan(null);
    setLoading("query");
    setError("");
    setSafety("");
    setLimit("");
    try {
      const data = await postForm("/api/sql/query", {
        query_text: question,
        meaning: confirmed,
        previous,
        question_id: questionId,
        steps: JSON.stringify(steps),
        reuse_id: reuseId,
      });
      if (data.fallback) {
        setLoading("");
        requestPlan(confirmed, []);
        return;
      }
      // the meaning asks for a value the data doesn't have: offer the meanings again, told the real
      // values (once: at most MAX_MEANING_TRIES meanings in all, as in config.py)
      const asked = history.filter((round) => round.fromCheck).length;
      if (data.rethink_meaning && asked < MAX_MEANING_TRIES - 1) {
        const pastRounds = [...history, { options: shownOptions, comment: data.rethink_meaning, fromCheck: true }];
        setHistory(pastRounds);
        setDataNote(data.rethink_meaning);
        setLoading("");
        interpret(question, pastRounds);
        return;
      }
      if (data.safety) {
        setSafety(data.safety); // blocked: nothing was run, so there is no result to show
      } else if (data.limit) {
        setLimit(data.limit); // too big or too slow: nothing to show either
      } else {
        if (data.error) setError(data.error);
        setResult(data);
        if (data.sql && !data.error) {
          setLastAnswer({ question, meaning: confirmed, sql: data.sql });
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
    setPlan(null);
    setPlanHistory([]);
    setMeaning("");
    setHistory([]);
    setDataNote("");
    setPicked(null);
    setComment("");
    setError("");
    setSafety("");
    setLimit("");
    setFollowUp(false);
    setQuestionId("");
    inputRef.current?.focus();
  }

  const busy = loading !== "";
  const idle = !busy && !clarify && !plan && !result && !error && !safety && !limit;

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
      {loading === "plan" && <Loading text="Breaking it into steps..." />}
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
            {dataNote && <p className="alert alert-warning">{dataNote}</p>}
            <div className="options">
              {clarify.options.map((option, i) => (
                <label key={i} className={`option ${picked === i ? "picked" : ""}`}>
                  <input type="radio" name="meaning" checked={picked === i} onChange={() => setPicked(i)} />
                  <div>
                    <div className="option-text">
                      {option.meaning}
                    </div>
                    {option.choices.length > 0 && (
                      <div className="option-choices">
                        {option.choices.map((c) => (
                          <span key={c} className="tag">
                            {c}
                          </span>
                        ))}
                      </div>
                    )}
                    {option.warning && (
                      <p className="option-warning">
                        <Icon name="alert" size={13} />
                        {option.warning}
                      </p>
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
                {picked === "none" ? "Show new options" : "Confirm meaning"}
                <Icon name="arrow" size={14} />
              </button>
            </div>
          </div>
        </form>
      )}

      {plan && <StepsCheck key={plan.round} plan={plan} onRerun={rerunSteps} onConfirm={() => runQuery()} />}

      {result?.sql && <SqlResult key={result.sql} result={result} meaning={meaning} questionId={questionId} />}
    </section>
  );
}

// Version-10: the steps to check. Click a step to say what's wrong with it; "Re-run steps"
// writes only the steps again with those comments; "Confirm steps" writes and runs the SQL.
function StepsCheck({ plan, onRerun, onConfirm }) {
  const [comments, setComments] = useState({}); // step index -> comment
  const [open, setOpen] = useState(null); // the step being commented on

  const flagged = Object.entries(comments).filter(([, c]) => c.trim());
  const lastTry = plan.round >= plan.max_rounds;

  function rerun() {
    onRerun(flagged.map(([i, comment]) => ({ step: Number(i) + 1, comment: comment.trim() })));
  }

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">
          <Icon name="layers" />
          Check the steps
        </h3>
        <span className="tag">
          Try {plan.round} of {plan.max_rounds}
        </span>
      </div>
      <div className="card-body">
        <p className="muted steps-intro">
          This is how Scout will answer. Click a step that isn't right and say what's wrong.
        </p>
        <ol className="plan-steps">
          {plan.steps.map((text, i) => {
            const note = comments[i] || "";
            return (
              <li key={i} className={`plan-step ${note.trim() ? "flagged" : ""} ${open === i ? "open" : ""}`}>
                <button type="button" className="plan-step-head" onClick={() => setOpen(open === i ? null : i)}>
                  <span className="plan-step-no">{i + 1}</span>
                  <span className="plan-step-text">
                    {text}
                  </span>
                  <span className="plan-step-action">
                    {note.trim() ? (
                      <>
                        <Icon name="alert" size={13} /> Flagged
                      </>
                    ) : (
                      <>
                        <Icon name="edit" size={13} /> Not right?
                      </>
                    )}
                  </span>
                </button>
                {open === i && (
                  <textarea
                    value={note}
                    onChange={(e) => setComments({ ...comments, [i]: e.target.value })}
                    placeholder="What's wrong with this step? e.g. count cancelled orders too"
                    rows={2}
                    autoFocus
                  />
                )}
              </li>
            );
          })}
        </ol>

        <div className="review-actions steps-actions">
          {flagged.length > 0 ? (
            <button type="button" className="btn btn-primary btn-sm" onClick={rerun} disabled={lastTry}>
              <Icon name="refresh" size={14} />
              Re-run steps ({flagged.length} comment{flagged.length === 1 ? "" : "s"})
            </button>
          ) : (
            <button type="button" className="btn btn-primary btn-sm" onClick={onConfirm}>
              <Icon name="check" size={14} />
              Steps look right: answer
            </button>
          )}
          {lastTry && flagged.length > 0 && (
            <span className="muted">That was the last try. Remove the comments to answer, or rephrase the question.</span>
          )}
        </div>
      </div>
    </div>
  );
}

function SqlResult({ result, meaning, questionId }) {
  // no query worked: the plain-English message above is all the user needs (Version-9)
  if (result.error) return null;
  return <Answer result={result} confirmedMeaning={meaning} questionId={questionId} />;
}

function Answer({ result, confirmedMeaning, questionId }) {
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
            <div className="option-text">
              {meaning.meaning}
            </div>
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
        {result.steps?.length > 0 && (
          <div className="meaning-box steps-followed">
            <div className="label">How this was answered</div>
            <ol className="steps-list">
              {result.steps.map((s, i) => (
                <li key={i}>
                  {s}
                </li>
              ))}
            </ol>
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
          <AnswerFeedback result={result} meaning={confirmedMeaning} questionId={questionId} rows={view.rows} />
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
    </>
  );
}

// Report issue on the answer (Version-10; Version-12: an answer nobody reports counts as right,
// so there's no "Correct" button). A report needs a comment, and is logged with what the user
// saw (steps, SQL, first rows) as an issue on the Dashboard, where the Failure Analyst agent can
// investigate it (Version-11). A reported answer is never reused.
function AnswerFeedback({ result, meaning, questionId, rows }) {
  const [rating, setRating] = useState(null); // sent: "up" | "down"
  const [writing, setWriting] = useState(false);
  const [comment, setComment] = useState("");
  const [error, setError] = useState("");

  async function send(value, text = "") {
    setError("");
    try {
      await postForm("/api/sql/feedback", {
        question_id: questionId,
        rating: value,
        comment: text,
        details: JSON.stringify({
          meaning,
          steps: result.steps,
          sql: result.sql,
          columns: result.columns,
          rows: rows.slice(0, 10),
          row_count: result.row_count,
        }),
      });
      setRating(value);
      setWriting(false);
    } catch (err) {
      setError(err.message);
    }
  }

  if (!questionId) return null;
  return (
    <div className="answer-feedback">
      {rating ? (
        <span className="muted">Thanks. Logged as an issue on the Dashboard.</span>
      ) : writing ? (
        <form
          className="feedback-form"
          onSubmit={(e) => {
            e.preventDefault();
            send("down", comment);
          }}
        >
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="What's wrong with this answer?"
            rows={2}
            autoFocus
            required
          />
          <div className="review-actions">
            <button type="submit" className="btn btn-primary btn-sm" disabled={!comment.trim()}>
              Send
            </button>
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => setWriting(false)}>
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <>
          <span className="muted">Something wrong with this answer?</span>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => setWriting(true)}>
            <Icon name="flag" size={14} />
            Report issue
          </button>
        </>
      )}
      {error && <span className="problems">{error}</span>}
    </div>
  );
}
