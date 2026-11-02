import { useState } from "react";
import { postForm } from "../api.js";

// The model's meaning plus its choices, as one text for the SQL step.
function meaningText(option) {
  return [option.meaning, ...option.choices.map((c) => `- ${c}`)].join("\n");
}

// Version-5: ask a question, pick what it means (or explain), then the LLM writes SQL,
// the backend runs it, and we show the table.
export default function SqlTab() {
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("");
  const [result, setResult] = useState(null);
  const [clarify, setClarify] = useState(null); // {options, round, max_rounds}
  const [history, setHistory] = useState([]); // rejected sets of options + the user's comment
  const [picked, setPicked] = useState(null); // option index, or "none"
  const [comment, setComment] = useState("");

  async function interpret(pastRounds) {
    setStatus("Working out what you mean...");
    setClarify(null);
    setPicked(null);
    setComment("");
    try {
      const data = await postForm("/api/sql/interpret", {
        query_text: question,
        history: JSON.stringify(pastRounds),
      });
      if (data.error) {
        setStatus(data.error);
        return;
      }
      setClarify(data);
      setStatus("");
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    setResult(null);
    setHistory([]);
    interpret([]);
  }

  async function handleConfirm(event) {
    event.preventDefault();
    if (picked === "none") {
      const pastRounds = [...history, { options: clarify.options, comment }];
      setHistory(pastRounds);
      interpret(pastRounds);
      return;
    }

    const meaning = meaningText(clarify.options[picked]);
    setClarify(null);
    setStatus("Loading...");
    try {
      const data = await postForm("/api/sql/query", { query_text: question, meaning });
      setResult(data);
      setStatus(data.error ? data.error : `${data.rows.length} row(s)`);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  }

  return (
    <section>
      <form onSubmit={handleSubmit} className="inline">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your data"
          required
        />
        <button type="submit">Ask</button>
      </form>

      {status && <p className="status">{status}</p>}

      {clarify && (
        <form onSubmit={handleConfirm} className="clarify">
          <p>
            <strong>Which did you mean?</strong>{" "}
            <span className="muted">
              (try {clarify.round} of {clarify.max_rounds})
            </span>
          </p>
          {clarify.options.map((option, i) => (
            <label key={i} className={`option ${picked === i ? "picked" : ""}`}>
              <input type="radio" name="meaning" checked={picked === i} onChange={() => setPicked(i)} />
              <span>
                {option.meaning}
                {option.choices.length > 0 && (
                  <ul>
                    {option.choices.map((c) => (
                      <li key={c}>{c}</li>
                    ))}
                  </ul>
                )}
              </span>
            </label>
          ))}
          <label className={`option ${picked === "none" ? "picked" : ""}`}>
            <input type="radio" name="meaning" checked={picked === "none"} onChange={() => setPicked("none")} />
            <span>None of these</span>
          </label>
          {picked === "none" && (
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Explain what you meant"
              rows={3}
              required
            />
          )}
          <button type="submit" disabled={picked === null}>
            {picked === "none" ? "Show new options" : "Confirm"}
          </button>
        </form>
      )}

      {result?.meaning && (
        <p className="tables-used">
          Meaning: <span className="meaning">{result.meaning}</span>
        </p>
      )}

      {result?.tables?.length > 0 && (
        <p className="tables-used">
          Tables used: {result.tables.join(", ")}
          {result.confidence && (
            <>
              {" · "}Confidence: <span className={`confidence ${result.confidence}`}>{result.confidence}</span>
            </>
          )}
          {" · "}LLM calls: {result.llm_calls}
        </p>
      )}

      {result?.warnings?.length > 0 && (
        <ul className="warnings">
          {result.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}

      {result?.attempts?.length > 0 && (
        <details className="attempts">
          <summary>All {result.attempts.length} attempts</summary>
          <ol>
            {result.attempts.map((a, i) => (
              <li key={i}>
                Round {a.round}: <strong>{a.status}</strong>
                {a.problems.length > 0 && <> ({a.problems.join(" ")})</>}
                {a.sql && <pre className="code">{a.sql}</pre>}
              </li>
            ))}
          </ol>
        </details>
      )}

      {result?.sql && (
        <>
          <h3>SQL</h3>
          <pre className="code">{result.sql}</pre>

          <h3>Answer</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  {result.columns.map((col) => (
                    <th key={col}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row, i) => (
                  <tr key={i}>
                    {row.map((value, j) => (
                      <td key={j}>{value === null ? "NULL" : String(value)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
