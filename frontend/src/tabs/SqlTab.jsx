import { useState } from "react";
import { postForm } from "../api.js";

// Version-2: ask a question; the LLM writes SQL, the backend runs it, we show the table.
export default function SqlTab() {
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("");
  const [result, setResult] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    setStatus("Loading...");
    setResult(null);

    try {
      const data = await postForm("/api/sql/query", { query_text: question });
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

      {result?.tables?.length > 0 && (
        <p className="tables-used">
          Tables used: {result.tables.join(", ")}
        </p>
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
