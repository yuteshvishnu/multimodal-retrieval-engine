import { useEffect, useRef, useState } from "react";
import { getJson, postForm } from "../api.js";
import { DataTable, ErrorAlert, Icon, Loading, Pagination } from "../components/ui.jsx";

// a description is either confirmed by the user or not
const STATUS = {
  reviewed: { label: "Reviewed", className: "high" },
  ai: { label: "Not reviewed", className: "medium" },
  basic: { label: "Not reviewed", className: "medium" },
};

// Text-to-SQL, Tables view: every table Rexie knows, what it understands about each one, a way to
// correct that understanding, and adding a new .sqlite database.
export default function TablesTab() {
  const [list, setList] = useState(null); // {items, page, pages, total}
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState(""); // search actually applied
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState(null); // {text, warning}
  const fileInput = useRef(null);

  async function load(nextPage = page, nextQuery = query) {
    setLoading(true);
    setError("");
    try {
      const data = await getJson("/api/sql/tables", { page: nextPage, search: nextQuery });
      setList(data);
      setPage(data.page);
    } catch (err) {
      setError(`Could not load tables: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load(1, "");
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  function handleSearch(event) {
    event.preventDefault();
    setQuery(search);
    load(1, search);
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    event.target.value = ""; // so the same file can be picked again
    if (!file) return;
    setUploading(true);
    setNotice(null);
    setError("");
    try {
      const data = await postForm("/api/sql/tables/upload", { file });
      setNotice({
        text: `Added ${data.database} with ${data.tables.length} table${data.tables.length === 1 ? "" : "s"}: ${data.tables.join(", ")}.`,
        warning: data.warning,
      });
      setSearch("");
      setQuery("");
      load(1, "");
    } catch (err) {
      setError(`Could not add the database: ${err.message}`);
    } finally {
      setUploading(false);
    }
  }

  function replaceTable(updated) {
    setList((current) => ({
      ...current,
      items: current.items.map((t) =>
        t.database === updated.database && t.name === updated.name ? updated : t
      ),
    }));
  }

  return (
    <section>
      <div className="page-header page-header-row">
        <div>
          <h1>Know your tables.</h1>
          <p>Everything Rexie understands about your data. Spot something wrong? Tell Rexie and it rewrites it.</p>
        </div>
        <button
          type="button"
          className="btn btn-primary btn-sm"
          onClick={() => fileInput.current?.click()}
          disabled={uploading}
        >
          <Icon name="upload" size={14} />
          Add database
        </button>
        <input ref={fileInput} type="file" accept=".sqlite" hidden onChange={handleUpload} />
      </div>

      <form onSubmit={handleSearch} className="card composer">
        <div className="composer-row">
          <label className="search-field">
            <Icon name="search" size={18} />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search tables by name or meaning, e.g. revenue"
              aria-label="Search tables"
            />
          </label>
          <button type="submit" className="btn btn-secondary">
            Search
          </button>
        </div>
      </form>

      {uploading && <Loading text="Adding the database and writing a description for each table..." />}
      {notice && (
        <div className="alert alert-success" role="status">
          <Icon name="check" />
          <span>
            {notice.text}
            {notice.warning && <> {notice.warning}</>}
          </span>
        </div>
      )}
      {error && <ErrorAlert message={error} />}
      {loading && !list && <Loading text="Loading tables..." />}

      {list && (
        <>
          <p className="muted list-count">
            {list.total} table{list.total === 1 ? "" : "s"}
            {query && <> matching “{query}”</>}
          </p>

          {list.items.length === 0 && (
            <div className="card empty">
              <div className="empty-icon">
                <Icon name="table" size={22} />
              </div>
              <h3>No tables found</h3>
              <p>{query ? "Try another search." : "Add a .sqlite database to get started."}</p>
            </div>
          )}

          {list.items.map((table) => (
            <TableItem key={`${table.database}.${table.name}`} table={table} onSaved={replaceTable} />
          ))}

          <Pagination page={list.page} pages={list.pages} total={list.total} label="tables" onPage={(p) => load(p)} />
        </>
      )}
    </section>
  );
}

function TableItem({ table, onSaved }) {
  const [open, setOpen] = useState(false);
  const status = STATUS[table.status];

  return (
    <div className={`card table-item ${open ? "open" : ""}`}>
      <button type="button" className="table-item-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="table-item-icon">
          <Icon name="table" size={16} />
        </span>
        <span className="table-item-title">
          <span className="table-item-name">{table.name}</span>
          <span className="muted">
            {table.database} · {table.row_count.toLocaleString()} rows · {table.columns.length} columns
          </span>
        </span>
        <span className={`badge ${status.className}`}>{status.label}</span>
        <span className="chevron">
          <Icon name="chevron" />
        </span>
      </button>

      {open && (
        <div className="table-item-body">
          <Review table={table} onSaved={onSaved} />

          <div className="label">Columns</div>
          <div className="option-choices column-list">
            {table.columns.map((c) => (
              <span key={c.name} className="tag">
                {c.name}
                {c.type && <span className="muted"> {c.type.toLowerCase()}</span>}
              </span>
            ))}
          </div>

          <RowPreview table={table} />
        </div>
      )}
    </div>
  );
}

// Shows the description; "Needs a fix" -> say what's wrong -> Rexie rewrites (1 LLM call)
// -> save it, or say what's still wrong and try again, as many times as needed.
function Review({ table, onSaved }) {
  const [mode, setMode] = useState("view"); // view | editing | rewriting | draft | saving
  const [comment, setComment] = useState("");
  const [corrections, setCorrections] = useState([]); // this session's corrections
  const [draft, setDraft] = useState(null); // {description, problems}
  const [error, setError] = useState("");
  const tries = corrections.length;

  const url = `/api/sql/tables/${encodeURIComponent(table.database)}/${encodeURIComponent(table.name)}`;
  const allFeedback = [...table.feedback, ...corrections];

  async function rewrite(event) {
    event.preventDefault();
    const next = [...corrections, comment.trim()];
    setMode("rewriting");
    setError("");
    try {
      const data = await postForm(`${url}/rewrite`, {
        current: draft?.description || table.description,
        feedback: JSON.stringify([...table.feedback, ...next]),
      });
      setCorrections(next);
      setDraft(data);
      setComment("");
      setMode("draft");
    } catch (err) {
      setError(err.message);
      setMode("editing");
    }
  }

  async function save(description) {
    setMode("saving");
    setError("");
    try {
      const data = await postForm(`${url}/save`, { description, feedback: JSON.stringify(allFeedback) });
      setCorrections([]);
      setDraft(null);
      setMode("view");
      onSaved(data.table);
    } catch (err) {
      setError(err.message);
      setMode(draft ? "draft" : "view");
    }
  }

  function cancel() {
    setCorrections([]);
    setDraft(null);
    setComment("");
    setError("");
    setMode("view");
  }

  return (
    <div className="review">
      <div className="label">What Rexie understands</div>

      {mode !== "draft" && mode !== "saving" && <p className="description">{draft?.description || table.description}</p>}

      {(mode === "draft" || (mode === "saving" && draft)) && (
        <div className="draft">
          <div className="draft-head">
            <span className="badge info">New version · try {tries}</span>
            <span className="muted">Not saved yet</span>
          </div>
          <p className="description">{draft.description}</p>
          {draft.problems.length > 0 && (
            <p className="problems">Heads up: this version {draft.problems.join("; ")}.</p>
          )}
        </div>
      )}

      {allFeedback.length > 0 && (
        <details className="feedback-list">
          <summary className="muted">Your corrections ({allFeedback.length})</summary>
          <ul>
            {allFeedback.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
        </details>
      )}

      {error && <ErrorAlert message={error} />}

      {mode === "view" && (
        <div className="review-actions">
          {table.status !== "reviewed" && (
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => save(table.description)}>
              <Icon name="check" size={14} />
              Looks right
            </button>
          )}
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => setMode("editing")}>
            <Icon name="edit" size={14} />
            Needs a fix
          </button>
        </div>
      )}

      {mode === "draft" && (
        <div className="review-actions">
          <button type="button" className="btn btn-primary btn-sm" onClick={() => save(draft.description)}>
            <Icon name="check" size={14} />
            Save this version
          </button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => setMode("editing")}>
            <Icon name="edit" size={14} />
            Still wrong
          </button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={cancel}>
            Discard
          </button>
        </div>
      )}

      {mode === "editing" && (
        <form onSubmit={rewrite} className="review-form">
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="What's wrong? e.g. cancelled orders still count as orders an employee handled"
            rows={3}
            autoFocus
            required
          />
          <div className="review-actions">
            <button type="submit" className="btn btn-primary btn-sm" disabled={!comment.trim()}>
              <Icon name="sparkle" size={14} />
              Rewrite
            </button>
            <button type="button" className="btn btn-secondary btn-sm" onClick={cancel}>
              Cancel
            </button>
          </div>
        </form>
      )}

      {mode === "rewriting" && <Loading text="Rexie is rewriting the description..." />}
      {mode === "saving" && <Loading text="Saving and updating table search..." />}
    </div>
  );
}

// 10 rows at a time, loaded only when asked for.
function RowPreview({ table }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function load(page) {
    setLoading(true);
    setError("");
    try {
      setData(
        await getJson(
          `/api/sql/tables/${encodeURIComponent(table.database)}/${encodeURIComponent(table.name)}/rows`,
          { page }
        )
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  if (!data) {
    return (
      <div className="preview-start">
        <button type="button" className="btn btn-secondary btn-sm" onClick={() => load(1)} disabled={loading}>
          <Icon name="eye" size={14} />
          {loading ? "Loading..." : "Preview rows"}
        </button>
        {error && <ErrorAlert message={error} />}
      </div>
    );
  }

  return (
    <div className="preview">
      <div className="label">Rows</div>
      <div className={`preview-table ${loading ? "is-loading" : ""}`}>
        <DataTable columns={data.columns} rows={data.rows} firstIndex={(data.page - 1) * 10 + 1} />
        <Pagination page={data.page} pages={data.pages} total={data.total} onPage={load} />
      </div>
      {error && <ErrorAlert message={error} />}
    </div>
  );
}
