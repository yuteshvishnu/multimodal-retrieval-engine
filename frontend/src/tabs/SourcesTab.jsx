import { useEffect, useRef, useState } from "react";
import { deleteJson, getJson, postForm } from "../api.js";
import { ErrorAlert, Icon, Loading, Pagination } from "../components/ui.jsx";

const ACCEPT = ".txt,.md,.pdf,.docx";

function formatSize(bytes) {
  if (bytes == null) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

// Rexie Scribe, Sources view: every document Scribe answers from, the passages it was
// split into, and adding new documents. onAdded tells the Ask view to refresh its picker.
export default function SourcesTab({ onAdded }) {
  const [list, setList] = useState(null); // {items, page, pages, total}
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState("");
  const [warning, setWarning] = useState(""); // e.g. a document with instruction-like text
  const fileInput = useRef(null);

  async function load(page = 1, nextQuery = query) {
    setLoading(true);
    setError("");
    try {
      setList(await getJson("/api/rag/documents", { page, search: nextQuery }));
    } catch (err) {
      setError(`Could not load documents: ${err.message}`);
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
    const files = [...(event.target.files || [])];
    event.target.value = ""; // so the same file can be picked again
    if (files.length === 0) return;
    setUploading(true);
    setNotice("");
    setError("");

    const added = [];
    const failed = [];
    const warnings = [];
    for (const file of files) {
      try {
        const data = await postForm("/api/rag/documents/upload", { file });
        added.push(`${data.document.name} (${data.document.chunks} passages)`);
        if (data.document.warning) warnings.push(`${data.document.name}: ${data.document.warning}`);
      } catch (err) {
        failed.push(`${file.name}: ${err.message}`);
      }
    }

    if (added.length) {
      setNotice(`Added ${added.join(", ")}. You can ask about ${added.length === 1 ? "it" : "them"} right away.`);
      onAdded?.();
    }
    if (failed.length) setError(`Could not add ${failed.join("; ")}`);
    setWarning(warnings.join(" "));
    setUploading(false);
    setSearch("");
    setQuery("");
    load(1, "");
  }

  function handleDeleted(name) {
    setNotice(`Removed ${name}. Scribe no longer answers from it.`);
    setWarning("");
    onAdded?.(); // refresh the Ask view's document picker
    load(list?.page || 1);
  }

  return (
    <section>
      <div className="page-header page-header-row">
        <div>
          <h1>Know your sources.</h1>
          <p>Everything Scribe answers from. Add PDF, Word, Markdown or text files and ask about them right away.</p>
        </div>
        <button
          type="button"
          className="btn btn-primary btn-sm"
          onClick={() => fileInput.current?.click()}
          disabled={uploading}
        >
          <Icon name="upload" size={14} />
          Add documents
        </button>
        <input ref={fileInput} type="file" accept={ACCEPT} multiple hidden onChange={handleUpload} />
      </div>

      <form onSubmit={handleSearch} className="card composer">
        <div className="composer-row">
          <label className="search-field">
            <Icon name="search" size={18} />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search documents by name or content"
              aria-label="Search documents"
            />
          </label>
          <button type="submit" className="btn btn-secondary">
            Search
          </button>
        </div>
      </form>

      {uploading && <Loading text="Reading, splitting and indexing your documents..." />}
      {notice && (
        <div className="alert alert-success" role="status">
          <Icon name="check" />
          <span>{notice}</span>
        </div>
      )}
      {warning && (
        <div className="alert alert-warning" role="status">
          <Icon name="alert" />
          <span>{warning}</span>
        </div>
      )}
      {error && <ErrorAlert message={error} />}
      {loading && !list && <Loading text="Loading documents..." />}

      {list && (
        <>
          <p className="muted list-count">
            {list.total} document{list.total === 1 ? "" : "s"}
            {query && <> matching “{query}”</>}
          </p>

          {list.items.length === 0 && (
            <div className="card empty">
              <div className="empty-icon">
                <Icon name="doc" size={22} />
              </div>
              <h3>No documents found</h3>
              <p>{query ? "Try another search." : "Add a PDF, Word, Markdown or text file to get started."}</p>
            </div>
          )}

          {list.items.map((doc) => (
            <DocumentItem key={doc.name} doc={doc} onDeleted={handleDeleted} />
          ))}

          <Pagination page={list.page} pages={list.pages} total={list.total} label="documents" onPage={(p) => load(p)} />
        </>
      )}
    </section>
  );
}

function DocumentItem({ doc, onDeleted }) {
  const [open, setOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState("");

  async function remove() {
    if (!window.confirm(`Remove “${doc.name}”? Its file and passages are deleted, and Scribe stops answering from it.`)) {
      return;
    }
    setDeleting(true);
    setError("");
    try {
      await deleteJson(`/api/rag/documents/${encodeURIComponent(doc.name)}`);
      onDeleted(doc.name);
    } catch (err) {
      setError(err.message);
      setDeleting(false);
    }
  }

  return (
    <div className={`card table-item ${open ? "open" : ""}`}>
      <button type="button" className="table-item-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="table-item-icon">
          <Icon name="doc" size={16} />
        </span>
        <span className="table-item-title">
          <span className="table-item-name">{doc.name}</span>
          <span className="muted">
            {doc.file} · {doc.chunks} passage{doc.chunks === 1 ? "" : "s"}
            {doc.size != null && <> · {formatSize(doc.size)}</>}
          </span>
        </span>
        <span className="badge info">{doc.type}</span>
        <span className="chevron">
          <Icon name="chevron" />
        </span>
      </button>

      {open && (
        <div className="table-item-body">
          <Passages name={doc.name} />
          {error && <ErrorAlert message={error} />}
          <div className="danger-zone">
            <button type="button" className="btn btn-danger btn-sm" onClick={remove} disabled={deleting}>
              <Icon name="trash" size={14} />
              {deleting ? "Removing..." : "Remove document"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// The passages a document was split into (what Scribe searches and cites), 10 at a time.
function Passages({ name }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function load(page) {
    setLoading(true);
    setError("");
    try {
      setData(await getJson(`/api/rag/documents/${encodeURIComponent(name)}/chunks`, { page }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load(1);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (error) return <ErrorAlert message={error} />;
  if (!data) return <Loading text="Loading passages..." />;

  return (
    <>
      <div className="label">What Scribe searches</div>
      <div className={`passages ${loading ? "is-loading" : ""}`}>
        {data.items.map((p, i) => (
          <div key={p.id} className="passage">
            <span className="citation-id">{(data.page - 1) * 10 + i + 1}</span>
            <div>
              <div className="citation-meta">
                <span className="citation-chunk">{p.id}</span>
              </div>
              <p className="citation-snippet">{p.text}</p>
            </div>
          </div>
        ))}
        <Pagination page={data.page} pages={data.pages} total={data.total} label="passages" onPage={load} />
      </div>
    </>
  );
}
