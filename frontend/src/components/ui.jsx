import { useState } from "react";

// Small shared pieces: icons, loading, alerts, SQL highlighting, copy button.

const ICONS = {
  search: <><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></>,
  doc: <><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" /><path d="M14 3v6h6" /><path d="M8 13h8M8 17h5" /></>,
  database: <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5" /><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></>,
  sparkle: <path d="M12 3l1.9 5.6L19.5 10l-5.6 1.9L12 17.5l-1.9-5.6L4.5 10l5.6-1.4z" />,
  code: <path d="m8 7-5 5 5 5M16 7l5 5-5 5" />,
  table: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 10h18M9 10v10" /></>,
  layers: <><path d="m12 3 9 5-9 5-9-5z" /><path d="m3 13 9 5 9-5" /></>,
  link: <path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" />,
  bulb: <><path d="M9 18h6M10 21h4" /><path d="M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0 0 12 3z" /></>,
  alert: <><circle cx="12" cy="12" r="9" /><path d="M12 8v5M12 16h.01" /></>,
  copy: <><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V5a2 2 0 0 1 2-2h8" /></>,
  check: <path d="m5 12 5 5 9-10" />,
  chevron: <path d="m9 6 6 6-6 6" />,
  sliders: <path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0M14 4v4M8 10v4M16 16v4" />,
  arrow: <path d="M5 12h14M13 6l6 6-6 6" />,
  upload: <path d="M12 16V4M7 9l5-5 5 5M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />,
  edit: <path d="M4 20h4L19 9a2.8 2.8 0 0 0-4-4L4 16zM13.5 6.5l4 4" />,
  left: <path d="m15 6-6 6 6 6" />,
  eye: <><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></>,
  sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
  moon: <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />,
  monitor: <><rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" /></>,
};

export function Icon({ name, size = 16 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {ICONS[name]}
    </svg>
  );
}

export function Loading({ text }) {
  return (
    <div className="card loading">
      <span className="spinner" />
      {text}
    </div>
  );
}

export function ErrorAlert({ message }) {
  return (
    <div className="alert alert-error" role="alert">
      <Icon name="alert" />
      <span>{message}</span>
    </div>
  );
}

export function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard blocked: nothing to do
    }
  }

  return (
    <button type="button" className="icon-btn" onClick={copy}>
      <Icon name={copied ? "check" : "copy"} size={13} />
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

const SQL_KEYWORDS = new Set(
  (
    "select from where join left right inner outer full cross on and or not in is null as with group by order " +
    "having limit offset distinct union all case when then else end asc desc count sum avg min max over " +
    "partition rank row_number dense_rank between like exists cast round coalesce strftime substr"
  ).split(" ")
);

// Light syntax colouring for SQL: keywords, strings, numbers.
export function SqlCode({ sql, className = "code" }) {
  const parts = sql.split(/('(?:[^']|'')*'|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]+\b)/g);
  return (
    <pre className={className}>
      {parts.map((part, i) => {
        if (part.startsWith("'")) return <span key={i} className="sql-str">{part}</span>;
        if (/^\d/.test(part)) return <span key={i} className="sql-num">{part}</span>;
        if (SQL_KEYWORDS.has(part.toLowerCase())) return <span key={i} className="sql-kw">{part}</span>;
        return part;
      })}
    </pre>
  );
}

// Previous / page x of y / next. Renders nothing when everything fits on one page.
export function Pagination({ page, pages, total, label = "rows", onPage }) {
  if (pages <= 1) return null;
  return (
    <div className="pagination">
      <span className="muted">
        Page {page} of {pages} · {total} {label}
      </span>
      <div className="pagination-buttons">
        <button type="button" className="icon-btn" disabled={page <= 1} onClick={() => onPage(page - 1)}>
          <Icon name="left" size={13} />
          Previous
        </button>
        <button type="button" className="icon-btn" disabled={page >= pages} onClick={() => onPage(page + 1)}>
          Next
          <Icon name="chevron" size={13} />
        </button>
      </div>
    </div>
  );
}

// A data table with numbers right-aligned and NULLs greyed out. firstIndex numbers the rows.
export function DataTable({ columns, rows, firstIndex = 1 }) {
  const numeric = columns.map(
    (_, j) => rows.length > 0 && rows.every((r) => r[j] === null || typeof r[j] === "number")
  );
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th className="row-index">#</th>
            {columns.map((col, j) => (
              <th key={col + j} className={numeric[j] ? "num" : ""}>
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              <td className="row-index">{firstIndex + i}</td>
              {row.map((value, j) => (
                <td key={j} className={value === null ? "null" : numeric[j] ? "num" : ""}>
                  {value === null ? "NULL" : String(value)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
