import { useEffect, useRef, useState } from "react";
import { getJson, postForm } from "../api.js";
import { ErrorAlert, Icon, Loading, Pagination } from "../components/ui.jsx";

// Dashboard -> Issues (Version-11): Scout's failures, and the Failure Analyst, an AI agent that
// investigates them with read-only tools and suggests a fix the developer approves or rejects.

const SOURCES = [
  { id: "", label: "All" },
  { id: "eval", label: "Eval" },
  { id: "user_report", label: "User report" },
  { id: "sql_error", label: "SQL error" },
];
const SOURCE_LABEL = { eval: "Eval", user_report: "User report", sql_error: "SQL error" };
const SOURCE_HELP = {
  eval: "A test question answered wrong (the correct answer is known)",
  user_report: "A user reported the answer as wrong",
  sql_error: "SQLite rejected every SQL try",
};

const STATUS = {
  new: { label: "Not investigated", style: "info" },
  investigating: { label: "Investigating…", style: "medium" },
  report_ready: { label: "Report ready", style: "high" },
  approved: { label: "Approved", style: "high" },
  rejected: { label: "Rejected", style: "low" },
  failed: { label: "Agent stopped", style: "low" },
};

// The kind of fix the agent suggests: it decides what Approve does.
const FIX = {
  error_book: {
    label: "Error book",
    help: "A new known-error entry: a rule for the SQL prompt and a hint for when SQLite shows that error",
    button: "Approve: add to error book",
    approve: (r) =>
      `Adds "${r.error_book_entry?.id || "the entry"}" to the error book (sql_errors.json). From the next question, ` +
      "its rule goes into the SQL prompt, and its hint is given to the model when SQLite shows this error.",
  },
  table_description: {
    label: "Table description",
    help: "A note added to a table's description, e.g. how a column's values are spelled",
    button: "Approve: add note to description",
    approve: (r) =>
      `Adds the note to the description of ${r.description_note?.table || "the table"} and re-embeds it. ` +
      "Scout reads it from the next question.",
  },
  prompt: {
    label: "Prompt change",
    help: "A rule for the steps or SQL prompt; you make the change",
    button: "Approve: mark as to-do",
    approve: () => "Changes nothing in Scout. It marks the issue Approved as a to-do: you change the steps or SQL prompt yourself.",
  },
  code: {
    label: "Code change",
    help: "A bug in Scout itself; you make the change",
    button: "Approve: mark as to-do",
    approve: () => "Changes nothing in Scout. It marks the issue Approved as a to-do: you change the code yourself.",
  },
  none: {
    label: "No fix",
    help: "Scout was right, or the test question is wrong",
    button: "Approve: close issue",
    approve: () => "Changes nothing. It closes the issue: the agent found nothing to fix.",
  },
};
// Where it broke, from the agent's report.
const CAUSE_LABEL = {
  meaning: "Meaning",
  steps: "Steps",
  sql: "SQL",
  data: "Data",
  description: "Table description",
  retrieval: "Tables picked",
  other: "Other",
};

const POLL_MS = 1500;

function formatTime(iso) {
  return iso ? new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "";
}

function formatCost(value) {
  if (!value) return "$0";
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(3)}`;
}

// Which issues a view shows: the Dashboard shows what users hit (reports, SQL errors);
// Evaluation shows the eval's wrong answers. One source -> no source tabs.
const DASHBOARD_SCOPE = ["user_report", "sql_error"];

export default function IssuesView({
  active,
  scope = DASHBOARD_SCOPE,
  run = "", // Version-15: only this evaluation run's issues
  header = null, // shown above the agent intro (Evaluation: the run picker)
  title = "Find out why it failed.",
  intro = "Every failure users hit in Scout (their reports and SQL that SQLite keeps rejecting), investigated by an AI agent that explains what went wrong and suggests a fix.",
  empty = <>Issues appear when a user clicks <strong>Report issue</strong>, or when SQLite keeps rejecting Scout's SQL. Eval failures are under <strong>Evaluation</strong>.</>,
}) {
  const sources = SOURCES.filter((s) => !s.id || scope.includes(s.id));
  const [list, setList] = useState(null);
  const [source, setSource] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function load(page = list?.page || 1, nextSource = source, nextStatus = status) {
    try {
      setList(await getJson("/api/issues", { page, source: nextSource, status: nextStatus, scope: scope.join(","), run }));
    } catch (err) {
      setError(`Could not load the issues: ${err.message}`);
    }
  }

  useEffect(() => {
    if (active) load(1);
  }, [active, run]); // eslint-disable-line react-hooks/exhaustive-deps

  // while the agent works on anything shown, keep the list's statuses fresh
  const running = list?.items.some((i) => i.status === "investigating");
  useEffect(() => {
    if (!active || !running) return undefined;
    const timer = setInterval(() => load(), POLL_MS * 2);
    return () => clearInterval(timer);
  }, [active, running]); // eslint-disable-line react-hooks/exhaustive-deps

  async function investigateNew() {
    setBusy(true);
    setError("");
    try {
      const { started } = await postForm("/api/issues/investigate-new", { scope: scope.join(","), run });
      if (started.length === 0) setError("No new issues to investigate.");
      await load();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const agent = list?.agent;
  const newCount = list?.by_status?.new || 0;

  return (
    <>
      <div className="page-header page-header-row">
        <div>
          <h1>{title}</h1>
          <p>{intro}</p>
        </div>
        <button type="button" className="btn btn-secondary btn-sm" onClick={() => load()}>
          <Icon name="refresh" size={14} />
          Refresh
        </button>
      </div>

      {header}
      <AgentIntro agent={agent} />

      {error && <ErrorAlert message={error} />}
      {!list && <Loading text="Loading the issues..." />}

      {list && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              <Icon name="flag" />
              Issues
            </h3>
            <span className="tag">{list.total}</span>
          </div>

          <div className="card-body issue-filters">
            {sources.length > 2 && (
            <div className="tabs" role="group" aria-label="Source">
              {sources.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  className={s.id === source ? "tab active" : "tab"}
                  aria-pressed={s.id === source}
                  onClick={() => {
                    setSource(s.id);
                    load(1, s.id, status);
                  }}
                >
                  {s.label}
                  {s.id && <span className="tab-count">{list.by_source?.[s.id] || 0}</span>}
                </button>
              ))}
            </div>
            )}
            <select
              className="select"
              value={status}
              aria-label="Status"
              onChange={(e) => {
                setStatus(e.target.value);
                load(1, source, e.target.value);
              }}
            >
              <option value="">Any status</option>
              {Object.entries(STATUS).map(([id, s]) => (
                <option key={id} value={id}>
                  {s.label} ({list.by_status?.[id] || 0})
                </option>
              ))}
            </select>
            <button
              type="button"
              className="btn btn-primary btn-sm issue-batch"
              onClick={investigateNew}
              disabled={busy || newCount === 0 || !agent?.ready}
              title={agent?.ready ? "" : "Set the LLM API key in .env to run the agent"}
            >
              <Icon name="agent" size={14} />
              Investigate all new{newCount ? ` (${Math.min(newCount, agent?.batch || 5)})` : ""}
            </button>
          </div>

          {list.items.length === 0 && (
            <div className="empty">
              <h3>{source || status ? "No issues match" : "No issues yet"}</h3>
              <p>{empty}</p>
            </div>
          )}

          {list.items.map((issue) => (
            <IssueRow key={issue.id} issue={issue} ready={agent?.ready} onChange={() => load()} />
          ))}
          <Pagination page={list.page} pages={list.pages} total={list.total} label="issues" onPage={(p) => load(p)} always />
        </div>
      )}
    </>
  );
}

// What the agent is, how the logs feed it, how it reasons, and what stays with the developer.
function AgentIntro({ agent }) {
  const steps = [
    { icon: "layers", title: "Logged", text: "Every failure is saved with what Scout did: wrong eval answers, users' reports, SQL errors." },
    { icon: "search", title: "Investigates", text: `The agent decides what to check, calls a read-only tool, reads the result and picks the next: up to ${agent?.max_tool_calls || 8} calls.` },
    { icon: "bulb", title: "Reports", text: "What happened, why, and a fix. Our code (not the LLM) tests the corrected SQL." },
    { icon: "shield", title: "You decide", text: "Each fix has a type, and the type decides what Approve does. Nothing changes without you." },
  ];
  return (
    <div className="card agent-intro">
      <div className="agent-intro-head">
        <span className="agent-avatar" aria-hidden="true">
          <Icon name="agent" size={26} />
        </span>
        <div>
          <h2>
            Failure Analyst <span className="agent-badge">AI agent</span>
          </h2>
          <p>
            Most of Scout is a fixed workflow: our code decides every step. The Failure Analyst is different. It's an{" "}
            <strong>agent</strong>: given a failure from the logs, the LLM decides for itself what to look at, using
            tools that can only read, until it can explain the cause.
          </p>
        </div>
      </div>
      <ol className="agent-flow">
        {steps.map((s, i) => (
          <li key={s.title}>
            <span className="agent-flow-icon">
              <Icon name={s.icon} size={16} />
            </span>
            <div>
              <strong>
                {i + 1}. {s.title}
              </strong>
              <p>{s.text}</p>
            </div>
          </li>
        ))}
      </ol>
      <div className="fix-legend">
        <span className="muted small">Fix types (what Approve does):</span>
        {Object.entries(FIX).map(([id, f]) => (
          <span key={id} className={`fix-tag ${id}`} title={f.help}>
            {f.label}
          </span>
        ))}
      </div>
      <p className="agent-tools muted">
        <Icon name="code" size={13} />
        <span>
          Tools: list tables · describe a table · column values · run a test query · read the error book. Model:{" "}
          {agent?.model || "…"}. Queries pass the same safety, size and time checks as Scout's own.
        </span>
      </p>
    </div>
  );
}

// One issue: a summary line that opens to the failure, the agent's steps (live) and its report.
function IssueRow({ issue, ready, onChange }) {
  const [open, setOpen] = useState(false);
  const status = STATUS[issue.status] || STATUS.new;

  return (
    <div className={`dash-row ${open ? "open" : ""}`}>
      <button type="button" className="dash-row-head issue-row-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className={`source-tag ${issue.source}`} title={SOURCE_HELP[issue.source]}>
          {SOURCE_LABEL[issue.source]}
        </span>
        {issue.report && FIX[issue.report.fix_type] && (
          <span className={`fix-tag ${issue.report.fix_type}`} title={`Suggested fix: ${FIX[issue.report.fix_type].help}`}>
            {FIX[issue.report.fix_type].label}
          </span>
        )}
        <span className="dash-question">{issue.question || "(no question)"}</span>
        <span className={`badge ${status.style}`}>{status.label}</span>
        <span className="dash-time muted">{formatTime(issue.updated_at)}</span>
        <span className="chevron">
          <Icon name="chevron" />
        </span>
      </button>
      {open && <IssueDetail id={issue.id} ready={ready} onChange={onChange} />}
    </div>
  );
}

function IssueDetail({ id, ready, onChange }) {
  const [issue, setIssue] = useState(null);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const lastStatus = useRef(null);

  async function load() {
    try {
      const data = await getJson(`/api/issues/${id}`);
      setIssue(data);
      if (lastStatus.current && lastStatus.current !== data.status) onChange();
      lastStatus.current = data.status;
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    load();
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  // live: re-read the steps while the agent works
  const running = issue?.status === "investigating";
  useEffect(() => {
    if (!running) return undefined;
    const timer = setInterval(load, POLL_MS);
    return () => clearInterval(timer);
  }, [running]); // eslint-disable-line react-hooks/exhaustive-deps

  async function act(path, fields = {}) {
    setBusy(true);
    setError("");
    try {
      const result = await postForm(`/api/issues/${id}/${path}`, fields);
      if (result.status) setIssue({ ...issue, ...result });
      else await load();
      onChange();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  if (!issue) return error ? <ErrorAlert message={error} /> : <div className="issue-detail muted">Loading…</div>;

  const canInvestigate = ["new", "failed", "rejected"].includes(issue.status);
  const fix = issue.report && FIX[issue.report.fix_type];
  // Version-18: a fix that changes Scout is never applied from a BIRD question (it would teach Scout the test)
  const benchmarkLocked = issue.details?.suite === "bird_minidev" && ["error_book", "table_description"].includes(issue.report?.fix_type);
  return (
    <div className="issue-detail">
      <FailureSection issue={issue} />

      <section className="issue-section">
        <div className="issue-section-head">
          <h4>
            <Icon name="agent" size={15} />
            How the agent worked
          </h4>
          {issue.agent_calls > 0 && (
            <span className="muted">
              {issue.agent_calls} LLM call{issue.agent_calls === 1 ? "" : "s"} · {formatCost(issue.agent_cost)}
            </span>
          )}
        </div>
        {issue.steps.length === 0 && !running && (
          <p className="muted">Not investigated yet. The agent runs only when you ask it to.</p>
        )}
        {issue.steps.length > 0 && <AgentTrace steps={issue.steps} running={running} />}
        {running && issue.steps.length === 0 && <p className="muted agent-waiting"><span className="spinner" /> Waiting for its turn…</p>}
        {canInvestigate && (
          <button type="button" className="btn btn-primary btn-sm" onClick={() => act("investigate")} disabled={busy || !ready}>
            <Icon name="agent" size={14} />
            {issue.steps.length ? "Investigate again" : "Investigate"}
          </button>
        )}
      </section>

      {issue.report && <ReportSection report={issue.report} />}

      {issue.status === "report_ready" && (
        <section className="issue-section issue-decision">
          <h4>Your decision</h4>
          {fix && benchmarkLocked && (
            <div className="approve-effect">
              <span className={`fix-tag ${issue.report.fix_type}`}>{fix.label}</span>
              <p>
                <strong>Can't be approved: </strong>
                this question is from BIRD, and the fix was found by looking at BIRD's correct answer. Applying it
                would teach Scout the test and inflate its BIRD score. Use the report to see what went wrong, then fix
                that in general.
              </p>
              <p className="muted small">Reject keeps your note so you remember why.</p>
            </div>
          )}
          {fix && !benchmarkLocked && (
            <div className="approve-effect">
              <span className={`fix-tag ${issue.report.fix_type}`}>{fix.label}</span>
              <p>
                <strong>When you approve: </strong>
                {fix.approve(issue.report)}
              </p>
              <p className="muted small">Reject changes nothing; it keeps your note so you remember why.</p>
            </div>
          )}
          <textarea value={note} onChange={(e) => setNote(e.target.value)} placeholder="Note (optional)" rows={2} />
          <div className="review-actions">
            <button type="button" className="btn btn-primary btn-sm" onClick={() => act("approve", { note })} disabled={busy || benchmarkLocked}>
              <Icon name="check" size={14} />
              {fix?.button || "Approve"}
            </button>
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => act("reject", { note })} disabled={busy}>
              <Icon name="x" size={14} />
              Reject
            </button>
          </div>
        </section>
      )}

      {(issue.status === "approved" || issue.status === "rejected") && (
        <section className="issue-section">
          <h4>Decision</h4>
          <p>
            <strong>{issue.status === "approved" ? "Approved" : "Rejected"}</strong>
            {issue.report?.applied && issue.status === "approved" && <> · {issue.report.applied}</>}
          </p>
          {issue.note && <p className="muted">“{issue.note}”</p>}
        </section>
      )}

      {error && <ErrorAlert message={error} />}
    </div>
  );
}

// What failed, as it was logged.
function FailureSection({ issue }) {
  const d = issue.details || {};
  return (
    <section className="issue-section">
      <h4>
        <Icon name="alert" size={15} />
        The failure <span className="muted">· {SOURCE_HELP[issue.source]}</span>
      </h4>
      {d.comment && <p className="dash-issue-comment">“{d.comment}”</p>}
      {d.eval_id && (
        <p>
          <span className="muted">Eval question </span>#{d.eval_id} ({d.difficulty})
          {d.trap && <span className="muted"> · trap: {d.trap}</span>}
        </p>
      )}
      {d.meaning && (
        <p>
          <span className="muted">Meaning: </span>
          {d.meaning.split("\n")[0]}
        </p>
      )}
      {d.steps?.length > 0 && (
        <ol className="steps-list">
          {d.steps.map((s, i) => (
            <li key={i}>{s.replace(/\[\[|\]\]/g, "")}</li>
          ))}
        </ol>
      )}
      {d.sql && (
        <>
          <div className="label">Scout's SQL</div>
          <pre className="code dash-issue-sql">{d.sql}</pre>
        </>
      )}
      {d.attempts?.map((a, i) => (
        <div key={i}>
          <div className="label">Try {i + 1} (round {a.round})</div>
          <pre className="code dash-issue-sql">{a.sql || "(no SQL)"}</pre>
          <p className="problems">{a.problems.join(" ")}</p>
        </div>
      ))}
      {d.error && !d.attempts && <p className="problems">{d.error}</p>}
      {d.gold_sql && (
        <>
          <div className="label">Correct SQL</div>
          <pre className="code dash-issue-sql">{d.gold_sql}</pre>
        </>
      )}
    </section>
  );
}

const STEP_ICON = { start: "doc", reasoning: "bulb", tool: "code", report: "edit", test: "check", test_failed: "x", error: "alert" };

// The agent's investigation, step by step: what it thought, which tool it called, what it saw.
function AgentTrace({ steps, running }) {
  return (
    <ol className="agent-trace">
      {steps.map((s, i) => (
        <li key={i} className={`trace-step ${s.kind}`}>
          <span className="trace-dot">
            <Icon name={STEP_ICON[s.kind] || "chevron"} size={13} />
          </span>
          <div className="trace-body">
            <TraceStep step={s} />
          </div>
        </li>
      ))}
      {running && (
        <li className="trace-step pending">
          <span className="trace-dot">
            <span className="spinner" />
          </span>
          <div className="trace-body muted">Deciding the next step…</div>
        </li>
      )}
    </ol>
  );
}

function TraceStep({ step }) {
  const [open, setOpen] = useState(["reasoning", "test", "test_failed", "error"].includes(step.kind));
  if (!step.detail) return <strong>{step.title}</strong>;
  return (
    <>
      <button type="button" className="trace-title" onClick={() => setOpen(!open)} aria-expanded={open}>
        {step.title}
        <span className={`chevron ${open ? "up" : ""}`}>
          <Icon name="chevron" size={12} />
        </span>
      </button>
      {open && (step.kind === "reasoning" ? <p className="trace-thought">{step.detail}</p> : <pre className="trace-detail">{step.detail}</pre>)}
    </>
  );
}

// The agent's report, and the result of the test our code ran on its fix.
function ReportSection({ report }) {
  const test = report.test || {};
  const verdict = { true: ["high", "Fix tested: passed"], false: ["low", "Fix tested: failed"] }[String(test.passed)] || [
    "info",
    "Needs a person to judge",
  ];
  return (
    <section className="issue-section report">
      <div className="issue-section-head">
        <h4>
          <Icon name="bulb" size={15} />
          The agent's report
        </h4>
        <span className="option-choices">
          <span className="tag" title="Where it broke">Broke at: {CAUSE_LABEL[report.root_cause] || report.root_cause}</span>
          <span className={`fix-tag ${report.fix_type}`} title={FIX[report.fix_type]?.help}>
            Fix: {FIX[report.fix_type]?.label || report.fix_type}
          </span>
          <span className="tag">{report.confidence} confidence</span>
        </span>
      </div>
      <dl className="report-grid">
        <dt>What happened</dt>
        <dd>{report.what_happened}</dd>
        <dt>Why</dt>
        <dd>{report.why}</dd>
        <dt>Suggested fix</dt>
        <dd>{report.fix}</dd>
      </dl>
      {report.error_book_entry && report.fix_type === "error_book" && (
        <>
          <div className="label">Error-book entry</div>
          <pre className="code dash-issue-sql">{JSON.stringify(report.error_book_entry, null, 2)}</pre>
        </>
      )}
      {report.description_note && report.fix_type === "table_description" && (
        <p>
          <span className="muted">Note for </span>
          <strong>{report.description_note.table}</strong>: {report.description_note.note}
        </p>
      )}
      {report.fixed_sql && (
        <>
          <div className="label">Corrected SQL</div>
          <pre className="code dash-issue-sql">{report.fixed_sql}</pre>
        </>
      )}
      <div className="report-test">
        <span className={`badge ${verdict[0]}`}>{verdict[1]}</span>
        <ul>
          {(test.checks || []).map((c, i) => (
            <li key={i} className={c.ok ? "ok" : "bad"}>
              <Icon name={c.ok ? "check" : "x"} size={13} />
              {c.text}
            </li>
          ))}
        </ul>
        {test.rows?.length > 0 && (
          <p className="muted small">
            First rows: {test.rows.slice(0, 3).map((r) => r.join(", ")).join(" · ")}
          </p>
        )}
      </div>
    </section>
  );
}
