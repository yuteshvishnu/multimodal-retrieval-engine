import { useEffect, useRef, useState } from "react";
import { getJson, postForm } from "../api.js";
import { AccuracyRing, ErrorAlert, Icon, Loading, Pagination, SqlCode } from "../components/ui.jsx";
import CompareView from "./CompareView.jsx";
import IssuesView, { RecheckSection } from "./IssuesView.jsx";

// Evaluation (Version-13): how accurate Scout is on a benchmark (BIRD), and why it got questions
// wrong. Two views: Accuracy check (the score and every question) and Issues (the eval's wrong
// answers, investigated by the Failure Analyst, as on the Dashboard).

const VIEWS = [
  { id: "accuracy", label: "Accuracy check", icon: "target" },
  { id: "compare", label: "Comparison", icon: "chart" }, // Version-22
  { id: "issues", label: "Issues", icon: "agent" },
];

const DIFFICULTY_LABEL = { simple: "Simple", moderate: "Moderate", challenging: "Challenging" };

// what happened to a question, and how each looks
const OUTCOME = {
  right: { label: "Right", style: "high" },
  "wrong rows": { label: "Wrong rows", style: "low" },
  "SQL error": { label: "SQL error", style: "low" }, // SQLite rejected the SQL
  "failed checks": { label: "Failed checks", style: "low" }, // valid SQL, but Scout's checks rejected every try
  "value not in data": { label: "Value not in data", style: "low" }, // the meaning asked for a value the data lacks, twice
  "no answer": { label: "No answer", style: "medium" },
};

const ISSUE_STATUS = {
  new: "Not investigated",
  investigating: "Investigating…",
  report_ready: "Report ready",
  approved: "Approved",
  rejected: "Rejected",
  failed: "Agent stopped",
};

function formatCost(value) {
  if (!value) return "$0";
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(3)}`;
}

function formatDate(iso) {
  return iso ? new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "";
}

export default function EvaluationTab({ active, onRunFinished }) {
  const [view, setView] = useState("accuracy");
  const [runsVersion, setRunsVersion] = useState(0); // bumped when a run ends, so the run list reloads
  const [run, setRun] = useState(""); // the evaluation run whose issues are shown

  return (
    <>
      <nav className="subnav" aria-label="Evaluation">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            className={v.id === view ? "subnav-item active" : "subnav-item"}
            onClick={() => setView(v.id)}
            aria-current={v.id === view ? "page" : undefined}
          >
            <Icon name={v.icon} size={14} />
            {v.label}
          </button>
        ))}
      </nav>

      <section hidden={view !== "accuracy"}>
        <AccuracyView
          active={active && view === "accuracy"}
          onOpenIssues={() => setView("issues")}
          onRunFinished={() => {
            setRunsVersion((n) => n + 1);
            onRunFinished?.();
          }}
        />
      </section>
      <section hidden={view !== "compare"}>
        <CompareView active={active && view === "compare"} />
      </section>
      <section hidden={view !== "issues"}>
        <IssuesView
          active={active && view === "issues" && run !== ""}
          scope={["eval"]}
          run={run}
          header={<RunPicker active={active && view === "issues"} version={runsVersion} value={run} onChange={setRun} />}
          title="Why did Scout get these wrong?"
          intro="Every question the evaluation got wrong, investigated by an AI agent that compares Scout's answer with the correct one, explains the cause and suggests a fix."
          empty={<>Wrong answers from an evaluation run appear here. Start one with <strong>Run evaluation</strong> under Accuracy check.</>}
        />
      </section>
    </>
  );
}

function AccuracyView({ active, onOpenIssues, onRunFinished }) {
  const [summary, setSummary] = useState(null);
  const [list, setList] = useState(null);
  const [filter, setFilter] = useState("all");
  const [difficulty, setDifficulty] = useState("");
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");

  async function loadResults(page = 1, next = {}) {
    const params = { filter, difficulty, search, ...next };
    try {
      setList(await getJson("/api/eval/results", { page, ...params }));
    } catch (err) {
      setError(`Could not load the questions: ${err.message}`);
    }
  }

  function loadSummary() {
    getJson("/api/eval/summary")
      .then(setSummary)
      .catch((err) => setError(`Could not load the evaluation: ${err.message}`));
  }

  useEffect(() => {
    if (!active) return;
    loadSummary();
    loadResults(list?.page || 1);
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  // a run just ended: its answers are the latest results now
  function runEnded() {
    loadSummary();
    loadResults(1);
    onRunFinished?.();
  }

  const wrong = summary ? summary.total - summary.correct : 0;
  const filters = [
    { id: "all", label: "All", count: summary?.total },
    { id: "right", label: "Right", count: summary?.correct },
    { id: "wrong", label: "Wrong", count: wrong },
    // Version-19: wrong answers split into Scout's mistakes and BIRD answers we believe are wrong
    ...(summary?.doubtful
      ? [
          { id: "scout", label: "Scout's mistakes", count: wrong - summary.doubtful },
          { id: "doubtful", label: "BIRD doubtful", count: summary.doubtful },
        ]
      : []),
  ];

  return (
    <>
      <div className="page-header">
        <h1>How accurate is Scout?</h1>
        <p>
          Scout answers every question of a benchmark on its own, and each answer is compared with the correct one: the
          same rows means right.
        </p>
      </div>

      <RunControl active={active} onEnded={runEnded} />

      {error && <ErrorAlert message={error} />}
      {!summary && !error && <Loading text="Loading the evaluation..." />}

      {summary?.sample && (
        <div className="alert alert-warning eval-sample">
          <Icon name="alert" />
          <span>
            <strong>Sample data.</strong> {summary.note}
          </span>
        </div>
      )}

      {summary && <Scorecard summary={summary} />}

      {summary && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              <Icon name="table" />
              Questions
            </h3>
            {wrong > 0 && (
              <button type="button" className="btn btn-secondary btn-sm" onClick={onOpenIssues}>
                <Icon name="agent" size={14} />
                Investigate wrong answers
              </button>
            )}
          </div>

          <form
            className="card-body eval-filters"
            onSubmit={(e) => {
              e.preventDefault();
              loadResults(1);
            }}
          >
            <div className="tabs" role="group" aria-label="Right or wrong">
              {filters.map((f) => (
                <button
                  key={f.id}
                  type="button"
                  className={f.id === filter ? "tab active" : "tab"}
                  aria-pressed={f.id === filter}
                  onClick={() => {
                    setFilter(f.id);
                    loadResults(1, { filter: f.id });
                  }}
                >
                  {f.label}
                  <span className="tab-count">{f.count ?? 0}</span>
                </button>
              ))}
            </div>
            <select
              id="eval-difficulty"
              className="select"
              value={difficulty}
              aria-label="Difficulty"
              onChange={(e) => {
                setDifficulty(e.target.value);
                loadResults(1, { difficulty: e.target.value });
              }}
            >
              <option value="">Any difficulty</option>
              {Object.entries(DIFFICULTY_LABEL).map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
            <label className="search-field eval-search">
              <Icon name="search" size={15} />
              <input
                id="eval-search"
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search questions or databases"
                aria-label="Search questions"
              />
            </label>
          </form>

          {list?.items.length === 0 && (
            <div className="empty">
              <h3>No questions match</h3>
              <p>Try another filter or search.</p>
            </div>
          )}
          {list?.items.map((r) => (
            <ResultRow key={r.id} result={r} sample={summary.sample} suite={summary.suite} onOpenIssues={onOpenIssues} />
          ))}
          {list && (
            <Pagination page={list.page} pages={list.pages} total={list.total} label="questions" onPage={(p) => loadResults(p)} always />
          )}
        </div>
      )}
    </>
  );
}

// The score: the ring, how it splits by difficulty and by what went wrong, and what the run took.
function Scorecard({ summary }) {
  const outcomes = Object.entries(OUTCOME).filter(([id]) => summary.by_outcome?.[id]);
  return (
    <div className="card eval-score">
      <div className="eval-score-main">
        <AccuracyRing value={summary.accuracy} size={132} stroke={12}>
          <b>{summary.accuracy == null ? "—" : `${summary.accuracy}%`}</b>
          <small>accuracy</small>
        </AccuracyRing>
        <div className="eval-score-text">
          <div className="label">{summary.benchmark}</div>
          <h2>
            {summary.correct} of {summary.total} right
          </h2>
          <p className="muted">
            {summary.databases} database{summary.databases === 1 ? "" : "s"} · {summary.model}
            {summary.provider && <> via {summary.provider}</>}
            {summary.run_at && <> · {formatDate(summary.run_at)}</>}
          </p>
          {summary.mode && (
            <p className="muted small">
              {summary.flow && <>Flow: {summary.flow} · </>}
              {summary.mode}
            </p>
          )}
          <div className="eval-outcomes">
            {outcomes.map(([id, o]) => (
              <span key={id} className={`badge ${o.style}`}>
                {o.label} {summary.by_outcome[id]}
              </span>
            ))}
          </div>
          {summary.doubtful > 0 && (
            <p className="muted small">
              {summary.doubtful} of the {summary.total - summary.correct} wrong answers {summary.doubtful === 1 ? "has" : "have"} a
              BIRD answer we believe is wrong. {summary.doubtful === 1 ? "It still counts" : "They still count"} as wrong.
            </p>
          )}
        </div>
      </div>

      <div className="eval-difficulty">
        {Object.entries(DIFFICULTY_LABEL).map(([id, label]) => {
          const d = summary.by_difficulty?.[id] || {};
          return (
            <div key={id} className="eval-diff">
              <div className="eval-diff-head">
                <span>{label}</span>
                <b>{d.accuracy == null ? "—" : `${d.accuracy}%`}</b>
              </div>
              <div className="eval-bar" aria-hidden="true">
                <span style={{ width: `${d.accuracy || 0}%` }} />
              </div>
              <span className="muted small">
                {d.correct || 0} of {d.total || 0} right
              </span>
            </div>
          );
        })}
      </div>

      <div className="stats eval-stats">
        <div className="stat">
          <div className="label">Questions</div>
          <div className="stat-value">{summary.total}</div>
        </div>
        <div className="stat">
          <div className="label">LLM calls</div>
          <div className="stat-value">{summary.llm_calls}</div>
          <span className="muted small">{summary.avg_llm_calls} per question</span>
        </div>
        <div className="stat">
          <div className="label">Cost</div>
          <div className="stat-value">{formatCost(summary.cost)}</div>
        </div>
        <div className="stat">
          <div className="label">Wrong</div>
          <div className="stat-value">{summary.total - summary.correct}</div>
        </div>
      </div>
    </div>
  );
}

// One question: right or wrong at a glance; opens to Scout's SQL beside the correct SQL.
function ResultRow({ result: r, sample, suite, onOpenIssues }) {
  const [open, setOpen] = useState(false);
  const outcome = OUTCOME[r.outcome] || OUTCOME["wrong rows"];
  return (
    <div className={`dash-row ${open ? "open" : ""}`}>
      <button type="button" className="dash-row-head eval-row-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className={`eval-mark ${r.correct ? "right" : "wrong"}`} aria-label={r.correct ? "Right" : "Wrong"}>
          <Icon name={r.correct ? "check" : "x"} size={13} />
        </span>
        <span className="dash-question">{r.question}</span>
        <span className="eval-db muted">{r.db_id}</span>
        <span className={`eval-diff-tag ${r.difficulty}`}>{DIFFICULTY_LABEL[r.difficulty]}</span>
        {!r.correct && <span className={`badge ${outcome.style}`}>{outcome.label}</span>}
        {r.doubtful && <span className="badge medium" title="We checked BIRD's answer by hand and believe it's wrong. Still counted as wrong.">BIRD answer doubtful</span>}
        <span className="chevron">
          <Icon name="chevron" />
        </span>
      </button>
      {open && (
        <div className="eval-detail">
          {r.evidence && (
            <p>
              <span className="muted">Hint given with the question: </span>
              {r.evidence}
            </p>
          )}
          {r.error && <p className="problems">{r.error}</p>}
          {r.doubtful && (
            <div className="alert alert-warning">
              <Icon name="alert" />
              <span>
                <strong>BIRD's answer looks wrong. </strong>
                {r.doubtful} <span className="muted">(Still counted as wrong; listed in data/text_to_sql/eval/bird_doubtful.json.)</span>
              </span>
            </div>
          )}
          <div className="eval-compare">
            <div>
              <div className="label">Scout's SQL</div>
              {r.sql ? <SqlCode sql={r.sql} /> : <p className="muted">No query ran.</p>}
              {!r.correct && r.rows && <RowsLine label="Returned" rows={r.rows} />}
            </div>
            <div>
              <div className="label">Correct SQL</div>
              <SqlCode sql={r.gold_sql} />
              {!r.correct && <RowsLine label="Expected" rows={r.expected_rows} />}
            </div>
          </div>
          {r.chains?.length > 0 && (
            <div className="eval-chains">
              <div className="label">Every try or chain</div>
              {r.chains.map((c) => (
                <details key={c.chain} className="eval-chain">
                  <summary>
                    <span className={`eval-mark ${c.correct ? "right" : "wrong"}`}>
                      <Icon name={c.correct ? "check" : "x"} size={12} />
                    </span>
                    <span>{c.chain}</span>
                    <span className="muted small">
                      {c.status === "ok" ? (c.rows || []).slice(0, 2).map((row) => row.join(", ")).join(" · ") || "no rows" : c.problems?.[0] || c.status}
                    </span>
                  </summary>
                  {c.meaning && <p className="muted small">{c.meaning}</p>}
                  {c.sentence && <p className="small"><b>Sentence:</b> {c.sentence}</p>}
                  {c.sql && <SqlCode sql={c.sql} />}
                </details>
              ))}
            </div>
          )}
          <p className="muted small">
            {r.llm_calls} LLM calls · {formatCost(r.cost)}
            {r.confidence && <> · {r.confidence} confidence</>}
          </p>
          {!r.correct && (
            <p className="eval-issue">
              {r.issue ? (
                <>
                  <span className="muted">Issue: </span>
                  <strong>{ISSUE_STATUS[r.issue.status] || r.issue.status}</strong>{" "}
                  <button type="button" className="btn btn-secondary btn-sm" onClick={onOpenIssues}>
                    <Icon name="agent" size={13} />
                    Open in Issues
                  </button>
                </>
              ) : (
                <span className="muted">
                  {sample ? "Sample questions aren't logged as issues." : "No issue logged for this question yet."}
                </span>
              )}
            </p>
          )}
          {!sample && suite && (
            <div className="eval-recheck">
              <RecheckSection questionId={r.id} suite={suite} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RowsLine({ label, rows }) {
  if (!rows?.length) return null;
  return (
    <p className="muted small eval-rows">
      {label}: {rows.slice(0, 3).map((row) => row.join(", ")).join(" · ")}
      {rows.length > 3 ? " …" : ""}
    </p>
  );
}

// ---------- Version-14: start a run and watch it live ----------

const RUN_STATUS = {
  running: { label: "Running", style: "info" },
  finished: { label: "Finished", style: "high" },
  stopped: { label: "Stopped", style: "medium" },
  failed: { label: "Failed", style: "low" },
};
const STAGES = [
  { id: "meaning", label: "Meaning" },
  { id: "steps", label: "Steps" },
  { id: "sql", label: "SQL" },
];
const LIMITS = [5, 10, 20, 50];
const RUN_POLL_MS = 1500;

function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rest = s % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${rest}s` : `${rest}s`;
}

// "Run evaluation": pick a suite and how many questions, see the estimate, start; then the live card.
function RunControl({ active, onEnded }) {
  const [run, setRun] = useState(null);
  const [setup, setSetup] = useState(false);
  const [hidden, setHidden] = useState(""); // a finished run's card the user closed
  const [error, setError] = useState("");
  const lastStatus = useRef(null);

  async function poll() {
    try {
      const { run: next } = await getJson("/api/eval/runs/current");
      setRun(next);
      if (lastStatus.current === "running" && next?.status !== "running") onEnded();
      lastStatus.current = next?.status || null;
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    if (active) poll();
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  const running = run?.status === "running";
  useEffect(() => {
    if (!active || !running) return undefined;
    const timer = setInterval(poll, RUN_POLL_MS);
    return () => clearInterval(timer);
  }, [active, running]); // eslint-disable-line react-hooks/exhaustive-deps

  async function act(path) {
    setError("");
    try {
      await postForm(`/api/eval/runs/${run.id}/${path}`, {});
      await poll();
    } catch (err) {
      setError(err.message);
    }
  }

  async function start(suite, first, last, flow, model) {
    setError("");
    try {
      const started = await postForm("/api/eval/runs", { suite, first, last, flow, model });
      setSetup(false);
      setRun(started);
      lastStatus.current = "running";
    } catch (err) {
      setError(err.message);
    }
  }

  const showRun = run && (running || hidden !== run.id);
  return (
    <>
      {error && <ErrorAlert message={error} />}
      {!running && !setup && (
        <div className="eval-run-bar">
          <button type="button" className="btn btn-primary" onClick={() => setSetup(true)}>
            <Icon name="target" size={15} />
            Run evaluation
          </button>
          <span className="muted small">Scout answers each question for real, using LLM calls. You'll see it live.</span>
        </div>
      )}
      {setup && <RunSetup onStart={start} onCancel={() => setSetup(false)} />}
      {showRun && (
        <LiveRun run={run} onStop={() => act("stop")} onResume={() => act("resume")} onClose={() => setHidden(run.id)} />
      )}
    </>
  );
}

function RunSetup({ onStart, onCancel }) {
  const [suites, setSuites] = useState(null);
  const [suite, setSuite] = useState("practice");
  const [flows, setFlows] = useState([]);
  const [flow, setFlow] = useState("original"); // Version-22: how Scout answers
  const [models, setModels] = useState([]);
  const [model, setModel] = useState(""); // Version-22: which LLM answers (empty: the default)
  // Version-18: questions first..last (1-based, both included), in the suite's own order
  const [first, setFirst] = useState(1);
  const [last, setLast] = useState(5);
  const [estimate, setEstimate] = useState(null);
  const [starting, setStarting] = useState(false);

  const chosen = suites?.find((s) => s.id === suite);
  const total = chosen?.questions || 0;
  const valid = total > 0 && first >= 1 && last >= first && last <= total;
  const count = valid ? last - first + 1 : 0;

  function pickSuite(id) {
    setSuite(id);
    setFirst(1);
    setLast(Math.min(5, suites?.find((s) => s.id === id)?.questions || 5));
  }

  useEffect(() => {
    getJson("/api/eval/suites", { questions: count || 1, flows_to_run: flow, ...(model ? { model } : {}) })
      .then((data) => {
        setSuites(data.suites);
        setFlows(data.flows || []);
        setModels(data.models || []);
        setModel((current) => current || data.default_model || "");
        setEstimate(count ? data.estimate : null);
      })
      .catch(() => setSuites([]));
  }, [count, flow, model]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="card eval-setup">
      <div className="card-header">
        <h3 className="card-title">
          <Icon name="target" />
          Run an evaluation
        </h3>
      </div>
      <div className="card-body">
        <div className="label">Test suite</div>
        <div className="options">
          {(suites || []).map((s) => (
            <label key={s.id} className={`option ${suite === s.id ? "picked" : ""} ${s.ready ? "" : "disabled"}`}>
              <input
                type="radio"
                name="eval-suite"
                id={`eval-suite-${s.id}`}
                checked={suite === s.id}
                disabled={!s.ready}
                onChange={() => pickSuite(s.id)}
              />
              <div>
                <div className="option-text">
                  {s.name} {s.ready && <span className="muted">· {s.questions} questions</span>}
                </div>
                <p className="muted small">{s.ready ? s.about : s.missing}</p>
              </div>
            </label>
          ))}
        </div>

        <div className="label">Flow</div>
        <div className="options">
          {flows.map((f) => (
            <label key={f.id} className={`option ${flow === f.id ? "picked" : ""} ${f.ready ? "" : "disabled"}`}>
              <input
                type="radio"
                name="eval-flow"
                id={`eval-flow-${f.id}`}
                checked={flow === f.id}
                disabled={!f.ready}
                onChange={() => setFlow(f.id)}
              />
              <div>
                <div className="option-text">
                  {f.name} {!f.ready && <span className="muted">· {f.note || "not built yet"}</span>}
                </div>
                <p className="muted small">{f.about}</p>
              </div>
            </label>
          ))}
        </div>

        <div className="label">Model</div>
        <select className="select eval-model" aria-label="Model" value={model} onChange={(e) => setModel(e.target.value)}>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name} · ${m.price_in} in / ${m.price_out} out per million tokens
            </option>
          ))}
        </select>

        <div className="label">Which questions</div>
        <div className="eval-range">
          <label>
            From
            <input type="number" min={1} max={total || 1} value={first} onChange={(e) => setFirst(Number(e.target.value))} />
          </label>
          <label>
            to
            <input type="number" min={1} max={total || 1} value={last} onChange={(e) => setLast(Number(e.target.value))} />
          </label>
          <span className="muted small">of {total} · in the suite's own order</span>
          <div className="tabs eval-limits" role="group" aria-label="Quick picks">
            {[...LIMITS.filter((n) => n < total), total].map((n) => (
              <button
                key={n}
                type="button"
                className={first === 1 && last === n ? "tab active" : "tab"}
                aria-pressed={first === 1 && last === n}
                onClick={() => {
                  setFirst(1);
                  setLast(n);
                }}
              >
                {n === total ? `All ${total}` : `First ${n}`}
              </button>
            ))}
          </div>
        </div>
        {!valid && total > 0 && <p className="muted small">Pick a range between 1 and {total}, with From no larger than to.</p>}

        {estimate && (
          <div className="eval-estimate">
            <span>
              About <b>{estimate.llm_calls}</b> LLM calls
            </span>
            <span>
              <b>{(estimate.tokens / 1000).toFixed(0)}K</b> tokens
            </span>
            <span>
              <b>{formatCost(estimate.cost)}</b>
            </span>
            <span>
              <b>~{estimate.minutes}</b> min
            </span>
            <span className="muted">via {estimate.provider}</span>
          </div>
        )}
        {estimate?.over_daily_limit && (
          <div className="alert alert-warning">
            <Icon name="alert" />
            <span>
              That's more than the provider's daily limit ({(estimate.daily_limit / 1000).toFixed(0)}K tokens). The run
              will stop when the limit is reached; resume it the next day.
            </span>
          </div>
        )}

        <div className="review-actions">
          <button
            type="button"
            className="btn btn-primary btn-sm"
            disabled={!chosen?.ready || starting || !valid}
            onClick={async () => {
              setStarting(true);
              await onStart(suite, first, last, flow, model);
              setStarting(false);
            }}
          >
            <Icon name="arrow" size={14} />
            Start {count ? `${count} question${count === 1 ? "" : "s"}` : ""}
          </button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

// The run, live: progress, what it's doing now, accuracy so far, and each answer as it lands.
// also used by Evaluation -> Comparison for the run a comparison is on (Version-22)
export function LiveRun({ run, onStop, onResume, onClose }) {
  const [, tick] = useState(0);
  const running = run.status === "running";
  useEffect(() => {
    if (!running) return undefined;
    const timer = setInterval(() => tick((n) => n + 1), 1000); // live clocks between polls
    return () => clearInterval(timer);
  }, [running]);

  const status = RUN_STATUS[run.status] || RUN_STATUS.running;
  const share = run.total ? (100 * run.done) / run.total : 0;
  const current = run.current;
  const nowSeconds = Date.now() / 1000;
  const onQuestion = running && current?.started ? nowSeconds - current.started : 0;
  const elapsed = run.elapsed_seconds + onQuestion;
  const left =
    run.seconds_per_question && running ? run.seconds_per_question * (run.total - run.done) - onQuestion : null;
  const waiting = running && run.wait_until ? run.wait_until - nowSeconds : null;
  const canResume = run.status !== "running" && run.status !== "finished" && run.done < run.total;

  return (
    <div className={`card eval-live ${running ? "running" : ""}`}>
      <div className="card-header">
        <h3 className="card-title">
          {running ? <span className="spinner" /> : <Icon name="target" />}
          {run.benchmark}
        </h3>
        <span className={`badge ${status.style}`}>{status.label}</span>
      </div>

      <div className="card-body eval-live-body">
        <div className="eval-progress">
          <div className="eval-progress-head">
            <b>
              {run.done} of {run.total}
            </b>
            <span className="muted">{Math.round(share)}%</span>
          </div>
          <div className="eval-bar large" aria-hidden="true">
            <span style={{ width: `${share}%` }} />
          </div>
        </div>

        <div className="eval-live-stats">
          <div className="eval-live-acc">
            <AccuracyRing value={run.accuracy} size={54} stroke={6}>
              <b className="small-ring">{run.accuracy == null ? "—" : `${Math.round(run.accuracy)}%`}</b>
            </AccuracyRing>
            <div>
              <div className="label">Accuracy so far</div>
              <span>
                {run.correct} of {run.done} right
              </span>
            </div>
          </div>
          <div>
            <div className="label">Elapsed</div>
            <span className="eval-live-value">{formatDuration(elapsed)}</span>
          </div>
          <div>
            <div className="label">Time left</div>
            <span className="eval-live-value">{running ? formatDuration(left) : "—"}</span>
          </div>
          <div>
            <div className="label">LLM calls</div>
            <span className="eval-live-value">{run.llm_calls}</span>
          </div>
          <div>
            <div className="label">Tokens</div>
            <span className="eval-live-value">{(run.tokens / 1000).toFixed(1)}K</span>
          </div>
          <div>
            <div className="label">Cost</div>
            <span className="eval-live-value">{formatCost(run.cost)}</span>
          </div>
        </div>

        {running && current && (
          <div className="eval-now">
            <div className="label">
              Now · question {current.index} of {run.total}
            </div>
            <p className="eval-now-q">{current.question}</p>
            {waiting > 0 ? (
              <p className="eval-wait">
                <Icon name="clock" size={14} />
                Waiting for the LLM provider's rate limit: trying again in {Math.ceil(waiting)} s
              </p>
            ) : (
              <ol className="eval-stages">
                {STAGES.map((st, i) => {
                  const at = STAGES.findIndex((x) => x.id === current.stage);
                  return (
                    <li key={st.id} className={i < at ? "done" : i === at ? "active" : ""}>
                      {i < at ? <Icon name="check" size={12} /> : i === at ? <span className="spinner" /> : null}
                      {st.label}
                    </li>
                  );
                })}
              </ol>
            )}
          </div>
        )}

        {run.message && <p className={run.status === "failed" ? "problems" : "muted"}>{run.message}</p>}

        {run.recent?.length > 0 && (
          <div className="eval-recent">
            <div className="label">Latest answers</div>
            <ul>
              {run.recent.map((r) => (
                <li key={r.id}>
                  <span className={`eval-mark ${r.correct ? "right" : "wrong"}`}>
                    <Icon name={r.correct ? "check" : "x"} size={12} />
                  </span>
                  <span className="eval-recent-q">{r.question}</span>
                  {!r.correct && <span className="muted small">{OUTCOME[r.outcome]?.label || r.outcome}</span>}
                  <span className="muted small eval-recent-meta">
                    {r.seconds}s · {r.llm_calls} calls · {formatCost(r.cost)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="review-actions">
          {running && (
            <button type="button" className="btn btn-secondary btn-sm" onClick={onStop}>
              <Icon name="x" size={14} />
              Stop after this question
            </button>
          )}
          {canResume && (
            <button type="button" className="btn btn-primary btn-sm" onClick={onResume}>
              <Icon name="arrow" size={14} />
              Resume ({run.total - run.done} left)
            </button>
          )}
          {!running && (
            <button type="button" className="btn btn-secondary btn-sm" onClick={onClose}>
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------- Version-15: pick the evaluation run the agent works on ----------

// Every finished run, newest first; its wrong answers are its issues. Issues the command-line
// eval script logged have no run, and are listed apart.
function RunPicker({ active, version, value, onChange }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!active) return;
    getJson("/api/eval/runs")
      .then((next) => {
        setData(next);
        const ids = next.runs.map((r) => r.id).concat(next.unlinked.total ? ["none"] : []);
        if (!ids.includes(value)) onChange(ids[0] || "none"); // the newest run by default
      })
      .catch((err) => setError(err.message));
  }, [active, version]); // eslint-disable-line react-hooks/exhaustive-deps

  if (error) return <ErrorAlert message={`Could not load the runs: ${error}`} />;
  if (!data) return null;
  const chosen = data.runs.find((r) => r.id === value);
  const nothing = data.runs.length === 0 && !data.unlinked.total;

  return (
    <div className="card eval-run-picker">
      <div className="card-body">
        <label className="label" htmlFor="eval-run">
          Evaluation run
        </label>
        {nothing ? (
          <p className="muted">
            No evaluation has finished yet. Start one with <strong>Run evaluation</strong> under Accuracy check; when it
            ends, it appears here with its wrong answers to investigate.
          </p>
        ) : (
          <>
            <select id="eval-run" className="select" value={value} onChange={(e) => onChange(e.target.value)}>
              {data.runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.short} · {formatDate(r.run_at)} · {r.correct}/{r.done} right ({r.accuracy}%) · {r.issues.total} issue
                  {r.issues.total === 1 ? "" : "s"}
                  {r.issues.new ? ` (${r.issues.new} new)` : ""}
                </option>
              ))}
              {data.unlinked.total > 0 && (
                <option value="none">
                  From the command-line eval script · {data.unlinked.total} issue{data.unlinked.total === 1 ? "" : "s"}
                </option>
              )}
            </select>
            {chosen && (
              <p className="muted small eval-run-meta">
                Run <code>{chosen.id}</code> · {chosen.benchmark} · {chosen.status === "finished" ? "finished" : `stopped at ${chosen.done} of ${chosen.total}`}
                {" · "}
                {chosen.wrong} wrong answer{chosen.wrong === 1 ? "" : "s"}, each an issue below
              </p>
            )}
            {value === "none" && (
              <p className="muted small eval-run-meta">
                Wrong answers logged by <code>python -m scripts.text_to_sql.evaluate</code>, which doesn't keep a run.
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
}
