import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { getJson } from "../api.js";
import { ErrorAlert, Icon, Loading, Pagination } from "../components/ui.jsx";
import IssuesView from "./IssuesView.jsx";

// ---------- formatting ----------

// older records marked names as [[name]] (blue tags, since removed): show them as plain text
function untag(text) {
  return String(text ?? "").replace(/\[\[|\]\]/g, "");
}

function formatCost(value) {
  if (!value) return "$0";
  if (value < 0.00001) return "<$0.00001";
  if (value < 1) return `$${Number(value.toPrecision(3))}`;
  return `$${value.toFixed(2)}`;
}

function formatCount(value) {
  return (value || 0).toLocaleString();
}

function formatMs(ms) {
  if (!ms) return "0 ms";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function formatTime(iso) {
  return new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

// dates as YYYY-MM-DD in local time (what the date inputs and the API use)
function isoDate(d) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function today() {
  return isoDate(new Date());
}

// the same period, one step earlier (-1) or later (+1)
function shift(day, period, step) {
  const d = new Date(`${day}T00:00:00`);
  if (period === "day") d.setDate(d.getDate() + step);
  if (period === "week") d.setDate(d.getDate() + 7 * step);
  if (period === "month") d.setMonth(d.getMonth() + step, 1);
  if (period === "year") d.setFullYear(d.getFullYear() + step, 0, 1);
  return isoDate(d);
}

const PERIODS = [
  { id: "day", label: "Day", slot: "hour" },
  { id: "week", label: "Week", slot: "day" },
  { id: "month", label: "Month", slot: "day" },
  { id: "year", label: "Year", slot: "month" },
];

const OUTCOME_STYLE = {
  answered: "high",
  "answered (reused)": "high",
  "waiting for meaning": "info",
  "blocked (write)": "medium",
  "too heavy": "medium",
  failed: "low",
  "no matching passages": "medium",
  "in progress": "info",
};

// Dashboard: what every question cost, in LLM calls, tokens, time and money.
// `active` reloads the data whenever the tab is opened.
const VIEWS = [
  { id: "usage", label: "Usage", icon: "chart" },
  { id: "issues", label: "Issues", icon: "agent" },
];

// Two views: Usage (costs, questions) and Issues (failures and the Failure Analyst agent, Version-11).
export default function DashboardTab({ active }) {
  const [view, setView] = useState("usage");

  return (
    <>
      <nav className="subnav" aria-label="Dashboard">
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

      {/* both stay mounted, so switching keeps each view's state; each loads when it's shown */}
      <section hidden={view !== "usage"}>
        <UsageView active={active && view === "usage"} />
      </section>
      <section hidden={view !== "issues"}>
        <IssuesView active={active && view === "issues"} />
      </section>
    </>
  );
}

function UsageView({ active }) {
  const [summary, setSummary] = useState(null);
  const [period, setPeriod] = useState("day");
  const [day, setDay] = useState(today());
  const [costs, setCosts] = useState(null);
  const [list, setList] = useState(null);
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const [onlyIssues, setOnlyIssues] = useState(false); // show only the reported answers (Version-10)
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function loadCosts(nextPeriod = period, nextDay = day) {
    try {
      setCosts(await getJson("/api/usage/costs", { period: nextPeriod, day: nextDay }));
    } catch (err) {
      setError(`Could not load costs: ${err.message}`);
    }
  }

  async function loadPage(page, nextQuery = query, issues = onlyIssues) {
    try {
      setList(await getJson("/api/usage/questions", { page, search: nextQuery, feedback: issues ? "down" : "" }));
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadAll() {
    setLoading(true);
    setError("");
    try {
      setSummary(await getJson("/api/usage/summary"));
      await Promise.all([loadCosts(), loadPage(list?.page || 1)]);
    } catch (err) {
      setError(`Could not load the dashboard: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (active) loadAll();
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  function choose(nextPeriod, nextDay) {
    setPeriod(nextPeriod);
    setDay(nextDay);
    loadCosts(nextPeriod, nextDay);
  }

  function handleSearch(event) {
    event.preventDefault();
    setQuery(search);
    loadPage(1, search);
  }

  const slot = PERIODS.find((p) => p.id === period).slot;

  return (
    <>
      <div className="page-header page-header-row">
        <div>
          <h1>Know your costs.</h1>
          <p>Every question, what it took to answer, and what it cost in LLM calls, tokens, time and money.</p>
        </div>
        <button type="button" className="btn btn-secondary btn-sm" onClick={loadAll} disabled={loading}>
          <Icon name="refresh" size={14} />
          Refresh
        </button>
      </div>

      {error && <ErrorAlert message={error} />}
      {!summary && loading && <Loading text="Loading the dashboard..." />}

      {summary && (
        <div className="card stats stats-5">
          <Stat label="Questions asked" value={formatCount(summary.questions)} />
          <Stat label="Input tokens" value={formatCount(summary.input_tokens)} />
          <Stat
            label="Output tokens"
            value={formatCount(summary.output_tokens)}
            note={summary.reasoning_tokens ? `incl. ${formatCount(summary.reasoning_tokens)} reasoning` : null}
          />
          <Stat
            label="Reported"
            value={`⚑ ${formatCount(summary.thumbs_down)}`}
            note={`answers reported as wrong${
              summary.open_issues ? ` · ${summary.open_issues} open issue${summary.open_issues === 1 ? "" : "s"}` : ""
            }`}
          />
          <Stat
            label="Total cost"
            value={formatCost(summary.cost)}
            note={`all ${formatCount(summary.llm_calls)} LLM calls${
              summary.other_calls ? `, incl. ${summary.other_calls} outside questions (table descriptions, eval, agent)` : ""
            }`}
          />
        </div>
      )}

      {summary && (
        <div className="card">
          <div className="card-header chart-header">
            <h3 className="card-title">
              <Icon name="chart" />
              LLM cost per {slot}
            </h3>
            <span className="muted">
              {costs ? (
                <>
                  {costs.title} · <strong>{formatCost(costs.total_cost)}</strong> · {costs.total_calls} calls
                </>
              ) : (
                "…"
              )}
            </span>
          </div>

          <div className="chart-controls">
            <div className="tabs period-tabs" role="group" aria-label="Period">
              {PERIODS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  className={p.id === period ? "tab active" : "tab"}
                  onClick={() => choose(p.id, day)}
                  aria-pressed={p.id === period}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <div className="date-nav">
              <button type="button" className="icon-btn" onClick={() => choose(period, shift(day, period, -1))} aria-label="Earlier">
                <Icon name="left" size={13} />
              </button>
              <PeriodInput period={period} day={day} onChange={(d) => choose(period, d)} />
              <button type="button" className="icon-btn" onClick={() => choose(period, shift(day, period, 1))} aria-label="Later">
                <Icon name="chevron" size={13} />
              </button>
              <button type="button" className="icon-btn" onClick={() => choose(period, today())}>
                Today
              </button>
            </div>
          </div>

          {costs && <CostChart data={costs} />}
          <p className="chart-note muted">
            Prices per million tokens:{" "}
            {Object.entries(summary.prices)
              .map(([model, p]) => `${model} $${p.input} in / $${p.output} out`)
              .join(" · ")}
            . Set in backend/config.py.
          </p>
        </div>
      )}

      {list && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              <Icon name="layers" />
              Questions
            </h3>
            <span className="tag">{formatCount(list.total)}</span>
          </div>
          <form onSubmit={handleSearch} className="card-body dash-search">
            <label className="search-field">
              <Icon name="search" size={18} />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search questions or ids"
                aria-label="Search questions"
              />
            </label>
            <button type="submit" className="btn btn-secondary">
              Search
            </button>
            <button
              type="button"
              className={`btn btn-secondary issues-toggle ${onlyIssues ? "on" : ""}`}
              onClick={() => {
                setOnlyIssues(!onlyIssues);
                loadPage(1, query, !onlyIssues);
              }}
              aria-pressed={onlyIssues}
            >
              ⚑ Reported only
            </button>
          </form>

          {list.items.length === 0 && (
            <div className="empty">
              <h3>{query || onlyIssues ? "No questions match" : "No questions yet"}</h3>
              <p>{query ? "Try another search." : "Ask something in Scout and it shows up here."}</p>
            </div>
          )}

          {list.items.map((q) => (
            <QuestionRow key={q.id} q={q} />
          ))}
          <Pagination page={list.page} pages={list.pages} total={list.total} label="questions" onPage={loadPage} always />
        </div>
      )}
    </>
  );
}

// the right picker for the period: a date (day, week), a month, or a year
function PeriodInput({ period, day, onChange }) {
  if (period === "month") {
    return (
      <input
        className="date-input"
        type="month"
        value={day.slice(0, 7)}
        onChange={(e) => e.target.value && onChange(`${e.target.value}-01`)}
        aria-label="Month"
      />
    );
  }
  if (period === "year") {
    const year = Number(day.slice(0, 4));
    const years = Array.from({ length: 7 }, (_, i) => year - 3 + i);
    return (
      <select className="date-input" value={year} onChange={(e) => onChange(`${e.target.value}-01-01`)} aria-label="Year">
        {years.map((y) => (
          <option key={y} value={y}>
            {y}
          </option>
        ))}
      </select>
    );
  }
  return (
    <input
      className="date-input"
      type="date"
      value={day}
      onChange={(e) => e.target.value && onChange(e.target.value)}
      aria-label={period === "week" ? "Any day in the week" : "Day"}
    />
  );
}

function Stat({ label, value, note }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="stat-value">{value}</div>
      {note && <div className="stat-note muted">{note}</div>}
    </div>
  );
}

// One question: a summary line that opens to show its metrics.
function QuestionRow({ q }) {
  const [open, setOpen] = useState(false);

  return (
    <div className={`dash-row ${open ? "open" : ""}`}>
      <button type="button" className="dash-row-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="dash-id" title={q.question}>
          {q.id}
        </span>
        <span className="dash-question">{q.question || "(empty question)"}</span>
        <span className={`badge ${OUTCOME_STYLE[q.outcome] || "info"}`}>{q.outcome}</span>
        <span className="dash-vote" title={q.feedback?.comment || ""}>
          {q.feedback ? (q.feedback.rating === "up" ? "✓" : "⚑") : ""}
        </span>
        <span className="dash-cost">{formatCost(q.cost)}</span>
        <span className="chevron">
          <Icon name="chevron" />
        </span>
      </button>

      {open && (
        <dl className="dash-metrics">
          <Metric label="Asked">{formatTime(q.started_at)}</Metric>
          <Metric label="LLM calls per try">
            {q.calls_per_try.length === 0 ? (
              <span className="muted">none (stopped before any LLM call)</span>
            ) : (
              <span className="option-choices">
                {q.calls_per_try.map((s) => (
                  <span key={s.step} className="tag">
                    {s.step}: {s.calls}
                  </span>
                ))}
              </span>
            )}
          </Metric>
          <Metric label="Retries">
            {q.retries}
            {q.retries > 0 && <span className="muted"> (meaning re-asks, step re-runs, SQL round 2)</span>}
          </Metric>
          <Metric label="Total input tokens">{formatCount(q.input_tokens)}</Metric>
          <Metric label="Total output tokens">
            {formatCount(q.output_tokens)}
            {q.reasoning_tokens > 0 && <span className="muted"> (incl. {formatCount(q.reasoning_tokens)} reasoning)</span>}
          </Metric>
          <Metric label="LLM latency">
            {formatMs(q.llm_latency_ms)}
            {q.llm_calls > 1 && <span className="muted"> (sum of {q.llm_calls} calls; tries run in parallel)</span>}
          </Metric>
          <Metric label="Total latency">{formatMs(q.total_latency_ms)}</Metric>
          <Metric label="Outcome">
            {q.outcome}
            {q.confidence && <span className="muted"> · {q.confidence} confidence</span>}
          </Metric>
          <Metric label="Total LLM cost">
            <strong>{formatCost(q.cost)}</strong>
            {q.failed_calls > 0 && <span className="muted"> · {q.failed_calls} call(s) failed</span>}
          </Metric>
          <Metric label="Feedback">
            {q.feedback ? (
              <>
                {q.feedback.rating === "up" ? "✓ Correct" : "⚑ Issue reported"}
                <span className="muted"> · {formatTime(q.feedback.created_at)}</span>
              </>
            ) : (
              <span className="muted">none</span>
            )}
          </Metric>
          {q.feedback?.rating === "down" && <Issue feedback={q.feedback} />}
        </dl>
      )}
    </div>
  );
}

// A reported issue: the user's comment and what they saw, to reproduce and fix it (Version-10).
function Issue({ feedback }) {
  const d = feedback.details || {};
  return (
    <div className="dash-issue">
      <div className="label">Issue reported</div>
      <p className="dash-issue-comment">“{feedback.comment}”</p>
      {d.meaning && (
        <p>
          <span className="muted">Meaning: </span>
          {untag(d.meaning.split("\n")[0])}
        </p>
      )}
      {d.steps?.length > 0 && (
        <ol className="steps-list">
          {d.steps.map((s, i) => (
            <li key={i}>
              {untag(s)}
            </li>
          ))}
        </ol>
      )}
      {d.sql && <pre className="code dash-issue-sql">{d.sql}</pre>}
      {d.rows?.length > 0 && (
        <p className="muted">
          First rows ({d.row_count} in all): {d.rows.slice(0, 3).map((r) => r.join(", ")).join(" · ")}
        </p>
      )}
      {d.answer && (
        <p>
          <span className="muted">Answer: </span>
          {d.answer}
        </p>
      )}
    </div>
  );
}

function Metric({ label, children }) {
  return (
    <div className="dash-metric">
      <dt className="label">{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

// ---------- the chart: a fixed time axis (24 hours, 7 days, a month's days or 12 months) ----------

const CHART_HEIGHT = 240;
const PAD = { top: 16, right: 16, bottom: 28, left: 56 };

function niceMax(value) {
  if (value <= 0) return 0.001;
  const power = 10 ** Math.floor(Math.log10(value));
  const step = [1, 2, 2.5, 5, 10].find((s) => s * power >= value);
  return step * power;
}

// which slots get an axis label, so they never crowd
function tickIndexes(period, count, narrow) {
  const all = Array.from({ length: count }, (_, i) => i);
  if (period === "day") return all.filter((i) => i % (narrow ? 6 : 3) === 0);
  if (period === "month") return all.filter((i) => i === 0 || (i + 1) % 5 === 0);
  if (period === "year" && narrow) return all.filter((i) => i % 3 === 0);
  return all;
}

function slotTitle(period, bucket, next) {
  if (period === "day") return `${bucket.label} – ${next ? next.label : "24:00"}`;
  const d = new Date(bucket.start);
  if (period === "year") return d.toLocaleDateString(undefined, { month: "long", year: "numeric" });
  return d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
}

function CostChart({ data }) {
  const box = useRef(null);
  const tip = useRef(null);
  const [width, setWidth] = useState(600);
  const [hover, setHover] = useState(null); // index of the hovered slot
  const [tipSize, setTipSize] = useState({ w: 280, h: 100 });

  useLayoutEffect(() => {
    if (!box.current) return undefined;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(box.current);
    return () => observer.disconnect();
  }, []);

  // measure the tooltip, so it can be placed fully inside the chart
  useLayoutEffect(() => {
    if (tip.current) setTipSize({ w: tip.current.offsetWidth, h: tip.current.offsetHeight });
  }, [hover]);

  const { buckets, period } = data;
  const plotW = Math.max(40, width - PAD.left - PAD.right);
  const plotH = CHART_HEIGHT - PAD.top - PAD.bottom;
  const max = niceMax(Math.max(...buckets.map((b) => b.cost)));
  const slotW = plotW / buckets.length;
  const barW = Math.max(2, Math.min(36, slotW - 2)); // 2px surface gap between bars
  const y = (v) => PAD.top + plotH - (v / max) * plotH;
  const ticks = [0, max / 2, max];
  const labels = tickIndexes(period, buckets.length, width < 520);
  const hovered = hover != null ? buckets[hover] : null;

  // tooltip: above the bar if it fits, otherwise below its top; always inside the chart
  let tipLeft = 0;
  let tipTop = 0;
  if (hovered) {
    const center = PAD.left + hover * slotW + slotW / 2;
    const barTop = y(hovered.cost);
    tipLeft = Math.min(Math.max(center - tipSize.w / 2, 4), width - tipSize.w - 4);
    tipTop = barTop - tipSize.h - 10 >= 0 ? barTop - tipSize.h - 10 : Math.min(barTop + 10, CHART_HEIGHT - tipSize.h);
    tipTop = Math.max(0, tipTop);
  }

  return (
    <div className="chart" ref={box}>
      <svg width={width} height={CHART_HEIGHT} role="img" aria-label={`LLM cost per slot, ${data.title}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line className="chart-grid" x1={PAD.left} x2={PAD.left + plotW} y1={y(t)} y2={y(t)} />
            <text className="chart-axis" x={PAD.left - 8} y={y(t)} dy="0.32em" textAnchor="end">
              {formatCost(t)}
            </text>
          </g>
        ))}
        {buckets.map((b, i) => {
          const x = PAD.left + i * slotW + (slotW - barW) / 2;
          const h = b.cost > 0 ? Math.max(2, (b.cost / max) * plotH) : 0;
          const r = Math.min(4, barW / 2, h);
          return (
            <g key={b.start}>
              {h > 0 && (
                <path
                  className={`chart-bar ${hover === i ? "hovered" : ""}`}
                  d={`M${x},${y(0)} V${y(0) - h + r} Q${x},${y(0) - h} ${x + r},${y(0) - h} H${x + barW - r} Q${x + barW},${y(0) - h} ${x + barW},${y(0) - h + r} V${y(0)} Z`}
                />
              )}
              {/* hit target: the whole column, wider than the bar */}
              <rect
                x={PAD.left + i * slotW}
                y={PAD.top}
                width={slotW}
                height={plotH}
                fill="transparent"
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
              />
            </g>
          );
        })}
        <line className="chart-baseline" x1={PAD.left} x2={PAD.left + plotW} y1={y(0)} y2={y(0)} />
        {labels.map((i) => (
          <text key={i} className="chart-axis" x={PAD.left + i * slotW + slotW / 2} y={CHART_HEIGHT - 8} textAnchor="middle">
            {buckets[i].label}
          </text>
        ))}
      </svg>

      {data.total_calls === 0 && <div className="chart-empty muted">No LLM calls in this period.</div>}

      {hovered && (
        <div className="chart-tooltip" ref={tip} style={{ left: tipLeft, top: tipTop }}>
          <div className="chart-tooltip-title">{slotTitle(period, hovered, buckets[hover + 1])}</div>
          <div>
            <strong>{formatCost(hovered.cost)}</strong> · {hovered.calls} LLM call{hovered.calls === 1 ? "" : "s"}
          </div>
          {hovered.questions.slice(0, 4).map((q) => (
            <div key={q.id} className="chart-tooltip-q">
              <span>{q.question || "(empty question)"}</span>
              <span className="muted">{formatCost(q.cost)}</span>
            </div>
          ))}
          {hovered.questions.length > 4 && <div className="muted">+{hovered.questions.length - 4} more questions</div>}
          {hovered.other_calls > 0 && (
            <div className="muted">
              {hovered.other_calls} call{hovered.other_calls === 1 ? "" : "s"} for table descriptions
            </div>
          )}
        </div>
      )}
    </div>
  );
}
