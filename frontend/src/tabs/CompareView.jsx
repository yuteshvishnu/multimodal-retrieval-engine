import { useEffect, useRef, useState } from "react";
import { getJson, postForm } from "../api.js";
import { AccuracyRing, ErrorAlert, Icon, Loading } from "../components/ui.jsx";
import { LiveRun } from "./EvaluationTab.jsx";

// Evaluation -> Comparison (Version-22): how each flow scores on exactly the same questions
// (any range of a suite: From and To), one ring per flow, and a "Run comparison" that runs the
// chosen flows one after another. Full's saved chains also give, with no LLM calls, a
// weak-supervision ring and an Adaptive ring (marked "replayed").

const QUICK = [50, 100, 250];
const POLL_MS = 2000;
const ITEM_STATUS = {
  waiting: { label: "Waiting", style: "info" },
  running: { label: "Running", style: "medium" },
  done: { label: "Done", style: "high" },
  reused: { label: "Existing run", style: "high" },
  stopped: { label: "Stopped", style: "medium" },
  failed: { label: "Failed", style: "low" },
};

function formatCost(value) {
  if (!value) return "$0";
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(3)}`;
}

function formatDate(iso) {
  return iso ? new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "";
}

function rangeLabel(first, last) {
  return `Questions ${first}–${last}`;
}

export default function CompareView({ active }) {
  const [suites, setSuites] = useState(null);
  const [flows, setFlows] = useState([]);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState(""); // the rings show only this model's runs
  const [suite, setSuite] = useState("");
  // questions first..last, both included, in the suite's own order
  const [first, setFirst] = useState(1);
  const [last, setLast] = useState(50);
  const [data, setData] = useState(null);
  const [queue, setQueue] = useState(null);
  const [busy, setBusy] = useState(false);
  const [setup, setSetup] = useState(false);
  const [error, setError] = useState("");
  const [live, setLive] = useState(null); // the run the comparison is on, as the Accuracy check shows it
  const [hiddenRun, setHiddenRun] = useState("");
  const lastQueue = useRef("");

  useEffect(() => {
    if (!active || suites) return;
    getJson("/api/eval/suites")
      .then((d) => {
        const ready = d.suites.filter((s) => s.ready);
        setSuites(ready);
        setFlows(d.flows || []);
        setModels(d.models || []);
        setModel(d.default_model || "");
        // the corrected BIRD answers are the fairest yardstick (Version-21), then BIRD, then practice
        const pick = ["bird_corrected", "bird_minidev", "practice"].find((id) => ready.some((s) => s.id === id));
        setSuite(pick || ready[0]?.id || "");
      })
      .catch((err) => setError(err.message));
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  const total = suites?.find((s) => s.id === suite)?.questions || 0;
  const quick = [...QUICK.filter((n) => n < total), total].filter(Boolean);
  const valid = total > 0 && first >= 1 && last >= first && last <= total;

  async function loadCompare() {
    if (!suite || !valid || !model) return;
    try {
      setData(await getJson("/api/eval/compare", { suite, first, last, model }));
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadQueue() {
    try {
      const d = await getJson("/api/eval/comparisons/current");
      setQueue(d.comparison);
      setBusy(d.busy);
      if (d.busy || d.comparison?.status === "running") {
        const { run } = await getJson("/api/eval/runs/current");
        setLive(run);
      } else {
        setLive(null);
      }
      // a run in the comparison moved on: its ring may have changed
      const key = JSON.stringify(d.comparison?.items?.map((i) => [i.status, i.done]) || []) + (d.busy ? "" : "idle");
      if (key !== lastQueue.current) {
        lastQueue.current = key;
        loadCompare();
      }
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    if (active && suite) {
      loadCompare();
      loadQueue();
    }
  }, [active, suite, first, last, total, model]); // eslint-disable-line react-hooks/exhaustive-deps

  const running = queue?.status === "running";
  useEffect(() => {
    if (!active || !(running || busy)) return undefined;
    const timer = setInterval(loadQueue, POLL_MS);
    return () => clearInterval(timer);
  }, [active, running, busy]); // eslint-disable-line react-hooks/exhaustive-deps

  async function queueAction(path) {
    setError("");
    try {
      await postForm(`/api/eval/comparisons/${path}`, {});
      await loadQueue();
    } catch (err) {
      setError(err.message);
    }
  }

  async function start(chosen, reuse) {
    setError("");
    try {
      const q = await postForm("/api/eval/comparisons", { suite, first, last, flows: chosen.join(","), reuse, model });
      setQueue(q);
      setBusy(true);
      setSetup(false);
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <>
      <div className="page-header">
        <h1>Compare flows</h1>
        <p>
          How each way of answering scores on exactly the same questions. A flow's ring uses its newest run that answered
          every one of them.
        </p>
      </div>

      {error && <ErrorAlert message={error} />}
      {!suites && !error && <Loading text="Loading..." />}

      {suites && (
        <div className="card">
          <div className="card-body compare-controls">
            <select
              className="select"
              aria-label="Test suite"
              value={suite}
              onChange={(e) => {
                const next = suites.find((x) => x.id === e.target.value);
                setSuite(e.target.value);
                setFirst(1);
                setLast(Math.min(50, next?.questions || 50));
              }}
            >
              {suites.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} · {s.questions} questions
                </option>
              ))}
            </select>
            <select className="select" aria-label="Model" value={model} onChange={(e) => setModel(e.target.value)}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} · ${m.price_in} / ${m.price_out}
                </option>
              ))}
            </select>
            <div className="eval-range compare-range">
              <label>
                From
                <input type="number" min={1} max={total || 1} value={first} onChange={(e) => setFirst(Number(e.target.value))} />
              </label>
              <label>
                to
                <input type="number" min={1} max={total || 1} value={last} onChange={(e) => setLast(Number(e.target.value))} />
              </label>
              <div className="tabs" role="group" aria-label="Quick picks">
                {quick.map((n) => (
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
            {!setup && (
              <button type="button" className="btn btn-primary btn-sm" disabled={busy || !valid} onClick={() => setSetup(true)}>
                <Icon name="target" size={14} />
                Run comparison
              </button>
            )}
          </div>
          {!valid && total > 0 && <p className="card-body muted small">Pick a range between 1 and {total}, with From no larger than to.</p>}
          {busy && !running && <p className="card-body muted small">An evaluation run is going: a comparison can start when it finishes.</p>}
        </div>
      )}

      {setup && (
        <CompareSetup
          flows={flows}
          compared={data}
          label={`${rangeLabel(first, last)} · ${models.find((m) => m.id === model)?.name || model}`}
          model={model}
          suiteName={suites?.find((s) => s.id === suite)?.name}
          questions={data?.questions || 0}
          onStart={start}
          onCancel={() => setSetup(false)}
        />
      )}

      {queue && (
        <QueueCard
          queue={queue}
          flows={flows}
          models={models}
          live={live}
          onStop={() => queueAction("stop")}
          onResume={() => queueAction("resume")}
          busy={busy}
        />
      )}
      {live && live.status === "running" && hiddenRun !== live.id && (
        <LiveRun
          run={live}
          onStop={() => queueAction("stop")}
          onResume={() => queueAction("resume")}
          onClose={() => setHiddenRun(live.id)}
        />
      )}

      {data && (
        <div className="compare-rings">
          {data.flows.map((f) => (
            <FlowRing key={f.flow} flow={f} questions={data.questions} />
          ))}
        </div>
      )}

      {data && <DiffList data={data} flows={flows} />}
    </>
  );
}

// One flow's score on the chosen questions, or why there's none.
function FlowRing({ flow, questions }) {
  const r = flow.run;
  return (
    <div className={`card compare-ring ${r ? "" : "empty-ring"}`} title={flow.about}>
      <AccuracyRing value={r?.accuracy ?? null} size={112} stroke={10}>
        <b>{r ? `${r.accuracy}%` : "—"}</b>
        <small>{r ? "right" : "not run"}</small>
      </AccuracyRing>
      <h3>{flow.name}</h3>
      {flow.replayed && <span className="badge info">Replayed from Full: no extra LLM calls</span>}
      {["adaptive", "adaptive_english"].includes(flow.flow) && r && !flow.replayed && !flow.older && <span className="badge info">Its own live run</span>}
      {flow.older && <span className="badge medium">Older version, before the 4 chains</span>}
      {r ? (
        <>
          <p className="compare-score">
            {r.correct} of {r.total} right
          </p>
          <div className="compare-stats">
            <span>
              <b>{r.calls_per_question}</b> LLM calls per question
              {flow.stopped_early != null && <> · stopped early on {flow.stopped_early}</>}
            </span>
            <span>
              <b>{formatCost(r.cost)}</b> in all{flow.flow === "adaptive" && flow.replayed ? " (estimated)" : ""}
            </span>
            <span title="Every chain that answered agreed, and the answer was wrong: the problem the paper targets">
              <b>{r.confident_wrong}</b> wrong with high confidence
            </span>
            <span>
              High confidence was right <b>{r.confident_right} of {r.confident}</b>
            </span>
            {flow.misses != null && (
              <span title="Wrong answers where one of the tries or chains had the right rows, but another was picked">
                A right answer was there on <b>{flow.right_was_there} of {flow.misses}</b> misses
              </span>
            )}
            {flow.average_chain != null && (
              <span>
                One chain alone: <b>{flow.average_chain}%</b> on average
              </span>
            )}
            {flow.three_chains != null && (
              <span>
                With only 3 chains: <b>{flow.three_chains} of {r.total}</b>
              </span>
            )}
            {flow.accuracies && (
              <span title="Learned from how often the chains agree, never from correct answers">
                Learned trust:{" "}
                {Object.entries(flow.accuracies)
                  .map(([c, a]) => `${c} ${Math.round(a * 100)}%`)
                  .join(", ")}
              </span>
            )}
            {["full_ws", "mixed"].includes(flow.flow) && !flow.accuracies && <span>Too few questions to learn: equal weights</span>}
            {flow.majority != null && (
              <span>
                Plain majority vote: <b>{flow.majority} of {r.total}</b>
              </span>
            )}
            {flow.flow === "mixed" && flow.chains && (
              <span>
                Each model alone: {flow.chains.map((c) => `${c.chain} ${c.correct}`).join(", ")}
              </span>
            )}
          </div>
          <p className="muted small">
            Run {formatDate(r.run_at)}
            {r.model && <> · {r.model}</>}
          </p>
        </>
      ) : (
        <p className="muted small">
          {`No run of this flow answered all ${questions} questions yet. Use Run comparison.`}
        </p>
      )}
    </div>
  );
}

// Run comparison: which flows, reuse existing runs, the estimate, start.
function CompareSetup({ flows, compared, label, suiteName, questions, model, onStart, onCancel }) {
  // Mixed by default; tick the others to compare them on the same questions
  const [chosen, setChosen] = useState(flows.filter((f) => f.ready && f.id === "mixed").map((f) => f.id));
  const [reuse, setReuse] = useState(true);
  const [estimate, setEstimate] = useState(null);
  const [starting, setStarting] = useState(false);

  // an older Full (before the 4 chains) isn't reused
  const existing = new Set((compared?.flows || []).filter((f) => f.run && !f.older).map((f) => f.flow));
  const fullChosen = chosen.includes("full");
  const toRun = chosen.filter((id) => !(reuse && existing.has(id)) && !(id === "adaptive" && fullChosen));

  useEffect(() => {
    if (!toRun.length || !questions) {
      setEstimate(null);
      return;
    }
    getJson("/api/eval/suites", { questions, flows_to_run: toRun.join(","), model })
      .then((d) => setEstimate(d.estimate))
      .catch(() => setEstimate(null));
  }, [toRun.join(","), questions, model]); // eslint-disable-line react-hooks/exhaustive-deps

  function toggle(id) {
    const next = chosen.includes(id) ? chosen.filter((x) => x !== id) : [...chosen, id];
    // a Full run already gives Adaptive's result, so it isn't run twice
    setChosen(id === "full" && next.includes("full") ? next.filter((x) => x !== "adaptive") : next);
  }

  return (
    <div className="card eval-setup">
      <div className="card-header">
        <h3 className="card-title">
          <Icon name="target" />
          Run comparison · {suiteName} · {label}
        </h3>
      </div>
      <div className="card-body">
        <div className="label">Flows, run one after another</div>
        <div className="options">
          {flows.map((f) => (
            <label key={f.id} className={`option ${chosen.includes(f.id) ? "picked" : ""} ${f.ready ? "" : "disabled"}`}>
              <input
                type="checkbox"
                checked={chosen.includes(f.id)}
                disabled={!f.ready || (f.id === "adaptive" && fullChosen)}
                onChange={() => toggle(f.id)}
              />
              <div>
                <div className="option-text">
                  {f.name}
                  {!f.ready && <span className="muted"> · {f.note || "not built yet"}</span>}
                  {f.id === "adaptive" && fullChosen && <span className="muted"> · comes free with Full</span>}
                  {f.ready && !(f.id === "adaptive" && fullChosen) && reuse && existing.has(f.id) && (
                    <span className="muted"> · has a run: reused</span>
                  )}
                </div>
                <p className="muted small">{f.about}</p>
              </div>
            </label>
          ))}
        </div>
        <label className="compare-reuse">
          <input type="checkbox" checked={reuse} onChange={(e) => setReuse(e.target.checked)} />
          <span>
            Reuse existing runs <span className="muted small">(a flow that already answered these questions isn't run again)</span>
          </span>
        </label>

        {toRun.length > 0 && estimate && (
          <div className="eval-estimate">
            <span>
              <b>{toRun.length}</b> run{toRun.length === 1 ? "" : "s"} × {questions} questions
            </span>
            <span>
              About <b>{estimate.llm_calls}</b> LLM calls
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
        {chosen.length > 0 && toRun.length === 0 && (
          <p className="muted small">Every chosen flow already has a run of these questions: nothing new will run.</p>
        )}

        <div className="review-actions">
          <button
            type="button"
            className="btn btn-primary btn-sm"
            disabled={!chosen.length || !toRun.length || starting}
            onClick={async () => {
              setStarting(true);
              await onStart(chosen, reuse);
              setStarting(false);
            }}
          >
            <Icon name="arrow" size={14} />
            Start {toRun.length ? `${toRun.length} run${toRun.length === 1 ? "" : "s"}` : ""}
          </button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

// The comparison being run: each flow's run, live.
function QueueCard({ queue, flows, models, live, onStop, onResume, busy }) {
  const running = queue.status === "running";
  const name = (id) => flows.find((f) => f.id === id)?.name || id;
  return (
    <div className="card compare-queue">
      <div className="card-header">
        <h3 className="card-title">
          <Icon name="clock" />
          {running ? "Comparison running" : queue.status === "finished" ? "Last comparison" : "Comparison stopped"}
          <span className="muted small">
            {" "}
            · {queue.first ? rangeLabel(queue.first, queue.last) : `First ${queue.last}`}
            {queue.model && <> · {models.find((m) => m.id === queue.model)?.name || queue.model}</>} · {formatDate(queue.started_at)}
          </span>
        </h3>
        {running && (
          <button type="button" className="btn btn-secondary btn-sm" onClick={onStop}>
            <Icon name="x" size={13} />
            Stop
          </button>
        )}
        {(queue.status === "stopped" || queue.status === "failed") && !busy && (
          <button type="button" className="btn btn-primary btn-sm" onClick={onResume}>
            <Icon name="refresh" size={13} />
            Resume
          </button>
        )}
      </div>
      <div className="card-body compare-items">
        {queue.items.map((raw) => {
          // the running flow's numbers come from the live run (its id is known to the queue only at the end)
          const now = raw.status === "running" && live?.status === "running" && (live.flow || "original") === raw.flow;
          const item = now ? { ...raw, done: live.done, total: live.total, correct: live.correct } : raw;
          const st = ITEM_STATUS[item.status] || ITEM_STATUS.waiting;
          return (
            <div key={item.flow} className="compare-item">
              <span className="compare-item-name">{name(item.flow)}</span>
              <span className={`badge ${st.style}`}>{st.label}</span>
              {item.total ? (
                <span className="muted small">
                  {item.done} of {item.total} answered · {item.correct} right
                </span>
              ) : (
                <span className="muted small">—</span>
              )}
            </div>
          );
        })}
        {queue.message && <p className="muted small">{queue.message}</p>}
      </div>
    </div>
  );
}

// Questions where the flows disagree on right or wrong: where one flow's change mattered.
function DiffList({ data, flows }) {
  const compared = data.flows.filter((f) => f.run);
  if (compared.length < 2) {
    return (
      <p className="muted small compare-note">
        Once two or more flows have answered these questions, the questions where they differ are listed here.
      </p>
    );
  }
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">
          <Icon name="table" />
          Where the flows differ <span className="muted small">· {data.differ.length} question{data.differ.length === 1 ? "" : "s"}</span>
        </h3>
      </div>
      {data.differ.length === 0 && <p className="card-body muted">Every flow got the same questions right.</p>}
      {data.differ.map((q) => (
        <div key={q.id} className="compare-diff">
          <span className="muted small">#{q.n}</span>
          <span className="dash-question">{q.question}</span>
          <span className="compare-marks">
            {compared.map((f) => (
              <span key={f.flow} className={`eval-mark ${q.right[f.flow] ? "right" : "wrong"}`} title={`${f.name}: ${q.right[f.flow] ? "right" : "wrong"}`}>
                <Icon name={q.right[f.flow] ? "check" : "x"} size={12} />
              </span>
            ))}
          </span>
        </div>
      ))}
      <p className="card-body muted small">Marks in order: {compared.map((f) => f.name).join(", ")}.</p>
    </div>
  );
}
