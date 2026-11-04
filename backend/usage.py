"""
Usage tracking for the Dashboard: every LLM call, grouped by question, with its tokens,
time and price. Stored in a small SQLite file (data/usage/usage.sqlite, git-ignored), so
the dashboard can sum, group and search with plain SQL.

    questions   one row per question asked (Scout or Scribe): when, outcome, total time
    llm_calls   one row per LLM call: which question, which job (stage), round, model,
                input / output / reasoning tokens (from the API's own count), time, price, cost

Cost of a call = input tokens x input price + output tokens x output price (per million
tokens, from LLM_PRICES in config). Reasoning tokens are part of the output and billed as
such. The prices used are saved with each call, so old costs stay right when prices change.
"""
import calendar
import json
import math
import sqlite3
import threading
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from backend.config import LLM_PRICES

USAGE_DB = Path("data/usage/usage.sqlite")
PAGE_SIZE = 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    product TEXT NOT NULL,              -- scout | scribe
    question TEXT NOT NULL,
    started_at TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'in progress',
    confidence TEXT,
    total_latency_ms REAL NOT NULL DEFAULT 0   -- time spent processing, summed over its requests
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id TEXT,                   -- NULL for calls not tied to a question (table descriptions)
    stage TEXT NOT NULL,                -- meaning | sql | answer | table_description
    round INTEGER,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL,               -- ok | error
    price_in REAL NOT NULL DEFAULT 0,   -- $ per million input tokens, at the time of the call
    price_out REAL NOT NULL DEFAULT 0,
    cost REAL NOT NULL DEFAULT 0,       -- $
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS llm_calls_question ON llm_calls (question_id);
CREATE TABLE IF NOT EXISTS feedback (
    question_id TEXT PRIMARY KEY,       -- one rating per question: the latest one counts
    rating TEXT NOT NULL,               -- up (Correct) | down (Report issue)
    comment TEXT,
    details TEXT,                       -- JSON: what the user saw (steps, SQL, first rows), to reproduce a report
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS issues (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,               -- eval | user_report | sql_error
    key TEXT NOT NULL UNIQUE,           -- source (+ eval run) + question: the same failure again updates this issue
    question TEXT NOT NULL,
    question_id TEXT,
    details TEXT NOT NULL,              -- JSON: what failed (SQL, error, comment, correct answer...)
    status TEXT NOT NULL DEFAULT 'new', -- new | investigating | report_ready | approved | rejected | failed
    report TEXT,                        -- JSON: the Failure Analyst's report
    note TEXT,                          -- the developer's note when approving or rejecting
    agent_calls INTEGER NOT NULL DEFAULT 0,
    agent_cost REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id TEXT NOT NULL,
    kind TEXT NOT NULL,                 -- start | reasoning | tool | report | test | test_failed | error
    title TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS agent_steps_issue ON agent_steps (issue_id);
"""

ISSUE_SOURCES = ("eval", "user_report", "sql_error")


def issue_key(source: str, question: str, run_id: Optional[str] = None) -> str:
    """The same failure again updates its issue: same source, question (and eval run, if any)."""
    norm = " ".join(question.lower().split())
    return f"{source}:{run_id}:{norm}" if run_id else f"{source}:{norm}"
ISSUE_STATUSES = ("new", "investigating", "report_ready", "approved", "rejected", "failed")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def call_cost(model: str, input_tokens: int, output_tokens: int) -> tuple:
    """(price in, price out, cost in $) for one call; unknown models cost 0 until priced."""
    price_in, price_out = LLM_PRICES.get(model, (0.0, 0.0))
    return price_in, price_out, (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class UsageStore:
    def __init__(self, path: Path = USAGE_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        # Version-15: eval issues belong to the evaluation run that found them (older databases lack the column)
        if "run_id" not in {r["name"] for r in self._conn.execute("PRAGMA table_info(issues)")}:
            self._conn.execute("ALTER TABLE issues ADD COLUMN run_id TEXT")
            self._conn.commit()
        self._lock = threading.Lock()
        # an investigation can't survive a restart: let it be started again
        self._write("UPDATE issues SET status = 'new' WHERE status = 'investigating'")

    def _write(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _read(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ---------- recording ----------

    def start_question(self, product: str, question: str, question_id: Optional[str] = None) -> Optional[str]:
        """A new question (or the existing one, when question_id is given and known). None for an empty one."""
        if question_id and self._read("SELECT 1 FROM questions WHERE id = ?", (question_id,)):
            return question_id
        if not question.strip():
            return None  # nothing was asked: nothing to record
        question_id = uuid.uuid4().hex[:10]
        self._write(
            "INSERT INTO questions (id, product, question, started_at) VALUES (?, ?, ?, ?)",
            (question_id, product, question, now()),
        )
        return question_id

    def add_time(self, question_id: Optional[str], seconds: float) -> None:
        if question_id:
            self._write(
                "UPDATE questions SET total_latency_ms = total_latency_ms + ? WHERE id = ?",
                (seconds * 1000, question_id),
            )

    def finish(self, question_id: Optional[str], outcome: str, confidence: Optional[str] = None) -> None:
        if question_id:
            self._write("UPDATE questions SET outcome = ?, confidence = ? WHERE id = ?", (outcome, confidence, question_id))

    def add_feedback(self, question_id: str, rating: str, comment: str = "", details: Optional[Dict] = None) -> None:
        """👍 / 👎 on an answer. A 👎 with its comment and details is an issue for the developer."""
        if rating not in ("up", "down"):
            raise ValueError("rating must be 'up' or 'down'")
        if not self._read("SELECT 1 FROM questions WHERE id = ?", (question_id,)):
            raise KeyError(question_id)
        self._write(
            "INSERT OR REPLACE INTO feedback (question_id, rating, comment, details, created_at) VALUES (?, ?, ?, ?, ?)",
            (question_id, rating, comment.strip() or None, json.dumps(details or {}, ensure_ascii=False), now()),
        )

    def record_call(self, tag: Optional[Dict], model: str, usage, latency: float, status: str) -> None:
        """One LLM call. usage is the API's usage object (None if the call failed)."""
        tag = tag or {}
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
        price_in, price_out, cost = call_cost(model, input_tokens, output_tokens)
        # OpenRouter reports what the call really cost (it depends on the host that served it)
        reported = getattr(usage, "cost", None)
        if reported is None:
            reported = (getattr(usage, "model_extra", None) or {}).get("cost")
        if isinstance(reported, (int, float)):
            cost = float(reported)
        self._write(
            "INSERT INTO llm_calls (question_id, stage, round, model, input_tokens, output_tokens, reasoning_tokens,"
            " latency_ms, status, price_in, price_out, cost, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tag.get("question_id"), tag.get("stage", "other"), tag.get("round"), model, input_tokens, output_tokens,
             reasoning, latency * 1000, status, price_in, price_out, cost, now()),
        )

    # ---------- reading (the dashboard) ----------

    def summary(self) -> Dict:
        q = self._read("SELECT COUNT(*) FROM questions")[0][0]
        row = self._read(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), "
            "COALESCE(SUM(reasoning_tokens), 0), COALESCE(SUM(cost), 0), "
            "SUM(CASE WHEN question_id IS NULL THEN 1 ELSE 0 END) FROM llm_calls"
        )[0]
        votes = dict(self._read("SELECT rating, COUNT(*) FROM feedback GROUP BY rating"))
        open_issues = self._read("SELECT COUNT(*) FROM issues WHERE status IN ('new', 'investigating', 'report_ready')")[0][0]
        return {
            "questions": q, "llm_calls": row[0], "input_tokens": row[1], "output_tokens": row[2],
            "reasoning_tokens": row[3], "cost": row[4], "other_calls": row[5] or 0,
            "thumbs_up": votes.get("up", 0), "thumbs_down": votes.get("down", 0), "open_issues": open_issues,
        }

    def costs(self, period: str, day: date) -> Dict:
        """
        LLM cost over one period, in fixed time slots, so the time axis never changes:
            day   -> 24 hours (00-24)        week -> 7 days (Mon-Sun)
            month -> every day of the month   year -> 12 months
        Every call counts (also table descriptions); each slot lists its questions and their cost.
        """
        if period == "day":
            start = datetime(day.year, day.month, day.day)
            slots = [start + timedelta(hours=h) for h in range(25)]
            labels = [f"{h:02d}:00" for h in range(24)]
            title = start.strftime("%a, %b %-d, %Y")
        elif period == "week":
            monday = day - timedelta(days=day.weekday())
            start = datetime(monday.year, monday.month, monday.day)
            slots = [start + timedelta(days=d) for d in range(8)]
            labels = [(start + timedelta(days=d)).strftime("%a %-d") for d in range(7)]
            title = f"{slots[0]:%b %-d} – {slots[6]:%b %-d, %Y}"
        elif period == "month":
            days = calendar.monthrange(day.year, day.month)[1]
            start = datetime(day.year, day.month, 1)
            slots = [start + timedelta(days=d) for d in range(days + 1)]
            labels = [str(d + 1) for d in range(days)]
            title = start.strftime("%B %Y")
        elif period == "year":
            slots = [datetime(day.year, m, 1) for m in range(1, 13)] + [datetime(day.year + 1, 1, 1)]
            labels = [calendar.month_abbr[m] for m in range(1, 13)]
            title = str(day.year)
        else:
            raise ValueError("period must be day, week, month or year")

        buckets = [{"label": label, "start": slots[i].isoformat(), "cost": 0.0, "calls": 0, "questions": {},
                    "other_calls": 0} for i, label in enumerate(labels)]
        rows = self._read(
            "SELECT c.created_at, c.cost, c.question_id, q.question FROM llm_calls c "
            "LEFT JOIN questions q ON q.id = c.question_id WHERE c.created_at >= ? AND c.created_at < ?",
            (slots[0].isoformat(timespec="seconds"), slots[-1].isoformat(timespec="seconds")),
        )
        for row in rows:
            at = datetime.fromisoformat(row["created_at"])
            i = next(i for i in range(len(labels)) if slots[i] <= at < slots[i + 1])
            b = buckets[i]
            b["cost"] += row["cost"]
            b["calls"] += 1
            if row["question_id"]:
                q = b["questions"].setdefault(row["question_id"], {"id": row["question_id"], "question": row["question"], "cost": 0.0})
                q["cost"] += row["cost"]
            else:
                b["other_calls"] += 1
        for b in buckets:
            b["questions"] = sorted(b["questions"].values(), key=lambda q: -q["cost"])
        return {
            "period": period, "title": title, "date": day.isoformat(), "buckets": buckets,
            "total_cost": sum(b["cost"] for b in buckets), "total_calls": sum(b["calls"] for b in buckets),
        }

    def questions(self, page: int = 1, search: str = "", feedback: str = "") -> Dict:
        """Questions, newest first, 10 per page, each with its metrics. feedback="down": only 👎 (the issues)."""
        conditions, params = [], ()
        if search.strip():
            conditions.append("(q.question LIKE ? OR q.id LIKE ?)")
            params += (f"%{search.strip()}%",) * 2
        if feedback in ("up", "down"):
            conditions.append("q.id IN (SELECT question_id FROM feedback WHERE rating = ?)")
            params += (feedback,)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        total = self._read(f"SELECT COUNT(*) FROM questions q {where}", params)[0][0]
        pages = max(1, math.ceil(total / PAGE_SIZE))
        page = min(max(1, page), pages)
        rows = self._read(
            f"SELECT q.* FROM questions q {where} ORDER BY q.started_at DESC, q.rowid DESC LIMIT ? OFFSET ?",
            params + (PAGE_SIZE, (page - 1) * PAGE_SIZE),
        )
        return {"items": [self._question_metrics(dict(r)) for r in rows], "page": page, "pages": pages, "total": total}

    def _question_metrics(self, q: Dict) -> Dict:
        calls = [dict(r) for r in self._read(
            "SELECT stage, round, model, input_tokens, output_tokens, reasoning_tokens, latency_ms, status, cost "
            "FROM llm_calls WHERE question_id = ? ORDER BY id", (q["id"],)
        )]
        # calls per try: the meaning step's tries, then each SQL round
        steps: Dict[str, int] = {}
        for c in calls:
            label = {"meaning": f"Meaning, try {c['round']}", "steps": f"Steps, try {c['round']}",
                     "sql": f"SQL, round {c['round']}", "answer": "Answer"}.get(c["stage"], c["stage"])
            steps[label] = steps.get(label, 0) + 1
        meaning_tries = len({c["round"] for c in calls if c["stage"] == "meaning"})
        step_tries = len({c["round"] for c in calls if c["stage"] == "steps"})
        sql_retries = sum(1 for c in calls if c["stage"] == "sql" and (c["round"] or 1) > 1)
        return {
            **q,
            "llm_calls": len(calls),
            "calls_per_try": [{"step": k, "calls": v} for k, v in steps.items()],
            "retries": max(0, meaning_tries - 1) + max(0, step_tries - 1) + sql_retries,
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "reasoning_tokens": sum(c["reasoning_tokens"] for c in calls),
            "llm_latency_ms": sum(c["latency_ms"] for c in calls),
            "cost": sum(c["cost"] for c in calls),
            "failed_calls": sum(1 for c in calls if c["status"] != "ok"),
            "feedback": self._feedback_for(q["id"]),
        }

    # ---------- issues: failures for the Failure Analyst agent (Version-11) ----------

    def add_issue(
        self, source: str, question: str, details: Dict, question_id: Optional[str] = None, run_id: Optional[str] = None,
    ) -> str:
        """
        A failure to investigate: a wrong eval answer, a user's report, or SQL that SQLite
        rejected. The same failure again (same source and question) updates the open issue
        and makes it new again, instead of adding a copy. An eval failure from a run (run_id)
        belongs to that run: the same question in another run is its own issue.
        """
        if source not in ISSUE_SOURCES:
            raise ValueError(f"source must be one of {ISSUE_SOURCES}")
        if not question.strip() and question_id:
            rows = self._read("SELECT question FROM questions WHERE id = ?", (question_id,))
            question = rows[0][0] if rows else ""
        key = issue_key(source, question_id or question, run_id)
        found = self._read("SELECT id FROM issues WHERE key = ?", (key,))
        details_json = json.dumps(details, ensure_ascii=False, default=str)
        if found:
            issue_id = found[0][0]
            self._write(
                "UPDATE issues SET details = ?, status = 'new', report = NULL, note = NULL, updated_at = ? WHERE id = ?",
                (details_json, now(), issue_id),
            )
            self._write("DELETE FROM agent_steps WHERE issue_id = ?", (issue_id,))
            return issue_id
        issue_id = uuid.uuid4().hex[:10]
        self._write(
            "INSERT INTO issues (id, source, key, question, question_id, details, run_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (issue_id, source, key, question.strip(), question_id, details_json, run_id, now(), now()),
        )
        return issue_id

    def issues(
        self, page: int = 1, source: str = "", status: str = "", scope: Sequence[str] = (), run: Optional[str] = None,
    ) -> Dict:
        """
        Issues, newest first, 10 per page, with counts per source and per status (for the filters).
        scope: the sources this view shows (the Dashboard: user reports and SQL errors; Evaluation:
        eval failures); the counts cover only those.
        run: only the issues of this evaluation run ("none": eval issues logged without a run).
        """
        conditions, params = [], ()
        scoped, scope_params = "", ()
        if scope:
            scoped = f"source IN ({', '.join('?' for _ in scope)})"
            scope_params = tuple(scope)
        if run:
            scoped = " AND ".join(filter(None, [scoped, "run_id IS NULL" if run == "none" else "run_id = ?"]))
            scope_params += () if run == "none" else (run,)
        if scoped:
            conditions.append(scoped)
            params += scope_params
        if source:
            conditions.append("source = ?")
            params += (source,)
        if status:
            conditions.append("status = ?")
            params += (status,)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        total = self._read(f"SELECT COUNT(*) FROM issues {where}", params)[0][0]
        pages = max(1, math.ceil(total / PAGE_SIZE))
        page = min(max(1, page), pages)
        rows = self._read(
            f"SELECT * FROM issues {where} ORDER BY updated_at DESC, rowid DESC LIMIT ? OFFSET ?",
            params + (PAGE_SIZE, (page - 1) * PAGE_SIZE),
        )
        return {
            "items": [self._issue_row(r) for r in rows], "page": page, "pages": pages, "total": total,
            "by_source": dict(self._read(f"SELECT source, COUNT(*) FROM issues {'WHERE ' + scoped if scoped else ''} GROUP BY source", scope_params)),
            "by_status": dict(self._read(f"SELECT status, COUNT(*) FROM issues {'WHERE ' + scoped if scoped else ''} GROUP BY status", scope_params)),
        }

    def calls_for(self, question_id: str) -> Dict:
        """LLM calls, tokens and cost recorded under one tag (an evaluation question, Version-14)."""
        row = self._read(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens + output_tokens), 0), COALESCE(SUM(cost), 0) "
            "FROM llm_calls WHERE question_id = ?", (question_id,)
        )[0]
        return {"llm_calls": row[0], "tokens": row[1], "cost": round(row[2], 6)}

    def issue_ids_for(self, source: str, questions: Sequence[str], run_id: Optional[str] = None) -> Dict[str, Dict]:
        """The issue (id, status) logged for each of these questions (in this run), keyed by question."""
        keys = {issue_key(source, q, run_id): q for q in questions}
        if not keys:
            return {}
        rows = self._read(
            f"SELECT key, id, status FROM issues WHERE key IN ({', '.join('?' for _ in keys)})", tuple(keys)
        )
        return {keys[r["key"]]: {"id": r["id"], "status": r["status"]} for r in rows}

    def issue(self, issue_id: str) -> Optional[Dict]:
        """One issue with everything the agent did on it, in order."""
        rows = self._read("SELECT * FROM issues WHERE id = ?", (issue_id,))
        if not rows:
            return None
        steps = self._read(
            "SELECT kind, title, detail, created_at FROM agent_steps WHERE issue_id = ? ORDER BY id", (issue_id,)
        )
        return {**self._issue_row(rows[0]), "steps": [dict(s) for s in steps]}

    def new_issue_ids(self, limit: int, scope: Sequence[str] = (), run: Optional[str] = None) -> List[str]:
        """The oldest issues nobody has investigated yet (only these sources, and this eval run, if given)."""
        scoped = f" AND source IN ({', '.join('?' for _ in scope)})" if scope else ""
        params = tuple(scope)
        if run:
            scoped += " AND run_id IS NULL" if run == "none" else " AND run_id = ?"
            params += () if run == "none" else (run,)
        return [r[0] for r in self._read(
            f"SELECT id FROM issues WHERE status = 'new'{scoped} ORDER BY created_at LIMIT ?", params + (limit,)
        )]

    def issue_counts_by_run(self) -> Dict[str, Dict[str, int]]:
        """Eval issues per run ("none": logged without a run): how many, and how many are new."""
        rows = self._read(
            "SELECT COALESCE(run_id, 'none') AS run, COUNT(*) AS total, SUM(status = 'new') AS new "
            "FROM issues WHERE source = 'eval' GROUP BY COALESCE(run_id, 'none')"
        )
        return {r["run"]: {"total": r["total"], "new": r["new"]} for r in rows}

    def link_issues_to_run(self, run_id: str, questions: Sequence[str], since: str) -> int:
        """
        Eval issues a run logged before issues had run ids: the same questions, logged during
        the run, without a run. Moves them to the run (once), so they appear under it.
        """
        moved = 0
        for question in questions:
            old, new = issue_key("eval", question), issue_key("eval", question, run_id)
            rows = self._read("SELECT id FROM issues WHERE key = ? AND run_id IS NULL AND updated_at >= ?", (old, since))
            if rows and not self._read("SELECT 1 FROM issues WHERE key = ?", (new,)):
                self._write("UPDATE issues SET run_id = ?, key = ? WHERE id = ?", (run_id, new, rows[0]["id"]))
                moved += 1
        return moved

    def start_investigation(self, issue_id: str) -> bool:
        """Marks an issue as being investigated, with a clean slate; False if it already is."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE issues SET status = 'investigating', report = NULL, note = NULL, agent_calls = 0, agent_cost = 0,"
                " updated_at = ? WHERE id = ? AND status != 'investigating'",
                (now(), issue_id),
            )
            if cur.rowcount:
                self._conn.execute("DELETE FROM agent_steps WHERE issue_id = ?", (issue_id,))
            self._conn.commit()
            return bool(cur.rowcount)

    def add_agent_step(self, issue_id: str, kind: str, title: str, detail: str = "") -> None:
        self._write(
            "INSERT INTO agent_steps (issue_id, kind, title, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (issue_id, kind, title, detail, now()),
        )

    def add_agent_usage(self, issue_id: str, cost: float) -> None:
        self._write("UPDATE issues SET agent_calls = agent_calls + 1, agent_cost = agent_cost + ? WHERE id = ?", (cost, issue_id))

    def set_issue(self, issue_id: str, status: str, report: Optional[Dict] = None, note: Optional[str] = None) -> None:
        """Sets the status (and the report or the developer's note, when given)."""
        if status not in ISSUE_STATUSES:
            raise ValueError(f"status must be one of {ISSUE_STATUSES}")
        sets, params = ["status = ?", "updated_at = ?"], [status, now()]
        if report is not None:
            sets.append("report = ?")
            params.append(json.dumps(report, ensure_ascii=False, default=str))
        if note is not None:
            sets.append("note = ?")
            params.append(note.strip() or None)
        self._write(f"UPDATE issues SET {', '.join(sets)} WHERE id = ?", tuple(params) + (issue_id,))

    @staticmethod
    def _issue_row(row: sqlite3.Row) -> Dict:
        item = dict(row)
        item.pop("key", None)
        item["details"] = json.loads(item["details"] or "{}")
        item["report"] = json.loads(item["report"]) if item["report"] else None
        return item

    def _feedback_for(self, question_id: str) -> Optional[Dict]:
        rows = self._read("SELECT rating, comment, details, created_at FROM feedback WHERE question_id = ?", (question_id,))
        if not rows:
            return None
        row = dict(rows[0])
        row["details"] = json.loads(row["details"] or "{}")
        return row


class Timer:
    """Measures a request's processing time and adds it to the question's total."""

    def __init__(self, store: "UsageStore"):
        self.store = store
        self.question_id: Optional[str] = None

    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.store.add_time(self.question_id, time.monotonic() - self.start)
        return False


# the app's store; tests replace it with one on a temporary file
usage_store = UsageStore()
