import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional

from backend.config import (
    MAX_CLARIFY_ROUNDS, MAX_ESTIMATED_READS, MAX_LLM_CALLS_PER_QUESTION, MAX_RESULT_ROWS, PAGE_SIZE,
    QUERY_TIMEOUT_SECONDS, RESULT_CACHE_SIZE, SQL_CALLS_PER_ROUND, SQL_TEMPERATURE,
)
from backend.text_to_sql.checks import (
    check_result, check_sql, date_ranges, describe_rows, explain_disagreement, question_problems, result_signature,
)
from backend.text_to_sql.database import QueryTimeout, SQLiteDatabase
from backend.text_to_sql.results import ResultCache
from backend.text_to_sql.interpreter import QuestionInterpreter
from backend.text_to_sql.safety import write_message, write_problems, write_request
from backend.text_to_sql.schema_index import SchemaIndex
from backend.text_to_sql.sql_generator import CallBudget, CallLimitReached, SQLGenerator

# how many tables the embedding search picks (bridge tables are added on top)
TOP_K_TABLES = 4


class TextToSQLPipeline:
    """
    Question -> pick relevant tables (embeddings) -> confirm the meaning with the user (Version-5)
    -> SQL (LLM, in rounds) -> checks -> result table.

    interpret(): up to 3 meanings for the user to pick from (1 LLM call per set of options).
    run(meaning=...): writes SQL that follows the meaning the user picked.

    SQL rounds (Version-4), with a hard stop of MAX_LLM_CALLS_PER_QUESTION calls
    (SQL_CALLS_PER_ROUND calls per round, both set in backend/config.py):
      round 1: each SQL is checked, then run. At least two with the same result -> done
               (high confidence if all agree, medium if only some do).
      round 2: told exactly how the attempts differed, with the descriptions of the columns
               involved. Any two answers agree -> done (medium). Otherwise -> the latest
               valid answer (low confidence). Stop.

    use_retrieval=False sends the full schema instead (versions 1-2 behaviour), for comparison.
    """

    def __init__(self, use_retrieval: bool = True, top_k: int = TOP_K_TABLES):
        self.generator = SQLGenerator()
        self.interpreter = QuestionInterpreter()
        self.top_k = top_k
        self.use_retrieval = use_retrieval
        self._reload_lock = threading.Lock()
        self.results = ResultCache(RESULT_CACHE_SIZE)  # Version-8: recent results, for paging
        self.reload()

    def reload(self) -> None:
        """
        Reads every .sqlite file, its descriptions and summaries, and re-embeds what changed.
        Called at startup, and after a summary is saved or a database is added (Version-6).
        The new state is built first and swapped in at once, so a question already running
        keeps using the old one.
        """
        with self._reload_lock:
            database = SQLiteDatabase()
            tables = database.tables
            index = SchemaIndex(tables) if self.use_retrieval and tables else None
            self.database, self.tables, self.index = database, tables, index
            self.row_counts = {t["name"].lower(): t["row_count"] for t in tables}

    def find_table(self, database: str, name: str) -> Optional[dict]:
        return next((t for t in self.tables if t["database"] == database and t["name"] == name), None)

    def select_tables(self, query_text: str) -> List[dict]:
        if self.index is None:
            return self.tables
        return self.index.retrieve(query_text, self.top_k)

    def interpret(self, query_text: str, history: Optional[List[dict]] = None, previous: Optional[dict] = None) -> dict:
        """
        Up to 3 meanings of the question for the user to pick from.
        history: earlier sets of options the user rejected, each {"options": [...], "comment": "..."}.
        previous: the answer this question follows up on, {"question", "meaning", "sql"}, or None.
        """
        history = history or []
        error = self._question_error(query_text)
        if error:
            return {"error": error, "llm_calls": 0}

        # safety, before any LLM call: the question asks to change data, or is write SQL
        reason = write_request(query_text)
        if reason:
            return self._blocked(write_message(reason), llm_calls=0)
        if len(history) >= MAX_CLARIFY_ROUNDS:
            return {
                "error": f"Still unclear after {MAX_CLARIFY_ROUNDS} tries. Please rephrase the question.",
                "llm_calls": 0,
            }

        tables, schema, context, _ = self._prepare(query_text, previous)
        try:
            options = self.interpreter.options(query_text, schema, context, history)
        except Exception as e:
            return {"error": f"Could not understand the question: {e}", "llm_calls": 1}

        # safety, after the LLM read the question: any meaning that would change data
        for option in options:
            reason = "changes data" if option["changes_data"] else write_request(option["meaning"])
            if reason:
                return self._blocked(
                    write_message(reason).replace("This question", f"This question (understood as: “{option['meaning']}”)"),
                    llm_calls=1,
                )
        return {
            "options": options,
            "round": len(history) + 1,
            "max_rounds": MAX_CLARIFY_ROUNDS,
            "tables": [table["name"] for table in tables],
            "llm_calls": 1,
        }

    def run(self, query_text: str, meaning: str = "", previous: Optional[dict] = None) -> dict:
        error = self._question_error(query_text)
        if error:
            return {"error": error, "llm_calls": 0}

        # safety: checked again here, since /query can be called without /interpret
        for text, where in ((query_text, "This question"), (meaning, "The confirmed meaning")):
            reason = write_request(text) if text else None
            if reason:
                return self._blocked(write_message(reason, where), llm_calls=0)

        # 1) pick the relevant tables, 2) context: what "now" is, and what dates the data covers
        tables, schema, context, now = self._prepare(query_text, previous)
        table_names = [table["name"] for table in tables]

        # 3) SQL in rounds, with a hard stop on LLM calls
        budget = CallBudget(MAX_LLM_CALLS_PER_QUESTION)
        attempts, chosen, confidence, feedback, feedback_sent = [], None, None, "", []

        while budget.left > 0 and chosen is None:
            round_no = len(attempts) // SQL_CALLS_PER_ROUND + 1
            calls = min(SQL_CALLS_PER_ROUND, budget.left)
            with ThreadPoolExecutor(max_workers=calls) as pool:
                futures = [
                    pool.submit(
                        self._attempt, query_text, meaning, schema, context, feedback, budget, tables, now, round_no
                    )
                    for _ in range(calls)
                ]
                new = [f.result() for f in futures]
            attempts += new

            # safety: SQL that could write stops everything at once, with no more LLM calls
            unsafe = next((a for a in new if a["status"] == "unsafe"), None)
            if unsafe:
                result = self._blocked(
                    write_message(", ".join(unsafe["problems"]), "The SQL written for this question"),
                    llm_calls=budget.used,
                )
                result.update(sql=unsafe["sql"], tables=table_names, attempts=[
                    {"round": a["round"], "sql": a["sql"], "status": a["status"], "problems": a["problems"], "rows": []}
                    for a in attempts
                ])
                return result

            # performance: when no try in this round worked because they were too big or too
            # slow, stop instead of spending more LLM calls on similar heavy SQL
            heavy = [a for a in new if a["status"] in ("too_big", "timeout")]
            if heavy and not any(a["status"] == "ok" for a in new):
                result = self._too_heavy(heavy, budget.used)
                result.update(sql=heavy[-1]["sql"], tables=table_names, attempts=[
                    {"round": a["round"], "sql": a["sql"], "status": a["status"], "problems": a["problems"], "rows": []}
                    for a in attempts
                ])
                return result

            valid = [a for a in attempts if a["status"] == "ok"]
            if round_no == 1:
                group = self._largest_agreeing_group([a for a in new if a["status"] == "ok"])
                if len(group) >= 2:
                    chosen = group[-1]
                    confidence = "high" if len(group) == len(new) else "medium"
            else:
                # any two answers (from either round) with the same result
                group = self._largest_agreeing_group(valid)
                if len(group) >= 2:
                    chosen, confidence = group[-1], "medium"
                elif valid:
                    chosen, confidence = valid[-1], "low"
                break  # round 2 is the last round, whatever happened

            feedback = self._feedback(new, tables)
            feedback_sent.append(feedback)
            if budget.left == 0 and chosen is None and valid:
                chosen, confidence = valid[-1], "low"

        summary = [
            {
                "round": a["round"], "sql": a["sql"], "status": a["status"], "problems": a["problems"],
                "rows": a.get("rows", [])[:3],
            }
            for a in attempts
        ]
        print(f"[Pipeline] {budget.used} LLM calls, confidence: {confidence}")

        if chosen is None:
            last = next((a for a in reversed(attempts) if a["sql"]), None)
            reasons = attempts[-1]["problems"] if attempts else []
            return {
                "sql": last["sql"] if last else None,
                "tables": table_names,
                "error": "Could not produce a valid query. " + " ".join(reasons),
                "llm_calls": budget.used,
                "attempts": summary,
                "feedback": "\n\n".join(f for f in feedback_sent if f),
            }

        # 4) checks on the chosen result
        warnings = chosen["warnings"] + check_result(
            query_text, chosen["sql"], chosen["columns"], chosen["rows"], tables
        )
        if confidence == "low":
            warnings.insert(0, "The model's answers disagreed with each other; this is its latest answer.")
        if chosen["more"]:
            warnings.insert(0, f"Showing the first {MAX_RESULT_ROWS} rows; there are more. Narrow the question to see everything.")

        # Version-8: keep the result on the server; the UI gets page 1 and asks for the rest
        result_id = self.results.put(chosen["sql"], chosen["columns"], chosen["rows"], chosen["more"])

        return {
            "sql": chosen["sql"],
            "meaning": meaning or None,
            "tables": table_names,
            "columns": chosen["columns"],
            "rows": chosen["rows"][:PAGE_SIZE],
            "row_count": len(chosen["rows"]),
            "more": chosen["more"],
            "page_size": PAGE_SIZE,
            "result_id": result_id,
            "confidence": confidence,
            "warnings": list(dict.fromkeys(warnings)),
            "llm_calls": budget.used,
            "attempts": summary,
            "feedback": "\n\n".join(f for f in feedback_sent if f),
        }

    @staticmethod
    def _too_heavy(attempts: List[dict], llm_calls: int) -> dict:
        """A question stopped because its SQL was too big or too slow (Version-8)."""
        too_big = [a for a in attempts if a["status"] == "too_big"]
        if too_big:
            reads = max(a["reads"] for a in too_big)
            message = (
                f"This query could read about {reads:,} rows (the limit is {MAX_ESTIMATED_READS:,}), so it wasn't run. "
                "Narrow it down, e.g. add a date range, a store or a category."
            )
        else:
            message = (
                f"This query took too long (over {QUERY_TIMEOUT_SECONDS} s), so it was stopped. "
                "Narrow it down, e.g. add a date range, a store or a category."
            )
        print(f"[Performance] Stopped: {message}")
        return {"error": message, "limit": message, "llm_calls": llm_calls}

    @staticmethod
    def _blocked(message: str, llm_calls: int) -> dict:
        """A request stopped because it could change data (Version-7)."""
        print(f"[Safety] Blocked: {message}")
        return {"error": message, "safety": message, "llm_calls": llm_calls}

    def _question_error(self, query_text: str) -> Optional[str]:
        """Problems found before any LLM call: no question, no tables, an impossible request."""
        if not query_text.strip():
            return "Please enter a question."
        if not self.tables:
            return "No tables found. Add .sqlite files to the data folder and restart the server."
        problems = question_problems(query_text)
        return " ".join(problems) if problems else None

    def _prepare(self, query_text: str, previous: Optional[dict] = None):
        """
        The tables for this question, their schema, and the context: the date, the data's
        date range, and for a follow-up ("now sort that by city") the earlier question and SQL.
        """
        previous = {k: str((previous or {}).get(k, ""))[:2000] for k in ("question", "meaning", "sql")}
        # a follow-up is about the same data, so the earlier question helps pick the tables
        tables = self.select_tables(f"{query_text} {previous['question']}".strip())
        schema = "\n\n".join(table["schema"] for table in tables)
        print(f"[Pipeline] Using {len(tables)}/{len(self.tables)} tables: {[t['name'] for t in tables]}")

        now = datetime.now()
        context = f"Current date and time: {now:%Y-%m-%d %H:%M} ({now:%A})."
        ranges = date_ranges(tables)
        if ranges:
            context += "\nDates in the data:\n" + "\n".join(f"- {line}" for line in ranges)
        if previous["question"]:
            context += (
                "\nThis question follows up on an earlier one; read it in that light "
                "(\"that\", \"those\", \"now sort by ...\" refer to the earlier answer):"
                f"\n- Earlier question: {previous['question']}"
                + (f"\n- What it meant: {previous['meaning']}" if previous["meaning"] else "")
                + (f"\n- SQL used:\n{previous['sql']}" if previous["sql"] else "")
            )
        return tables, schema, context, now

    def _attempt(self, question, meaning, schema, context, feedback, budget, tables, now, round_no) -> dict:
        """One LLM call, then the before-run checks, then run it."""
        attempt = {"round": round_no, "sql": None, "status": "failed", "problems": [], "warnings": []}
        try:
            attempt["sql"] = self.generator.generate(
                question, schema, budget, context=context, feedback=feedback, meaning=meaning,
                temperature=SQL_TEMPERATURE,
            )
        except CallLimitReached as e:
            attempt["problems"] = [str(e)]
            return attempt
        except Exception as e:
            attempt["problems"] = [f"Could not generate SQL: {e}"]
            return attempt

        sql = attempt["sql"]

        # safety first: read with the SQL parser; anything but one plain read query is never run
        problems = write_problems(sql)
        if problems:
            attempt.update(status="unsafe", problems=problems)
            return attempt

        try:
            self.database.explain(sql)
        except Exception as e:
            attempt.update(status="rejected", problems=[f"Invalid SQL: {e}"])
            return attempt

        errors, warnings = check_sql(sql, tables, now.date())
        if errors:
            attempt.update(status="rejected", problems=errors)
            return attempt

        # performance, before running: how many rows could this read?
        try:
            reads = self.database.estimate_reads(sql, self.row_counts)
        except Exception:
            reads = 0  # no plan: the time limit still applies
        if reads > MAX_ESTIMATED_READS:
            attempt.update(status="too_big", reads=reads, problems=[
                f"could read about {reads:,} rows (limit {MAX_ESTIMATED_READS:,})"
            ])
            return attempt

        try:
            columns, rows, more = self.database.query(sql)
        except QueryTimeout as e:
            attempt.update(status="timeout", problems=[str(e)])
            return attempt
        except Exception as e:
            attempt.update(status="failed", problems=[f"Query failed: {e}"])
            return attempt

        attempt.update(
            status="ok", columns=columns, rows=rows, more=more, warnings=warnings, signature=result_signature(rows)
        )
        return attempt

    def result_page(self, result_id: str, page: int) -> dict:
        """
        One page of a saved result (Version-8). If the result was dropped from the cache,
        its SQL is run once more (with the same safety and time checks) and cached again.
        """
        result = self.results.get(result_id)
        if result is None:
            sql = self.results.sql_for(result_id)
            if sql is None:
                return {"error": "This result has expired. Please ask the question again."}
            if write_problems(sql):
                return {"error": "This result can't be loaded again."}
            try:
                columns, rows, more = self.database.query(sql)
            except QueryTimeout:
                return {"error": f"Reloading this result took longer than {QUERY_TIMEOUT_SECONDS} s. Please ask again."}
            self.results.put(sql, columns, rows, more, result_id=result_id)
            result = self.results.get(result_id)

        rows = result["rows"]
        pages = max(1, -(-len(rows) // PAGE_SIZE))
        page = min(max(1, page), pages)
        start = (page - 1) * PAGE_SIZE
        return {
            "columns": result["columns"], "rows": rows[start:start + PAGE_SIZE], "page": page, "pages": pages,
            "row_count": len(rows), "more": result["more"], "page_size": PAGE_SIZE,
        }

    def _feedback(self, attempts: List[dict], tables: List[dict]) -> str:
        """
        What to tell the next round: the problems found by the checks, and, if attempts
        disagreed, exactly where their SQL differs plus the descriptions of those columns.
        """
        lines = list(dict.fromkeys(f"- {problem}" for a in attempts for problem in a["problems"]))
        ok = [a for a in attempts if a["status"] == "ok"]
        if len({a["signature"] for a in ok}) > 1:
            lines.append(
                explain_disagreement(ok, tables)
                or "The earlier attempts returned different results: "
                + "; ".join(f"attempt {i} returned {describe_rows(a['rows'])}" for i, a in enumerate(ok, 1))
            )
        return "\n".join(lines)

    @staticmethod
    def _largest_agreeing_group(attempts: List[dict]) -> List[dict]:
        """The biggest set of attempts with the same result (ties: the one with the latest attempt)."""
        groups = {}
        for a in attempts:
            groups.setdefault(a["signature"], []).append(a)
        if not groups:
            return []
        return max(groups.values(), key=lambda g: (len(g), attempts.index(g[-1])))
