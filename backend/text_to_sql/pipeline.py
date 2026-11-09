import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional

from backend.config import (
    ANSWER_CACHE_SIZE, BLOCK_ESTIMATED_READS, MAX_CLARIFY_ROUNDS, MAX_STEP_ROUNDS, MAX_ESTIMATED_READS, MAX_LLM_CALLS_PER_QUESTION, MAX_RESULT_ROWS, PAGE_SIZE,
    QUERY_TIMEOUT_SECONDS, RESULT_CACHE_SIZE, SHARE_ROWS_WITH_LLM, SQL_CALLS_PER_ROUND, SQL_TEMPERATURE,
)
from backend.rag.encoders.text_encoder import get_text_encoder
from backend.text_to_sql.answer_cache import AnswerCache
from backend.text_to_sql.checks import (
    MISSING_VALUE, check_result, check_sql, describe_rows, merged_records, explain_disagreement, question_problems,
    result_signature, tables_in_sql,
)
from backend.text_to_sql.database import QueryTimeout, SQLiteDatabase
from backend.text_to_sql.results import ResultCache
from backend.text_to_sql.sql_errors import ErrorBook, normalize
from backend.text_to_sql.wording import plain_words, schema_names
from backend.text_to_sql.interpreter import MAX_OPTIONS, QuestionInterpreter
from backend.text_to_sql.planner import StepPlanner
from backend.text_to_sql.safety import write_message, write_problems, write_request
from backend.text_to_sql.schema_index import SchemaIndex
from backend.text_to_sql.sql_generator import CallBudget, CallLimitReached, SQLGenerator

def meaning_text(option: dict) -> str:
    """A meaning with its choices, as the UI sends it once confirmed."""
    return "\n".join([option["meaning"]] + [f"- {c}" for c in option["choices"]])


# how many tables the embedding search picks (bridge tables are added on top)
TOP_K_TABLES = 4
# shown on a meaning that sounds like a change to the data (it never blocks the question)
WRITE_WARNING = ("This sounds like it could change data. Scout only reads data: if the final query "
                 "would change anything, it's rejected before it runs.")


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

    def __init__(self, use_retrieval: bool = True, top_k: int = TOP_K_TABLES, data_dir: str = "data/text_to_sql"):
        """data_dir: the folder whose .sqlite files this Scout can see (an evaluation gives BIRD one database)."""
        self.data_dir = data_dir
        self.generator = SQLGenerator()
        self.interpreter = QuestionInterpreter()
        self.planner = StepPlanner()  # Version-10: the steps the user checks
        # Version-9: SQLite errors already seen: rules for the prompt, hints for retries
        self.errors = ErrorBook()
        self.generator.rules = self.errors.rules
        self.top_k = top_k
        self.use_retrieval = use_retrieval
        self._reload_lock = threading.Lock()
        self.results = ResultCache(RESULT_CACHE_SIZE)  # Version-8: recent results, for paging
        # Version-12: recent answers, reused when the same question comes back in other words
        self.answers = AnswerCache(ANSWER_CACHE_SIZE, lambda text: get_text_encoder().encode(text))
        self.reload()

    def reload(self) -> None:
        """
        Reads every .sqlite file, its descriptions and summaries, and re-embeds what changed.
        Called at startup, and after a summary is saved or a database is added (Version-6).
        The new state is built first and swapped in at once, so a question already running
        keeps using the old one.
        """
        with self._reload_lock:
            database = SQLiteDatabase(self.data_dir)
            tables = database.tables
            index = SchemaIndex(tables) if self.use_retrieval and tables else None
            self.database, self.tables, self.index = database, tables, index
            self.row_counts = {t["name"].lower(): t["row_count"] for t in tables}
            # a table, its description or a database changed: saved SQL may no longer fit
            self.answers.clear()

    def find_table(self, database: str, name: str) -> Optional[dict]:
        return next((t for t in self.tables if t["database"] == database and t["name"] == name), None)

    def select_tables(self, query_text: str) -> List[dict]:
        if self.index is None:
            return self.tables
        return self.index.retrieve(query_text, self.top_k)

    def interpret(
        self, query_text: str, history: Optional[List[dict]] = None, previous: Optional[dict] = None,
        question_id: Optional[str] = None, use_saved: bool = True,
    ) -> dict:
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
        # Version-12: the same question asked before (in any words): its meaning goes first
        # (an evaluation run passes use_saved=False: it must answer every question itself)
        saved = self.answers.match(query_text) if use_saved and not history and not previous else None
        try:
            options = self.interpreter.options(
                query_text, schema, context, history,
                tag={"question_id": question_id, "stage": "meaning", "round": len(history) + 1},
                known=saved["meaning"] if saved else "",
            )
        except Exception as e:
            return {"error": f"Could not understand the question: {e}", "llm_calls": 1}

        # what the user reads: everyday words, never database names (order_date -> order date)
        names = schema_names(self.tables)
        for option in options:
            option["meaning"] = plain_words(option["meaning"], names)
            option["choices"] = [plain_words(c, names) for c in option["choices"]]

        if saved:
            first, *choices = saved["meaning"].split("\n")
            reuse = {"meaning": first, "choices": [c.removeprefix("- ") for c in choices], "changes_data": False,
                     "reuse_id": saved["id"]}
            # drop any option that only rewords the saved meaning
            others = [o for o in options if not self.answers.repeats(saved["meaning"], meaning_text(o))]
            options = [reuse] + others[: MAX_OPTIONS - 1]

        # the LLM's wording never blocks a question: "Add together the values..." is arithmetic, and
        # only the final SQL shows what would really run (it's checked by the parser before running).
        # A meaning that sounds like a change to the data only gets a warning.
        for option in options:
            if option["changes_data"] or write_request(option["meaning"]):
                option["warning"] = WRITE_WARNING
        return {
            "options": options,
            "round": len(history) + 1,
            "max_rounds": MAX_CLARIFY_ROUNDS,
            "tables": [table["name"] for table in tables],
            "llm_calls": 1,
        }

    def plan(
        self, query_text: str, meaning: str, previous: Optional[dict] = None, question_id: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> dict:
        """
        Version-10: the steps for the confirmed meaning, for the user to check (1 LLM call).
        history: earlier tries with the user's comments on wrong steps; only this step re-runs.
        """
        history = history or []
        error = self._question_error(query_text)
        if error:
            return {"error": error, "llm_calls": 0}
        # the user's own words are checked; the confirmed meaning (LLM wording) never blocks
        reason = write_request(query_text)
        if reason:
            return self._blocked(write_message(reason), llm_calls=0)
        if len(history) >= MAX_STEP_ROUNDS:
            return {
                "error": f"The steps still aren't right after {MAX_STEP_ROUNDS} tries. Please rephrase the question.",
                "llm_calls": 0,
            }

        tables, schema, context, _ = self._prepare(query_text, previous)
        try:
            steps = self.planner.steps(
                query_text, meaning, schema, context, history,
                tag={"question_id": question_id, "stage": "steps", "round": len(history) + 1},
            )
        except Exception as e:
            return {"error": f"Could not work out the steps: {e}", "llm_calls": 1}
        names = schema_names(self.tables)
        return {
            # everyday words, never database names
            "steps": [plain_words(step, names) for step in steps],
            "round": len(history) + 1,
            "max_rounds": MAX_STEP_ROUNDS,
            "tables": [table["name"] for table in tables],
            "llm_calls": 1,
        }

    def run(
        self, query_text: str, meaning: str = "", previous: Optional[dict] = None, question_id: Optional[str] = None,
        steps: Optional[List[str]] = None, save: bool = True,
    ) -> dict:
        error = self._question_error(query_text)
        if error:
            return {"error": error, "llm_calls": 0}

        # safety: the user's words are checked again here, since /query can be called without
        # /interpret; the meaning (LLM wording) never blocks: the SQL itself is checked before it runs
        reason = write_request(query_text)
        if reason:
            return self._blocked(write_message(reason), llm_calls=0)

        # 1) pick the relevant tables, 2) context: what "now" is, and what dates the data covers
        tables, schema, context, now = self._prepare(query_text, previous)
        table_names = [table["name"] for table in tables]

        # 3) SQL in rounds, with a hard stop on LLM calls
        budget = CallBudget(MAX_LLM_CALLS_PER_QUESTION)
        attempts, chosen, confidence, feedback, feedback_sent = [], None, None, "", []
        stopped_early = False

        while budget.left > 0 and chosen is None:
            round_no = len(attempts) // SQL_CALLS_PER_ROUND + 1
            calls = min(SQL_CALLS_PER_ROUND, budget.left)
            with ThreadPoolExecutor(max_workers=calls) as pool:
                futures = [
                    pool.submit(
                        self._attempt, query_text, meaning, schema, context, feedback, budget, tables, now, round_no,
                        {"question_id": question_id, "stage": "sql", "round": round_no}, steps,
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

            # Version-9: look up every SQLite error in the error book, and log it
            for a in new:
                if a.get("error"):
                    known = self.errors.match(a["error"], a["sql"] or "")
                    a["known_error"] = known["id"] if known else None
                    a["hint"] = known["hint"] if known else None
                    self.errors.log({
                        "type": "error", "question": query_text, "meaning": meaning, "round": round_no,
                        "sql": a["sql"], "error": a["error"], "normalized": normalize(a["error"]),
                        "known_id": a["known_error"],
                    })

            # every round-1 try hit the same error and nobody has a fix for it yet:
            # round 2 would only repeat it, so stop and save the LLM calls
            if (
                round_no == 1 and new and all(a.get("error") for a in new)
                and not any(a.get("hint") for a in new)
                and len({normalize(a["error"]) for a in new}) == 1
            ):
                stopped_early = True
                break

            valid = [a for a in attempts if a["status"] == "ok"]
            if round_no == 1:
                group = self._largest_agreeing_group([a for a in new if a["status"] == "ok"])
                if len(group) >= 2:
                    chosen = group[-1]
                    # Version-22, the same rule in every flow: every try that answered agrees
                    confidence = "high" if len(group) == sum(a["status"] == "ok" for a in new) else "medium"
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

        self._log_outcome(query_text, meaning, attempts, chosen is not None, budget.used, stopped_early)

        if chosen is None:
            last = next((a for a in reversed(attempts) if a["sql"]), None)
            reasons = attempts[-1]["problems"] if attempts else []
            return {
                "sql": last["sql"] if last else None,
                "tables": table_names,
                "error": self._failure_message(attempts, stopped_early),
                "failure_reason": " ".join(reasons),
                # the confirmed meaning asks for a value the data doesn't have: the SQL step can't fix
                # that (it must follow the meaning), so the meaning is asked again with the real values
                "rethink_meaning": self._missing_values(attempts),
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
        # Version-12: saved for reuse (unless it's a follow-up, which depends on the earlier answer)
        if save and not (previous or {}).get("question"):
            self.answers.add(query_text, meaning, steps or [], chosen["sql"], question_id, confidence)

        return {
            "sql": chosen["sql"],
            "meaning": meaning or None,
            "steps": steps or [],
            # the tables the answer really used, read from its SQL (not every table offered to the LLM)
            "tables": tables_in_sql(chosen["sql"], [t["name"] for t in self.tables]) or table_names,
            "tables_offered": table_names,
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

    def reuse(self, entry_id: str, question_id: Optional[str] = None) -> dict:
        """
        Version-12: answers from a saved answer the user picked as the meaning. Its SQL runs
        again (fresh rows, same safety and time checks), with no LLM calls. If it's gone or
        no longer runs, {"fallback": True}: the usual steps and SQL take over.
        """
        entry = self.answers.get(entry_id)
        if entry is None or write_problems(entry["sql"]):
            return {"fallback": True, "llm_calls": 0}
        try:
            columns, rows, more = self.database.query(entry["sql"])
        except Exception:
            return {"fallback": True, "llm_calls": 0}
        self.answers.link(entry_id, question_id)
        result_id = self.results.put(entry["sql"], columns, rows, more)
        warnings = [f"Showing the first {MAX_RESULT_ROWS} rows; there are more. Narrow the question to see everything."] if more else []
        return {
            "sql": entry["sql"],
            "meaning": entry["meaning"],
            "steps": entry["steps"],
            "tables": tables_in_sql(entry["sql"], [t["name"] for t in self.tables]),
            "columns": columns,
            "rows": rows[:PAGE_SIZE],
            "row_count": len(rows),
            "more": more,
            "page_size": PAGE_SIZE,
            "result_id": result_id,
            "confidence": entry["confidence"],
            "warnings": warnings,
            "llm_calls": 0,
            "reused": True,
        }

    @staticmethod
    def _missing_values(attempts: List[dict]) -> str:
        """The checks' "<column> has no value ..." problems, as a note for the meaning step ("" if none)."""
        found = list(dict.fromkeys(p for a in attempts for p in a["problems"] if MISSING_VALUE.search(p)))
        if not found:
            return ""
        return ("Scout checked the data: " + " ".join(found)
                + " Use the values the data actually stores.")

    @staticmethod
    def _failure_message(attempts: List[dict], stopped_early: bool) -> str:
        """A plain-English failure the user can act on; the raw errors stay in the attempts."""
        errors = [normalize(a["error"]) for a in attempts if a.get("error")]
        tries = len(attempts)
        if errors and len(errors) == tries and len(set(errors)) == 1:
            message = (
                f"Scout couldn't build a working query for this question: SQLite rejected all {tries} "
                f"tries for the same reason ({errors[0]})."
            )
            if stopped_early:
                message += " It stopped after the first round to save LLM calls, since there's no known fix for this error yet."
        else:
            reason = Counter(errors).most_common(1)[0][0] if errors else "no valid query"
            message = f"Scout couldn't build a working query for this question after {tries} tries ({reason})."
        return message + " Try simpler wording, or ask for fewer things at once."

    def _log_outcome(
        self, question: str, meaning: str, attempts: List[dict], fixed: bool, llm_calls: int, stopped_early: bool
    ) -> None:
        """How a question that hit SQL errors ended: did the hints help? (for the report)"""
        errored = [a for a in attempts if a.get("error")]
        if not errored:
            return
        self.errors.log({
            "type": "outcome", "question": question, "meaning": meaning, "fixed": fixed, "llm_calls": llm_calls,
            "stopped_early": stopped_early,
            "errors": sorted({normalize(a["error"]) for a in errored}),
            "known_ids": sorted({a["known_error"] for a in errored if a.get("known_error")}),
            "first_round_errors": sum(1 for a in errored if a["round"] == 1),
        })

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
        The tables for this question, their schema (from the catalog), and the context: the date,
        the stored values the question mentions (value index), and for a follow-up ("now sort
        that by city") the earlier question and SQL.
        """
        previous = {k: str((previous or {}).get(k, ""))[:2000] for k in ("question", "meaning", "sql")}
        # a follow-up is about the same data, so the earlier question helps pick the tables
        tables = self.select_tables(f"{query_text} {previous['question']}".strip())
        schema = "\n\n".join(table["schema"] for table in tables)
        print(f"[Pipeline] Using {len(tables)}/{len(self.tables)} tables: {[t['name'] for t in tables]}")

        now = datetime.now()
        context = f"Current date and time: {now:%Y-%m-%d %H:%M} ({now:%A})."
        # (each date column's format and range is in the schema now: Version-20 catalog)
        # Version-20: stored values the question mentions, spelled exactly as in the data
        if SHARE_ROWS_WITH_LLM:
            matches = self.database.values.match(query_text, [(t["database"], t["name"]) for t in tables])
            if matches:
                context += ("\nValues stored in the data that match words in the question (spelled exactly "
                            "as stored; use them if they're what the question means):\n"
                            + "\n".join(f"- {m['table']}.{m['column']} = '{m['value']}'" for m in matches))
        if previous["question"]:
            context += (
                "\nThis question follows up on an earlier one; read it in that light "
                "(\"that\", \"those\", \"now sort by ...\" refer to the earlier answer):"
                f"\n- Earlier question: {previous['question']}"
                + (f"\n- What it meant: {previous['meaning']}" if previous["meaning"] else "")
                + (f"\n- SQL used:\n{previous['sql']}" if previous["sql"] else "")
            )
        return tables, schema, context, now

    def _attempt(
        self, question, meaning, schema, context, feedback, budget, tables, now, round_no, tag=None, steps=None
    ) -> dict:
        """One LLM call, then the before-run checks, then run it."""
        attempt = {"round": round_no, "sql": None, "status": "failed", "problems": [], "warnings": []}
        try:
            attempt["sql"] = self.generator.generate(
                question, schema, budget, context=context, feedback=feedback, meaning=meaning,
                temperature=SQL_TEMPERATURE, tag=tag, steps=steps,
            )
        except CallLimitReached as e:
            attempt["problems"] = [str(e)]
            return attempt
        except Exception as e:
            attempt["problems"] = [f"Could not generate SQL: {e}"]
            return attempt

        return self.check_and_run(attempt, tables, now)

    def check_and_run(self, attempt: dict, tables: List[dict], now) -> dict:
        """
        The before-run checks on attempt["sql"], then run it (Version-22: also used by the Full and
        Adaptive flows, so every flow's SQL passes exactly the same checks).
        """
        sql = attempt["sql"]

        # safety first: read with the SQL parser; anything but one plain read query is never run
        problems = write_problems(sql)
        if problems:
            attempt.update(status="unsafe", problems=problems)
            return attempt

        try:
            self.database.explain(sql)
        except Exception as e:
            attempt.update(status="rejected", problems=[f"Invalid SQL: {e}"], error=str(e))
            return attempt

        errors, warnings = check_sql(sql, tables, now.date(), self.database.values)
        errors += merged_records(sql, tables, self.database.run)  # Version-20: grouped by a shared name
        if errors:
            attempt.update(status="rejected", problems=errors)
            return attempt

        # performance, before running: how many rows could this read?
        try:
            reads = self.database.estimate_reads(sql, self.row_counts)
        except Exception:
            reads = 0  # no plan: the time limit still applies
        heavy_warning = []
        if reads > MAX_ESTIMATED_READS:
            if BLOCK_ESTIMATED_READS:
                attempt.update(status="too_big", reads=reads, problems=[
                    f"could read about {reads:,} rows (limit {MAX_ESTIMATED_READS:,})"
                ])
                return attempt
            # Version-20: the estimate only warns; the time limit stops a query that's really too slow
            heavy_warning = [f"The query was estimated to read about {reads:,} rows; it ran within the time limit."]

        try:
            columns, rows, more = self.database.query(sql)
        except QueryTimeout as e:
            attempt.update(status="timeout", problems=[str(e)])
            return attempt
        except Exception as e:
            attempt.update(status="failed", problems=[f"Query failed: {e}"], error=str(e))
            return attempt

        attempt.update(
            status="ok", columns=columns, rows=rows, more=more, warnings=warnings + heavy_warning,
            signature=result_signature(rows),
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
        # Version-9: for errors in the error book, say exactly how to fix them in SQLite
        hints = list(dict.fromkeys(a["hint"] for a in attempts if a.get("hint")))
        if hints:
            lines.append("How to fix this in SQLite:\n" + "\n".join(f"- {hint}" for hint in hints))
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
