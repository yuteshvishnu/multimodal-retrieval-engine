from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional

from backend.config import (
    MAX_CLARIFY_ROUNDS, MAX_LLM_CALLS_PER_QUESTION, SQL_CALLS_PER_ROUND, SQL_TEMPERATURE,
)
from backend.text_to_sql.checks import (
    check_result, check_sql, date_ranges, explain_disagreement, question_problems, result_signature,
)
from backend.text_to_sql.database import SQLiteDatabase
from backend.text_to_sql.interpreter import QuestionInterpreter
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

    use_retrieval=False sends the full schema instead (Version-2 behaviour), for comparison.
    """

    def __init__(self, use_retrieval: bool = True, top_k: int = TOP_K_TABLES):
        self.database = SQLiteDatabase()
        self.generator = SQLGenerator()
        self.interpreter = QuestionInterpreter()
        self.top_k = top_k

        # tables are read once at startup; restart the server after adding .sqlite files
        self.tables = self.database.tables
        self.index = SchemaIndex(self.tables) if use_retrieval and self.tables else None

    def select_tables(self, query_text: str) -> List[dict]:
        if self.index is None:
            return self.tables
        return self.index.retrieve(query_text, self.top_k)

    def interpret(self, query_text: str, history: Optional[List[dict]] = None) -> dict:
        """
        Up to 3 meanings of the question for the user to pick from.
        history: earlier sets of options the user rejected, each {"options": [...], "comment": "..."}.
        """
        history = history or []
        error = self._question_error(query_text)
        if error:
            return {"error": error, "llm_calls": 0}
        if len(history) >= MAX_CLARIFY_ROUNDS:
            return {
                "error": f"Still unclear after {MAX_CLARIFY_ROUNDS} tries. Please rephrase the question.",
                "llm_calls": 0,
            }

        tables, schema, context, _ = self._prepare(query_text)
        try:
            options = self.interpreter.options(query_text, schema, context, history)
        except Exception as e:
            return {"error": f"Could not understand the question: {e}", "llm_calls": 1}
        return {
            "options": options,
            "round": len(history) + 1,
            "max_rounds": MAX_CLARIFY_ROUNDS,
            "tables": [table["name"] for table in tables],
            "llm_calls": 1,
        }

    def run(self, query_text: str, meaning: str = "") -> dict:
        error = self._question_error(query_text)
        if error:
            return {"error": error, "llm_calls": 0}

        # 1) pick the relevant tables, 2) context: what "now" is, and what dates the data covers
        tables, schema, context, now = self._prepare(query_text)
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

        return {
            "sql": chosen["sql"],
            "meaning": meaning or None,
            "tables": table_names,
            "columns": chosen["columns"],
            "rows": chosen["rows"],
            "confidence": confidence,
            "warnings": list(dict.fromkeys(warnings)),
            "llm_calls": budget.used,
            "attempts": summary,
            "feedback": "\n\n".join(f for f in feedback_sent if f),
        }

    def _question_error(self, query_text: str) -> Optional[str]:
        """Problems found before any LLM call: no question, no tables, an impossible request."""
        if not query_text.strip():
            return "Please enter a question."
        if not self.tables:
            return "No tables found. Add .sqlite files to the data folder and restart the server."
        problems = question_problems(query_text)
        return " ".join(problems) if problems else None

    def _prepare(self, query_text: str):
        """The tables for this question, their schema, and the date context."""
        tables = self.select_tables(query_text)
        schema = "\n\n".join(table["schema"] for table in tables)
        print(f"[Pipeline] Using {len(tables)}/{len(self.tables)} tables: {[t['name'] for t in tables]}")

        now = datetime.now()
        context = f"Current date and time: {now:%Y-%m-%d %H:%M} ({now:%A})."
        ranges = date_ranges(tables)
        if ranges:
            context += "\nDates in the data:\n" + "\n".join(f"- {line}" for line in ranges)
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
        try:
            self.database.explain(sql)
        except Exception as e:
            attempt.update(status="rejected", problems=[f"Invalid SQL: {e}"])
            return attempt

        errors, warnings = check_sql(sql, tables, now.date())
        if errors:
            attempt.update(status="rejected", problems=errors)
            return attempt

        try:
            columns, rows = self.database.run(sql)
        except Exception as e:
            attempt.update(status="failed", problems=[f"Query failed: {e}"])
            return attempt

        attempt.update(
            status="ok", columns=columns, rows=rows, warnings=warnings, signature=result_signature(rows)
        )
        return attempt

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
                + "; ".join(f"attempt {i} returned {a['rows'][:3]}" for i, a in enumerate(ok, 1))
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
