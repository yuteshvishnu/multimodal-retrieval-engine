"""
The Failure Analyst's tools: everything it can do, and nothing that writes.

    list_tables     every table, its rows and what it holds
    describe_table  a table's columns with their descriptions
    column_values   the most common values of one column (spellings, formats, case)
    run_query       one small read query, to test an idea
    known_errors    the error book's rules (to avoid suggesting an entry that exists)

There is no tool that changes anything, and run_query goes through the same checks as
Scout's own SQL: the allow-list (one plain read query), the size estimate, the time limit,
and the read-only connection. The agent only ever sees what these return.
"""
import json
from typing import Dict, List

from backend.config import MAX_ESTIMATED_READS
from backend.text_to_sql.safety import write_problems

RESULT_ROWS = 20       # rows the agent sees from one query
MAX_RESULT_CHARS = 3000

SPECS = [
    {"type": "function", "function": {
        "name": "list_tables",
        "description": "Every table: its name, number of rows and a short description.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "describe_table",
        "description": "One table's CREATE TABLE statement with every column's description, and its summary.",
        "parameters": {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]},
    }},
    {"type": "function", "function": {
        "name": "column_values",
        "description": "The most common values of one column with their counts, and how many distinct values it has. "
                       "Use it to check spellings, letter case, date formats and units.",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string"}, "column": {"type": "string"},
        }, "required": ["table", "column"]},
    }},
    {"type": "function", "function": {
        "name": "run_query",
        "description": f"Runs one read-only SQLite SELECT query and returns its first {RESULT_ROWS} rows. "
                       "Use it to test an idea or a corrected query.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
    }},
    {"type": "function", "function": {
        "name": "known_errors",
        "description": "The SQLite errors Scout already knows how to fix (the error book): id and rule of each.",
        "parameters": {"type": "object", "properties": {}},
    }},
]


class ReadOnlyTools:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def call(self, name: str, args: Dict) -> str:
        """Runs one tool; any problem comes back as text for the agent to read, never as an exception."""
        handler = {
            "list_tables": self.list_tables, "describe_table": self.describe_table,
            "column_values": self.column_values, "run_query": self.run_query, "known_errors": self.known_errors,
        }.get(name)
        if handler is None:
            return f"Unknown tool '{name}'. Use one of: {', '.join(s['function']['name'] for s in SPECS)}."
        try:
            text = handler(**(args or {}))
        except TypeError as e:
            text = f"Wrong arguments for {name}: {e}"
        except Exception as e:
            text = f"Error: {e}"
        return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "\n... (cut)"

    # ---------- the tools ----------

    def list_tables(self) -> str:
        return json.dumps([
            {"table": t["name"], "rows": t["row_count"], "about": (t.get("summary") or "")[:200]}
            for t in self.pipeline.tables
        ], ensure_ascii=False)

    def describe_table(self, table: str) -> str:
        t = self._table(table)
        return f"{t['schema']}\n\nSummary: {t.get('summary') or '(none)'}"

    def column_values(self, table: str, column: str) -> str:
        t = self._table(table)
        real = {c["name"].lower(): c["name"] for c in t["columns"]}
        if column.lower() not in real:
            return f"No column '{column}' in {t['name']}. Columns: {', '.join(real.values())}."
        col, source = real[column.lower()], f'"{t["database"]}"."{t["name"]}"'
        _, rows, _ = self.pipeline.database.query(
            f'SELECT "{col}", COUNT(*) AS n FROM {source} GROUP BY 1 ORDER BY n DESC LIMIT 15'
        )
        _, distinct, _ = self.pipeline.database.query(f'SELECT COUNT(DISTINCT "{col}") FROM {source}')
        return json.dumps({"distinct_values": distinct[0][0], "most_common": rows}, ensure_ascii=False, default=str)

    def run_query(self, sql: str) -> str:
        problems = write_problems(sql)
        if problems:
            return f"Refused: only one plain read query may run ({', '.join(problems)})."
        try:
            reads = self.pipeline.database.estimate_reads(sql, self.pipeline.row_counts)
        except Exception:
            reads = 0
        if reads > MAX_ESTIMATED_READS:
            return f"Refused: this could read about {reads:,} rows (limit {MAX_ESTIMATED_READS:,}). Narrow it."
        try:
            columns, rows, more = self.pipeline.database.query(sql)
        except Exception as e:
            return f"SQLite error: {e}"
        return json.dumps({
            "columns": columns, "row_count": f"{len(rows)}{'+' if more else ''}", "rows": rows[:RESULT_ROWS],
        }, ensure_ascii=False, default=str)

    def known_errors(self) -> str:
        return json.dumps([{"id": e["id"], "rule": e["rule"]} for e in self.pipeline.errors.entries], ensure_ascii=False)

    def _table(self, name: str) -> Dict:
        found = [t for t in self.pipeline.tables if t["name"].lower() == str(name).lower()]
        if not found:
            raise ValueError(f"No table '{name}'. Tables: {', '.join(self.table_names())}.")
        return found[0]

    def table_names(self) -> List[str]:
        return [t["name"] for t in self.pipeline.tables]
