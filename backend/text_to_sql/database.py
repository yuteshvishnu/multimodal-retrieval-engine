import queue
import re
import sqlite3
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import sqlglot
from sqlglot import exp

from backend.config import MAX_RESULT_ROWS, QUERY_CONNECTIONS, QUERY_TIMEOUT_SECONDS
from backend.text_to_sql.checks import profile_table
from backend.text_to_sql.descriptions import annotated_schema, load_descriptions, table_document
from backend.text_to_sql.table_summaries import fresh_summary, load_summaries


class QueryTimeout(Exception):
    """A query ran longer than QUERY_TIMEOUT_SECONDS and was stopped (Version-8)."""


class SQLiteDatabase:
    """
    Read-only view over every .sqlite file in the data folder (and its subfolders,
    e.g. data/text_to_sql/retail_store/retail_store.sqlite).

    Each file is attached to an in-memory connection, so a query can use
    and join tables from any file by their plain table names.

    Queries use a small pool of such connections (QUERY_CONNECTIONS), so one slow query
    doesn't make every other question wait; the main connection reads table metadata.
    """

    def __init__(self, data_dir: str = "data/text_to_sql", max_rows: int = MAX_RESULT_ROWS):
        self.data_dir = Path(data_dir)
        self.max_rows = max_rows
        self.files = sorted(self.data_dir.rglob("*.sqlite"))

        self.aliases: List[str] = [path.stem.replace('"', "") for path in self.files]
        self.lock = threading.Lock()  # guards the main connection

        self.conn = self._connect()
        self._pool: "queue.Queue[sqlite3.Connection]" = queue.Queue()
        for _ in range(QUERY_CONNECTIONS):
            self._pool.put(self._connect())
        print(f"[SQLiteDatabase] Attached {len(self.aliases)} databases from {self.data_dir}")

    def _connect(self) -> sqlite3.Connection:
        """An in-memory connection with every .sqlite file attached read-only."""
        conn = sqlite3.connect("file::memory:", uri=True, check_same_thread=False)
        for path, alias in zip(self.files, self.aliases):
            uri = path.resolve().as_uri() + "?mode=ro"
            conn.execute(f'ATTACH DATABASE ? AS "{alias}"', (uri,))
        # block any write, even if a file was opened writable
        conn.execute("PRAGMA query_only = ON")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Borrow a query connection from the pool (waits if all are busy)."""
        conn = self._pool.get()
        try:
            yield conn
        finally:
            self._pool.put(conn)

    @cached_property
    def tables(self) -> List[dict]:
        """
        One entry per table, in every attached file:

            name      table name
            database  name of the attached database it lives in
            folder    folder of its .sqlite file (where its index files live)
            schema    CREATE TABLE statement, from the best source available:
                      1. database_description/ (BIRD format): every column annotated
                      2. schema.sql next to the .sqlite file: hand-written comments
                      3. plain CREATE TABLE statement from the database
            doc       text embedded to find relevant tables (Version-3): the LLM summary
                      from table_summaries.json if it's up to date, else a column template
            links     names of the tables it references with foreign keys
            columns   [{"name", "type"}] in table order
            summary   the LLM summary (None if missing or out of date); reviewed: the user
                      confirmed or corrected it (Version-6); feedback: the user's corrections
            row_count, profile   what each column really contains (Version-4 checks)

        Built once, then reused.
        """
        tables = []
        for path, alias in zip(self.files, self.aliases):
            desc_dir = path.parent / "database_description"
            schema_file = path.parent / "schema.sql"

            descriptions = None
            statements: Dict[str, str] = {}
            if desc_dir.is_dir():
                descriptions = load_descriptions(desc_dir)
                statements, warnings = annotated_schema(self.conn, alias, descriptions)
                for warning in warnings:
                    print(f"[Descriptions] WARNING {alias}: {warning}")
                print(f"[Descriptions] Loaded descriptions for {alias}")
            elif schema_file.exists():
                statements = _split_schema_sql(schema_file.read_text(encoding="utf-8"))

            summaries = load_summaries(path.parent)
            summarized = 0

            rows = self.conn.execute(
                f"SELECT name, sql FROM \"{alias}\".sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY rowid"
            ).fetchall()
            for name, plain_sql in rows:
                info = self.conn.execute(f'PRAGMA "{alias}".table_info("{name}")').fetchall()
                columns = [col[1] for col in info]
                links = {fk[2] for fk in self.conn.execute(f'PRAGMA "{alias}".foreign_key_list("{name}")')}
                table = {
                    "name": name,
                    "database": alias,
                    "folder": path.parent,
                    "schema": statements.get(name, plain_sql),
                    "links": links - {name},
                    "columns": [{"name": col[1], "type": col[2] or ""} for col in info],
                }
                table["row_count"], table["profile"] = profile_table(self.conn, alias, name)

                summary = fresh_summary(table, summaries)
                entry = summaries.get(name, {})
                table["summary"] = summary
                table["reviewed"] = bool(summary and entry.get("reviewed"))
                table["feedback"] = entry.get("feedback", [])
                if summary:
                    table["doc"] = f"Table {name.replace('_', ' ')}: {summary}"
                    summarized += 1
                else:
                    table["doc"] = table_document(name, columns, (descriptions or {}).get(name.lower()))
                tables.append(table)

            print(f"[Summaries] {alias}: {summarized}/{len(rows)} tables have an up-to-date LLM summary")
        return tables

    def preview(self, alias: str, table: str, page: int, page_size: int) -> Tuple[List[str], List[List[Any]]]:
        """One page of a table's rows, for the Tables tab."""
        offset = (page - 1) * page_size
        return self.run(f'SELECT * FROM "{alias}"."{table}" LIMIT {int(page_size)} OFFSET {int(offset)}')

    def close(self) -> None:
        with self.lock:
            self.conn.close()
        while not self._pool.empty():
            self._pool.get().close()

    def schema(self) -> str:
        """
        The full schema: every table's CREATE TABLE statement.
        """
        return "\n\n".join(table["schema"] for table in self.tables)

    def explain(self, sql: str) -> None:
        """
        Checks a query without running it (syntax, unknown tables or columns).
        Raises sqlite3.Error if it is invalid.
        """
        if not sql.lstrip().lower().startswith(("select", "with")):
            raise ValueError("Only SELECT queries are allowed.")
        with self._connection() as conn:
            conn.execute(f"EXPLAIN {sql}").fetchall()

    def run(self, sql: str) -> Tuple[List[str], List[List[Any]]]:
        """
        Run a single SELECT query and return (columns, rows): at most max_rows rows,
        stopped with QueryTimeout if it runs too long.
        """
        columns, rows, _ = self.query(sql)
        return columns, rows

    def query(self, sql: str, timeout: float = QUERY_TIMEOUT_SECONDS) -> Tuple[List[str], List[List[Any]], bool]:
        """
        Run a single SELECT query and return (columns, rows, more): at most max_rows rows,
        and more=True if there were more than that (one extra row is fetched to know).
        Raises QueryTimeout if it runs longer than `timeout` seconds (Version-8).
        """
        if not sql.lstrip().lower().startswith(("select", "with")):
            raise ValueError("Only SELECT queries are allowed.")

        with self._connection() as conn:
            # the clock starts once this query has a connection, not while it waits for one
            deadline = time.monotonic() + timeout

            # SQLite has no per-query time limit: this handler runs every few thousand steps
            # of the query, and returning True stops it
            def too_slow() -> bool:
                return time.monotonic() > deadline

            conn.set_progress_handler(too_slow, 10_000)
            try:
                cursor = conn.execute(sql)
                columns = [col[0] for col in cursor.description]
                rows = cursor.fetchmany(self.max_rows + 1)
            except sqlite3.OperationalError as e:
                if "interrupted" in str(e).lower():
                    raise QueryTimeout(f"took longer than {timeout:g} s") from None
                raise
            finally:
                conn.set_progress_handler(None, 0)

        more = len(rows) > self.max_rows
        # BLOBs can't be sent as JSON
        clean_rows = [
            [f"<{len(v)} bytes>" if isinstance(v, bytes) else v for v in row]
            for row in rows[: self.max_rows]
        ]
        return columns, clean_rows, more

    def estimate_reads(self, sql: str, row_counts: Dict[str, int]) -> int:
        """
        Roughly how many rows a query reads, worked out before running it (Version-8).
        SQLite's EXPLAIN QUERY PLAN has no row estimates, so we use its steps with the
        tables' row counts:
            SCAN t                      reads every row of t
            SEARCH t USING ... (rowid=?) / primary key equality: about 1 row per lookup
            SEARCH t USING INDEX ...    assumed ~1% of t per lookup
        Steps at the same level are nested loops (a join), so their sizes multiply;
        a WITH step's estimated size is used wherever it's scanned later.
        """
        with self._connection() as conn:
            plan = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()

        # the plan names tables by alias ("SCAN o"): map aliases back to table names
        aliases = {}
        try:
            for table in sqlglot.parse_one(sql, read="sqlite").find_all(exp.Table):
                aliases[table.alias_or_name.lower()] = table.name.lower()
        except sqlglot.errors.ParseError:
            pass

        children = defaultdict(list)
        for node_id, parent, _, detail in plan:
            children[parent].append((node_id, detail))
        steps: Dict[str, int] = {}  # WITH step / subquery name -> estimated rows

        def level(parent: int) -> Tuple[int, int]:
            """(rows read at this level and below, rows this level produces)"""
            reads, product, loops = 0, 1, False
            for node_id, detail in children[parent]:
                access = re.match(r"(SCAN|SEARCH) (\S+)", detail)
                if access and not detail.startswith("SCAN CONSTANT"):
                    name = access.group(2).lower()
                    size = steps.get(name) or row_counts.get(aliases.get(name, name), 1)
                    if access.group(1) == "SEARCH":
                        exact = re.search(r"\((rowid|\w+)=\?\)", detail) and "PRIMARY KEY" in detail
                        size = 1 if exact else max(1, size // 100)
                    product *= max(1, size)
                    loops = True
                else:
                    sub_reads, sub_rows = level(node_id)
                    reads += sub_reads
                    step = re.match(r"(CO-ROUTINE|MATERIALIZE) (\S+)", detail)
                    if step:
                        steps[step.group(2).lower()] = max(1, sub_rows)
            produced = product if loops else 0
            return reads + produced, produced

        return level(0)[0]


def _split_schema_sql(text: str) -> Dict[str, str]:
    """Splits a schema.sql file into {table_name: CREATE TABLE statement}."""
    statements = {}
    for match in re.finditer(r'CREATE TABLE\s+"?(\w+)"?\s*\(.*?\n\s*\);?', text, re.DOTALL | re.IGNORECASE):
        statements[match.group(1)] = match.group(0).strip()
    return statements
