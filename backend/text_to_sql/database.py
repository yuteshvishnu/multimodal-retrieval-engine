import re
import sqlite3
import threading
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.text_to_sql.checks import profile_table
from backend.text_to_sql.descriptions import annotated_schema, load_descriptions, table_document
from backend.text_to_sql.table_summaries import fresh_summary, load_summaries


class SQLiteDatabase:
    """
    Read-only view over every .sqlite file in the data folder (and its subfolders,
    e.g. data/text_to_sql/retail_store/retail_store.sqlite).

    Each file is attached to one in-memory connection, so a query can use
    and join tables from any file by their plain table names.
    """

    def __init__(self, data_dir: str = "data/text_to_sql", max_rows: int = 500):
        self.data_dir = Path(data_dir)
        self.max_rows = max_rows
        self.files = sorted(self.data_dir.rglob("*.sqlite"))

        self.conn = sqlite3.connect("file::memory:", uri=True, check_same_thread=False)
        self.lock = threading.Lock()

        self.aliases: List[str] = []
        for path in self.files:
            alias = path.stem.replace('"', "")
            uri = path.resolve().as_uri() + "?mode=ro"
            self.conn.execute(f'ATTACH DATABASE ? AS "{alias}"', (uri,))
            self.aliases.append(alias)

        # block any write, even if a file was opened writable
        self.conn.execute("PRAGMA query_only = ON")
        print(f"[SQLiteDatabase] Attached {len(self.aliases)} databases from {self.data_dir}")

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
                columns = [col[1] for col in self.conn.execute(f'PRAGMA "{alias}".table_info("{name}")')]
                links = {fk[2] for fk in self.conn.execute(f'PRAGMA "{alias}".foreign_key_list("{name}")')}
                table = {
                    "name": name,
                    "database": alias,
                    "folder": path.parent,
                    "schema": statements.get(name, plain_sql),
                    "links": links - {name},
                }
                table["row_count"], table["profile"] = profile_table(self.conn, alias, name)

                summary = fresh_summary(table, summaries)
                if summary:
                    table["doc"] = f"Table {name.replace('_', ' ')}: {summary}"
                    summarized += 1
                else:
                    table["doc"] = table_document(name, columns, (descriptions or {}).get(name.lower()))
                tables.append(table)

            print(f"[Summaries] {alias}: {summarized}/{len(rows)} tables have an up-to-date LLM summary")
        return tables

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
        with self.lock:
            self.conn.execute(f"EXPLAIN {sql}").fetchall()

    def run(self, sql: str) -> Tuple[List[str], List[List[Any]]]:
        """
        Run a single SELECT query and return (columns, rows).
        """
        if not sql.lstrip().lower().startswith(("select", "with")):
            raise ValueError("Only SELECT queries are allowed.")

        with self.lock:
            cursor = self.conn.execute(sql)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchmany(self.max_rows)

        # BLOBs can't be sent as JSON
        clean_rows = [
            [f"<{len(v)} bytes>" if isinstance(v, bytes) else v for v in row]
            for row in rows
        ]
        return columns, clean_rows


def _split_schema_sql(text: str) -> Dict[str, str]:
    """Splits a schema.sql file into {table_name: CREATE TABLE statement}."""
    statements = {}
    for match in re.finditer(r'CREATE TABLE\s+"?(\w+)"?\s*\(.*?\n\s*\);?', text, re.DOTALL | re.IGNORECASE):
        statements[match.group(1)] = match.group(0).strip()
    return statements
