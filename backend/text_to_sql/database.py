import sqlite3
import threading
from pathlib import Path
from typing import Any, List, Tuple

from backend.text_to_sql.descriptions import annotated_schema, load_descriptions


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

    def schema(self) -> str:
        """
        Schema for every table in every attached file, from the best source available:

        1. database_description/ next to the .sqlite file (BIRD format): every column
           annotated with its meaning, format and allowed values
        2. schema.sql next to the .sqlite file: hand-written comments
        3. plain CREATE TABLE statements from the database
        """
        statements = []
        for path, alias in zip(self.files, self.aliases):
            desc_dir = path.parent / "database_description"
            schema_file = path.parent / "schema.sql"

            if desc_dir.is_dir():
                text, warnings = annotated_schema(self.conn, alias, load_descriptions(desc_dir))
                for warning in warnings:
                    print(f"[Descriptions] WARNING {alias}: {warning}")
                print(f"[Descriptions] Loaded descriptions for {alias}")
                statements.append(text)
                continue

            if schema_file.exists():
                statements.append(schema_file.read_text(encoding="utf-8").strip())
                continue

            rows = self.conn.execute(
                f"SELECT sql FROM \"{alias}\".sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"
            ).fetchall()
            statements.extend(row[0] for row in rows)
        return "\n\n".join(statements)

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
