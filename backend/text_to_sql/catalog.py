"""
Text-to-SQL step 6: what Rexie knows about each table, shown in the Tables view, and adding a new
.sqlite database without restarting the server.
"""
import math
import re
from pathlib import Path
from typing import List

SQLITE_HEADER = b"SQLite format 3\x00"
PAGE_SIZE = 10


def page_of(items: list, page: int, page_size: int = PAGE_SIZE) -> dict:
    pages = max(1, math.ceil(len(items) / page_size))
    page = min(max(1, page), pages)
    start = (page - 1) * page_size
    return {"items": items[start:start + page_size], "page": page, "pages": pages, "total": len(items)}


def table_card(table: dict) -> dict:
    """What the Tables tab shows for one table."""
    if table["summary"]:
        status = "reviewed" if table["reviewed"] else "ai"
        description = table["summary"]
    else:
        status = "basic"  # no LLM summary yet: the description is built from column names
        description = table["doc"]
    return {
        "name": table["name"],
        "database": table["database"],
        "row_count": table["row_count"],
        "columns": table["columns"],
        "description": description,
        "status": status,
        "feedback": table["feedback"],
    }


def list_tables(tables: List[dict], page: int, search: str = "") -> dict:
    words = search.lower().split()
    matching = [
        t for t in tables
        if all(w in f"{t['name']} {t['database']} {t['summary'] or ''}".lower() for w in words)
    ]
    result = page_of(matching, page)
    result["items"] = [table_card(t) for t in result["items"]]
    return result


def save_upload(data_dir: Path, filename: str, content: bytes, tables: List[dict]) -> Path:
    """
    Checks an uploaded file and saves it as data_dir/<name>/<name>.sqlite.
    Raises ValueError with a message for the user if it can't be added.
    """
    stem = Path(filename or "").stem
    if not filename or Path(filename).suffix.lower() != ".sqlite":
        raise ValueError("Please upload a .sqlite file.")
    if not content.startswith(SQLITE_HEADER):
        raise ValueError("This file is not a SQLite database.")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", stem):
        raise ValueError("Use a file name with only letters, numbers and _ (e.g. sales_2026.sqlite).")

    folder = Path(data_dir) / stem
    if folder.exists() or any(t["database"].lower() == stem.lower() for t in tables):
        raise ValueError(f"A database called '{stem}' already exists.")

    folder.mkdir(parents=True)
    path = folder / f"{stem}.sqlite"
    path.write_bytes(content)
    return path


def check_new_tables(path: Path, tables: List[dict]) -> List[str]:
    """
    The new file's table names. Questions use plain table names, so a name that already
    exists in another database would be ambiguous: refuse it.
    """
    import sqlite3

    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )]
    except sqlite3.DatabaseError as e:
        raise ValueError(f"Could not read the database: {e}")
    finally:
        conn.close()
    if not names:
        raise ValueError("The database has no tables.")
    existing = {t["name"].lower() for t in tables}
    clashes = sorted(n for n in names if n.lower() in existing)
    if clashes:
        raise ValueError("These tables already exist in another database: " + ", ".join(clashes))
    return names
