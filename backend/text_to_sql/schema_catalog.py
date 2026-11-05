"""
Version-20: the schema catalog. Everything Scout knows about a database, in one place.

Offline onboarding (once per database, again only when its file or descriptions change) reads
the data and writes data/catalogs/<database>-<id>/catalog.json:

    per table    row count, primary key, links to other tables (declared foreign keys, and links
                 inferred from the data where the database declares none)
    per column   type; meaning (the description file, else schema.sql comments, else an
                 optional AI note); and a profile chosen by the column's kind:

        kind        what it is                         what the profile keeps
        key         ids and links                      nothing more (values say nothing)
        category    20 or fewer different values       every value, with how often it appears
        text        more than 20 different values      medium (<= 1,000): the most common values;
                                                       high: a few examples; all looked up in the
                                                       value index (value_index.py) instead
        long_text   free text (long on average)        nothing: never listed or indexed
        number      numbers                            range (and every value if 20 or fewer)
        date        dates and times                    format and range
        empty       no values at all                   that it's empty

Online question answering only reads it: render_schema() writes the CREATE TABLE text the LLM
sees, with each column's meaning and profile as a comment, and legacy_profile() gives the
deterministic checks (checks.py) what they need. No LLM call is needed for any of this.
"""
import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from backend.config import CATALOG_DIR
from backend.text_to_sql.checks import DATE_FORMATS

CATALOG_VERSION = 1      # bump to rebuild every catalog after a change to what's stored
CATEGORY_MAX = 20        # low cardinality: every value is listed
MEDIUM_MAX = 1000        # medium cardinality: the most common values are listed
TOP_VALUES = 8           # how many common values a medium column lists
EXAMPLES = 3             # how many examples a high-cardinality column lists
LONG_TEXT = 80           # average length above which text is free text
VALUE_CHARS = 40         # a listed value is cut to this many characters
SAMPLE_ROWS = 2000       # rows read to decide a column's type and format
LINK_SAMPLE = 200        # distinct values tested when inferring a link
LINK_MATCH = 0.9         # share of them that must exist in the other table

NUMBER_TYPES = ("INT", "REAL", "NUM", "DEC", "FLOA", "DOUB", "BOOL")
# id-like names: "id", "user_id", "CustomerID", "OwnerUserId" (not "paid" or "valid")
KEY_NAME = re.compile(r"(^(id|Id|ID)$|_(id|Id|ID)$|[a-z0-9](Id|ID)$)")

# dates the deterministic checks understand (they parse and compare them): checks.py's formats
CHECKED_DATES = DATE_FORMATS
# more date and time formats, shown to the LLM (compared as text: their order is the time order)
SHOWN_DATES = {
    "YYYY-MM-DD HH:MM:SS": re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?$"),
    "YYYYMM": re.compile(r"^(19|20)\d{2}(0[1-9]|1[0-2])$"),
    "HH:MM:SS": re.compile(r"^\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?$"),
}


# ---------- where a catalog lives, and when it's out of date ----------

def catalog_dir(path: Path, alias: str) -> Path:
    """data/catalogs/<alias>-<8 chars of the file's path>: one folder per database file. A linked
    file keeps its own folder (Version-21: corrected BIRD links each database with other descriptions)."""
    digest = hashlib.sha1(str(Path(path).parent.resolve() / Path(path).name).encode("utf-8")).hexdigest()[:8]
    return CATALOG_DIR / f"{alias}-{digest}"


def fingerprint(path: Path) -> str:
    """Changes when the database file, its descriptions or its schema.sql change."""
    parts = [str(CATALOG_VERSION)]
    for p in [Path(path), Path(path).parent / "schema.sql", *sorted((Path(path).parent / "database_description").glob("*.csv"))]:
        if p.exists():
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def load(path: Path, alias: str) -> Optional[dict]:
    """The saved catalog if it's up to date, else None."""
    file = catalog_dir(path, alias) / "catalog.json"
    if not file.exists():
        return None
    try:
        catalog = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return catalog if catalog.get("fingerprint") == fingerprint(path) else None


def save(catalog: dict, path: Path, alias: str) -> Path:
    folder = catalog_dir(path, alias)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "catalog.json.tmp"
    tmp.write_text(json.dumps(catalog, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(folder / "catalog.json")  # never a half-written catalog
    return folder


# ---------- offline: build ----------

def build(conn: sqlite3.Connection, alias: str, path: Path, meanings: Dict[str, Dict[str, str]]) -> dict:
    """
    Reads the whole database once. meanings: {table: {column: description text}} from the
    description files or schema.sql (lowercased names). Read-only: nothing is written to the
    database.
    """
    started = time.monotonic()
    tables = {}
    names = [r[0] for r in conn.execute(
        f'SELECT name FROM "{alias}".sqlite_master '
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY rowid"
    )]
    for name in names:
        tables[name] = _table(conn, alias, name, meanings.get(name.lower(), {}))
    _infer_links(conn, alias, tables)
    for table in tables.values():  # a column that links to another table is a key
        for link in table["links"]:
            column = _column(table, link["column"])
            if column and column["kind"] != "empty":
                column["kind"] = "key"
    return {
        "version": CATALOG_VERSION, "database": alias, "file": str(path), "fingerprint": fingerprint(path),
        "built_at": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.monotonic() - started, 1),
        "tables": tables,
    }


def _table(conn, alias: str, name: str, meanings: Dict[str, str]) -> dict:
    source = f'"{alias}"."{name}"'
    row_count = conn.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0]
    info = conn.execute(f'PRAGMA "{alias}".table_info("{name}")').fetchall()
    primary_key = [col[1] for col in sorted(info, key=lambda c: c[5]) if col[5]]
    links = [
        # a foreign key may leave out the column it points to: it then means that table's primary key
        {"column": fk[3], "table": fk[2], "to": fk[4] or "", "inferred": False}
        for fk in conn.execute(f'PRAGMA "{alias}".foreign_key_list("{name}")')
    ]
    columns = []
    for _, column, col_type, _, _, pk in info:
        entry = {"name": column, "type": col_type or "", "description": meanings.get(column.lower(), "")}
        entry.update(_profile(conn, source, column, col_type or "", row_count, is_key=bool(pk) and len(primary_key) == 1))
        columns.append(entry)
    return {"row_count": row_count, "primary_key": primary_key, "links": links, "columns": columns}


def _profile(conn, source: str, column: str, col_type: str, row_count: int, is_key: bool) -> dict:
    col = f'"{column}"'
    non_null = conn.execute(f"SELECT COUNT({col}) FROM {source}").fetchone()[0]
    profile = {"nulls": row_count - non_null, "non_null": non_null}
    if non_null == 0:
        return {**profile, "kind": "empty", "distinct": 0}
    # distinct values, counted up to one past the medium limit (enough to choose a strategy)
    distinct = conn.execute(
        f"SELECT COUNT(*) FROM (SELECT DISTINCT {col} FROM {source} WHERE {col} IS NOT NULL LIMIT {MEDIUM_MAX + 1})"
    ).fetchone()[0]
    profile["distinct"] = distinct  # MEDIUM_MAX + 1 means "more than MEDIUM_MAX"

    sample = [r[0] for r in conn.execute(
        f"SELECT DISTINCT {col} FROM {source} WHERE {col} IS NOT NULL LIMIT {SAMPLE_ROWS}"
    )]
    texts = [v for v in sample if isinstance(v, str)]
    numeric = all(isinstance(v, (int, float)) for v in sample)

    if is_key or (KEY_NAME.search(column) and numeric and distinct > CATEGORY_MAX):
        return {**profile, "kind": "key"}

    if texts and len(texts) == len(sample):
        dated = _date_profile(conn, source, col, texts, distinct)
        if dated:
            return {**profile, **dated}

    if numeric:
        low, high = conn.execute(f"SELECT MIN({col}), MAX({col}) FROM {source}").fetchone()
        profile.update(kind="number", min=low, max=high)
        if distinct <= CATEGORY_MAX:
            profile["values"] = _counts(conn, source, col, CATEGORY_MAX)
        return profile

    average = sum(len(str(v)) for v in sample) / len(sample)
    if average > LONG_TEXT:
        return {**profile, "kind": "long_text", "average_length": round(average)}
    if distinct <= CATEGORY_MAX:
        return {**profile, "kind": "category", "values": _counts(conn, source, col, CATEGORY_MAX)}
    profile["kind"] = "text"
    top = _counts(conn, source, col, TOP_VALUES) if distinct <= MEDIUM_MAX else []
    if top and top[0][1] > 1:
        profile["top"] = top  # values repeat: the most common ones say what's typical
    else:
        profile["examples"] = [str(v) for v in sample[:EXAMPLES]]  # (nearly) every value is unique
    return profile


def _date_profile(conn, source: str, col: str, texts: List[str], distinct: int) -> Optional[dict]:
    for name, (pattern, fmt) in CHECKED_DATES.items():
        if all(pattern.match(v) for v in texts):
            values = texts
            if distinct > len(texts):  # not every date was in the sample: read them all
                values = [r[0] for r in conn.execute(f"SELECT DISTINCT {col} FROM {source} WHERE {col} IS NOT NULL")]
                if not all(isinstance(v, str) and pattern.match(v) for v in values):
                    return None
            parsed = sorted(datetime.strptime(v, fmt).date() for v in values)
            return {"kind": "date", "date_format": name, "date_min": parsed[0].isoformat(), "date_max": parsed[-1].isoformat()}
    for name, pattern in SHOWN_DATES.items():
        if all(pattern.match(v) for v in texts):
            low, high = conn.execute(f"SELECT MIN({col}), MAX({col}) FROM {source} WHERE {col} IS NOT NULL").fetchone()
            return {"kind": "date", "time_format": name, "min": low, "max": high}
    return None


def _counts(conn, source: str, col: str, limit: int) -> List[list]:
    return [[v, n] for v, n in conn.execute(
        f"SELECT {col}, COUNT(*) FROM {source} WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY 2 DESC, 1 LIMIT {limit}"
    )]


def _column(table: dict, name: str) -> Optional[dict]:
    return next((c for c in table["columns"] if c["name"].lower() == str(name).lower()), None)


def _infer_links(conn, alias: str, tables: Dict[str, dict]) -> None:
    """
    Links the database doesn't declare, found from names and data: a column with the same name
    as another table's one-column primary key (or <table>_id / <table>id pointing at its "id"),
    whose values (a sample) nearly all exist there. Marked inferred, with the reason.
    """
    keys = {}  # lowercased primary key name -> [(table, column)]
    for name, table in tables.items():
        if len(table["primary_key"]) == 1:
            keys.setdefault(table["primary_key"][0].lower(), []).append((name, table["primary_key"][0]))
    for name, table in tables.items():
        declared = {link["column"].lower() for link in table["links"]}
        for column in table["columns"]:
            low = column["name"].lower()
            if low in declared or column["kind"] == "empty":
                continue
            candidates = [(t, c) for t, c in keys.get(low, []) if t != name]
            for other, pk in keys.get("id", []):
                singular = other.lower().rstrip("s")
                if other != name and low in (f"{singular}_id", f"{singular}id", f"{other.lower()}_id"):
                    candidates.append((other, pk))
            if table["primary_key"] == [column["name"]] and len(table["primary_key"]) == 1:
                continue  # its own key: a link only if another table points here, found from that side
            for other, pk in candidates:
                share = _share_found(conn, alias, name, column["name"], other, pk)
                if share >= LINK_MATCH:
                    table["links"].append({
                        "column": column["name"], "table": other, "to": pk, "inferred": True,
                        "why": f"same name as {other}.{pk}; {round(100 * share)}% of its values exist there",
                    })
                    break


def _share_found(conn, alias: str, table: str, column: str, other: str, key: str) -> float:
    values = [r[0] for r in conn.execute(
        f'SELECT DISTINCT "{column}" FROM "{alias}"."{table}" WHERE "{column}" IS NOT NULL LIMIT {LINK_SAMPLE}'
    )]
    if not values:
        return 0.0
    marks = ",".join("?" * len(values))
    found = conn.execute(
        f'SELECT COUNT(DISTINCT "{key}") FROM "{alias}"."{other}" WHERE "{key}" IN ({marks})', values
    ).fetchone()[0]
    return found / len(values)


# ---------- online: read ----------

def legacy_profile(column: dict) -> dict:
    """What the Version-4 checks read (checks.py): category text values, number ranges, checked dates."""
    profile = {}
    kind = column["kind"]
    if kind == "category":
        values = [v for v, _ in column["values"]]
        if all(isinstance(v, str) for v in values):
            profile["values"] = set(values)
    if kind == "number" and column.get("min") is not None:
        profile.update(min=column["min"], max=column["max"])
    if kind == "date" and column.get("date_format"):
        profile.update(
            date_format=column["date_format"],
            date_min=datetime.fromisoformat(column["date_min"]).date(),
            date_max=datetime.fromisoformat(column["date_max"]).date(),
        )
    return profile


def column_note(column: dict, share_values: bool) -> str:
    """
    What the LLM is told about a column's contents, chosen by its kind. Values and examples
    are data, so they're left out when share_values is False (SHARE_ROWS_WITH_LLM); formats,
    ranges and counts stay.
    """
    kind = column["kind"]
    notes = []
    if kind == "empty":
        return "always empty"
    if kind == "category" or (kind == "number" and column.get("values")):
        if share_values:
            notes.append("values: " + ", ".join(_quote_value(v) for v, _ in column["values"]))
        else:
            notes.append(f"{column['distinct']} different values")
    elif kind == "number":
        notes.append(f"from {_number(column['min'])} to {_number(column['max'])}")
    elif kind == "date":
        fmt = column.get("date_format") or column.get("time_format")
        if column.get("date_format"):
            fmt_py = CHECKED_DATES[fmt][1]
            low = datetime.fromisoformat(column["date_min"]).strftime(fmt_py)
            high = datetime.fromisoformat(column["date_max"]).strftime(fmt_py)
        else:
            low, high = column["min"], column["max"]
        said = fmt in (column.get("description") or "")  # the description already gives the format
        notes.append(f"from {low} to {high}" if said else f"format {fmt}, from {low} to {high}")
    elif kind == "text":
        many = "more than 1,000" if column["distinct"] > MEDIUM_MAX else f"{column['distinct']}"
        if share_values and column.get("top"):
            notes.append(f"{many} different values, most common: " + ", ".join(_quote_value(v) for v, _ in column["top"]))
        elif share_values and column.get("examples"):
            notes.append(f"{many} different values, e.g. " + ", ".join(_quote_value(v) for v in column["examples"]))
        else:
            notes.append(f"{many} different values")
    elif kind == "long_text":
        notes.append("free text")
    total = column.get("nulls", 0) + column.get("non_null", 0)
    if kind != "key" and total and column.get("nulls", 0) / total >= 0.5:
        notes.append(f"mostly empty ({round(100 * column['nulls'] / total)}% missing)")
    return "; ".join(notes)


def render_schema(name: str, table: dict, share_values: bool, quote) -> str:
    """
    The CREATE TABLE text the LLM sees: every column with its meaning and what's in it, declared
    links as REFERENCES, inferred links named in the comment.
    """
    declared = {link["column"]: link for link in table["links"] if not link["inferred"]}
    inferred = {link["column"]: link for link in table["links"] if link["inferred"]}
    single_pk = table["primary_key"][0] if len(table["primary_key"]) == 1 else None
    lines = []
    for column in table["columns"]:
        definition = f"{quote(column['name'])} {column['type']}".rstrip()
        if column["name"] == single_pk:
            definition += " PRIMARY KEY"
        if column["name"] in declared:
            link = declared[column["name"]]
            definition += f" REFERENCES {quote(link['table'])}" + (f"({quote(link['to'])})" if link["to"] else "")
        # the description file's meaning; else an optional AI note (column_notes.py), marked as such
        meaning = column["description"] or (f"{column['ai_note']} (AI note)" if column.get("ai_note") else "")
        parts = [meaning] if meaning else []
        note = column_note(column, share_values)
        if note:
            parts.append(note)
        if column["name"] in inferred:
            link = inferred[column["name"]]
            parts.append(f"links to {link['table']}.{link['to']} (not declared; matched by name and values)")
        lines.append((definition, " ".join("; ".join(parts).split())))
    if len(table["primary_key"]) > 1:
        lines.append((f"PRIMARY KEY ({', '.join(quote(c) for c in table['primary_key'])})", ""))
    body = []
    for i, (definition, comment) in enumerate(lines):
        comma = "," if i < len(lines) - 1 else ""
        body.append(f"    {definition}{comma}" + (f"  -- {comment}" if comment else ""))
    return f"CREATE TABLE {quote(name)} (  -- {table['row_count']:,} rows\n" + "\n".join(body) + "\n);"


def _quote_value(value) -> str:
    if isinstance(value, (int, float)):
        return _number(value)
    text = str(value)
    if len(text) > VALUE_CHARS:
        text = text[:VALUE_CHARS] + "…"
    return "'" + text.replace("'", "''") + "'"


def _number(value) -> str:
    """Plain numbers, as they'd be written in SQL: 2052194.34, not 2.05219e+06."""
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))
    return str(value)
