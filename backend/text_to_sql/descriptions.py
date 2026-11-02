"""
Database descriptions in BIRD format: one CSV per table in a database_description/
folder next to the .sqlite file, with these columns:

    original_column_name, column_name, column_description, data_format, value_description

They say what each column means (units, date formats, allowed values, business rules),
which the raw schema can't. We attach each description to its column in the schema
the LLM sees, and check the descriptions against the actual data.
"""
import csv
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

# formats we can verify against the data
DATE_PATTERNS = {
    "YYYY-MM-DD": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "MM/DD/YYYY": re.compile(r"^\d{2}/\d{2}/\d{4}$"),
}


def load_descriptions(desc_dir: Path) -> Dict[str, Dict[str, dict]]:
    """
    Returns {table_name: {column_name: description_row}}, with names lowercased.
    """
    tables = {}
    for csv_path in sorted(desc_dir.glob("*.csv")):
        # BIRD files are sometimes saved with a BOM or non-UTF-8 characters
        with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
            rows = list(csv.DictReader(f))

        columns = {}
        for row in rows:
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            name = row.get("original_column_name", "")
            if name:
                columns[name.lower()] = row
        tables[csv_path.stem.strip().lower()] = columns
    return tables


def annotated_schema(
    conn: sqlite3.Connection, alias: str, descriptions: Dict[str, Dict[str, dict]]
) -> Tuple[Dict[str, str], List[str]]:
    """
    Builds CREATE TABLE statements for one attached database, with each column's
    description as a comment, e.g.

        order_date TEXT,  -- date the order was placed; stored as TEXT in MM/DD/YYYY ...

    Returns ({table_name: statement}, warnings).
    """
    statements, warnings = {}, []
    tables = [
        row[0]
        for row in conn.execute(
            f"SELECT name FROM \"{alias}\".sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        )
    ]

    for table in tables:
        # cid, name, type, notnull, default, pk
        columns = conn.execute(f'PRAGMA "{alias}".table_info("{table}")').fetchall()
        foreign_keys = {
            fk[3]: f"{_quote(fk[2])}({_quote(fk[4])})"
            for fk in conn.execute(f'PRAGMA "{alias}".foreign_key_list("{table}")')
        }
        pk_columns = [col[1] for col in sorted(columns, key=lambda c: c[5]) if col[5]]

        table_desc = descriptions.get(table.lower())
        if table_desc is None:
            warnings.append(f"{table}: no description file")
            table_desc = {}

        lines = []
        for _, name, col_type, _, _, _ in columns:
            definition = f"{_quote(name)} {col_type}".rstrip()
            if len(pk_columns) == 1 and name == pk_columns[0]:
                definition += " PRIMARY KEY"
            if name in foreign_keys:
                definition += f" REFERENCES {foreign_keys[name]}"

            desc = table_desc.get(name.lower())
            if desc is None:
                if descriptions.get(table.lower()) is not None:
                    warnings.append(f"{table}.{name}: no description")
                lines.append((definition, ""))
                continue

            lines.append((definition, _comment(desc, col_type)))
            warnings.extend(_check_values(conn, alias, table, name, desc))

        if len(pk_columns) > 1:
            lines.append((f"PRIMARY KEY ({', '.join(_quote(c) for c in pk_columns)})", ""))

        real_columns = {col[1].lower() for col in columns}
        for extra in sorted(set(table_desc) - real_columns):
            warnings.append(f"{table}.{extra}: described but not in the database")

        body = []
        for i, (definition, comment) in enumerate(lines):
            comma = "," if i < len(lines) - 1 else ""
            body.append(f"    {definition}{comma}" + (f"  -- {comment}" if comment else ""))
        statements[table] = f"CREATE TABLE {_quote(table)} (\n" + "\n".join(body) + "\n);"

    known_tables = {t.lower() for t in tables}
    for extra in sorted(set(descriptions) - known_tables):
        warnings.append(f"{extra}.csv: no such table in the database")

    return statements, warnings


def table_document(table: str, columns: List[str], table_desc: Dict[str, dict] | None) -> str:
    """
    One short line describing a table, used for embedding-based table search (step 3).
    Built from the table name, its column names and their descriptions, e.g.

        Table orders: order id (unique id of the order), order date (date the order was placed), ...
    """
    parts = []
    for column in columns:
        name = column.replace("_", " ")
        meaning = (table_desc or {}).get(column.lower(), {}).get("column_description", "")
        parts.append(f"{name} ({meaning})" if meaning and meaning.lower() != name.lower() else name)
    return f"Table {table.replace('_', ' ')}: " + ", ".join(parts)


def _comment(desc: dict, col_type: str) -> str:
    """One line: what the column means, plus its format and values."""
    text = desc.get("column_description", "")
    data_format = desc.get("data_format", "")
    if data_format and data_format.lower() != col_type.lower():
        text += f" ({data_format})"
    value_description = desc.get("value_description", "")
    if value_description:
        text += f"; {value_description}" if text else value_description
    return " ".join(text.split())  # keep it on one line


def _check_values(
    conn: sqlite3.Connection, alias: str, table: str, column: str, desc: dict
) -> List[str]:
    """
    Read-only checks that the description matches the data:
    - "one of: 'a', 'b'" lists contain every value in the column
    - stated date formats (YYYY-MM-DD, MM/DD/YYYY) match every value
    """
    value_description = desc.get("value_description", "")
    warnings = []

    values = None

    def distinct_values():
        return {
            row[0]
            for row in conn.execute(
                f'SELECT DISTINCT "{column}" FROM "{alias}"."{table}" WHERE "{column}" IS NOT NULL'
            )
        }

    one_of = re.search(r"one of:(.*)", value_description, re.IGNORECASE)
    if one_of:
        allowed = set(re.findall(r"'([^']*)'", one_of.group(1)))
        if allowed:
            values = distinct_values()
            unexpected = sorted(str(v) for v in values if str(v) not in allowed)
            if unexpected:
                warnings.append(f"{table}.{column}: values not listed in description: {unexpected[:5]}")

    for fmt, pattern in DATE_PATTERNS.items():
        if fmt in value_description:
            values = values if values is not None else distinct_values()
            bad = sorted(str(v) for v in values if not pattern.match(str(v)))
            if bad:
                warnings.append(f"{table}.{column}: {len(bad)} values don't match {fmt}, e.g. {bad[:3]}")
            break

    return warnings


def _quote(name: str) -> str:
    """Quote names with spaces or symbols (BIRD has some), leave simple ones as is."""
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else f'"{name}"'
