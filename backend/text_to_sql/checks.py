"""
Text-to-SQL Version-4: deterministic checks around the LLM's SQL. No LLM calls, no guessing:
each check is a plain rule over the question, the SQL, or the result.

    profile_table   once at startup: what each column really contains
    question_problems   before any LLM call: impossible requests ("top -5")
    check_sql           before running: errors (reject the SQL) and warnings
    check_result        after running: warnings about a suspicious result
    result_signature    to compare two answers by their results, not their SQL text
    explain_disagreement  when answers differ: where their SQL differs (read with the sqlglot parser)
"""
import hashlib
import re
import sqlite3
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp

from backend.config import SHARE_ROWS_WITH_LLM

# columns with at most this many distinct values are treated as categories
MAX_CATEGORY_VALUES = 20
# distinct values read per text column when looking for dates (keeps startup fast on big tables)
MAX_VALUES_SCANNED = 1000

DATE_FORMATS = {
    "YYYY-MM-DD": (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
    "MM/DD/YYYY": (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "%m/%d/%Y"),
}

# window functions that pick "the top one(s)", and how they treat ties
RANKING_FUNCTIONS = {exp.RowNumber: "ROW_NUMBER", exp.Rank: "RANK", exp.DenseRank: "DENSE_RANK"}
RANKING_NOTE = "ROW_NUMBER keeps exactly one row per group and drops ties; RANK and DENSE_RANK keep ties"

MONEY_WORDS = ("amount", "price", "revenue", "total", "spent", "sales", "cost", "salary")


# ---------- profiling (once, at startup) ----------

def profile_table(conn: sqlite3.Connection, alias: str, table: str) -> Tuple[int, Dict[str, dict]]:
    """
    Returns (row_count, {column: profile}). A profile can have:
        values       the full set of values, for category columns (<= 20 distinct)
        min, max     for number columns
        date_format  "YYYY-MM-DD" or "MM/DD/YYYY", if every value is a date in that format
        date_min, date_max   earliest and latest date (as date objects)
    """
    source = f'"{alias}"."{table}"'
    row_count = conn.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0]

    profiles = {}
    for _, column, col_type, _, _, _ in conn.execute(f'PRAGMA "{alias}".table_info("{table}")'):
        col = f'"{column}"'
        profile = {}
        col_type = (col_type or "").upper()

        if any(t in col_type for t in ("INT", "REAL", "NUM", "DEC", "FLOA", "DOUB")):
            low, high = conn.execute(f"SELECT MIN({col}), MAX({col}) FROM {source}").fetchone()
            if low is not None:
                profile.update(min=low, max=high)
        else:
            values = [r[0] for r in conn.execute(
                f"SELECT DISTINCT {col} FROM {source} WHERE {col} IS NOT NULL LIMIT {MAX_VALUES_SCANNED}"
            )]
            date_format = _date_format(values)
            if date_format:
                parsed = [_parse_date(v, date_format) for v in values]
                profile.update(date_format=date_format, date_min=min(parsed), date_max=max(parsed))
            elif 0 < len(values) <= MAX_CATEGORY_VALUES and all(isinstance(v, str) for v in values):
                profile["values"] = set(values)

        profiles[column] = profile
    return row_count, profiles


def _date_format(values: list) -> Optional[str]:
    if not values or not all(isinstance(v, str) for v in values):
        return None
    for name, (pattern, _) in DATE_FORMATS.items():
        if all(pattern.match(v) for v in values):
            return name
    return None


def _parse_date(text: str, date_format: str) -> date:
    return datetime.strptime(text, DATE_FORMATS[date_format][1]).date()


def date_ranges(tables: List[dict]) -> List[str]:
    """Lines like 'orders.order_date: 01/02/2026 to 06/28/2026 (MM/DD/YYYY)'."""
    lines = []
    for table in tables:
        for column, p in table["profile"].items():
            if "date_format" in p:
                fmt = DATE_FORMATS[p["date_format"]][1]
                lines.append(
                    f"{table['name']}.{column}: {p['date_min'].strftime(fmt)} to "
                    f"{p['date_max'].strftime(fmt)} ({p['date_format']})"
                )
    return lines


# ---------- before any LLM call: the question itself ----------

def question_problems(question: str) -> List[str]:
    """Requests that can never be answered, caught for free before calling the LLM."""
    problems = []
    if re.search(r"\b(top|first|last|bottom|best|worst)\s+(-\s*\d+|0)\b", question, re.IGNORECASE):
        problems.append("The number of results asked for must be positive (e.g. 'top 5', not 'top -5' or 'top 0').")
    return problems


# ---------- before running: the SQL ----------

def check_sql(sql: str, tables: List[dict], today: date) -> Tuple[List[str], List[str]]:
    """
    Returns (errors, warnings).
    Errors mean the SQL is wrong for sure: it is rejected and never run.
    Warnings mean it may be fine but deserves attention.
    """
    errors, warnings = [], []
    profiles = _profiles_by_column(tables)

    limit = re.search(r"\bLIMIT\s+(-?\d+)", sql, re.IGNORECASE)
    if limit and int(limit.group(1)) <= 0:
        errors.append(f"LIMIT {limit.group(1)} returns nothing; it must be a positive number.")

    if re.search(r"cents", sql, re.IGNORECASE) and re.search(r"/\s*100(?![\d.])", sql):
        errors.append("Dividing cents by the integer 100 drops the cents in SQLite; divide by 100.0 instead.")

    # column = 'text' / column != 'text' / column IN ('a', 'b')
    comparisons = [(m.group(1), m.group(2), [m.group(3)]) for m in re.finditer(
        r"\b(\w+)\s*(=|!=|<>|<=|>=|<|>)\s*'([^']*)'", sql)]
    comparisons += [(m.group(1), "IN", re.findall(r"'([^']*)'", m.group(2))) for m in re.finditer(
        r"\b(\w+)\s+(?:NOT\s+)?IN\s*\(([^)]*)\)", sql, re.IGNORECASE)]

    for column, op, values in comparisons:
        for p in profiles.get(column.lower(), []):
            if "values" in p and op in ("=", "!=", "<>", "IN"):
                unknown = [v for v in values if v not in p["values"]]
                if unknown and not any(set(values) <= q.get("values", set()) for q in profiles[column.lower()]):
                    errors.append(
                        f"{column} has no value {', '.join(repr(v) for v in unknown)}; "
                        f"its values are {', '.join(repr(v) for v in sorted(p['values']))}."
                    )
                    break
            if "date_format" in p:
                for value in values:
                    if not DATE_FORMATS[p["date_format"]][0].match(value):
                        other = next((n for n, (pat, _) in DATE_FORMATS.items() if pat.match(value)), None)
                        if other:
                            errors.append(
                                f"{column} is stored as {p['date_format']}, but it is compared with "
                                f"'{value}' ({other}); convert the column or the value first."
                            )
                    else:
                        if _parse_date(value, p["date_format"]) > today:
                            warnings.append(f"The query uses a future date ('{value}').")

    # column <op> number, where no row can match
    for m in re.finditer(r"\b(\w+)\s*(=|<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)(?![\w.'])", sql):
        column, op, number = m.group(1), m.group(2), float(m.group(3))
        for p in profiles.get(column.lower(), []):
            if "min" not in p:
                continue
            low, high = p["min"], p["max"]
            impossible = (
                (op == "=" and not low <= number <= high)
                or (op == ">" and number >= high) or (op == ">=" and number > high)
                or (op == "<" and number <= low) or (op == "<=" and number < low)
            )
            if impossible:
                warnings.append(f"No row has {column} {op} {m.group(3)}; its values range from {low} to {high}.")
                break

    return _unique(errors), _unique(warnings)


# ---------- after running: the result ----------

def check_result(question: str, sql: str, columns: List[str], rows: List[list], tables: List[dict]) -> List[str]:
    warnings = []
    used = [t for t in tables if re.search(rf"\b{re.escape(t['name'])}\b", sql, re.IGNORECASE)]

    if not rows:
        message = "The result is empty. A filter may not match the data."
        ranges = date_ranges(used)
        if ranges:
            message += " The data covers: " + "; ".join(ranges) + "."
        warnings.append(message)
        return warnings

    has_filter = re.search(r"\b(WHERE|LIMIT|GROUP\s+BY|HAVING|DISTINCT)\b", sql, re.IGNORECASE)
    has_join = re.search(r"\bJOIN\b", sql, re.IGNORECASE)
    asks_for_all = re.search(r"\b(all|every|list|show)\b", question, re.IGNORECASE)
    if len(used) == 1 and not has_join and not has_filter and not asks_for_all:
        table = used[0]
        if table["row_count"] > 1 and len(rows) == table["row_count"]:
            warnings.append(f"The result has every row of {table['name']}; a filter may be missing.")

    max_rows = max((t["row_count"] for t in used), default=0)
    for i, column in enumerate(columns):
        values = [row[i] for row in rows if isinstance(row[i], (int, float))]
        name = column.lower()
        if "count" in name and any(v < 0 or (max_rows and v > max_rows) for v in values):
            warnings.append(f"{column} is negative or larger than the table ({max_rows} rows); a join may have multiplied rows.")
        if any(word in name for word in MONEY_WORDS) and any(v < 0 for v in values):
            warnings.append(f"{column} has negative values.")

    # identical rows that include an id column: the same record came back more than once,
    # which usually means a join multiplied rows (repeated cities or names are normal)
    has_id = any(c.lower() == "id" or c.lower().endswith("_id") for c in columns)
    if has_join and has_id and len(rows) != len({tuple(map(str, row)) for row in rows}):
        warnings.append("The same record appears more than once; a join may have multiplied rows.")

    asks_for_one = re.search(
        r"\b(the most|the highest|the lowest|the least|the best|the worst|most expensive|cheapest)\b", question, re.IGNORECASE)
    asks_for_many = re.search(r"\b(top|bottom)\s+\d+|\beach\b|\bevery\b|\bper\b|\ball\b", question, re.IGNORECASE)
    if asks_for_one and not asks_for_many and len(rows) > 1:
        warnings.append(f"The question asks for a single item, but {len(rows)} rows were returned.")

    return _unique(warnings)


def describe_rows(rows: List[list]) -> str:
    """What the LLM is told about a result: the first rows, or only the count if rows aren't shared."""
    if not SHARE_ROWS_WITH_LLM:
        return f"{len(rows)} rows (values not shared)"
    return f"{rows[:3]}{' ...' if len(rows) > 3 else ''}"


def result_signature(rows: List[list]) -> Tuple[int, str]:
    """
    Two answers agree if they return the same rows (order and column names ignored).
    Version-8: compared as (row count, fingerprint): each row is hashed, the hashes are
    sorted (so row order doesn't matter) and hashed together. Exact up to a SHA-256 collision.
    """
    row_hashes = sorted(
        hashlib.sha256(repr(tuple(round(v, 4) if isinstance(v, float) else v for v in row)).encode()).hexdigest()
        for row in rows
    )
    return len(rows), hashlib.sha256("".join(row_hashes).encode()).hexdigest()


# ---------- when attempts disagree: say exactly how ----------

def explain_disagreement(attempts: List[dict], tables: List[dict]) -> str:
    """
    Compares SQL attempts part by part and returns feedback for the next round: what each
    attempt did differently, what it returned, and what the data descriptions say about the
    columns involved. The SQL is read with a parser (sqlglot), so multi-step queries (WITH),
    subqueries and window functions are split into their real parts. No LLM call.
    Returns "" if the attempts don't differ in any of these parts (or can't be parsed).
    """
    ok = [a for a in attempts if a.get("status") == "ok"]
    if len(ok) < 2:
        return ""

    real_columns = {column.lower() for table in tables for column in table["profile"]}
    try:
        parts = [_sql_parts(a["sql"], real_columns) for a in ok]
    except sqlglot.errors.ParseError:
        return ""

    lines, columns = [], set()
    for name in ("filters", "tables", "joins", "grouping", "ranking", "limit"):
        values = [p[name] for p in parts]
        if all(v == values[0] for v in values):
            continue
        common = set.intersection(*values)
        lines.append(f"They differ in {name}:")
        for i, value in enumerate(values, 1):
            shown = sorted(value - common)
            lines.append(f"  - attempt {i}: {', '.join(shown) or '(none)'}")
            for item in shown:
                columns |= set(re.findall(r"[A-Za-z_]\w*", item))
        if name == "ranking":
            lines.append(f"  ({RANKING_NOTE})")

    if not lines:
        return ""

    out = ["The earlier attempts disagreed. Here is exactly how:"]
    for i, a in enumerate(ok, 1):
        out.append(f"- attempt {i} returned {describe_rows(a['rows'])}")
    out += lines

    notes = _column_notes(columns, tables, set().union(*(p["tables"] for p in parts)))
    if notes:
        out.append("What the data descriptions say about these columns:")
        out += [f"  - {note}" for note in notes]
    out.append("Decide which version matches the question, using these descriptions, then write the SQL.")
    return "\n".join(out)


def _sql_parts(sql: str, real_columns: set) -> dict:
    """
    The parts of a query we compare, collected from every step of it (WITH steps and
    subqueries included). Table aliases are dropped (o.status -> status), and only filters
    on real table columns are kept: "rn = 1" filters a step's own ranking, not the data.
    """
    tree = sqlglot.parse_one(sql, read="sqlite")
    steps = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for column in tree.find_all(exp.Column):
        column.set("table", None)

    filters = set()
    for clause in list(tree.find_all(exp.Where)) + list(tree.find_all(exp.Having)):
        condition = clause.this.unnest()
        for part in (condition.flatten() if isinstance(condition, exp.And) else [condition]):
            part = part.unnest()
            if any(c.name.lower() in real_columns for c in part.find_all(exp.Column)):
                filters.add(part.sql(dialect="sqlite"))

    joins = set()
    for join in tree.find_all(exp.Join):
        kind = " ".join(word for word in (join.side, join.kind) if word) or "INNER"
        table = join.this.find(exp.Table)
        if table is not None and table.name.lower() not in steps:
            joins.add(f"{kind} JOIN {table.name.lower()}")

    grouping = {
        column.name.lower()
        for group in tree.find_all(exp.Group)
        for column in group.find_all(exp.Column)
    }
    ranking = {RANKING_FUNCTIONS[type(w.this)] for w in tree.find_all(exp.Window) if type(w.this) in RANKING_FUNCTIONS}
    limit = tree.args.get("limit")
    return {
        "filters": filters,
        "tables": {t.name.lower() for t in tree.find_all(exp.Table) if t.name.lower() not in steps},
        "joins": joins,
        "grouping": grouping,
        "ranking": ranking,
        "limit": {limit.expression.sql()} if limit else set(),
    }


def _column_notes(names: set, tables: List[dict], used: set) -> List[str]:
    """The description comment of each named column, read from the schema shown to the LLM."""
    notes = []
    for table in tables:
        if table["name"].lower() not in used:
            continue
        for name in sorted(names):
            if name not in table["profile"]:
                continue
            match = re.search(rf"^\s*\"?{re.escape(name)}\"?\s+\w+.*?--\s*(.+)$", table["schema"], re.MULTILINE)
            if match:
                notes.append(f"{table['name']}.{name}: {match.group(1).strip()}")
    return _unique(notes)


def _profiles_by_column(tables: List[dict]) -> Dict[str, List[dict]]:
    by_column = {}
    for table in tables:
        for column, profile in table["profile"].items():
            by_column.setdefault(column.lower(), []).append(profile)
    return by_column


def _unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))
