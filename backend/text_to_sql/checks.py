"""
Text-to-SQL Version-4: deterministic checks around the LLM's SQL. No LLM calls, no guessing:
each check is a plain rule over the question, the SQL, or the result.

    (what each column contains is profiled once per database by schema_catalog.py, Version-20)
    question_problems   before any LLM call: impossible requests ("top -5")
    check_sql           before running: errors (reject the SQL) and warnings
    check_result        after running: warnings about a suspicious result
    result_signature    to compare two answers by their results, not their SQL text
    explain_disagreement  when answers differ: where their SQL differs (read with the sqlglot parser)
"""
import hashlib
import re
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp

from backend.config import SHARE_ROWS_WITH_LLM

# check_sql's problem when the SQL compares a column with a value it never has ("city has no value 'X'")
MISSING_VALUE = re.compile(r"^[\w.]+ has no value ")


DATE_FORMATS = {
    "YYYY-MM-DD": (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
    "MM/DD/YYYY": (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "%m/%d/%Y"),
}

# window functions that pick "the top one(s)", and how they treat ties
RANKING_FUNCTIONS = {exp.RowNumber: "ROW_NUMBER", exp.Rank: "RANK", exp.DenseRank: "DENSE_RANK"}
RANKING_NOTE = "ROW_NUMBER keeps exactly one row per group and drops ties; RANK and DENSE_RANK keep ties"

MONEY_WORDS = ("amount", "price", "revenue", "total", "spent", "sales", "cost", "salary")


# ---------- dates (profiled by schema_catalog.py) ----------

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

def check_sql(sql: str, tables: List[dict], today: date, values=None) -> Tuple[List[str], List[str]]:
    """
    Returns (errors, warnings).
    values: the value index (Version-20), to check text columns with too many values to list.
    Errors mean the SQL is wrong for sure: it is rejected and never run.
    Warnings mean it may be fine but deserves attention.
    """
    errors, warnings = [], []
    # Version-20: only the tables the SQL reads (a column name like Segment can exist in two tables)
    used = tables_in_sql(sql, [t["name"] for t in tables])
    tables = [t for t in tables if t["name"] in used] or tables
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

    for column, op, literals in comparisons:
        for p in profiles.get(column.lower(), []):
            if "values" in p and op in ("=", "!=", "<>", "IN"):
                unknown = [v for v in literals if v not in p["values"]]
                if unknown and not any(set(literals) <= q.get("values", set()) for q in profiles[column.lower()]):
                    errors.append(
                        f"{column} has no value {', '.join(repr(v) for v in unknown)}; "
                        f"its values are {', '.join(repr(v) for v in sorted(p['values']))}."
                    )
                    break
            if "date_format" in p:
                for value in literals:
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
        if values is not None and op in ("=", "IN") and not any("values" in p for p in profiles.get(column.lower(), [])):
            problem = _missing_in_index(column, literals, tables, values)
            if problem:
                errors.append(problem)

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


def tables_in_sql(sql: str, known: List[str]) -> List[str]:
    """
    The real tables a query reads, from the SQL itself (read with the parser): no WITH step
    names, no guessing. In the order they appear; only names of tables that exist.
    """
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:  # ParseError, and TokenError (an unclosed quote)
        return []
    steps = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    exact = {name.lower(): name for name in known}
    used = []
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if name in exact and name not in steps and exact[name] not in used:
            used.append(exact[name])
    return used


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
    except sqlglot.errors.SqlglotError:  # ParseError, and TokenError (an unclosed quote)
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


def _missing_in_index(column: str, literals: List[str], tables: List[dict], values) -> Optional[str]:
    """
    Version-20: for a text column with too many values to list, look each compared value up in
    the value index. A problem only when the column is indexed and none of its tables has it.
    """
    owners = [(t["name"], c["name"]) for t in tables for c in t["columns"]
              if c["name"].lower() == column.lower() and values.indexed(t["name"], c["name"])]
    if not owners:
        return None
    missing = [v for v in literals if not any(values.has(t, c, v) for t, c in owners)]
    if not missing:
        return None
    similar = list(dict.fromkeys(s for v in missing for t, c in owners for s in values.similar(t, c, v)))[:6]
    hint = f"similar values: {', '.join(repr(v) for v in similar)}" if similar else "no similar value was found"
    return f"{column} has no value {', '.join(repr(v) for v in missing)}; {hint}."


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


# ---------- Version-20: grouping by a name merges different records ----------

# grouped columns "nearly identify" a record when at least this share of rows has its own combination
NEARLY_UNIQUE = 0.9

def merged_records(sql: str, tables: List[dict], run) -> List[str]:
    """
    Errors for GROUP BYs that group a table's rows by columns that *nearly identify* a record
    (like first_name, last_name: almost every customer has their own) without the table's key,
    when the data really has different records with the same values (two customers called Ananya
    Kim): their rows would be added together as one. Grouping by a category (city, status:
    many records each) is meant to add them up and is never flagged, nor is grouping by a key or
    a link (orders.customer_id). Read with the parser; run(sql) -> (columns, rows) asks the data.
    """
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:  # ParseError, and TokenError (an unclosed quote)
        return []
    by_name = {t["name"].lower(): t for t in tables if t.get("catalog")}
    steps = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    aliases = {}  # alias or name -> real table
    for node in tree.find_all(exp.Table):
        real = by_name.get(node.name.lower())
        if real and node.name.lower() not in steps:
            aliases[node.alias_or_name.lower()] = real
            aliases[node.name.lower()] = real

    problems = []
    for group in tree.find_all(exp.Group):
        grouped = {}  # real table name -> its grouped column names
        for column in group.find_all(exp.Column):
            table = _owner(column, aliases)
            if table:
                grouped.setdefault(table["name"], set()).add(_real_column(table, column.name))
        for name, cols in grouped.items():
            table = by_name[name.lower()]
            key = table["catalog"]["primary_key"]
            if len(key) != 1 or key[0] in cols or _linked_key_grouped(table, grouped, by_name):
                continue
            names = sorted(c for c in cols if c)
            kinds = {c["name"]: c["kind"] for c in table["catalog"]["columns"]}
            if not names or any(kinds.get(c) != "text" for c in names):
                continue  # only name-like text columns: not keys or links (by records on purpose),
                # numbers, dates or free text
            quoted = ", ".join(f'"{c}"' for c in names)
            source = f'"{table["database"]}"."{table["name"]}"'
            try:
                _, (groups,) = run(f"SELECT COUNT(*) FROM (SELECT 1 FROM {source} GROUP BY {quoted})")
                if groups[0] < NEARLY_UNIQUE * table["catalog"]["row_count"]:
                    continue  # a category (many records per value): adding them up is the point
                _, rows = run(f'SELECT {quoted} FROM {source} GROUP BY {quoted} '
                              f'HAVING COUNT(DISTINCT "{key[0]}") > 1 LIMIT 1')
            except Exception:
                continue
            if rows:
                example = " ".join(str(v) for v in rows[0])
                problems.append(
                    f"Grouping {table['name']} by {', '.join(names)} adds different {table['name']} together "
                    f"when they share those values (e.g. '{example}' is more than one record). Group by "
                    f"{table['name']}.{key[0]} too; the result can still show only {', '.join(names)}."
                )
    return _unique(problems)


def _owner(column, aliases: Dict[str, dict]) -> Optional[dict]:
    """The real table a GROUP BY column belongs to: by its qualifier, or the one table that has it."""
    if column.table:
        return aliases.get(column.table.lower())
    owners = {t["name"]: t for t in aliases.values()
              if any(c["name"].lower() == column.name.lower() for c in t["catalog"]["columns"])}
    return next(iter(owners.values())) if len(owners) == 1 else None


def _real_column(table: dict, name: str) -> Optional[str]:
    return next((c["name"] for c in table["catalog"]["columns"] if c["name"].lower() == name.lower()), None)


def _linked_key_grouped(table: dict, grouped: Dict[str, set], by_name: Dict[str, dict]) -> bool:
    """Another grouped column links to this table's key (GROUP BY o.customer_id, c.first_name)."""
    for other, cols in grouped.items():
        for link in by_name[other.lower()]["catalog"]["links"]:
            if link["table"].lower() == table["name"].lower() and link["column"] in cols:
                return True
    return False
