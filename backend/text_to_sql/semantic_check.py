"""
Version-24: a structured meaning, and code checks that the SQL matches it. No AI.

The meaning step returns, beside its restatement, a small spec of what must be computed:

    measure         entity count, transaction count, money or other   (checked in measure_check.py)
    grain           the unit the question totals or ranks by: month, year, customer... or none
    extreme         highest, lowest or none
    date_part       year or month when the answer is that part of a date, else none
    values_per_row  how many values each result row holds, when the question fixes it (0: not fixed)

The spec goes to the SQL step as plain lines, and after the SQL has run, these checks compare it with the
SQL and its rows. Each flags only a clear case, and says what to change:

    METRIC_MISMATCH     a count is asked for but the SQL adds up values, or the reverse
    OUTPUT_DATE_PART    a month or year is asked for, but a column holds whole dates (201304, 2013-04-01)
    OUTPUT_SHAPE        one row of N values is asked for, but N rows came back
    AGGREGATION_GRAIN   the highest or lowest per month/customer..., but MAX, MIN or a top-1 sort runs on
                        single rows of a table that has several rows per unit, with no GROUP BY
    VALUE_GROUNDING     no rows (or only NULLs) came back, the SQL matches a number from the question through an
                        expression or column where the data doesn't have it, and the data has it elsewhere
    NUMERIC_TYPE        a division of two whole numbers (counts), which SQLite rounds down: 1 / 2 is 0

A flag gets one retry where the flow has retries; a flagged answer is never dropped, but it ranks below
unflagged ones in the vote, and Adaptive doesn't stop early on it.
"""
import re
from typing import Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp

from backend.text_to_sql.measure_check import counts, measure_problem, normalize

TIME_GRAINS = {"year", "month", "day", "week", "quarter", "date"}
DATE_PARTS = {"year", "month"}
WHOLE_DATE = re.compile(r"^(\d{4})(-?)(\d{2})(-?\d{2})?([ T].*)?$")
# only when the month is the answer: "which month", "what was the peak month" (not "at least one month with...")
ASKS_FOR_MONTH = re.compile(r"\b(which|what)\s+(\w+\s+){0,3}month\b|\bpeak\s+month\b", re.IGNORECASE)


def parse_spec(parsed: dict) -> Dict:
    """The meaning step's JSON, cleaned: unknown or missing values become 'none' / 0."""
    def word(key: str, allowed: Optional[set] = None) -> str:
        text = str(parsed.get(key) or "none").strip().lower()
        if text in ("", "null", "n/a"):
            return "none"
        return text if allowed is None or text in allowed else "none"

    try:
        values = int(parsed.get("values_per_row") or 0)
    except (TypeError, ValueError):
        values = 0
    return {"measure": normalize(parsed.get("measure")), "grain": word("grain"),
            "extreme": word("extreme", {"highest", "lowest"}), "date_part": word("date_part", DATE_PARTS),
            "values_per_row": values if 1 < values <= 10 else 0}


def default_spec() -> Dict:
    return parse_spec({})


def with_question(spec: Dict, question: str) -> Dict:
    """Code fills in what the question's own words make plain: 'which month', 'the peak month'."""
    if spec["date_part"] == "none" and ASKS_FOR_MONTH.search(question.split("(Hint:")[0]):
        return {**spec, "date_part": "month"}
    return spec


def spec_lines(spec: Dict) -> str:
    """The spec as lines for the SQL step: what must be computed and returned."""
    lines = []
    if spec["measure"] != "other":
        lines.append(f"The answer is a {spec['measure']}.")
    if spec["extreme"] != "none" and spec["grain"] != "none":
        lines.append(f"Take the {spec['extreme']} of the totals per {spec['grain']}, not of single rows, when "
                     f"several rows make up one {spec['grain']}.")
    if spec["date_part"] != "none":
        lines.append(f"Return the {spec['date_part']} only, not the whole date.")
    if spec["values_per_row"]:
        lines.append(f"Return one row with {spec['values_per_row']} values.")
    return "\n".join(f"- {line}" for line in lines)


def date_part_problem(spec: Dict, rows: List[list]) -> Optional[str]:
    if spec["date_part"] not in DATE_PARTS or not rows:
        return None
    sample = rows[:20]
    for col in range(len(sample[0])):
        values = [str(r[col]) for r in sample if r[col] is not None]
        if values and all(WHOLE_DATE.match(v) for v in values):
            return (f"OUTPUT_DATE_PART: the question asks for the {spec['date_part']}, but a column holds whole dates "
                    f"({values[0]}). Return only the {spec['date_part']}, taken from the date the way the hint or data "
                    f"defines it, and group by that.")
    return None


def shape_problem(spec: Dict, rows: List[list]) -> Optional[str]:
    n = spec["values_per_row"]
    if n and len(rows) == n and rows and len(rows[0]) < n:
        return (f"OUTPUT_SHAPE: the question asks for {n} values in one row, but {n} rows came back. "
                f"Return one row with the {n} values as separate columns, in the order the question lists them, "
                f"without labels.")
    return None


def _repeats(database, table: str, column: str, cache: dict) -> bool:
    key = (id(database), table.lower(), column.lower())
    if key not in cache:
        try:
            _, rows = database.run_all(f'SELECT COUNT(*) > COUNT(DISTINCT "{column}") FROM "{table}"')
            cache[key] = bool(rows and rows[0][0])
        except Exception:
            cache[key] = False
    return cache[key]


def _date_like(database, table: str, column: str, cache: dict) -> bool:
    key = ("date", id(database), table.lower(), column.lower())
    if key not in cache:
        try:
            _, rows = database.run_all(f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT 5')
            cache[key] = bool(rows) and all(WHOLE_DATE.match(str(r[0])) for r in rows)
        except Exception:
            cache[key] = False
    return cache[key]


def grain_problem(spec: Dict, sql: str, database, cache: dict) -> Optional[str]:
    if spec["extreme"] == "none" or spec["grain"] == "none" or database is None:
        return None
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if tree.find(exp.Group) or tree.find(exp.Window):
        return None  # grouped or ranked somewhere: this check can't tell, so it stays quiet
    picks_one = any(isinstance(a, (exp.Max, exp.Min)) for a in tree.find_all(exp.AggFunc)) or (
        tree.find(exp.Order) is not None and tree.find(exp.Limit) is not None)
    tables = list(tree.find_all(exp.Table))
    if not picks_one or len(tables) != 1:
        return None
    table = tables[0].name
    grain = spec["grain"].rstrip("s")
    if grain in TIME_GRAINS:
        candidates = [c.name for c in tree.find_all(exp.Column)]
        units = [c for c in dict.fromkeys(candidates) if _date_like(database, table, c, cache)]
    else:
        try:
            columns, _ = database.run_all(f'SELECT * FROM "{table}" LIMIT 1')
        except Exception:
            return None
        units = [c for c in columns if grain.replace(" ", "") in c.lower().replace("_", "")]
    for column in units:
        if _repeats(database, table, column, cache):
            return (f"AGGREGATION_GRAIN: the question asks for the {spec['extreme']} per {spec['grain']}, but {table} "
                    f"has several rows per {spec['grain']} ({column} repeats), and this SQL takes the {spec['extreme']} "
                    f"of single rows. Group by the {spec['grain']}, total the measure, then take the {spec['extreme']}.")
    return None


def _whole(node) -> bool:
    """A value SQLite keeps as a whole number: a count, a sum of 0/1, or an integer literal."""
    node = node.unnest() if isinstance(node, exp.Paren) else node
    if isinstance(node, exp.Neg):
        return _whole(node.this)
    if isinstance(node, (exp.Count, exp.Sum)):
        return counts(node)
    if isinstance(node, exp.Literal):
        return not node.is_string and "." not in node.this and "e" not in node.this.lower()
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul)):
        return _whole(node.left) and _whole(node.right)
    return False


def division_problem(sql: str) -> Optional[str]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    for div in tree.find_all(exp.Div):
        if _whole(div.left) and _whole(div.right):
            return (f"NUMERIC_TYPE: {div.sql(dialect='sqlite')} divides two whole numbers, which SQLite rounds down "
                    f"(1 / 2 is 0). Multiply by 1.0 or CAST to REAL before dividing.")
    return None


def _empty(rows: List[list]) -> bool:
    return not rows or all(value is None for row in rows for value in row)


def grounding_problem(sql: str, rows: List[list], database, cache: dict) -> Optional[str]:
    """No rows came back: is a number from the question matched where the data doesn't have it?"""
    if database is None or not _empty(rows):
        return None
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    lookups = []
    for eq in tree.find_all(exp.EQ):
        for side, other in ((eq.left, eq.right), (eq.right, eq.left)):
            if isinstance(side, exp.Literal) and not side.is_string and "." in side.this:
                lookups.append((side.this, other.sql(dialect="sqlite")))
    if not lookups:
        return None
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = list(dict.fromkeys(t.name for t in tree.find_all(exp.Table) if t.name.lower() not in ctes))[:4]
    for value, used in lookups:
        where = []
        for table in tables:
            try:
                columns, _ = database.run_all(f'SELECT * FROM "{table}" LIMIT 1')
            except Exception:
                continue
            for column in columns[:40]:
                key = ("value", id(database), table.lower(), column.lower(), value)
                if key not in cache:
                    try:
                        _, found = database.run_all(f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" = {value}')
                        cache[key] = found[0][0] if found else 0
                    except Exception:
                        cache[key] = 0
                if cache[key] and used.strip('"`').split(".")[-1].strip('"`').lower() != column.lower():
                    where.append(f"{table}.{column} ({cache[key]} rows)")
        if where:
            return (f"VALUE_GROUNDING: no rows came back, and the SQL looks for {value} as {used}, but in the data "
                    f"{value} appears in {', '.join(where[:3])}. Match the value where the data holds it, and check "
                    f"that it picks out the intended record before computing anything from it.")
    return None


def problems(spec: Dict, sql: str, rows: List[list], database=None, cache: Optional[dict] = None) -> List[Tuple[str, str]]:
    """(code, message) for each clear mismatch between the spec and the SQL or its rows."""
    cache = {} if cache is None else cache
    found = []
    measure = measure_problem(spec["measure"], sql)
    if measure:
        found.append(("METRIC_MISMATCH", "METRIC_MISMATCH: " + measure))
    for check in (date_part_problem(spec, rows), shape_problem(spec, rows), grain_problem(spec, sql, database, cache),
                  grounding_problem(sql, rows, database, cache), division_problem(sql)):
        if check:
            found.append((check.split(":")[0], check))
    return found
