"""
Version-24: does the SQL compute the kind of number the question asks for? A code check, no AI.

The meaning step names the measure: "entity count" (how many people or things), "transaction count" (how
many events or records), "money" (an amount paid, spent, earned or charged) or "other". This module reads
the SQL's totals with the parser and flags only clear mismatches:

    a count is asked for, and the SQL has totals but none of them counts: it adds up values instead
    money is asked for, and every total in the SQL only counts

COUNT(...) counts; so does SUM over a condition or over a CASE that gives only 0, 1 or -1. Anything else, a
question without totals, or SQL the parser can't read, is never flagged. A flag gets one retry where the
flow has retries, and is otherwise kept as a warning; it never removes an answer.
"""
from typing import List, Optional

import sqlglot
from sqlglot import exp

MEASURES = ("entity count", "transaction count", "money", "other")
COUNTS = ("entity count", "transaction count")


def normalize(measure) -> str:
    text = str(measure or "").strip().lower()
    return text if text in MEASURES else "other"


def _zero_or_one(node) -> bool:
    """0, 1, -1 (a count going down, as in 'how many more A than B') or nothing."""
    node = node.unnest() if isinstance(node, exp.Paren) else node
    if isinstance(node, exp.Neg):
        node = node.this
    return isinstance(node, exp.Literal) and not node.is_string and node.this in ("0", "1", "0.0", "1.0") \
        or isinstance(node, exp.Null) or node is None


def counts(agg) -> bool:
    """COUNT(...), or SUM over a condition, or SUM over CASE / IIF that only gives 0 or 1."""
    if isinstance(agg, exp.Count):
        return True
    if not isinstance(agg, exp.Sum):
        return False
    arg = agg.this.unnest() if isinstance(agg.this, exp.Paren) else agg.this
    if isinstance(arg, (exp.Predicate, exp.Not, exp.And, exp.Or)):
        return True
    if isinstance(arg, exp.Case):
        return all(_zero_or_one(i.args.get("true")) for i in arg.args.get("ifs") or []) and _zero_or_one(arg.args.get("default"))
    if isinstance(arg, exp.If):
        return _zero_or_one(arg.args.get("true")) and _zero_or_one(arg.args.get("false"))
    return _zero_or_one(arg)


def totals(sql: str) -> Optional[List[exp.Expression]]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    return [a for a in tree.find_all(exp.AggFunc) if isinstance(a, (exp.Count, exp.Sum, exp.Avg))]


def measure_problem(measure, sql: str) -> Optional[str]:
    """A plain-words problem when the SQL clearly computes another kind of number, else None."""
    measure = normalize(measure)
    if measure == "other" or not sql:
        return None
    found = totals(sql)
    if not found:
        return None
    counting = [a for a in found if counts(a)]
    if measure in COUNTS and not counting:
        example = found[0].sql(dialect="sqlite")
        return (f"The question asks for a {measure}, but this SQL adds up values ({example}) instead of counting. "
                "Count the rows or entities, e.g. COUNT(...) or SUM(CASE WHEN ... THEN 1 ELSE 0 END).")
    if measure == "money" and len(counting) == len(found):
        return ("The question asks for an amount of money, but this SQL only counts rows. "
                "Add up or average the money amounts instead.")
    return None
