"""
Scout: what the user reads (meaning options, steps) is in everyday words, never database
names. The prompts ask the model for that; this makes sure of it, deterministically:
any table or column name that still slips through is rewritten in plain words.

    order_date        -> order date
    employee_id       -> employee ID
    orders.status     -> status
    `unit_price_cents` -> unit price cents
"""
import re
from typing import Iterable, List

WORDS = {"id": "ID", "ids": "IDs", "url": "URL", "sku": "SKU"}


def humanize(name: str) -> str:
    """A database name as words: order_date -> 'order date', employee_id -> 'employee ID'."""
    parts = [p for p in re.split(r"_+", name) if p]
    return " ".join(WORDS.get(p.lower(), p.lower()) for p in parts)


def plain_words(text: str, names: Iterable[str]) -> str:
    """
    Rewrites database names in text the user reads. Names that look like code (with an
    underscore, written as table.column, or in backticks) become words; plain single words
    (status, city) are already words.
    """
    text = re.sub(r"`([^`]+)`", r"\1", text)  # no code formatting
    known = {n.lower() for n in names}

    # table.column -> column (as words)
    text = re.sub(
        r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b",
        lambda m: humanize(m.group(2)) if m.group(1).lower() in known or m.group(2).lower() in known else m.group(0),
        text,
    )
    # snake_case names the database uses
    return re.sub(r"\b[A-Za-z]\w*_\w*\b", lambda m: humanize(m.group(0)) if m.group(0).lower() in known else m.group(0), text)


def schema_names(tables: List[dict]) -> List[str]:
    """Every table and column name, to recognise them in text."""
    names = []
    for table in tables:
        names.append(table["name"])
        names += [c["name"] for c in table.get("columns", [])]
    return names
