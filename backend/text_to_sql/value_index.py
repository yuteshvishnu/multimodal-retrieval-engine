"""
Version-20: the value index. Every short text value in a database, searchable by the words of
a question, so Scout knows how the data spells what the user asked about.

Offline (onboarding, with the catalog): each category and text column's distinct values go into
data/catalogs/<database>-<id>/values.sqlite, a small SQLite file with a full-text index (FTS5).
Free text, ids, numbers and dates are left out.

Online, no LLM:
    match(question, tables)          question words -> stored values: "discount" finds
                                     gasstations.Segment = 'Discount'; shown to the LLM
    has(table, column, value)        does the column hold exactly this value? (the checks use it
                                     for columns with too many values to list)
    similar(table, column, value)    values that share a word with it, else the closest spellings,
                                     for the check's message
"""
import difflib
import re
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

MAX_PER_COLUMN = 100_000   # distinct values indexed per column
MAX_VALUE_CHARS = 200      # longer values are free text, not names or labels
MATCHES_PER_COLUMN = 3
MATCHES_TOTAL = 12
MIN_SCORE = 0.5            # share of a value's words that must appear in the question
FUZZY_MAX = 20_000         # values compared by spelling when no word is shared ("Danial" -> 'Daniel')

# words that say nothing about which value is meant
STOPWORDS = set("""
a an and are as at be been being but by can could did do does each for from had has have how i if in
into is it its list many me more most much name names not of on or per please show should so than that
the their them then there these they this those to total was we were what when where which who whose
why will with would you your all any between both during give find get number amount average count
less least lowest highest biggest smallest top bottom first last same other only over under
""".split())


def words(text: str) -> List[str]:
    """Lowercased words worth matching: 3+ letters (or letters and digits), not stopwords, not bare numbers."""
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(w) >= 3 and w not in STOPWORDS and not w.isdigit()]


# ---------- offline ----------

def build(conn: sqlite3.Connection, alias: str, catalog: dict, out: Path) -> int:
    """Writes values.sqlite for one database; returns how many values were indexed."""
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript("""
        CREATE TABLE vals (id INTEGER PRIMARY KEY, tbl TEXT, col TEXT, value TEXT, n INTEGER);
        CREATE INDEX vals_exact ON vals (tbl, col, value);
        CREATE VIRTUAL TABLE vals_fts USING fts5(value, content='vals', content_rowid='id',
                                                 tokenize="unicode61 remove_diacritics 2");
    """)
    total = 0
    for name, table in catalog["tables"].items():
        for column in table["columns"]:
            if column["kind"] not in ("category", "text"):
                continue
            col = f'"{column["name"]}"'
            rows = conn.execute(
                f'SELECT {col}, COUNT(*) FROM "{alias}"."{name}" WHERE typeof({col}) = \'text\' '
                f"AND length({col}) <= {MAX_VALUE_CHARS} GROUP BY {col} LIMIT {MAX_PER_COLUMN}"
            ).fetchall()
            db.executemany("INSERT INTO vals (tbl, col, value, n) VALUES (?, ?, ?, ?)",
                           [(name, column["name"], v, n) for v, n in rows])
            total += len(rows)
    db.execute("INSERT INTO vals_fts(vals_fts) VALUES ('rebuild')")
    db.commit()
    db.close()
    tmp.replace(out)
    return total


# ---------- online ----------

class ValueIndex:
    """The value indexes of every attached database, searched together (read-only)."""

    def __init__(self, files: Dict[str, Path]):
        self._files = {alias: Path(f) for alias, f in files.items() if Path(f).exists()}
        self._local = threading.local()  # one connection per thread per file

    def _db(self, alias: str) -> sqlite3.Connection:
        conns = getattr(self._local, "conns", None)
        if conns is None:
            conns = self._local.conns = {}
        if alias not in conns:
            conns[alias] = sqlite3.connect(f"{self._files[alias].resolve().as_uri()}?mode=ro", uri=True)
        return conns[alias]

    def match(self, question: str, tables: Optional[Iterable[Tuple[str, str]]] = None) -> List[dict]:
        """
        Stored values that the question mentions, best first: [{"table", "column", "value", "n", "score"}].
        tables: (database, table) pairs to search, or None for all. A value matches when at least
        half of its words appear in the question ("discount" -> 'Discount'; "value for money" ->
        'Value for money'); a word inside a long value doesn't.
        """
        asked = set(words(question))
        if not asked:
            return []
        allowed = {(d, t.lower()) for d, t in tables} if tables is not None else None
        found = {}
        for alias in self._files:
            if allowed is not None and not any(d == alias for d, _ in allowed):
                continue
            db = self._db(alias)
            for word in asked:
                for tbl, col, value, n in db.execute(
                    "SELECT v.tbl, v.col, v.value, v.n FROM vals_fts f JOIN vals v ON v.id = f.rowid "
                    "WHERE vals_fts MATCH ? LIMIT 200", (f'"{word}"',)
                ):
                    if allowed is not None and (alias, tbl.lower()) not in allowed:
                        continue
                    have = words(value)
                    if not have:
                        continue
                    score = len(set(have) & asked) / len(set(have))
                    if score >= MIN_SCORE:
                        found[(tbl, col, value)] = {"table": tbl, "column": col, "value": value, "n": n, "score": score}
        ranked = sorted(found.values(), key=lambda m: (-m["score"], -m["n"], m["table"], m["column"], m["value"]))
        per_column, out = {}, []
        for m in ranked:
            key = (m["table"], m["column"])
            if per_column.get(key, 0) < MATCHES_PER_COLUMN and len(out) < MATCHES_TOTAL:
                per_column[key] = per_column.get(key, 0) + 1
                out.append(m)
        return out

    def indexed(self, table: str, column: str) -> bool:
        return any(self._db(a).execute("SELECT 1 FROM vals WHERE tbl = ? AND col = ? LIMIT 1", (table, column)).fetchone()
                   for a in self._files)

    def has(self, table: str, column: str, value: str) -> bool:
        return any(self._db(a).execute("SELECT 1 FROM vals WHERE tbl = ? AND col = ? AND value = ? LIMIT 1",
                                       (table, column, value)).fetchone() for a in self._files)

    def similar(self, table: str, column: str, value: str, limit: int = 5) -> List[str]:
        """Stored values of this column sharing a word with value (any case), most common first."""
        out = {}
        for alias in self._files:
            for word in words(value) or [str(value).lower()]:
                for v, n in self._db(alias).execute(
                    "SELECT v.value, v.n FROM vals_fts f JOIN vals v ON v.id = f.rowid "
                    "WHERE vals_fts MATCH ? AND v.tbl = ? AND v.col = ? LIMIT 50", (f'"{word}"', table, column)
                ):
                    out[v] = max(n, out.get(v, 0))
        if out:
            return [v for v, _ in sorted(out.items(), key=lambda x: -x[1])][:limit]
        # no shared word ("Danial"): the closest spellings among the column's values
        stored = [v for a in self._files for (v,) in self._db(a).execute(
            "SELECT value FROM vals WHERE tbl = ? AND col = ? LIMIT ?", (table, column, FUZZY_MAX))]
        lowered = {v.lower(): v for v in stored}
        return [lowered[m] for m in difflib.get_close_matches(str(value).lower(), list(lowered), n=limit, cutoff=0.75)]
