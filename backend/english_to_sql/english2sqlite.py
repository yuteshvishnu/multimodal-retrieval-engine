#!/usr/bin/env python3
"""english2sqlite: compile a strict English format into exact SQLite commands.

The translation is deterministic and uses no AI: one sentence always gives the
same SQL, and anything ambiguous or unknown is an error rather than a guess.
See FORMAT.md for the language.

    python3 english2sqlite.py 'get name from users where age is at least 18'
    python3 english2sqlite.py --db shop.db 'how many users are there'
    python3 english2sqlite.py -f script.txt
    python3 english2sqlite.py                # interactive
"""
from __future__ import annotations

import argparse
import difflib
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SQLITE_KEYWORDS = set("""
ABORT ACTION ADD AFTER ALL ALTER ALWAYS ANALYZE AND AS ASC ATTACH AUTOINCREMENT
BEFORE BEGIN BETWEEN BY CASCADE CASE CAST CHECK COLLATE COLUMN COMMIT CONFLICT
CONSTRAINT CREATE CROSS CURRENT CURRENT_DATE CURRENT_TIME CURRENT_TIMESTAMP
DATABASE DEFAULT DEFERRABLE DEFERRED DELETE DESC DETACH DISTINCT DO DROP EACH
ELSE END ESCAPE EXCEPT EXCLUDE EXCLUSIVE EXISTS EXPLAIN FAIL FILTER FIRST
FOLLOWING FOR FOREIGN FROM FULL GENERATED GLOB GROUP GROUPS HAVING IF IGNORE
IMMEDIATE IN INDEX INDEXED INITIALLY INNER INSERT INSTEAD INTERSECT INTO IS
ISNULL JOIN KEY LAST LEFT LIKE LIMIT MATCH MATERIALIZED NATURAL NO NOT NOTHING
NOTNULL NULL NULLS OF OFFSET ON OR ORDER OTHERS OUTER OVER PARTITION PLAN PRAGMA
PRECEDING PRIMARY QUERY RAISE RANGE RECURSIVE REFERENCES REGEXP REINDEX RELEASE
RENAME REPLACE RESTRICT RETURNING RIGHT ROLLBACK ROW ROWS SAVEPOINT SELECT SET
TABLE TEMP TEMPORARY THEN TIES TO TRANSACTION TRIGGER UNBOUNDED UNION UNIQUE
UPDATE USING VACUUM VALUES VIEW VIRTUAL WHEN WHERE WINDOW WITH WITHOUT
""".split())

# Words that mean something in this format, so they cannot be bare names.
# Backticks turn any of them back into a name: `order`, `from`.
RESERVED = set("""
from where whose group grouped having sorted sort ordered order limit limited
skip offset union intersect except join inner left right full cross
natural on using as called named and or not is in like glob regexp between
asc ascending desc descending then when else end set values into all everything
distinct unique of by to for per with returning do null true false case if
exists now today
""".split())

CLAUSE_WORDS = set("""
from where whose group grouped having sorted sort ordered order limit limited
skip offset for per returning union intersect except
""".split())

QUERY_VERBS = ["give me", "what is", "what are", "get", "select", "show", "find",
               "list", "fetch", "retrieve", "display", "return"]
ORDER_PHRASES = ["sorted by", "sort by", "ordered by", "order by"]
GROUP_PHRASES = ["grouped by", "group by", "for each", "per"]
LIMIT_PHRASES = ["limited to", "limit to", "limit"]
ALIAS_WORDS = ["as", "called", "named"]

JOINS = {
    "join": "JOIN", "joined with": "JOIN", "inner join": "INNER JOIN",
    "left join": "LEFT JOIN", "left outer join": "LEFT JOIN",
    "right join": "RIGHT JOIN", "right outer join": "RIGHT JOIN",
    "full join": "FULL JOIN", "full outer join": "FULL JOIN",
    "cross join": "CROSS JOIN", "natural join": "NATURAL JOIN",
    "natural left join": "NATURAL LEFT JOIN",
}

COMPARE_BARE = {
    "equal to": "=", "at least": ">=", "at most": "<=",
    "no less than": ">=", "no more than": "<=",
    "greater than or equal to": ">=", "more than or equal to": ">=",
    "less than or equal to": "<=", "fewer than or equal to": "<=",
    "greater than": ">", "more than": ">", "larger than": ">",
    "bigger than": ">", "higher than": ">", "less than": "<",
    "fewer than": "<", "smaller than": "<", "lower than": "<",
}
COMPARE_AFTER_IS = {
    **COMPARE_BARE, "the same as": "=", "on or after": ">=", "on or before": "<=",
    "above": ">", "over": ">", "after": ">", "below": "<", "under": "<", "before": "<",
}
NEGATED = {"=": "<>", ">": "<=", "<": ">=", ">=": "<", "<=": ">"}

# Phrase -> (SQL template, may take DISTINCT, is an aggregate)
PHRASE_FUNCTIONS = {
    "count of": ("COUNT({})", True), "number of": ("COUNT({})", True),
    "sum of": ("SUM({})", True), "total of": ("SUM({})", True),
    "average of": ("AVG({})", True), "avg of": ("AVG({})", True),
    "mean of": ("AVG({})", True),
    "maximum of": ("MAX({})", False), "max of": ("MAX({})", False),
    "highest": ("MAX({})", False), "largest": ("MAX({})", False),
    "minimum of": ("MIN({})", False), "min of": ("MIN({})", False),
    "lowest": ("MIN({})", False), "smallest": ("MIN({})", False),
    "uppercase of": ("UPPER({})", False), "lowercase of": ("LOWER({})", False),
    "length of": ("LENGTH({})", False), "absolute value of": ("ABS({})", False),
    "trim of": ("TRIM({})", False), "date of": ("DATE({})", False),
    "time of": ("TIME({})", False),
    "year of": ("CAST(strftime('%Y', {}) AS INTEGER)", False),
    "month of": ("CAST(strftime('%m', {}) AS INTEGER)", False),
    "day of": ("CAST(strftime('%d', {}) AS INTEGER)", False),
    "hour of": ("CAST(strftime('%H', {}) AS INTEGER)", False),
    "minute of": ("CAST(strftime('%M', {}) AS INTEGER)", False),
    "second of": ("CAST(strftime('%S', {}) AS INTEGER)", False),
    "weekday of": ("CAST(strftime('%w', {}) AS INTEGER)", False),
}
BARE_PHRASES = {"highest", "largest", "lowest", "smallest"}
COUNT_ALL = ["count of all rows", "count of everything", "count of rows",
             "number of rows", "count of *"]
RANKING = {"row number": "ROW_NUMBER()", "dense rank": "DENSE_RANK()",
           "percent rank": "PERCENT_RANK()", "rank": "RANK()"}
ORDER_DIRECTION = {
    "ascending": "ASC", "asc": "ASC", "lowest first": "ASC", "smallest first": "ASC",
    "oldest first": "ASC", "a to z": "ASC",
    "descending": "DESC", "desc": "DESC", "highest first": "DESC",
    "largest first": "DESC", "newest first": "DESC", "z to a": "DESC",
}
TYPE_ALIASES = {"STRING": "TEXT", "NUMBER": "NUMERIC"}
COLUMN_TYPE_STOPS = set("""
primary not required mandatory unique default defaulting check references collate
generated as without strict to auto autoincrement
""".split())
REF_ACTIONS = {"cascade": "CASCADE", "set null": "SET NULL", "set default": "SET DEFAULT",
               "restrict": "RESTRICT", "no action": "NO ACTION"}
ARITHMETIC_WORDS = ["plus", "minus", "times", "multiplied by", "divided by",
                    "modulo", "mod", "concatenated with", "followed by"]

VOCABULARY = sorted(
    RESERVED
    | {w for group in (QUERY_VERBS, ORDER_PHRASES, GROUP_PHRASES, LIMIT_PHRASES,
                       JOINS, COMPARE_AFTER_IS, PHRASE_FUNCTIONS, ORDER_DIRECTION,
                       ARITHMETIC_WORDS) for p in group for w in p.split()}
    | set("contains includes starts begins ends matches exceeds equals does has with".split())
    | set("""add insert update change delete remove create drop alter rename begin start
             commit rollback savepoint release vacuum analyze reindex pragma attach detach
             explain import dump describe how many""".split())
)


class FormatError(Exception):
    """The input does not follow the format."""

    def __init__(self, message: str, source: str = "", pos: int | None = None):
        super().__init__(message)
        self.message, self.source, self.pos = message, source, pos

    def __str__(self) -> str:
        if self.pos is None or not self.source:
            return self.message
        start = self.source.rfind("\n", 0, self.pos) + 1
        end = self.source.find("\n", self.pos)
        line = self.source[start:end if end >= 0 else None]
        return f"{self.message}\n  {line}\n  {' ' * (self.pos - start)}^"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_PART = r"[A-Za-z_][A-Za-z0-9_$]*|`(?:[^`]|``)+`"
_TOKEN_RE = re.compile(rf"""
    (?P<space>\s+)
  | (?P<comment>(?:--|\#)[^\n]*)
  | (?P<blob>[xX]'[0-9a-fA-F]*')
  | (?P<str>"(?:[^"]|"")*"|'(?:[^']|'')*')
  | (?P<num>0[xX][0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>(?:{_PART})(?:\.(?:{_PART}|\*))*)
  | (?P<param>\?\d*|[:@$][A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>->>|->|\|\||<<|>>|<=|>=|==|!=|<>|[-+*/%&|~<>=(),;:])
""", re.VERBOSE)


@dataclass
class Token:
    kind: str  # word, ident, num, str, blob, param, op, eof
    value: str
    pos: int

    @property
    def low(self) -> str:
        return self.value.lower() if self.kind == "word" else ""


def tokenize(text: str) -> list[Token]:
    tokens, i = [], 0
    while i < len(text):
        m = _TOKEN_RE.match(text, i)
        if not m:
            if text[i] in "\"'`":
                raise FormatError("This quote is never closed.", text, i)
            raise FormatError(f"Unexpected character {text[i]!r}.", text, i)
        kind, value = m.lastgroup, m.group()
        if kind == "name":
            simple = "." not in value and "`" not in value
            tokens.append(Token("word" if simple else "ident", value, i))
        elif kind not in ("space", "comment"):
            tokens.append(Token(kind, value, i))
        i = m.end()
    # A trailing question mark is punctuation, not a bound parameter.
    if tokens and tokens[-1].value == "?":
        tokens.pop()
    tokens = [t for j, t in enumerate(tokens)
              if not (t.value == "?" and j + 1 < len(tokens) and tokens[j + 1].value == ";")]
    tokens.append(Token("eof", "", len(text)))
    return tokens


def quote_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def str_value(raw: str) -> str:
    q = raw[0]
    return raw[1:-1].replace(q * 2, q)


def quote_ident(name: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", name) and name.upper() not in SQLITE_KEYWORDS:
        return name
    return '"' + name.replace('"', '""') + '"'


def render_ident(raw: str) -> str:
    parts = re.findall(r"`(?:[^`]|``)+`|[^.`]+", raw)
    out = []
    for part in parts:
        if part == "*":
            out.append("*")
        elif part.startswith("`"):
            out.append(quote_ident(part[1:-1].replace("``", "`")))
        else:
            out.append(quote_ident(part))
    return ".".join(out)


# ---------------------------------------------------------------------------
# Parser: reads the English format and emits SQL text directly
# ---------------------------------------------------------------------------

def is_single_total(sql: str) -> bool:
    """True if the SQL is exactly one total, such as SUM(x) or COUNT(*), and nothing more."""
    if not re.match(r"(COUNT|SUM|AVG|MAX|MIN|TOTAL|GROUP_CONCAT)\(", sql, re.IGNORECASE):
        return False
    depth, quote = 0, None
    for position, ch in enumerate(sql):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"`":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return position == len(sql) - 1
    return False


class Parser:
    def __init__(self, text: str):
        self.text = text
        self.toks = tokenize(text)
        self.i = 0
        # True while "and" separates list items instead of meaning boolean AND.
        self.no_and = False

    # --- token helpers ---------------------------------------------------
    def peek(self, k: int = 0) -> Token:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self) -> Token:
        tok = self.peek()
        self.i += 1
        return tok

    def fail(self, message: str, tok: Token | None = None):
        tok = tok or self.peek()
        raise FormatError(message, self.text, tok.pos)

    def at(self, *words: str) -> bool:
        return all(self.peek(k).low == w for k, w in enumerate(words))

    def accept(self, *words: str) -> bool:
        if self.at(*words):
            self.i += len(words)
            return True
        return False

    def expect(self, *words: str) -> None:
        if not self.accept(*words):
            self.fail(f"Expected '{' '.join(words)}' but found {self.describe()}.{self.hint()}")

    def at_op(self, *ops: str) -> bool:
        tok = self.peek()
        return tok.kind == "op" and tok.value in ops

    def accept_op(self, *ops: str) -> bool:
        if self.at_op(*ops):
            self.i += 1
            return True
        return False

    def expect_op(self, op: str) -> None:
        if not self.accept_op(op):
            self.fail(f"Expected '{op}' but found {self.describe()}.")

    def at_phrase(self, phrases) -> str | None:
        for phrase in sorted(phrases, key=lambda p: -len(p.split())):
            if self.at(*phrase.split()):
                return phrase
        return None

    def accept_phrase(self, phrases) -> str | None:
        phrase = self.at_phrase(phrases)
        if phrase:
            self.i += len(phrase.split())
        return phrase

    def at_end(self, k: int = 0) -> bool:
        tok = self.peek(k)
        return tok.kind == "eof" or (tok.kind == "op" and tok.value == ";")

    def describe(self) -> str:
        tok = self.peek()
        return "the end of the sentence" if tok.kind == "eof" else f"'{tok.value}'"

    def hint(self) -> str:
        tok = self.peek()
        if tok.kind != "word":
            return ""
        close = difflib.get_close_matches(tok.low, VOCABULARY, n=1, cutoff=0.7)
        return f" Did you mean '{close[0]}'?" if close and close[0] != tok.low else ""

    # --- names and lists -------------------------------------------------
    def name(self, what: str = "name") -> str:
        while self.accept("the"):
            pass
        tok = self.peek()
        if tok.kind == "ident":
            self.next()
            return render_ident(tok.value)
        if tok.kind == "word":
            if tok.low in RESERVED:
                self.fail(f"'{tok.value}' is a word of this format. If you mean a {what} "
                          f"called that, write it in backticks: `{tok.value}`.")
            self.next()
            return quote_ident(tok.value)
        self.fail(f"Expected a {what} but found {self.describe()}.")

    def alias(self) -> str:
        if self.peek().kind == "str":
            return quote_ident(str_value(self.next().value))
        return self.name("alias")

    def more_items(self, and_sep: bool = True) -> bool:
        if self.at_op(","):
            following = self.peek(1)
            if following.low in CLAUSE_WORDS:
                return False
            self.next()
            self.accept("and")
            return True
        return and_sep and self.accept("and")

    def name_list_in_parens(self, what: str = "column") -> list[str]:
        self.expect_op("(")
        names = [self.name(what)]
        while self.accept_op(","):
            names.append(self.name(what))
        self.expect_op(")")
        return names

    # --- statements ------------------------------------------------------
    def program(self) -> list[str]:
        out = []
        while self.peek().kind != "eof":
            if self.accept_op(";"):
                continue
            out.append(self.statement())
            if not self.at_end():
                self.fail(f"I don't understand {self.describe()} here.{self.hint()}")
        return out

    def statement(self) -> str:
        if self.accept("explain"):
            prefix = "EXPLAIN QUERY PLAN " if self.accept("query", "plan") else "EXPLAIN "
            return prefix + self.statement()
        meta = self.meta_command()
        if meta:
            return meta
        if self.at("with"):
            return self.with_prefix() + self.statement_body()
        return self.statement_body()

    def statement_body(self) -> str:
        if self.at_query():
            return self.query()
        tok = self.peek()
        word = tok.low
        if self.at("add", "column"):
            return self.add_column()
        if word in ("add", "insert", "replace"):
            return self.insert()
        if word in ("update", "change"):
            return self.update()
        if word in ("delete", "remove"):
            return self.delete()
        if word == "create":
            return self.create()
        if word == "drop":
            return self.drop()
        if word in ("alter", "rename"):
            return self.alter()
        simple = self.utility()
        if simple:
            return simple
        self.fail(f"A sentence can't start with {self.describe()}. Start with get, add, "
                  f"update, delete, create, drop, how many, ...{self.hint()}")

    def meta_command(self) -> str | None:
        if (self.at("show", "tables") or self.at("list", "tables")) and self.at_end(2):
            self.i += 2
            return ".tables"
        if self.at("show", "databases") and self.at_end(2):
            self.i += 2
            return ".databases"
        if self.at("show", "schema") and (self.at_end(2) or self.peek(2).low in ("of", "for")):
            self.i += 2
            if self.at_end():
                return ".schema"
            self.next()
            return ".schema " + self.name("table name")
        if self.accept("describe"):
            self.accept("table")
            return ".schema " + self.name("table name")
        if self.at("show", "indexes") and (self.at_end(2) or self.peek(2).low in ("of", "on", "for")):
            self.i += 2
            if self.at_end():
                return ".indexes"
            self.next()
            return ".indexes " + self.name("table name")
        if self.at("show", "columns") and self.peek(2).low in ("of", "in", "from"):
            self.i += 3
            return f"PRAGMA table_info({self.name('table name')})"
        if self.accept("dump"):
            if self.accept("table"):
                return ".dump " + self.name("table name")
            self.accept("database")
            return ".dump"
        if self.accept("import"):
            path = self.string("file path")
            self.expect("into")
            return f".import --csv {shell_arg(path)} {self.name('table name')}"
        if self.accept("run", "file"):
            return ".read " + shell_arg(self.string("file path"))
        if self.accept("backup", "to"):
            return ".backup " + shell_arg(self.string("file path"))
        return None

    def string(self, what: str) -> str:
        if self.peek().kind != "str":
            self.fail(f"Expected a quoted {what} but found {self.describe()}.")
        return str_value(self.next().value)

    def utility(self) -> str | None:
        if self.accept("begin") or self.accept("start"):
            mode = ""
            for m in ("deferred", "immediate", "exclusive"):
                if self.accept(m):
                    mode = " " + m.upper()
            self.accept("transaction")
            return "BEGIN" + mode
        if self.accept("commit") or self.accept("save", "changes"):
            self.accept("transaction")
            return "COMMIT"
        if self.accept("rollback") or self.accept("undo", "changes"):
            if self.accept("to"):
                self.accept("savepoint")
                return "ROLLBACK TO " + self.name("savepoint")
            self.accept("transaction")
            return "ROLLBACK"
        if self.accept("savepoint"):
            return "SAVEPOINT " + self.name("savepoint")
        if self.accept("release"):
            self.accept("savepoint")
            return "RELEASE " + self.name("savepoint")
        if self.accept("vacuum"):
            if self.accept("into"):
                return "VACUUM INTO " + quote_str(self.string("file path"))
            return "VACUUM"
        for verb in ("analyze", "reindex"):
            if self.accept(verb):
                return verb.upper() + ("" if self.at_end() else " " + self.name("name"))
        if self.accept("pragma"):
            return self.pragma()
        if self.accept("attach"):
            path = self.string("file path")
            self.expect("as")
            return f"ATTACH {quote_str(path)} AS {self.name('database name')}"
        if self.accept("detach"):
            return "DETACH " + self.name("database name")
        return None

    def pragma(self) -> str:
        sql = "PRAGMA " + self.name("pragma name")
        if self.accept_op("="):
            return f"{sql} = {self.pragma_value()}"
        if self.accept_op("("):
            value = self.pragma_value()
            self.expect_op(")")
            return f"{sql}({value})"
        return sql

    def pragma_value(self) -> str:
        tok = self.next()
        if tok.kind == "str":
            return quote_str(str_value(tok.value))
        if tok.kind in ("word", "num"):
            return tok.value
        if tok.kind == "ident":
            return render_ident(tok.value)
        self.fail("Expected a pragma value.", tok)

    # --- queries ---------------------------------------------------------
    def at_query(self) -> bool:
        if self.at("how", "many") or self.at("with"):
            return True
        if self.at("values") and self.peek(1).value == "(":
            return True
        phrase = self.at_phrase(QUERY_VERBS)
        if not phrase:
            return False
        following = self.peek(len(phrase.split()))
        # "(list + 1)" is arithmetic on a column called list, not a query.
        return not (following.kind == "op" and following.value not in ("*", "(", "-"))

    def with_prefix(self) -> str:
        self.expect("with")
        head = "WITH RECURSIVE " if self.accept("recursive") else "WITH "
        ctes = []
        while True:
            name = self.name("name")
            if self.at_op("("):
                name += "(" + ", ".join(self.name_list_in_parens()) + ")"
            self.expect("as")
            self.expect_op("(")
            ctes.append(f"{name} AS ({self.query()})")
            self.expect_op(")")
            if not self.accept_op(","):
                break
        return head + ", ".join(ctes) + " "

    def query(self) -> str:
        saved, self.no_and = self.no_and, False
        prefix = self.with_prefix() if self.at("with") else ""
        core, tail = self.select_core()
        sql = prefix + core
        while True:
            op = self.accept_phrase(["union all", "union", "intersect", "except"])
            if not op:
                break
            if tail:
                self.fail("Sorting and limits must come after the last query of a "
                          f"{op}. Move them to the end.")
            core, tail = self.select_core()
            sql += f" {op.upper()} {core}"
        self.no_and = saved
        return sql + tail

    def select_core(self) -> tuple[str, str]:
        if self.accept("how", "many"):
            return self.how_many()
        if self.at("values"):
            self.next()
            return "VALUES " + self.row_list(), ""
        if not self.accept_phrase(QUERY_VERBS):
            self.fail(f"Expected a query (get ...) but found {self.describe()}.{self.hint()}")
        distinct = "DISTINCT " if self.accept("unique") or self.accept("distinct") else ""
        while self.accept("the"):
            pass
        limit = None
        if self.at("top") and self.peek(1).kind == "num":
            self.next()
            limit = self.next().value
        items = self.select_list()
        return self.clauses(f"SELECT {distinct}{items}", limit)

    def how_many(self) -> tuple[str, str]:
        if self.accept("unique") or self.accept("distinct"):
            column = self.name("column name")
            if not self.at("from"):
                self.fail("Write: how many unique <column> from <table>.")
            return self.clauses(f"SELECT COUNT(DISTINCT {column})", None)
        self.accept("rows")
        self.accept("in")
        source = self.table_ref()
        self.accept("are", "there")
        self.accept("exist")
        return self.clauses("SELECT COUNT(*)", None, given_from=source)

    def select_list(self) -> str:
        items = [self.select_item()]
        while self.more_items():
            items.append(self.select_item())
        return ", ".join(items)

    def select_item(self) -> str:
        if self.accept_op("*") or self.accept("everything"):
            return "*"
        if self.accept("every", "column") or self.accept("all", "columns"):
            return "*"
        if self.at("all") and self.peek(1).low == "from":
            self.next()
            return "*"
        expr = self.list_value()
        if self.accept_phrase(ALIAS_WORDS):
            expr += " AS " + self.alias()
        return expr

    def clauses(self, head: str, limit: str | None, given_from: str | None = None) -> tuple[str, str]:
        found: dict[str, str] = {}
        if given_from:
            found["from"] = given_from
        if limit:
            found["limit"] = limit

        def once(key: str):
            if key in found:
                self.fail(f"This sentence has two {key} parts.")

        while True:
            if self.at_op(",") and self.peek(1).low in CLAUSE_WORDS:
                self.next()
            if self.at("from"):
                once("from")
                self.next()
                found["from"] = self.from_clause()
            elif self.at("where") or self.at("whose"):
                once("where")
                self.next()
                found["where"] = self.full_expr()
            elif self.at_phrase(GROUP_PHRASES):
                once("group")
                self.accept_phrase(GROUP_PHRASES)
                found["group"] = self.expr_items()
            elif self.at("having"):
                once("having")
                self.next()
                found["having"] = self.full_expr()
            elif self.at_phrase(ORDER_PHRASES):
                once("order")
                self.accept_phrase(ORDER_PHRASES)
                found["order"] = self.order_items()
            elif self.at_phrase(LIMIT_PHRASES):
                once("limit")
                self.accept_phrase(LIMIT_PHRASES)
                found["limit"] = self.value()
                self.accept("rows")
            elif self.at("skip") or self.at("offset"):
                once("offset")
                self.next()
                found["offset"] = self.value()
                self.accept("rows")
            else:
                break

        sql = head
        if "from" in found:
            sql += " FROM " + found["from"]
        if "where" in found:
            sql += " WHERE " + found["where"]
        if "group" in found:
            sql += " GROUP BY " + found["group"]
        if "having" in found:
            sql += " HAVING " + found["having"]
        tail = ""
        if "order" in found:
            tail += " ORDER BY " + found["order"]
        if "limit" in found:
            tail += " LIMIT " + found["limit"]
        if "offset" in found:
            # SQLite only accepts OFFSET after a LIMIT; -1 means no limit.
            tail += ("" if "limit" in found else " LIMIT -1") + " OFFSET " + found["offset"]
        return sql, tail

    def from_clause(self) -> str:
        sql = self.table_ref()
        while True:
            if self.more_items(and_sep=False):
                sql += ", " + self.table_ref()
                continue
            join = self.accept_phrase(JOINS)
            if not join:
                return sql
            sql += f" {JOINS[join]} {self.table_ref()}"
            if self.accept("on"):
                sql += " ON " + self.full_expr()
            elif self.accept("using"):
                sql += " USING (" + ", ".join(self.name_list_in_parens()) + ")"

    def table_ref(self) -> str:
        if self.accept_op("("):
            sql = f"({self.query()})"
            self.expect_op(")")
        else:
            sql = self.name("table name")
            if self.at_op("("):
                self.next()
                args = [] if self.at_op(")") else self.expr_list()
                self.expect_op(")")
                sql += "(" + ", ".join(args) + ")"
        if self.accept_phrase(ALIAS_WORDS):
            sql += " AS " + self.alias()
        return sql

    def expr_list(self) -> list[str]:
        items = [self.full_expr()]
        while self.accept_op(","):
            items.append(self.full_expr())
        return items

    def expr_items(self) -> str:
        items = [self.list_value()]
        while self.more_items():
            items.append(self.list_value())
        return ", ".join(items)

    def order_items(self) -> str:
        items = [self.order_term()]
        while self.more_items() or self.accept("then"):
            items.append(self.order_term())
        return ", ".join(items)

    def order_term(self) -> str:
        sql = self.list_value()
        direction = self.accept_phrase(ORDER_DIRECTION)
        if direction:
            sql += " " + ORDER_DIRECTION[direction]
        if self.accept("nulls", "first"):
            sql += " NULLS FIRST"
        elif self.accept("nulls", "last"):
            sql += " NULLS LAST"
        return sql

    def row_list(self) -> str:
        rows = [self.row()]
        while self.accept_op(","):
            rows.append(self.row())
        return ", ".join(rows)

    def row(self) -> str:
        self.expect_op("(")
        items = self.expr_list()
        self.expect_op(")")
        return "(" + ", ".join(items) + ")"

    # --- expressions -----------------------------------------------------
    def full_expr(self) -> str:
        saved, self.no_and = self.no_and, False
        try:
            return self.or_expr()
        finally:
            self.no_and = saved

    def list_value(self) -> str:
        saved, self.no_and = self.no_and, True
        try:
            return self.or_expr()
        finally:
            self.no_and = saved

    def or_expr(self) -> str:
        sql = self.and_expr()
        while self.accept("or"):
            sql += " OR " + self.and_expr()
        return sql

    def and_expr(self) -> str:
        sql = self.not_expr()
        while not self.no_and and self.accept("and"):
            sql += " AND " + self.not_expr()
        return sql

    def not_expr(self) -> str:
        if self.accept("not"):
            return "NOT " + self.not_expr()
        return self.predicate()

    def predicate(self) -> str:
        left = self.value()
        tok = self.peek()
        if tok.kind == "op" and tok.value in ("=", "==", "!=", "<>", "<", "<=", ">", ">="):
            self.next()
            op = {"==": "=", "!=": "<>"}.get(tok.value, tok.value)
            return f"{left} {op} {self.value()}"
        if self.accept("is"):
            return self.after_is(left, self.accept("not"))
        if self.accept("does", "not"):
            if self.accept("equal"):
                return f"{left} <> {self.value()}"
            if self.accept("contain") or self.accept("include"):
                return self.like(left, True, "%", "%")
            if self.accept("start", "with") or self.accept("begin", "with"):
                return self.like(left, True, "", "%")
            if self.accept("end", "with"):
                return self.like(left, True, "%", "")
            if self.accept("match"):
                return f"{left} NOT LIKE {self.value()}"
            self.fail("After 'does not', write equal, contain, start with, end with or match.")
        if self.at("not") and self.peek(1).low in ("in", "like", "glob", "between", "regexp"):
            self.next()
            return self.membership(left, True)
        if self.peek().low in ("in", "like", "glob", "between", "regexp"):
            return self.membership(left, False)
        phrase = self.accept_phrase(COMPARE_BARE)
        if phrase:
            return f"{left} {COMPARE_BARE[phrase]} {self.value()}"
        if self.accept("equals"):
            return f"{left} = {self.value()}"
        if self.accept("exceeds"):
            return f"{left} > {self.value()}"
        if self.accept("contains") or self.accept("includes"):
            return self.like(left, False, "%", "%")
        if self.accept("starts", "with") or self.accept("begins", "with"):
            return self.like(left, False, "", "%")
        if self.accept("ends", "with"):
            return self.like(left, False, "%", "")
        if self.accept("matches"):
            return f"{left} LIKE {self.value()}"
        if self.accept("has", "no", "value"):
            return f"{left} IS NULL"
        if self.accept("has", "a", "value"):
            return f"{left} IS NOT NULL"
        return left

    def after_is(self, left: str, neg: bool) -> str:
        if self.accept("null") or self.accept("missing"):
            return f"{left} IS {'NOT ' if neg else ''}NULL"
        if self.accept("true"):
            return f"{left} IS {'NOT ' if neg else ''}TRUE"
        if self.accept("false"):
            return f"{left} IS {'NOT ' if neg else ''}FALSE"
        if self.accept("empty"):
            return f"{left} {'<>' if neg else '='} ''"
        if self.accept("distinct", "from"):
            return f"{left} IS {'NOT ' if neg else ''}DISTINCT FROM {self.value()}"
        if self.peek().low in ("in", "like", "glob", "between", "regexp") or self.at("one", "of"):
            return self.membership(left, neg)
        phrase = self.accept_phrase(COMPARE_AFTER_IS)
        if phrase:
            op = COMPARE_AFTER_IS[phrase]
            return f"{left} {NEGATED[op] if neg else op} {self.value()}"
        return f"{left} {'<>' if neg else '='} {self.value()}"

    def membership(self, left: str, neg: bool) -> str:
        not_ = "NOT " if neg else ""
        if self.accept("between"):
            low = self.value()
            self.expect("and")
            return f"{left} {not_}BETWEEN {low} AND {self.value()}"
        if self.accept("like"):
            return f"{left} {not_}LIKE {self.value()}"
        if self.accept("glob"):
            return f"{left} {not_}GLOB {self.value()}"
        if self.accept("regexp"):
            return f"{left} {not_}REGEXP {self.value()}"
        if not self.accept("in"):
            self.expect("one", "of")
        if not self.at_op("("):
            return f"{left} {not_}IN {self.name('table name')}"
        self.next()
        if self.at_query():
            inner = self.query()
        elif self.at_op(")"):
            inner = ""
        else:
            inner = ", ".join(self.expr_list())
        self.expect_op(")")
        return f"{left} {not_}IN ({inner})"

    def like(self, left: str, neg: bool, before: str, after: str) -> str:
        not_ = "NOT " if neg else ""
        if self.peek().kind == "str":
            text = str_value(self.next().value)
            special = any(c in text for c in "%_\\")
            for c in "\\%_":
                text = text.replace(c, "\\" + c)
            escape = " ESCAPE '\\'" if special else ""
            return f"{left} {not_}LIKE {quote_str(before + text + after)}{escape}"
        parts = [self.value()]
        if before:
            parts.insert(0, quote_str(before))
        if after:
            parts.append(quote_str(after))
        return f"{left} {not_}LIKE {' || '.join(parts)}"

    def value(self) -> str:
        """Arithmetic, plus the suffixes that apply to all of it."""
        sql = self.bitwise()
        while True:
            if self.accept("rounded"):
                if self.accept("to"):
                    digits = self.value_digits()
                    sql = f"ROUND({sql}, {digits})"
                else:
                    sql = f"ROUND({sql})"
            elif self.accept("converted", "to"):
                sql = f"CAST({sql} AS {self.type_name(single_word=True)})"
            else:
                return sql

    def value_digits(self) -> str:
        tok = self.next()
        if tok.kind != "num":
            self.fail("Write: rounded to <number> places.", tok)
        if not self.accept("decimal", "places"):
            if not self.accept("places"):
                self.accept("decimals")
        return tok.value

    def bitwise(self) -> str:
        sql = self.additive()
        while self.at_op("&", "|", "<<", ">>"):
            op = self.next().value
            sql += f" {op} {self.additive()}"
        return sql

    def additive(self) -> str:
        sql = self.multiplicative()
        while True:
            if self.at_op("+", "-"):
                op = self.next().value
            elif self.accept("plus"):
                op = "+"
            elif self.accept("minus"):
                op = "-"
            else:
                return sql
            sql += f" {op} {self.multiplicative()}"

    def multiplicative(self) -> str:
        sql = self.concat()
        while True:
            if self.at_op("*", "/", "%"):
                op = self.next().value
            elif self.accept("times") or self.accept("multiplied", "by"):
                op = "*"
            elif self.accept("divided", "by"):
                op = "/"
            elif self.accept("modulo") or self.accept("mod"):
                op = "%"
            else:
                return sql
            if op == "/":  # always a decimal result: in SQLite, 1 / 2 is 0
                op = "* 1.0 /"
            sql += f" {op} {self.concat()}"

    def concat(self) -> str:
        sql = self.unary()
        while True:
            if self.at_op("||", "->", "->>"):
                op = self.next().value
            elif self.accept("concatenated", "with") or self.accept("followed", "by"):
                op = "||"
            else:
                return sql
            sql += f" {op} {self.unary()}"

    def unary(self) -> str:
        if self.at_op("-", "+", "~"):
            return self.next().value + self.unary()
        sql = self.primary()
        while True:
            if self.accept("collate"):
                sql += " COLLATE " + self.name("collation")
            elif self.accept("ignoring", "case"):
                sql += " COLLATE NOCASE"
            else:
                return sql

    def primary(self) -> str:
        while self.accept("the"):
            pass
        tok = self.peek()
        if tok.kind == "num":
            self.next()
            return self.after_number(tok)
        if tok.kind == "str":
            self.next()
            return quote_str(str_value(tok.value))
        if tok.kind in ("blob", "param"):
            self.next()
            return tok.value
        if tok.kind == "ident":
            self.next()
            return render_ident(tok.value)
        if tok.kind == "op" and tok.value == "(":
            return self.parenthesized()
        if tok.kind == "word":
            return self.word_primary(tok)
        self.fail(f"Expected a value but found {self.describe()}.")

    def parenthesized(self) -> str:
        self.next()
        if self.at_query():
            sql = f"({self.query()})"
            self.expect_op(")")
            return sql
        items = self.expr_list()
        if len(items) == 1 and (self.at("where") or self.at("from")) and is_single_total(items[0]):
            return self.total_with_clauses(items[0])
        self.expect_op(")")
        return "(" + ", ".join(items) + ")"

    def total_with_clauses(self, total: str) -> str:
        """(sum of x where c) filters the query's own rows; (sum of x from t where c) is a query of its own."""
        saved, self.no_and = self.no_and, False
        if self.source_before_closing():
            core, tail = self.clauses(f"SELECT {total}", None)
            sql = f"({core}{tail})"
        else:
            self.expect("where")
            sql = f"({total} FILTER (WHERE {self.full_expr()}))"
        self.no_and = saved
        self.expect_op(")")
        return sql

    def source_before_closing(self) -> bool:
        """True if a 'from' comes before this group's closing parenthesis, outside any inner group."""
        depth, k = 0, 0
        while True:
            tok = self.peek(k)
            if tok.kind == "eof" or (depth == 0 and tok.kind == "op" and tok.value == ")"):
                return False
            if tok.kind == "op" and tok.value == "(":
                depth += 1
            elif tok.kind == "op" and tok.value == ")":
                depth -= 1
            elif depth == 0 and tok.kind == "word" and tok.low == "from":
                return True
            k += 1

    def after_number(self, tok: Token) -> str:
        unit = self.peek().low
        if re.fullmatch(r"(second|minute|hour|day|week|month|year)s?", unit):
            ago = self.peek(1).low == "ago"
            ahead = self.at(unit, "from", "now")
            if ago or ahead:
                self.i += 2 if ago else 3
                sign = "-" if ago else "+"
                unit = unit.rstrip("s")
                amount = tok.value
                if unit == "week":
                    amount, unit = str(float(amount) * 7).removesuffix(".0"), "day"
                fn = "datetime" if unit in ("second", "minute", "hour") else "date"
                return f"{fn}('now', '{sign}{amount} {unit}s')"
        return tok.value

    def word_primary(self, tok: Token) -> str:
        low = tok.low
        literals = {"null": "NULL", "true": "TRUE", "false": "FALSE",
                    "now": "CURRENT_TIMESTAMP", "today": "CURRENT_DATE"}
        if low in literals:
            self.next()
            return literals[low]
        for phrase, sql in (("current date", "CURRENT_DATE"), ("current time", "CURRENT_TIME"),
                            ("current timestamp", "CURRENT_TIMESTAMP")):
            if self.accept(*phrase.split()):
                return sql
        if low in ("current_date", "current_time", "current_timestamp"):
            self.next()
            return low.upper()
        if self.accept("case"):
            return self.case_expr()
        if self.accept("if"):
            return self.if_expr()
        if self.accept("exists"):
            self.expect_op("(")
            sql = f"EXISTS ({self.query()})"
            self.expect_op(")")
            return sql
        if self.accept("raw"):
            return self.string("raw SQL expression")
        if low == "cast" and self.peek(1).value == "(":
            return self.cast_call()
        if self.accept_phrase(COUNT_ALL):
            return self.window_suffix("COUNT(*)")
        rank = self.at_phrase(RANKING)
        if rank and self.peek(len(rank.split())).low == "over":
            self.accept_phrase([rank])
            return self.window_suffix(RANKING[rank])
        if self.at("grouped", "list", "of"):
            self.i += 3
            return self.group_concat()
        if self.at("substring", "of"):
            self.i += 2
            return self.substring()
        phrase = self.at_phrase(PHRASE_FUNCTIONS)
        if phrase and self.phrase_has_argument(phrase):
            self.accept_phrase([phrase])
            return self.phrase_call(phrase)
        if self.peek(1).value == "(" and low not in RESERVED:
            return self.function_call()
        if low in RESERVED:
            self.fail(f"Expected a value but found '{tok.value}'. If you mean a column called "
                      f"that, write it in backticks: `{tok.value}`.")
        self.next()
        return quote_ident(tok.value)

    def phrase_has_argument(self, phrase: str) -> bool:
        if phrase not in BARE_PHRASES:
            return True
        following = self.peek(1)
        if following.kind == "op":
            return following.value == "("
        return following.kind in ("word", "ident") and following.low not in RESERVED

    def phrase_call(self, phrase: str) -> str:
        template, allows_distinct = PHRASE_FUNCTIONS[phrase]
        distinct = ""
        if allows_distinct and (self.accept("unique") or self.accept("distinct")):
            distinct = "DISTINCT "
        start = self.i
        arg = self.unary()
        self.forbid_arithmetic(phrase)
        if self.wrapped_in_parens(start) and not arg.startswith(("(SELECT", "(WITH", "(VALUES")):
            arg = arg[1:-1]  # sum of (a * b) is SUM(a * b), not SUM((a * b))
        return self.window_suffix(template.format(distinct + arg))

    def wrapped_in_parens(self, start: int) -> bool:
        """True if the tokens from start up to here are one (...) group."""
        depth = 0
        for k in range(start, self.i):
            tok = self.toks[k]
            if tok.kind == "op" and tok.value == "(":
                depth += 1
            elif tok.kind == "op" and tok.value == ")":
                depth -= 1
            if depth == 0:
                return k == self.i - 1 and k > start
        return False

    def forbid_arithmetic(self, phrase: str) -> None:
        if self.at_op("+", "-", "*", "/", "%", "||") or self.at_phrase(ARITHMETIC_WORDS):
            self.fail(f"Ambiguous: does '{phrase}' cover only the next value, or everything "
                      f"after it? Use parentheses: {phrase} (a * b), or (({phrase} a) * b).")

    def group_concat(self) -> str:
        distinct = "DISTINCT " if self.accept("unique") or self.accept("distinct") else ""
        arg = self.unary()
        if self.accept("separated", "by"):
            arg += ", " + self.value()
        return self.window_suffix(f"GROUP_CONCAT({distinct}{arg})")

    def substring(self) -> str:
        arg = self.unary()
        self.expect("from")
        args = [arg, self.value()]
        if self.accept("for"):
            args.append(self.value())
        return "SUBSTR(" + ", ".join(args) + ")"

    def cast_call(self) -> str:
        self.i += 1
        self.expect_op("(")
        sql = self.full_expr()
        self.expect("as")
        kind = self.type_name()
        self.expect_op(")")
        return f"CAST({sql} AS {kind})"

    def function_call(self) -> str:
        fn = self.next().value
        self.expect_op("(")
        if self.accept_op("*"):
            args = "*"
        elif self.at_op(")"):
            args = ""
        else:
            prefix = "DISTINCT " if self.accept("distinct") or self.accept("unique") else ""
            args = prefix + ", ".join(self.expr_list())
        self.expect_op(")")
        return self.window_suffix(f"{fn}({args})")

    def window_suffix(self, sql: str) -> str:
        if self.accept("filter"):
            self.expect_op("(")
            self.expect("where")
            sql += f" FILTER (WHERE {self.full_expr()})"
            self.expect_op(")")
        if not self.accept("over"):
            return sql
        if not self.accept_op("("):
            return f"{sql} OVER {self.name('window name')}"
        parts = []
        saved, self.no_and = self.no_and, False
        if self.accept_phrase(["partition by", "partitioned by", *GROUP_PHRASES]):
            parts.append("PARTITION BY " + self.expr_items())
        if self.accept_phrase(ORDER_PHRASES):
            parts.append("ORDER BY " + self.order_items())
        if self.accept("frame"):
            parts.append(self.string("window frame"))
        self.no_and = saved
        self.expect_op(")")
        return f"{sql} OVER ({' '.join(parts)})"

    def case_expr(self) -> str:
        sql = "CASE"
        if not self.at("when"):
            sql += " " + self.full_expr()
        if not self.at("when"):
            self.fail(f"Expected 'when' but found {self.describe()}.")
        while self.accept("when"):
            sql += " WHEN " + self.full_expr()
            self.expect("then")
            sql += " THEN " + self.full_expr()
        if self.accept("else"):
            sql += " ELSE " + self.full_expr()
        self.expect("end")
        return sql + " END"

    def if_expr(self) -> str:
        sql = "CASE WHEN " + self.full_expr()
        self.expect("then")
        sql += " THEN " + self.or_expr()
        if self.accept("else"):
            sql += " ELSE " + self.or_expr()
        self.accept("end")
        return sql + " END"

    def type_name(self, single_word: bool = False) -> str:
        words = []
        while (self.peek().kind == "word" and self.peek().low not in COLUMN_TYPE_STOPS
               and self.peek().low not in RESERVED and not (single_word and words)):
            words.append(self.next().value.upper())
        if not words:
            self.fail(f"Expected a type but found {self.describe()}.")
        kind = TYPE_ALIASES.get(" ".join(words), " ".join(words))
        if self.at_op("("):
            self.next()
            sizes = [self.next().value]
            while self.accept_op(","):
                sizes.append(self.next().value)
            self.expect_op(")")
            kind += "(" + ", ".join(sizes) + ")"
        return kind

    # --- writing data ----------------------------------------------------
    def conflict_verb(self) -> str:
        if self.accept("or"):
            for action in ("replace", "ignore", "abort", "fail", "rollback"):
                if self.accept(action):
                    return " OR " + action.upper()
            self.fail("After 'or', write replace, ignore, abort, fail or rollback.")
        return ""

    def assignments(self) -> list[tuple[str, str]]:
        pairs = []
        while True:
            column = self.name("column name")
            if not (self.accept_op("=", ":") or self.accept("to") or self.accept("is")
                    or self.accept("as")):
                self.fail(f"Expected '=' after {column} but found {self.describe()}.")
            pairs.append((column, self.list_value()))
            if not self.more_items():
                return pairs

    def insert(self) -> str:
        if self.accept("replace"):
            self.expect("into")
            verb = "REPLACE"
        else:
            self.next()
            verb = "INSERT" + self.conflict_verb()
            self.accept("a", "row")
            self.accept("row")
            self.accept("rows")
            if not (self.accept("to") or self.accept("into")):
                self.fail(f"Expected 'to' or 'into' but found {self.describe()}.")
        table = self.name("table name")
        sql = f"{verb} INTO {table}"

        columns = None
        if self.at_op("(") and self.peek(1).low not in ("get", "select"):
            columns = self.name_list_in_parens()
            sql += " (" + ", ".join(columns) + ")"

        if self.accept("default", "values"):
            sql += " DEFAULT VALUES"
        elif self.accept("values") or self.accept("the", "rows") or self.accept("rows"):
            sql += " VALUES " + self.row_list()
        elif self.accept("set") or self.accept("with") or self.accept_op(":"):
            if columns:
                self.fail("Use either a column list or name = value pairs, not both.")
            pairs = self.assignments()
            sql += (" (" + ", ".join(c for c, _ in pairs) + ") VALUES ("
                    + ", ".join(v for _, v in pairs) + ")")
        elif self.at_op("("):
            self.next()
            sql += " " + self.query()
            self.expect_op(")")
        elif self.at_query():
            sql += " " + self.query()
        else:
            self.fail(f"Expected values, set, with or a query but found {self.describe()}.")

        if self.accept("on", "conflict"):
            sql += " ON CONFLICT"
            if self.at_op("("):
                sql += " (" + ", ".join(self.name_list_in_parens()) + ")"
            self.expect("do")
            if self.accept("nothing"):
                sql += " DO NOTHING"
            else:
                self.expect("update")
                self.expect("set")
                sql += " DO UPDATE SET " + self.render_assignments()
                if self.accept("where"):
                    sql += " WHERE " + self.full_expr()
        return sql + self.returning()

    def render_assignments(self) -> str:
        return ", ".join(f"{c} = {v}" for c, v in self.assignments())

    def returning(self) -> str:
        return " RETURNING " + self.select_list() if self.accept("returning") else ""

    def update(self) -> str:
        self.next()
        verb = "UPDATE" + self.conflict_verb()
        table = self.table_ref()
        self.expect("set")
        sql = f"{verb} {table} SET {self.render_assignments()}"
        if self.accept("from"):
            sql += " FROM " + self.from_clause()
        if self.accept("where") or self.accept("whose"):
            sql += " WHERE " + self.full_expr()
        return sql + self.returning()

    def delete(self) -> str:
        self.next()
        self.accept("everything") or self.accept("all", "rows") or self.accept("all") or self.accept("rows")
        self.expect("from")
        sql = "DELETE FROM " + self.table_ref()
        if self.accept("where") or self.accept("whose"):
            sql += " WHERE " + self.full_expr()
        return sql + self.returning()

    # --- schema ----------------------------------------------------------
    def if_not_exists(self) -> str:
        return " IF NOT EXISTS" if self.accept("if", "not", "exists") else ""

    def create(self) -> str:
        self.next()
        temp = " TEMPORARY" if self.accept("temporary") or self.accept("temp") else ""
        if self.accept("table"):
            return self.create_table(temp)
        if self.accept("view"):
            return self.create_view(temp)
        if self.accept("trigger"):
            return self.create_trigger(temp)
        if self.accept("unique", "index"):
            return self.create_index(" UNIQUE")
        if self.accept("index"):
            return self.create_index("")
        self.fail(f"Expected table, view, index or trigger after 'create' but found "
                  f"{self.describe()}. (For virtual tables use raw: ...){self.hint()}")

    def create_table(self, temp: str) -> str:
        head = f"CREATE{temp} TABLE{self.if_not_exists()} {self.name('table name')}"
        if self.accept("as"):
            return f"{head} AS {self.query_maybe_parens()}"
        if self.accept_op("("):
            body = self.table_items()
            self.expect_op(")")
        else:
            if not self.accept("with"):
                self.fail(f"Expected 'with' or '(' after the table name but found {self.describe()}.")
            self.accept("columns")
            self.accept_op(":")
            body = self.table_items()
        options = []
        while True:
            if self.accept("without", "rowid"):
                options.append("WITHOUT ROWID")
            elif self.accept("strict"):
                options.append("STRICT")
            elif not self.accept_op(","):
                break
        return f"{head} ({body})" + (" " + ", ".join(options) if options else "")

    def table_items(self) -> str:
        items = [self.table_item()]
        while self.at_op(",") and not (self.peek(1).low in ("without", "strict")):
            self.next()
            items.append(self.table_item())
        return ", ".join(items)

    def table_item(self) -> str:
        if self.at("primary", "key") and self.peek(2).value == "(":
            self.i += 2
            return "PRIMARY KEY (" + ", ".join(self.name_list_in_parens()) + ")"
        if self.at("unique") and self.peek(1).value == "(":
            self.next()
            return "UNIQUE (" + ", ".join(self.name_list_in_parens()) + ")"
        if self.at("check") and self.peek(1).value == "(":
            self.next()
            return self.check()
        if self.at("foreign", "key"):
            self.i += 2
            cols = ", ".join(self.name_list_in_parens())
            self.expect("references")
            return f"FOREIGN KEY ({cols}) " + self.references()
        return self.column_def()

    def check(self) -> str:
        self.expect_op("(")
        sql = f"CHECK ({self.full_expr()})"
        self.expect_op(")")
        return sql

    def references(self) -> str:
        sql = "REFERENCES " + self.name("table name")
        if self.at_op("("):
            sql += " (" + ", ".join(self.name_list_in_parens()) + ")"
        while self.at("on", "delete") or self.at("on", "update"):
            self.next()
            event = self.next().value.upper()
            action = self.accept_phrase(REF_ACTIONS)
            if not action:
                self.fail("Expected cascade, set null, set default, restrict or no action.")
            sql += f" ON {event} {REF_ACTIONS[action]}"
        return sql

    def column_def(self) -> str:
        sql = self.name("column name")
        if self.peek().kind == "word" and self.peek().low not in COLUMN_TYPE_STOPS:
            sql += " " + self.type_name()
        while True:
            if self.accept("primary", "key"):
                sql += " PRIMARY KEY"
                if self.accept("asc"):
                    sql += " ASC"
                elif self.accept("desc"):
                    sql += " DESC"
                if self.accept("auto", "increment") or self.accept("autoincrement"):
                    sql += " AUTOINCREMENT"
            elif self.accept("auto", "increment") or self.accept("autoincrement"):
                sql += " PRIMARY KEY AUTOINCREMENT"
            elif self.accept("not", "null") or self.accept("required") or self.accept("mandatory"):
                sql += " NOT NULL"
            elif self.accept("unique"):
                sql += " UNIQUE"
            elif self.accept("default") or self.accept("defaulting", "to"):
                sql += " DEFAULT " + self.default_value()
            elif self.at("check"):
                self.next()
                sql += " " + self.check()
            elif self.accept("references"):
                sql += " " + self.references()
            elif self.accept("collate"):
                sql += " COLLATE " + self.name("collation")
            elif self.accept("generated", "always", "as") or self.at("as"):
                self.accept("as")
                self.expect_op("(")
                sql += f" GENERATED ALWAYS AS ({self.full_expr()})"
                self.expect_op(")")
                if self.accept("stored"):
                    sql += " STORED"
                elif self.accept("virtual"):
                    sql += " VIRTUAL"
            else:
                return sql

    def default_value(self) -> str:
        value = self.unary()
        plain = re.fullmatch(r"[-+]?[\d.eE+-]+|'(?:[^']|'')*'|NULL|TRUE|FALSE|"
                             r"CURRENT_(?:DATE|TIME|TIMESTAMP)|\(.*\)", value)
        return value if plain else f"({value})"

    def create_view(self, temp: str) -> str:
        sql = f"CREATE{temp} VIEW{self.if_not_exists()} {self.name('view name')}"
        if self.at_op("("):
            sql += " (" + ", ".join(self.name_list_in_parens()) + ")"
        self.expect("as")
        return f"{sql} AS {self.query_maybe_parens()}"

    def query_maybe_parens(self) -> str:
        if self.accept_op("("):
            sql = self.query()
            self.expect_op(")")
            return sql
        return self.query()

    def create_index(self, unique: str) -> str:
        sql = f"CREATE{unique} INDEX{self.if_not_exists()} {self.name('index name')}"
        self.expect("on")
        sql += f" ON {self.name('table name')}"
        self.expect_op("(")
        sql += " (" + self.order_items() + ")"
        self.expect_op(")")
        if self.accept("where"):
            sql += " WHERE " + self.full_expr()
        return sql

    def create_trigger(self, temp: str) -> str:
        sql = f"CREATE{temp} TRIGGER{self.if_not_exists()} {self.name('trigger name')}"
        timing = self.accept_phrase(["before", "after", "instead of"])
        if timing:
            sql += " " + timing.upper()
        event = self.accept_phrase(["insert", "update", "delete"])
        if not event:
            self.fail(f"Expected insert, update or delete but found {self.describe()}.")
        sql += " " + event.upper()
        if event == "update" and self.accept("of"):
            cols = [self.name("column name")]
            while self.accept_op(","):
                cols.append(self.name("column name"))
            sql += " OF " + ", ".join(cols)
        self.expect("on")
        sql += " ON " + self.name("table name")
        if self.accept("for", "each", "row"):
            sql += " FOR EACH ROW"
        if self.accept("when"):
            sql += " WHEN " + self.full_expr()
        self.expect("do")
        self.expect_op("(")
        body = []
        while not self.at_op(")"):
            body.append(self.statement() + ";")
            self.accept_op(";")
        self.expect_op(")")
        return f"{sql} BEGIN {' '.join(body)} END"

    def drop(self) -> str:
        self.next()
        kind = self.accept_phrase(["table", "view", "index", "trigger"])
        if self.accept("column"):
            column = self.name("column name")
            self.expect("from")
            return f"ALTER TABLE {self.name('table name')} DROP COLUMN {column}"
        if not kind:
            self.fail(f"Expected table, view, index, trigger or column after 'drop' but found "
                      f"{self.describe()}.{self.hint()}")
        exists = " IF EXISTS" if self.accept("if", "exists") else ""
        return f"DROP {kind.upper()}{exists} {self.name(kind + ' name')}"

    def add_column(self) -> str:
        self.expect("add", "column")
        column = self.column_def()
        self.expect("to")
        return f"ALTER TABLE {self.name('table name')} ADD COLUMN {column}"

    def alter(self) -> str:
        if self.accept("rename", "table"):
            old = self.name("table name")
            self.expect("to")
            return f"ALTER TABLE {old} RENAME TO {self.name('table name')}"
        if self.accept("rename", "column"):
            old = self.name("column name")
            self.expect("to")
            new = self.name("column name")
            if not (self.accept("in") or self.accept("on")):
                self.fail(f"Expected 'in <table>' but found {self.describe()}.")
            return f"ALTER TABLE {self.name('table name')} RENAME COLUMN {old} TO {new}"
        self.expect("alter", "table")
        table = self.name("table name")
        if self.accept("rename", "to"):
            return f"ALTER TABLE {table} RENAME TO {self.name('table name')}"
        if self.accept("rename", "column"):
            old = self.name("column name")
            self.expect("to")
            return f"ALTER TABLE {table} RENAME COLUMN {old} TO {self.name('column name')}"
        if self.accept("add", "column"):
            return f"ALTER TABLE {table} ADD COLUMN {self.column_def()}"
        if self.accept("drop", "column"):
            return f"ALTER TABLE {table} DROP COLUMN {self.name('column name')}"
        self.fail(f"Expected rename, add column or drop column but found {self.describe()}.")


def shell_arg(path: str) -> str:
    return path if re.fullmatch(r"[\w./~-]+", path) else '"' + path.replace('"', '\\"') + '"'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def logical_statements(text: str) -> list[str]:
    """One statement per line; an indented line continues the previous one."""
    chunks: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("--", "#")):
            continue
        if chunks and line[0].isspace():
            chunks[-1] += "\n" + line
        else:
            chunks.append(stripped)
    return chunks


def compile_english(text: str) -> list[str]:
    """Translate format text into a list of SQLite commands."""
    commands: list[str] = []
    for chunk in logical_statements(text):
        if chunk.startswith("."):
            commands.append(chunk)  # sqlite3 shell command, passed through unchanged
            continue
        raw = re.match(r"raw\s*:\s*(.*)", chunk, re.DOTALL | re.IGNORECASE)
        if raw:
            body = raw.group(1).strip()
            commands.append(body if body.endswith(";") else body + ";")
            continue
        for sql in Parser(chunk).program():
            commands.append(sql if sql.startswith(".") else sql + ";")
    return commands


# ---------------------------------------------------------------------------
# Running commands
# ---------------------------------------------------------------------------

def print_table(columns: list[str], rows: list[tuple]) -> None:
    cells = [["NULL" if v is None else str(v) for v in row] for row in rows]
    widths = [max([len(c)] + [len(r[i]) for r in cells]) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print(f"({len(rows)} row{'' if len(rows) == 1 else 's'})")


def run_dot_command(conn: sqlite3.Connection, command: str, db: str) -> None:
    parts = command.split()
    name, arg = parts[0], parts[1].strip('"') if len(parts) > 1 else None
    if name == ".tables":
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        print("\n".join(r[0] for r in rows))
    elif name == ".schema":
        sql, params = "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL", ()
        if arg:
            sql, params = sql + " AND tbl_name = ? COLLATE NOCASE", (arg,)
        for (text,) in conn.execute(sql, params):
            print(text + ";")
    elif name == ".indexes":
        sql, params = "SELECT name FROM sqlite_master WHERE type = 'index'", ()
        if arg:
            sql, params = sql + " AND tbl_name = ? COLLATE NOCASE", (arg,)
        print("\n".join(r[0] for r in conn.execute(sql, params)))
    elif name == ".databases":
        for _, schema, path in conn.execute("PRAGMA database_list"):
            print(f"{schema}: {path or '(memory)'}")
    else:
        if db == ":memory:":
            raise RuntimeError(f"{name} needs a database file; pass --db <file>.")
        if not shutil.which("sqlite3"):
            raise RuntimeError(f"{name} needs the sqlite3 command-line tool, which isn't installed.")
        sys.stdout.flush()  # keep our output ahead of the child's
        subprocess.run(["sqlite3", db, command], check=True)


def run_commands(conn: sqlite3.Connection, commands: list[str], db: str, show_sql: bool = True) -> None:
    for command in commands:
        if show_sql:
            print(command)
        if command.startswith("."):
            run_dot_command(conn, command, db)
            continue
        cursor = conn.execute(command)
        if cursor.description:
            print_table([d[0] for d in cursor.description], cursor.fetchall())
        elif cursor.rowcount >= 0:
            print(f"({cursor.rowcount} row{'' if cursor.rowcount == 1 else 's'} changed)")


def repl(conn: sqlite3.Connection | None, db: str) -> int:
    print("Type a sentence in the format (see FORMAT.md). Press Ctrl-D to quit.")
    while True:
        try:
            line = input("english> ")
        except EOFError:
            print()
            return 0
        try:
            commands = compile_english(line)
            if conn:
                run_commands(conn, commands, db)
            else:
                print("\n".join(commands))
        except (FormatError, sqlite3.Error, RuntimeError, subprocess.CalledProcessError) as err:
            print(f"Error: {err}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compile a strict English format into SQLite commands.")
    ap.add_argument("sentence", nargs="*", help="sentence to translate (quote it in your shell)")
    ap.add_argument("-f", "--file", help="read sentences from a file, one per line")
    ap.add_argument("--db", help="also run the commands on this SQLite file and print the results; "
                                 "writes really change the file")
    args = ap.parse_args(argv)

    conn = sqlite3.connect(args.db, isolation_level=None) if args.db else None
    db = args.db or ""
    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            text = handle.read()
    elif args.sentence:
        text = " ".join(args.sentence)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        return repl(conn, db)

    try:
        commands = compile_english(text)
        if conn:
            run_commands(conn, commands, db)
        else:
            print("\n".join(commands))
    except (FormatError, sqlite3.Error, RuntimeError, subprocess.CalledProcessError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
