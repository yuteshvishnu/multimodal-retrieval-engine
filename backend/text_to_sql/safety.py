"""
Scout Version-7: Scout only reads data. Deterministic rules (no LLM deciding) that stop
anything that could change a database, at every point a write could slip in:

    write_request(text)   a question, or the meaning the user confirmed, that asks to change data
                          ("Delete the cancelled orders", "can you update the price...", or SQL
                          typed straight into the box) -> rejected before any SQL is written
    write_problems(sql)   the SQL the LLM wrote, read with the sqlglot parser: anything but one
                          plain read query (SELECT / WITH ... SELECT / UNION) -> rejected, never run

Behind these, the databases are attached read-only and the connection refuses writes
(PRAGMA query_only), so even a missed case can't change data.
"""
import re
from typing import List, Optional

import sqlglot
from sqlglot import exp

# statements that change data, the schema, the connection or transactions
WRITE_NODES = {
    exp.Insert: "INSERT",
    exp.Update: "UPDATE",
    exp.Delete: "DELETE",
    exp.Drop: "DROP",
    exp.Create: "CREATE",
    exp.Alter: "ALTER",
    exp.Merge: "MERGE",
    exp.TruncateTable: "TRUNCATE",
    exp.Into: "SELECT INTO",
    exp.Pragma: "PRAGMA",
    exp.Attach: "ATTACH",
    exp.Detach: "DETACH",
    exp.Transaction: "BEGIN",
    exp.Commit: "COMMIT",
    exp.Rollback: "ROLLBACK",
    exp.Analyze: "ANALYZE",
    exp.Grant: "GRANT",
    exp.Revoke: "REVOKE",
    exp.Copy: "COPY",
    exp.Set: "SET",
}

# SQLite functions that touch files or load code
UNSAFE_FUNCTIONS = {"load_extension", "writefile", "readfile", "edit", "fts3_tokenizer"}

# used when the parser can't read the SQL: any of these words outside a quoted string
WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|create|alter|replace\s+into|truncate|attach|detach|pragma|vacuum|"
    r"reindex|begin|commit|rollback|savepoint|release|analyze|merge|grant|revoke|upsert)\b",
    re.IGNORECASE,
)

# text that starts like SQL
SQL_START = re.compile(
    r"^\s*(select|with|insert|update|delete|drop|create|alter|replace|pragma|attach|detach|vacuum|"
    r"reindex|begin|commit|rollback|truncate|merge|analyze|savepoint|release)\b",
    re.IGNORECASE,
)

# a request that starts with one of these verbs asks to change data
WRITE_VERBS = {
    "delete", "remove", "drop", "truncate", "erase", "wipe", "purge", "destroy", "clear",
    "update", "edit", "change", "modify", "set", "reset", "overwrite", "replace", "rename",
    "insert", "add", "create", "alter", "merge", "upsert", "append",
    "increase", "decrease", "raise", "cut", "mark", "move", "assign", "grant", "revoke",
}
# polite openers before the verb: "please", "can you", "I want to", "let's"...
POLITE_START = re.compile(
    r"^\s*(please\s+|(can|could|would|will)\s+you\s+(please\s+)?|i\s+(want|need|would\s+like|'d\s+like)\s+to\s+|"
    r"let'?s\s+|go\s+ahead\s+and\s+|kindly\s+)",
    re.IGNORECASE,
)


def write_problems(sql: str) -> List[str]:
    """
    What makes this SQL more than one plain read query (empty list = safe to run).
    Allow-list: only SELECT, WITH ... SELECT and UNION / INTERSECT / EXCEPT pass.
    """
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError:
        # can't read it: reject if a write keyword appears outside quoted strings
        code = re.sub(r"'(?:[^']|'')*'|\"[^\"]*\"|--[^\n]*|/\*.*?\*/", " ", sql, flags=re.DOTALL)
        found = {m.group(1).split()[0].upper() for m in WRITE_KEYWORDS.finditer(code)}
        return sorted(found)

    if not statements:
        return ["no query"]
    problems = []
    if len(statements) > 1:
        problems.append("more than one statement")
    for statement in statements:
        for node in statement.walk():
            label = WRITE_NODES.get(type(node))
            if label:
                problems.append(label)
            elif isinstance(node, exp.Command):
                problems.append(str(node.this).upper())
            elif isinstance(node, (exp.Anonymous, exp.Func)) and node.name.lower() in UNSAFE_FUNCTIONS:
                problems.append(f"{node.name.lower()}()")
        if not isinstance(statement, exp.Query) and not problems:
            problems.append(f"not a read query ({type(statement).__name__})")
    return list(dict.fromkeys(problems))


def write_request(text: str) -> Optional[str]:
    """
    If this question (or confirmed meaning) asks to change data, the reason, e.g.
    "DELETE" or "update"; else None. Deterministic: SQL is read with the parser, and plain
    English is checked for a request that starts with a write verb.
    """
    if SQL_START.match(text):
        problems = write_problems(text)
        if problems:
            return ", ".join(problems)

    rest = text
    while True:
        stripped = POLITE_START.sub("", rest, count=1)
        if stripped == rest:
            break
        rest = stripped
    words = re.findall(r"[a-z']+", rest.lower())
    if words and words[0] in WRITE_VERBS:
        if words[0] == "add" and len(words) > 1 and words[1] == "up":
            return None  # "add up the revenue" is a sum, not a write
        return words[0]
    return None


def write_message(reason: str, where: str = "This question") -> str:
    return (
        f"{where} could be a potential write ({reason}). Scout only reads data and never changes it, "
        "so nothing was run. Please rewrite it as a question about the data, "
        "e.g. “How many cancelled orders are there?”"
    )
