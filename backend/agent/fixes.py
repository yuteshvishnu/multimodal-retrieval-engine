"""
What happens to the Failure Analyst's suggested fix. Deterministic: code tests it and code
applies it; the LLM only suggests.

    test_fix   runs the corrected SQL (read-only) and checks it: against the correct answer
               for an eval failure, "runs without an error" for a SQL error; checks that an
               error-book entry is complete, new, and recognises the logged error
    apply_fix  after the developer approves: a new error-book entry is added to
               data/text_to_sql/sql_errors.json, a table-description note is added to the
               table's description (and re-embedded). Prompt and code fixes are the developer's.
"""
import json
import re
from datetime import date
from typing import Dict, List

from backend.text_to_sql.safety import write_problems
from backend.text_to_sql.table_summaries import save_summary

FIX_TYPES = ("error_book", "table_description", "prompt", "code", "none")
ENTRY_FIELDS = ("id", "match", "hint", "rule")


def as_set(rows) -> set:
    """Rows in any order; floats rounded so 399.99 == 399.9900000001 (as in evaluate.py)."""
    return {tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in rows}


def logged_errors(details: Dict) -> List[str]:
    """The SQLite error messages a failure logged."""
    found = [details.get("error") or ""]
    for attempt in details.get("attempts", []):
        found += attempt.get("problems", [])
    return [e for e in found if e]


def test_fix(issue: Dict, report: Dict, pipeline) -> Dict:
    """{"passed": True / False / None (a person must judge), "checks": [{"ok", "text"}], "columns", "rows"}"""
    details, checks, out = issue["details"], [], {"columns": [], "rows": []}

    sql = (report.get("fixed_sql") or "").strip()
    if sql:
        problems = write_problems(sql)
        if problems:
            checks.append({"ok": False, "text": f"Corrected SQL is not one plain read query ({', '.join(problems)})."})
        else:
            try:
                columns, rows, _ = pipeline.database.query(sql)
                out.update(columns=columns, rows=rows[:5])
                checks.append({"ok": True, "text": f"Corrected SQL runs: {len(rows)} row(s)."})
                if details.get("gold_sql"):
                    _, gold, _ = pipeline.database.query(details["gold_sql"])
                    same = as_set(rows) == as_set(gold)
                    checks.append({"ok": same, "text": "Same rows as the correct answer." if same else
                                   f"Different rows from the correct answer ({len(gold)} expected)."})
            except Exception as e:
                checks.append({"ok": False, "text": f"Corrected SQL fails: {e}"})

    if report.get("fix_type") == "error_book":
        errors = logged_errors(details)
        if errors:
            checks += entry_checks(report.get("error_book_entry") or {}, pipeline, errors)
        else:
            # the error book reacts to SQLite's error messages; a query that ran gives none to match
            checks.append({"ok": False, "text": "No SQLite error was logged, so an error-book entry would never "
                                                "match: this needs a prompt rule instead."})
    if report.get("fix_type") == "table_description":
        note = report.get("description_note") or {}
        known = any(t["name"].lower() == str(note.get("table", "")).lower() for t in pipeline.tables)
        checks.append({"ok": known and bool(str(note.get("note", "")).strip()),
                       "text": f"Description note for table '{note.get('table')}'" + ("." if known else ": no such table.")})

    if not checks:
        out["passed"] = None
    elif not all(c["ok"] for c in checks):
        out["passed"] = False
    else:
        # a user's report has no correct answer to compare with: a person must judge the rows
        out["passed"] = None if issue["source"] == "user_report" and sql and len(checks) == 1 else True
    out["checks"] = checks
    return out


def entry_checks(entry: Dict, pipeline, errors: List[str]) -> List[Dict]:
    """An error-book entry must be complete, new, a valid pattern, and recognise the logged error."""
    missing = [f for f in ENTRY_FIELDS if not str(entry.get(f, "")).strip()]
    if missing:
        return [{"ok": False, "text": f"Error-book entry is missing: {', '.join(missing)}."}]
    checks = []
    taken = {e["id"] for e in pipeline.errors.entries}
    checks.append({"ok": entry["id"] not in taken, "text": f"Entry id '{entry['id']}'" +
                   (" is new." if entry["id"] not in taken else " already exists.")})
    try:
        pattern = re.compile(entry["match"], re.IGNORECASE)
    except re.error as e:
        return checks + [{"ok": False, "text": f"Pattern is not a valid regular expression: {e}"}]
    if errors:
        hit = any(pattern.search(e) for e in errors)
        checks.append({"ok": hit, "text": "Pattern recognises the logged error." if hit else
                       "Pattern does not match the logged error."})
    return checks


def apply_fix(issue: Dict, pipeline) -> str:
    """Applies an approved data fix; returns what was done, in a sentence."""
    report = issue.get("report") or {}
    fix_type = report.get("fix_type")

    if fix_type == "error_book":
        entry = {k: str(v).strip() for k, v in (report.get("error_book_entry") or {}).items() if str(v).strip()}
        failed = [c["text"] for c in entry_checks(entry, pipeline, []) if not c["ok"]]
        if failed:
            raise ValueError(" ".join(failed))
        path = pipeline.errors.known_file
        data = json.loads(path.read_text(encoding="utf-8"))
        data.append({**entry, "added": date.today().isoformat(), "source": f"Failure Analyst, issue {issue['id']}"})
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return f"Added '{entry['id']}' to the error book; Scout uses it from the next question."

    if fix_type == "table_description":
        note = report.get("description_note") or {}
        text = " ".join(str(note.get("note", "")).split())
        table = next((t for t in pipeline.tables if t["name"].lower() == str(note.get("table", "")).lower()), None)
        if table is None or not text:
            raise ValueError("The description note needs a known table and some text.")
        summary = f"{table.get('summary') or ''} {text}".strip()
        save_summary(table, summary, list(table.get("feedback") or []) + [text])
        pipeline.reload()
        return f"Added the note to the description of '{table['name']}' and re-embedded it."

    return "Approved. This fix is a prompt or code change, so it's for the developer to make."
