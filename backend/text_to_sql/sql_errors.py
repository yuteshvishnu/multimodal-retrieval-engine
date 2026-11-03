"""
Scout Version-9: an error book. SQLite errors Scout has already seen are handled before they
can be repeated. Deterministic: the LLM never decides what goes in it.

    data/text_to_sql/sql_errors.json        the KNOWN errors (in git, edited by hand): how to
                                            recognise each one, and the fix to tell the model
    data/text_to_sql/logs/sql_errors.jsonl  every rejected attempt and how the question ended
                                            (git-ignored: it holds users' questions)

It acts in two places:
    before: the most frequent errors' one-line rules go into the SQL prompt (prevention)
    after:  a matching error's hint goes into round 2's feedback (repair); an error with no
            hint that every round-1 try hit stops the question early (no wasted calls)

To grow it, read the report and add entries for new errors:
    python -m scripts.text_to_sql.sql_errors
"""
import json
import re
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

KNOWN_FILE = Path("data/text_to_sql/sql_errors.json")
LOG_FILE = Path("data/text_to_sql/logs/sql_errors.jsonl")
PROMPT_RULES = 10  # how many rules go into the SQL prompt


def normalize(message: str) -> str:
    """
    The kind of error, without the specifics, so the same kind groups together:
    'Invalid SQL: no such column: o.statuss' -> 'no such column: <name>'.
    """
    text = re.sub(r"^(Invalid SQL|Query failed):\s*", "", message.strip())
    text = re.sub(r"(no such (?:column|table)|ambiguous column name):\s*\S+", r"\1: <name>", text)
    text = re.sub(r'near "(\d+(?:\.\d+)?)"', 'near "<number>"', text)
    text = re.sub(r"near \"'[^']*'\"", 'near "<text>"', text)
    text = re.sub(r"'[^']*'", "<text>", text)
    text = re.sub(r"\b\d+\b", "<n>", text)
    return text


class ErrorBook:
    def __init__(self, known_file: Path = KNOWN_FILE, log_file: Path = LOG_FILE):
        self.known_file = Path(known_file)
        self.log_file = Path(log_file)
        self._entries: List[Dict] = []
        self._mtime = None
        self._lock = threading.Lock()
        self.counts: Counter = Counter()  # known error id -> times seen (from the log)
        self._count_log()

    @property
    def entries(self) -> List[Dict]:
        """The known errors, re-read whenever the file changes (no restart needed)."""
        try:
            mtime = self.known_file.stat().st_mtime
        except OSError:
            return []
        if mtime != self._mtime:
            data = json.loads(self.known_file.read_text(encoding="utf-8"))
            for entry in data:
                entry["_match"] = re.compile(entry["match"], re.IGNORECASE)
                entry["_sql"] = re.compile(entry["sql_match"], re.IGNORECASE) if entry.get("sql_match") else None
            self._entries, self._mtime = data, mtime
        return self._entries

    def match(self, message: str, sql: str = "") -> Optional[Dict]:
        """The first known error whose message pattern (and SQL pattern, if it has one) fits."""
        for entry in self.entries:
            if entry["_match"].search(message) and (entry["_sql"] is None or entry["_sql"].search(sql or "")):
                return entry
        return None

    def rules(self, limit: int = PROMPT_RULES) -> List[str]:
        """One-line rules for the SQL prompt: the most often seen errors first, then file order."""
        entries = self.entries
        order = sorted(range(len(entries)), key=lambda i: (-self.counts[entries[i]["id"]], i))
        return [entries[i]["rule"] for i in order[:limit]]

    def log(self, event: Dict) -> None:
        """Appends one event to the log (one JSON object per line)."""
        event = {"time": datetime.now().isoformat(timespec="seconds"), **event}
        with self._lock:
            if event.get("type") == "error" and event.get("known_id"):
                self.counts[event["known_id"]] += 1
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with self.log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            except OSError as e:
                print(f"[SQL errors] Could not write the log: {e}")

    def read_log(self) -> List[Dict]:
        if not self.log_file.exists():
            return []
        events = []
        for line in self.log_file.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def _count_log(self) -> None:
        for event in self.read_log():
            if event.get("type") == "error" and event.get("known_id"):
                self.counts[event["known_id"]] += 1
