"""
Text-to-SQL Version-3: LLM-written table summaries, used for embedding-based table search.

A template built from column names only says WHAT COLUMNS a table has. People ask in
business words ("revenue", "spent", "sales"), so we ask an LLM, once and offline, to
describe what each table MEANS: what a row represents, what questions it answers,
and the words people use for it. Those summaries are then embedded (schema_index.py).

Rules that keep summaries correct as the database changes:
- Each summary describes ONLY its own table: no other table names, no joins.
  Relationships come from the live foreign keys instead (the bridge-table rule),
  so new, deleted or renamed tables can never leave a summary pointing at the wrong place.
- An automatic check rejects summaries that mention other tables or joins, and retries.
- Summaries are cached per table in table_summaries.json next to the database:
  only new or changed tables are summarized again, deleted tables are dropped,
  and the file can be read and edited by hand.

Build them all with the script:
    python -m scripts.text_to_sql.build_schema_index

Version-6: the user reviews summaries in the Tables view. When one is wrong, they say why;
rewrite() makes a new one from their corrections (1 LLM call per try) until they save it.
Their corrections are kept in table_summaries.json ("feedback") and reused whenever the
summary is written again, so they survive schema changes and rebuilds.
"""
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.llm import chat, make_client

from backend.config import LLM_MODEL_SUMMARY, SHARE_ROWS_WITH_LLM

SUMMARY_FILE = "table_summaries.json"
SAMPLE_ROWS = 5
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You are a data analyst who documents database tables.

Your description of a table will be turned into an embedding and used to decide whether \
this table is relevant to a question a business user asks in plain English. So it must \
capture the MEANING of the data in the words real people use, not just repeat column names.

Write ONE paragraph of plain English, about 120 to 200 words, that covers:
1. What one row represents in the real world (a person, a thing, an event, a transaction).
2. What the table is used for: the kinds of questions it can answer \
(totals, counts, trends over time, rankings, comparisons, lookups).
3. The everyday words and synonyms people would use when asking about this data, \
e.g. "sales", "revenue", "purchases", "spend", "staff", "headcount", "rating", "feedback".
4. The important columns, explained in business terms, including units \
(e.g. amounts in cents), date formats, and the categories or status values that appear.
5. Any rules for interpreting the values correctly \
(e.g. what a status means, what NULL means, what should be excluded from totals).

Strict rules:
- Describe ONLY this table. Never mention, name or guess any other table, \
and never describe joins, foreign keys or how this table connects to others.
- Use only facts supported by the schema, the column descriptions and the sample rows. \
Do not invent columns, values or meanings.
- Plain prose only: no lists, no headings, no markdown, no SQL, no table name in backticks."""


def fingerprint(table: dict) -> str:
    """Changes when the table's columns or descriptions change (not when rows change)."""
    return hashlib.sha256(table["schema"].encode("utf-8")).hexdigest()


def load_summaries(folder: Path) -> Dict[str, dict]:
    path = Path(folder) / SUMMARY_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fresh_summary(table: dict, saved: Dict[str, dict]) -> Optional[str]:
    """The saved summary for this table, if it was written for its current schema."""
    entry = saved.get(table["name"])
    if entry and entry.get("fingerprint") == fingerprint(table) and entry.get("summary"):
        return entry["summary"]
    return None


def save_summary(table: dict, summary: str, feedback: List[str]) -> None:
    """Saves a summary the user confirmed, with every correction they gave for it."""
    saved = load_summaries(table["folder"])
    saved[table["name"]] = {
        "summary": summary,
        "fingerprint": fingerprint(table),
        "reviewed": True,
        "feedback": feedback,
    }
    path = Path(table["folder"]) / SUMMARY_FILE
    path.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def problems_in(summary: str, table: dict, all_names: List[str]) -> List[str]:
    """
    Ways a summary breaks the "only this table" rule. Plain words like "customers" are
    fine in normal English, so we only flag clear references to another table: its exact
    name when it has an underscore, a name in backticks, "<name> table" / "table <name>",
    and any talk of joins or foreign keys.
    """
    problems = []
    for name in all_names:
        if name == table["name"]:
            continue
        n = re.escape(name)
        patterns = [rf"`{n}`", rf"\b{n}\s+table\b", rf"\btable\s+{n}\b"]
        if "_" in name:
            patterns.append(rf"\b{n}\b")
        if any(re.search(p, summary, re.IGNORECASE) for p in patterns):
            problems.append(f"mentions the table '{name}'")
    if re.search(r"\bjoin(s|ed|ing)?\b|foreign key", summary, re.IGNORECASE):
        problems.append("describes joins or foreign keys")
    return problems


class TableSummarizer:
    def __init__(self, database):
        self.database = database
        self.client = make_client()

    def build(self, rebuild: bool = False, folders: Optional[List[Path]] = None) -> None:
        """Summarizes new or changed tables in every database (or only in `folders`) and saves them."""
        tables = self.database.tables
        all_names = [t["name"] for t in tables]

        by_folder = {}
        for table in tables:
            if folders is None or Path(table["folder"]) in {Path(f) for f in folders}:
                by_folder.setdefault(table["folder"], []).append(table)

        for folder, group in by_folder.items():
            saved = load_summaries(folder)
            updated = {}
            for table in group:
                entry = saved.get(table["name"], {})
                summary = None if rebuild else fresh_summary(table, saved)
                if summary is None:
                    summary = self._summarize(table, all_names, entry.get("feedback", []))
                    if summary is None:
                        continue  # left out; the template is used for this table
                    entry = {k: v for k, v in entry.items() if k != "reviewed"}  # new text: not reviewed yet
                updated[table["name"]] = {**entry, "summary": summary, "fingerprint": fingerprint(table)}

            dropped = sorted(set(saved) - {t["name"] for t in group})
            for name in dropped:
                print(f"[Summaries] Dropped '{name}' (table no longer exists)")

            path = Path(folder) / SUMMARY_FILE
            path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"[Summaries] Saved {len(updated)}/{len(group)} summaries to {path}")

    def rewrite(self, table: dict, all_names: List[str], current: str, feedback: List[str]) -> Tuple[str, List[str]]:
        """
        A new summary that follows the user's corrections. One LLM call, no retries:
        returns (summary, problems), and the user decides whether to keep it.
        """
        if self.client is None:
            raise RuntimeError("The LLM API key is not set (LLM_PROVIDER and its key in .env).")
        prompt = self._table_prompt(table, feedback) + f"\n\nCurrent description (the user says it is wrong):\n{current}"
        completion = chat(
            self.client,
            {"stage": "table_description"},
            model=LLM_MODEL_SUMMARY,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        summary = " ".join((completion.choices[0].message.content or "").split())
        return summary, problems_in(summary, table, all_names)

    def _summarize(self, table: dict, all_names: List[str], feedback: List[str] = ()) -> Optional[str]:
        if self.client is None:
            raise RuntimeError("The LLM API key is not set (LLM_PROVIDER and its key in .env).")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._table_prompt(table, feedback)},
        ]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            completion = chat(
                self.client,
                {"stage": "table_description"},
                model=LLM_MODEL_SUMMARY,
                messages=messages,
                temperature=0.2,
            )
            summary = " ".join((completion.choices[0].message.content or "").split())
            problems = problems_in(summary, table, all_names)
            if summary and not problems:
                print(f"[Summaries] Summarized '{table['name']}'")
                return summary

            print(f"[Summaries] '{table['name']}' attempt {attempt} rejected: {problems or ['empty']}")
            messages += [
                {"role": "assistant", "content": summary},
                {"role": "user", "content": (
                    "Rewrite the description. Problems: " + "; ".join(problems or ["it was empty"]) + ". "
                    "Describe only this table, with no other table names and no joins."
                )},
            ]

        print(f"[Summaries] WARNING: no valid summary for '{table['name']}'; using the column template")
        return None

    def _table_prompt(self, table: dict, feedback: List[str] = ()) -> str:
        prompt = (
            f"Table name: {table['name']}\n\n"
            f"Schema, with a description of each column:\n{table['schema']}\n\n"
            f"Sample rows:\n{self._sample_rows(table)}"
        )
        if feedback:
            prompt += (
                "\n\nCorrections from the person who owns this data (oldest first). They are facts: "
                "follow every one, even over the sample rows or your own assumptions:\n"
                + "\n".join(f"- {item}" for item in feedback)
            )
        return prompt

    def _sample_rows(self, table: dict) -> str:
        if not SHARE_ROWS_WITH_LLM:
            return "(not shared: SHARE_ROWS_WITH_LLM is off, so describe the table from its schema only)"
        try:
            columns, rows = self.database.run(
                f'SELECT * FROM "{table["database"]}"."{table["name"]}" LIMIT {SAMPLE_ROWS}'
            )
        except Exception as e:
            return f"(could not read sample rows: {e})"

        def cell(value):
            text = "NULL" if value is None else str(value)
            return text if len(text) <= 60 else text[:57] + "..."

        lines = [" | ".join(columns)]
        lines += [" | ".join(cell(v) for v in row) for row in rows]
        return "\n".join(lines)
