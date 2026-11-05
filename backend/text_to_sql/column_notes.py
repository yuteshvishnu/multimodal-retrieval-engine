"""
Version-20: AI notes for columns nobody described (optional, offline, never automatic).

Some databases leave columns undescribed (BIRD's "Country" in gasstations has no description at
all). When asked to (python -m scripts.text_to_sql.onboard --describe), the LLM reads one table's
catalog entry (its schema with each column's profile: values, ranges, formats) and writes one
short sentence for each undescribed column. One LLM call per table that has such columns.

Saved to data/catalogs/<database>-<id>/column_notes.json, next to the catalog but in its own file,
so rebuilding the catalog keeps them. A described column is never given a note: a real
description always wins. Notes are shown to Scout's LLM marked "(AI note)", so they're never
mistaken for the database owner's own description. They come from the data only, never from any
question or answer, so they're allowed for benchmarks like BIRD too.
"""
import json
import re
from pathlib import Path
from typing import Dict, List

from backend.config import LLM_API_KEY, LLM_MODEL_SUMMARY
from backend.llm import chat, make_client
from backend.text_to_sql import schema_catalog

NOTES_FILE = "column_notes.json"
MAX_NOTE_CHARS = 160

SYSTEM_PROMPT = """You describe database columns for someone writing SQL about this data.
For each column listed, write one short sentence (at most 25 words) saying what it holds, from its
name, its table, the other columns, and its profile in the schema (values, ranges, formats).
If you can't tell what it means, write "unclear". Don't invent units or meanings the data doesn't
support. Return only JSON: {"columns": {"<column name>": "<sentence>"}} with exactly the columns listed."""


def load(path: Path, alias: str) -> Dict[str, Dict[str, str]]:
    """{table: {column: note}} for one database ({} if none were written)."""
    file = schema_catalog.catalog_dir(path, alias) / NOTES_FILE
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8"))


def undescribed(table: dict) -> List[str]:
    """Columns of a table entry with no description, that aren't keys (ids say what they are)."""
    return [c["name"] for c in table["catalog"]["columns"]
            if not c["description"] and c["kind"] not in ("key", "empty")]


class ColumnNoteWriter:
    def __init__(self, client=None):
        self.client = client

    def write(self, database, rebuild: bool = False) -> int:
        """Notes for every undescribed column of every table; returns how many LLM calls were made."""
        if self.client is None:
            if not LLM_API_KEY:
                raise RuntimeError("The LLM API key is not set (LLM_PROVIDER and its key in .env).")
            self.client = make_client(LLM_API_KEY)
        calls = 0
        by_database: Dict[str, List[dict]] = {}
        for table in database.tables:
            by_database.setdefault(table["database"], []).append(table)
        for alias, tables in by_database.items():
            path = next(p for p, a in zip(database.files, database.aliases) if a == alias)
            notes = {} if rebuild else load(path, alias)
            for table in tables:
                missing = [c for c in undescribed(table) if c not in notes.get(table["name"], {})]
                if not missing:
                    continue
                written = self._describe(table, missing)
                calls += 1
                notes.setdefault(table["name"], {}).update(written)
                print(f"[Column notes] {alias}.{table['name']}: {len(written)}/{len(missing)} columns described")
            folder = schema_catalog.catalog_dir(path, alias)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / NOTES_FILE).write_text(json.dumps(notes, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return calls

    def _describe(self, table: dict, columns: List[str]) -> Dict[str, str]:
        completion = chat(
            self.client,
            {"stage": "column_notes"},
            model=LLM_MODEL_SUMMARY,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Schema:\n{table['schema']}\n\nDescribe these columns of "
                                            f"{table['name']}: {', '.join(columns)}"},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return parse(completion.choices[0].message.content or "", columns)


def parse(text: str, columns: List[str]) -> Dict[str, str]:
    """Only the columns asked for; one line each, short, no "unclear" guesses, no SQL."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0) if match else text)
    except json.JSONDecodeError:
        return {}
    wanted = {c.lower(): c for c in columns}
    notes = {}
    for name, note in (data.get("columns") or {}).items():
        real = wanted.get(str(name).lower())
        note = " ".join(str(note).split())
        if (real and note and note.lower().rstrip(".") != "unclear" and len(note) <= MAX_NOTE_CHARS
                and not re.search(r"\b(SELECT|FROM|WHERE)\b", note)):
            notes[real] = note
    return notes
