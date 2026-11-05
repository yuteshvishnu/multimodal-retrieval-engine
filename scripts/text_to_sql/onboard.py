"""
Version-20: offline database onboarding. Builds each database's schema catalog and value index
(data/catalogs/), from the data only, with no LLM call. Scout also does this by itself the first
time it opens a database (or when the file changes); this script does it ahead of time.

Run from the project root:
    python -m scripts.text_to_sql.onboard                 # Scout's databases (data/text_to_sql/)
    python -m scripts.text_to_sql.onboard --bird          # every BIRD Mini-Dev database
    python -m scripts.text_to_sql.onboard --rebuild       # build again even if up to date
    python -m scripts.text_to_sql.onboard --describe      # also AI notes for undescribed columns
                                                          # (LLM calls: says how many and asks first)
"""
import argparse
import shutil
from pathlib import Path

from backend.config import BIRD_MINI_DEV_DIR
from backend.text_to_sql import schema_catalog
from backend.text_to_sql.column_notes import ColumnNoteWriter, undescribed
from backend.text_to_sql.database import SQLiteDatabase


def folders(bird: bool):
    if bird:
        # one at a time: each BIRD question uses only its own database (SQLite attaches 10 at most)
        return sorted(p for p in (BIRD_MINI_DEV_DIR / "dev_databases").iterdir() if p.is_dir())
    return [Path("data/text_to_sql")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird", action="store_true", help="BIRD Mini-Dev's databases instead of Scout's")
    parser.add_argument("--rebuild", action="store_true", help="build again even if up to date")
    parser.add_argument("--describe", action="store_true", help="AI notes for undescribed columns (LLM calls)")
    parser.add_argument("--yes", action="store_true", help="don't ask before the LLM calls")
    args = parser.parse_args()

    for folder in folders(args.bird):
        if args.rebuild:
            for path in folder.rglob("*.sqlite"):
                shutil.rmtree(schema_catalog.catalog_dir(path, path.stem.replace('"', "")), ignore_errors=True)
        database = SQLiteDatabase(str(folder))
        tables = database.tables  # builds whatever is missing or out of date
        kinds = {}
        for table in tables:
            for column in table["catalog"]["columns"]:
                kinds[column["kind"]] = kinds.get(column["kind"], 0) + 1
        inferred = sum(1 for t in tables for link in t["catalog"]["links"] if link["inferred"])
        print(f"[Onboard] {folder}: {len(tables)} tables; columns by kind {kinds}; {inferred} inferred links")

        if args.describe:
            pending = [t["name"] for t in tables if undescribed(t)]
            if not pending:
                print("[Onboard] every column is described already")
            elif args.yes or input(f"Write AI notes for {len(pending)} tables ({len(pending)} LLM calls)? [y/N] ").lower() == "y":
                print(f"[Onboard] {ColumnNoteWriter().write(database)} LLM calls")
        database.close()


if __name__ == "__main__":
    main()
