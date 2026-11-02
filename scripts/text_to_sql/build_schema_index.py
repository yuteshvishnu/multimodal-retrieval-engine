"""
Builds everything table search needs, ahead of time (Version-3):

1. Summaries: asks the LLM to describe each new or changed table
   (saved to table_summaries.json next to the database; unchanged tables are skipped).
2. Embeddings: embeds every table's summary (saved to table_embeddings.npz).

Run from the project root:
    python -m scripts.text_to_sql.build_schema_index                   # summarize new/changed tables, then embed
    python -m scripts.text_to_sql.build_schema_index --resummarize     # summarize every table again
    python -m scripts.text_to_sql.build_schema_index --skip-summaries  # embed only, no LLM calls
"""
import argparse

from backend.text_to_sql.database import SQLiteDatabase
from backend.text_to_sql.schema_index import SchemaIndex
from backend.text_to_sql.table_summaries import TableSummarizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resummarize", action="store_true", help="summarize every table again")
    parser.add_argument("--skip-summaries", action="store_true", help="no LLM calls; embed only")
    args = parser.parse_args()

    if not args.skip_summaries:
        TableSummarizer(SQLiteDatabase()).build(rebuild=args.resummarize)

    # reload, so the new summaries are what gets embedded
    tables = SQLiteDatabase().tables
    SchemaIndex(tables, rebuild=True)
    print(f"[Done] Indexed {len(tables)} tables")


if __name__ == "__main__":
    main()
