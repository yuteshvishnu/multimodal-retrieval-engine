"""
Scores the Text-to-SQL pipeline on a set of questions with known correct SQL.

Two scores:
- Schema recall (Version-3): did the picked tables include every table the correct
  SQL uses? Needs no LLM call.
- Execution accuracy (as in BIRD): the model's SQL and the correct ("gold") SQL are
  both run on the database and their result sets compared.

Run from the project root:
    python -m scripts.text_to_sql.evaluate                     # full run (calls the LLM)
    python -m scripts.text_to_sql.evaluate --retrieval-only    # table picking only, no LLM
    python -m scripts.text_to_sql.evaluate --k 3               # try a different top-k
    python -m scripts.text_to_sql.evaluate --full-schema       # Version-2: send all tables
    python -m scripts.text_to_sql.evaluate --questions path/to/questions.json
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from backend.text_to_sql.pipeline import TOP_K_TABLES, TextToSQLPipeline

DEFAULT_QUESTIONS = Path("data/text_to_sql/retail_store/questions.json")


def as_set(rows):
    # order-insensitive comparison; round floats so 399.99 == 399.9900000001
    return {tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in rows}


def tables_in(sql, table_names):
    """Names of the known tables that a SQL query mentions."""
    return {name for name in table_names if re.search(rf"\b{re.escape(name)}\b", sql, re.IGNORECASE)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--k", type=int, default=TOP_K_TABLES, help="tables picked by embedding search")
    parser.add_argument("--full-schema", action="store_true", help="send every table (no table search)")
    parser.add_argument("--retrieval-only", action="store_true", help="only score table picking; no LLM calls")
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    pipeline = TextToSQLPipeline(use_retrieval=not args.full_schema, top_k=args.k)
    all_names = [table["name"] for table in pipeline.tables]

    recall_hits, tables_sent, correct = 0, 0, 0
    by_difficulty = defaultdict(lambda: [0, 0])  # difficulty -> [correct, total]

    for q in questions:
        needed = tables_in(q["gold_sql"], all_names)
        picked = [table["name"] for table in pipeline.select_tables(q["question"])]
        missing = needed - set(picked)
        recall_hits += not missing
        tables_sent += len(picked)

        if args.retrieval_only:
            mark = "PASS" if not missing else "MISS"
            print(f"[{mark}] #{q['id']} ({q['difficulty']}) {q['question']}")
            print(f"  Picked:  {picked}")
            if missing:
                print(f"  Missing: {sorted(missing)}")
            continue

        result = pipeline.run(q["question"])
        _, gold_rows = pipeline.database.run(q["gold_sql"])

        ok = not result.get("error") and as_set(result.get("rows", [])) == as_set(gold_rows)
        correct += ok
        by_difficulty[q["difficulty"]][0] += ok
        by_difficulty[q["difficulty"]][1] += 1

        mark = "PASS" if ok else "FAIL"
        print(f"\n[{mark}] #{q['id']} ({q['difficulty']}) {q['question']}")
        if not ok:
            if missing:
                print(f"  Tables:    picked {picked}, missing {sorted(missing)}")
            print(f"  Model SQL: {result.get('sql')}")
            if result.get("error"):
                print(f"  Error:     {result['error']}")
            else:
                print(f"  Model got: {result.get('rows', [])[:5]}")
            print(f"  Gold SQL:  {q['gold_sql']}")
            print(f"  Expected:  {gold_rows[:5]}")
            if q.get("trap"):
                print(f"  Trap:      {q['trap']}")

    total = len(questions)
    mode = "full schema" if args.full_schema else f"top-{args.k} + bridge tables"
    print(f"\n========== SCHEMA RECALL ({mode}) ==========")
    print(f"all needed tables picked: {recall_hits}/{total} ({recall_hits / total * 100:.1f}%)")
    print(f"avg tables sent:          {tables_sent / total:.1f} of {len(all_names)}")

    if args.retrieval_only:
        return

    print("\n========== EXECUTION ACCURACY ==========")
    for difficulty in ("easy", "medium", "hard"):
        if difficulty in by_difficulty:
            c, t = by_difficulty[difficulty]
            print(f"{difficulty:<8}: {c}/{t}")
    print(f"{'overall':<8}: {correct}/{total} ({correct / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
