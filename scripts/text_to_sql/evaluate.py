"""
Scores the Text-to-SQL pipeline on a set of questions with known correct SQL.

For each question, the model's SQL and the correct ("gold") SQL are both run on
the database, and the two result sets are compared (Execution Accuracy, as in BIRD).

Run from the project root:
    python -m scripts.text_to_sql.evaluate
    python -m scripts.text_to_sql.evaluate --questions path/to/questions.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from backend.text_to_sql.pipeline import TextToSQLPipeline

DEFAULT_QUESTIONS = Path("data/text_to_sql/retail_store/questions.json")


def as_set(rows):
    # order-insensitive comparison; round floats so 399.99 == 399.9900000001
    return {tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    pipeline = TextToSQLPipeline()

    correct = 0
    by_difficulty = defaultdict(lambda: [0, 0])  # difficulty -> [correct, total]

    for q in questions:
        result = pipeline.run(q["question"])
        _, gold_rows = pipeline.database.run(q["gold_sql"])

        ok = not result.get("error") and as_set(result.get("rows", [])) == as_set(gold_rows)
        correct += ok
        by_difficulty[q["difficulty"]][0] += ok
        by_difficulty[q["difficulty"]][1] += 1

        mark = "PASS" if ok else "FAIL"
        print(f"\n[{mark}] #{q['id']} ({q['difficulty']}) {q['question']}")
        if not ok:
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
    print("\n========== EXECUTION ACCURACY ==========")
    for difficulty in ("easy", "medium", "hard"):
        if difficulty in by_difficulty:
            c, t = by_difficulty[difficulty]
            print(f"{difficulty:<8}: {c}/{t}")
    print(f"{'overall':<8}: {correct}/{total} ({correct / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
