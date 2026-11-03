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
    python -m scripts.text_to_sql.evaluate --full-schema       # versions 1-2: send all tables
    python -m scripts.text_to_sql.evaluate --ids 3,7,8         # only these questions
    python -m scripts.text_to_sql.evaluate --confirm           # the app's flow: confirm the meaning first
    python -m scripts.text_to_sql.evaluate --questions path/to/questions.json
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.rag.encoders.text_encoder import get_text_encoder
from backend.text_to_sql.pipeline import TOP_K_TABLES, TextToSQLPipeline

DEFAULT_QUESTIONS = Path("data/text_to_sql/retail_store/questions.json")


def as_set(rows):
    # order-insensitive comparison; round floats so 399.99 == 399.9900000001
    return {tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in rows}


def tables_in(sql, table_names):
    """Names of the known tables that a SQL query mentions."""
    return {name for name in table_names if re.search(rf"\b{re.escape(name)}\b", sql, re.IGNORECASE)}


def pick_meaning(options, q):
    """
    Plays the user in the confirm-the-meaning step (Version-5): picks the option closest in
    meaning (embeddings) to what the question intends, i.e. its "trap" note when it has one.
    A stand-in for a person, so a wrong pick here is a limit of the simulation, not of Scout.
    """
    if len(options) == 1:
        return options[0]
    encoder = get_text_encoder()
    target = encoder.encode(f"{q['question']} {q.get('trap', '')}")
    texts = [o["meaning"] + " " + " ".join(o["choices"]) for o in options]
    scores = [float(np.dot(target, v) / (np.linalg.norm(target) * np.linalg.norm(v) + 1e-8))
              for v in (encoder.encode(t) for t in texts)]
    return options[int(np.argmax(scores))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--k", type=int, default=TOP_K_TABLES, help="tables picked by embedding search")
    parser.add_argument("--full-schema", action="store_true", help="send every table (no table search)")
    parser.add_argument("--retrieval-only", action="store_true", help="only score table picking; no LLM calls")
    parser.add_argument("--ids", help="comma-separated question ids to run, e.g. 3,7,8")
    parser.add_argument("--confirm", action="store_true",
                        help="like the app: get the meanings, pick one (simulated user), then write SQL")
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.ids:
        wanted = {int(i) for i in args.ids.split(",")}
        questions = [q for q in questions if q["id"] in wanted]
    pipeline = TextToSQLPipeline(use_retrieval=not args.full_schema, top_k=args.k)
    all_names = [table["name"] for table in pipeline.tables]

    recall_hits, tables_sent, correct, llm_calls = 0, 0, 0, 0
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

        meaning, interpret_calls = "", 0
        if args.confirm:
            interpreted = pipeline.interpret(q["question"])
            interpret_calls = interpreted.get("llm_calls", 0)
            if interpreted.get("options"):
                picked_option = pick_meaning(interpreted["options"], q)
                meaning = "\n".join([picked_option["meaning"]] + [f"- {c}" for c in picked_option["choices"]])
        result = pipeline.run(q["question"], meaning)
        result["llm_calls"] = result.get("llm_calls", 0) + interpret_calls
        _, gold_rows = pipeline.database.run(q["gold_sql"])
        # the response holds only page 1; compare every row kept for the result
        saved = pipeline.results.get(result["result_id"]) if result.get("result_id") else None
        rows = saved["rows"] if saved else result.get("rows", [])

        ok = not result.get("error") and as_set(rows) == as_set(gold_rows)
        correct += ok
        by_difficulty[q["difficulty"]][0] += ok
        by_difficulty[q["difficulty"]][1] += 1

        llm_calls += result.get("llm_calls", 0)

        mark = "PASS" if ok else "FAIL"
        print(f"\n[{mark}] #{q['id']} ({q['difficulty']}) {q['question']}")
        print(f"  Confidence: {result.get('confidence')}, LLM calls: {result.get('llm_calls', 0)}")
        if meaning:
            print(f"  Meaning:   {meaning.splitlines()[0]}")
        for warning in result.get("warnings", []):
            print(f"  Warning:   {warning}")
        if not ok:
            if missing:
                print(f"  Tables:    picked {picked}, missing {sorted(missing)}")
            print(f"  Model SQL: {result.get('sql')}")
            if result.get("error"):
                print(f"  Error:     {result['error']}")
            else:
                print(f"  Model got: {rows[:5]}")
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
    print(f"LLM calls: {llm_calls} total, {llm_calls / total:.1f} per question")


if __name__ == "__main__":
    main()
