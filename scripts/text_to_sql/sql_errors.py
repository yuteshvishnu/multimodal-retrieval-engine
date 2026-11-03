"""
Report on Scout's SQL error log, to grow the error book (Version-9).

    python -m scripts.text_to_sql.sql_errors

UNKNOWN errors: no entry in data/text_to_sql/sql_errors.json yet, grouped and counted, each
with an example. Write a hint for the common ones and add them to the file (no restart needed).

KNOWN errors: how often each was seen, and how often the question still got an answer
afterwards. A hint that rarely helps needs rewriting.
"""
from collections import Counter, defaultdict

from backend.text_to_sql.sql_errors import ErrorBook


def main():
    book = ErrorBook()
    events = book.read_log()
    errors = [e for e in events if e.get("type") == "error"]
    outcomes = [e for e in events if e.get("type") == "outcome"]
    if not errors:
        print(f"No SQL errors logged yet ({book.log_file}).")
        return

    print(f"{len(errors)} rejected attempts in {len(outcomes)} questions ({book.log_file})\n")

    unknown = defaultdict(list)
    for e in errors:
        if not e.get("known_id"):
            unknown[e["normalized"]].append(e)
    print(f"========== UNKNOWN ERRORS ({len(unknown)}) - add these to {book.known_file} ==========")
    if not unknown:
        print("none")
    for normalized, group in sorted(unknown.items(), key=lambda kv: -len(kv[1])):
        last = group[-1]
        print(f"\n[{len(group)}x] {normalized}")
        print(f"  last seen: {last['time']}")
        print(f"  question:  {last['question']}")
        print(f"  error:     {last['error']}")
        print("  SQL:       " + " ".join((last.get("sql") or "").split())[:300])

    seen = Counter(e["known_id"] for e in errors if e.get("known_id"))
    asked, answered = Counter(), Counter()
    for o in outcomes:
        for known_id in o.get("known_ids", []):
            asked[known_id] += 1
            answered[known_id] += bool(o.get("fixed"))
    print("\n========== KNOWN ERRORS ==========")
    print(f"{'id':32} {'seen':>5} {'questions':>10} {'answered after hint':>20}")
    for entry in book.entries:
        n = asked[entry["id"]]
        rate = f"{answered[entry['id']]}/{n} ({answered[entry['id']] / n:.0%})" if n else "-"
        print(f"{entry['id']:32} {seen[entry['id']]:>5} {n:>10} {rate:>20}")

    early = sum(1 for o in outcomes if o.get("stopped_early"))
    print(f"\nQuestions with errors that still got an answer: {sum(1 for o in outcomes if o.get('fixed'))}/{len(outcomes)}")
    print(f"Stopped after round 1 (unknown error, round 2 skipped): {early}")


if __name__ == "__main__":
    main()
