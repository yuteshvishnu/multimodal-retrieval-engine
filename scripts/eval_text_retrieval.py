import json
from pathlib import Path
from typing import List, Dict

from backend.core.pipeline import MultimodalPipeline


EVAL_FILE = Path("data/eval/text_eval.json")
TOP_K = 5  # how many citations we look at for hit@k


def load_eval_cases() -> List[Dict]:
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Eval file not found: {EVAL_FILE}")
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    print(f"[Eval] Loading eval cases from {EVAL_FILE}...")
    cases = load_eval_cases()
    print(f"[Eval] Loaded {len(cases)} cases")

    pipeline = MultimodalPipeline()

    total = len(cases)
    source_hits = 0
    snippet_hits = 0

    for idx, case in enumerate(cases, start=1):
        query = case["query"]
        expected_source = case.get("expected_source")
        expected_name_contains = case.get("expected_name_contains")

        print(f"\n[Eval] Case {idx}/{total}")
        print(f"  Query: {query!r}")
        print(f"  Expected source: {expected_source!r}")
        print(f"  Expected snippet contains: {expected_name_contains!r}")

        # Run pipeline (text-only; no image, no source filter)
        result = pipeline.run(
            query_text=query,
            image_bytes=None,
            sources=None,
        )

        citations = result.get("citations", [])[:TOP_K]
        print(f"  Got {len(citations)} citations (top {TOP_K})")

        # Check source hit@k
        hit_source = False
        if expected_source is not None:
            for c in citations:
                if c.get("source") == expected_source:
                    hit_source = True
                    break
        else:
            hit_source = None  # no expectation

        # Check snippet substring hit@k
        hit_snippet = False
        if expected_name_contains is not None:
            for c in citations:
                if expected_name_contains in c.get("snippet", ""):
                    hit_snippet = True
                    break
        else:
            hit_snippet = None

        if hit_source:
            source_hits += 1
        if hit_snippet:
            snippet_hits += 1

        print(f"  hit@{TOP_K} (source): {hit_source}")
        print(f"  hit@{TOP_K} (snippet contains): {hit_snippet}")

    print("\n========== EVAL SUMMARY ==========")
    print(f"Total cases: {total}")

    # Only count cases where expectation exists
    source_cases = sum(1 for c in cases if c.get("expected_source") is not None)
    snippet_cases = sum(1 for c in cases if c.get("expected_name_contains") is not None)

    if source_cases > 0:
        print(
            f"hit@{TOP_K} (source): {source_hits}/{source_cases} "
            f"({(source_hits / source_cases) * 100:.1f}%)"
        )
    else:
        print(f"hit@{TOP_K} (source): n/a (no expected_source specified)")

    if snippet_cases > 0:
        print(
            f"hit@{TOP_K} (snippet contains): {snippet_hits}/{snippet_cases} "
            f"({(snippet_hits / snippet_cases) * 100:.1f}%)"
        )
    else:
        print(f"hit@{TOP_K} (snippet contains): n/a (no expected_name_contains specified)")


if __name__ == "__main__":
    main()