from typing import List, Dict


class ReRanker:
    """
    Second-stage retriever that reorders candidate chunks.

    Stage 1: VectorIndex.search() gives us top_N candidates (fast, embedding-based).
    Stage 2: ReRanker.rerank() reorders those candidates using a cheaper, more precise heuristic.

    For now we use a simple heuristic:
    - Count how many query words appear in each snippet
    - Break ties using the original similarity score
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def rerank(
        self,
        query_text: str,
        candidates: List[Dict],
        final_top_k: int = 5,
    ) -> List[Dict]:
        """
        Re-rank the candidate chunks and return the top `final_top_k`.

        Each candidate is a dict with at least:
          - "snippet": str
          - "score": float
          - "id": str

        If disabled or no candidates, just return the first `final_top_k`.
        """
        if not self.enabled or not candidates:
            return candidates[:final_top_k]

        # Basic term-based heuristic on top of embedding score
        query_terms = set(query_text.lower().split())

        def extra_signal(c: Dict) -> tuple[int, float]:
            snippet = c.get("snippet", "").lower()
            term_hits = sum(1 for t in query_terms if t in snippet)
            base_score = float(c.get("score", 0.0))
            return term_hits, base_score

        # Sort by:
        # 1) more query-term hits in snippet
        # 2) higher original similarity score
        sorted_candidates = sorted(
            candidates,
            key=extra_signal,
            reverse=True,
        )

        return sorted_candidates[:final_top_k]