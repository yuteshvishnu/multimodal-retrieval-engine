from typing import List, Dict, Tuple


class LLMReasoner:
    """
    Reasoning / answer layer.
    For now, it's a simple rule-based summarizer over retrieved snippets.
    Later, we'll replace the internals with a real LLM call.
    """

    def __init__(self):
        # in the future, initialize LLM client/config here
        pass

    def answer(
        self,
        query_text: str,
        retrieved_chunks: List[Dict],
        image_info: str | None = None,
    ) -> Tuple[str, str]:
        """
        Build a simple answer from retrieved chunks.

        Returns:
            answer: a short, human-readable answer string
            reasoning_summary: brief explanation of what was used
        """
        if not retrieved_chunks:
            return (
                "I couldn't find any relevant documents to answer your question.",
                "No retrieved chunks available.",
            )

        # naive rule-based "answer": stitch snippets together
        joined_snippets = " ".join(chunk["snippet"] for chunk in retrieved_chunks)
        # truncate to avoid massive responses
        joined_snippets = joined_snippets[:500]

        prefix = "Based on the retrieved documents"
        if image_info:
            prefix += f" and the image you provided ({image_info})"

        answer = (
            f"{prefix}, here is a summary related to your question "
            f"'{query_text}':\n\n{joined_snippets}"
        )

        ids = [chunk["id"] for chunk in retrieved_chunks]
        reasoning_summary = f"Answer constructed from {len(retrieved_chunks)} chunks: {', '.join(ids)}"

        return answer, reasoning_summary