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

                # Build a bullet-style summary from the top chunks
        bullets = []
        max_bullets = 3  # keep it short and readable

        for chunk in retrieved_chunks[:max_bullets]:
            text = chunk["snippet"].strip()

            # Trim to a reasonable length
            if len(text) > 200:
                text = text[:197].rsplit(" ", 1)[0] + "..."

            # Ensure it starts with a capital and no trailing spaces
            if text and not text[0].isupper():
                text = text[0].upper() + text[1:]

            bullets.append(f"• {text}")

        if not bullets:
            bullets_text = "(No relevant details could be extracted.)"
        else:
            bullets_text = "\n".join(bullets)

        prefix = "Based on the retrieved documents"
        if image_info:
            prefix += f" and the image you provided ({image_info})"

        answer = (
            f"{prefix}, here is a brief summary related to your question "
            f"'{query_text}':\n\n{bullets_text}"
        )

        ids = [chunk["id"] for chunk in retrieved_chunks]
        reasoning_summary = f"Answer constructed from {len(retrieved_chunks)} chunks: {', '.join(ids)}"

        return answer, reasoning_summary