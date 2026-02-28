from typing import List, Dict, Tuple

from transformers import pipeline, Pipeline


class LLMReasoner:
    """
    Reasoning / answer layer using a free, open-source local model via transformers.

    - By default, uses microsoft/Phi-3-mini-4k-instruct (SOTA small instruct model).
    - If model loading or generation fails, falls back to rule-based bullet summary.
    """

    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name
        self.generator: Pipeline | None = None

        # try:
        #     # text-generation pipeline (works for chat/instruct models too)
        #     self.generator = pipeline(
        #         "text-generation",
        #         model=self.model_name,
        #         tokenizer=self.model_name,
        #         max_new_tokens=256,
        #         device_map="auto",  # uses GPU if available, CPU otherwise
        #     )
        #     print(f"[LLMReasoner] Loaded local model: {self.model_name}")
        # except Exception as e:
        #     print(f"[LLMReasoner] Failed to load model {self.model_name}: {e}")
        #     print("[LLMReasoner] Falling back to rule-based reasoning only.")
        #     self.generator = None

    def answer(
        self,
        query_text: str,
        retrieved_chunks: List[Dict],
        image_info: str | None = None,
    ) -> Tuple[str, str]:
        """
        Build an answer from retrieved chunks.

        Returns:
            answer: a short, human-readable answer string
            reasoning_summary: brief explanation of what was used
        """
        if not retrieved_chunks:
            return (
                "I couldn't find any relevant documents to answer your question.",
                "No retrieved chunks available.",
            )

        # If we have a local LLM, try to use it
        if self.generator is not None:
            try:
                return self._answer_with_llm(query_text, retrieved_chunks, image_info)
            except Exception as e:
                print(f"[LLMReasoner] Local LLM generation failed: {e}. Falling back to rule-based summary.")

        # Fallback: rule-based bullet summary
        return self._answer_with_bullets(query_text, retrieved_chunks, image_info)

    def _build_context(self, retrieved_chunks: List[Dict], max_chunks: int = 5) -> str:
        parts = []
        for chunk in retrieved_chunks[:max_chunks]:
            parts.append(f"[{chunk['id']}] {chunk['snippet']}")
        return "\n".join(parts)

    def _answer_with_llm(
        self,
        query_text: str,
        retrieved_chunks: List[Dict],
        image_info: str | None = None,
    ) -> Tuple[str, str]:
        """
        Use a local transformers model to synthesize an answer from retrieved chunks.
        """
        context_text = self._build_context(retrieved_chunks)

        instructions = (
            "You are a helpful assistant answering questions based ONLY on the provided context.\n"
            "If the context does not contain the answer, say you are not sure.\n"
            "When you use information from a chunk, mention its ID in square brackets, like [doc1_chunk0].\n"
        )

        if image_info:
            instructions += (
                "\nThe user also provided an image, but you do not have access to the pixels, "
                "so do not invent visual details not present in the text.\n"
            )

        prompt = (
            f"{instructions}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query_text}\n\n"
            "Answer clearly and concisely:\n"
        )

        print("[LLMReasoner] Calling local transformers model for reasoning...")
        outputs = self.generator(
            prompt,
            do_sample=False,
            num_return_sequences=1,
        )
        generated = outputs[0]["generated_text"]

        # Try to return just the part after the prompt (model often echoes prompt)
        if "Answer clearly and concisely:" in generated:
            answer_text = generated.split("Answer clearly and concisely:", 1)[-1].strip()
        else:
            # fallback: return everything after "Question:"
            if "Question:" in generated:
                answer_text = generated.split("Question:", 1)[-1].strip()
            else:
                answer_text = generated.strip()

        ids = [chunk["id"] for chunk in retrieved_chunks]
        reasoning_summary = (
            "Local LLM-generated answer using retrieved chunks: "
            + ", ".join(ids)
        )

        return answer_text, reasoning_summary

    def _answer_with_bullets(
        self,
        query_text: str,
        retrieved_chunks: List[Dict],
        image_info: str | None = None,
    ) -> Tuple[str, str]:
        """
        Rule-based fallback: format top chunks as a bullet list.
        """
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
        reasoning_summary = f"Rule-based summary constructed from {len(retrieved_chunks)} chunks: {', '.join(ids)}"

        return answer, reasoning_summary