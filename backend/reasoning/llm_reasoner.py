import os
from typing import List, Dict, Tuple
from openai import OpenAI

class LLMReasoner:
    """
    Hybrid Reasoner: 
    1. Primary: Groq Cloud API (Llama 3.1 8B) for 500+ tokens/sec speed.
    2. Fallback: Local rule-based bullet summary if API fails.
    """

    def __init__(self, api_key: str = ""):
        # Points to Groq's high-speed infrastructure
        self.client = OpenAI(
            base_url="",
            api_key=api_key
        )
        self.model_name = "llama-3.1-8b-instant"
        # We set this to None so pipeline.py knows we aren't using a local model
        self.generator = None 

    def answer(
        self,
        query_text: str,
        retrieved_chunks: List[Dict],
        image_info: str | None = None,
    ) -> Tuple[str, str]:
        if not retrieved_chunks:
            return ("I couldn't find any relevant documents.", "No chunks available.")

        try:
            return self._answer_with_cloud(query_text, retrieved_chunks, image_info)
        except Exception as e:
            print(f"[LLMReasoner] Cloud failed: {e}. Falling back to rules.")
            return self._answer_with_bullets(query_text, retrieved_chunks, image_info)

    def _answer_with_cloud(self, query, chunks, img) -> Tuple[str, str]:
        context = "\n".join([f"[{c['id']}] {c['snippet']}" for c in chunks[:5]])
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Answer ONLY using context. Cite [id]."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content, f"Groq {self.model_name}"

    def _answer_with_bullets(self, query, chunks, img) -> Tuple[str, str]:
        bullets = "\n".join([f"• {c['snippet'][:200]}..." for c in chunks[:3]])
        return f"Summary for '{query}':\n\n{bullets}", "Rule-based fallback."