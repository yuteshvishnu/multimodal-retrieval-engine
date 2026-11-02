from backend.encoders.text_encoder import TextEncoder
from backend.retrieval.index import VectorIndex
from backend.retrieval.reranker import ReRanker
from backend.reasoning.llm_reasoner import LLMReasoner


class RAGPipeline:
    """
    Orchestrates text retrieval + reasoning.
    Updated to handle Cloud-Hybrid LLM reasoning without crashing.
    """

    def __init__(self):
        # encoder
        self.text_encoder = TextEncoder()

        # retrieval index (stage 1)
        self.index = VectorIndex()

        # re-ranking (stage 2)
        self.reranker = ReRanker(enabled=True)

        # reasoning layer
        self.reasoner = LLMReasoner()

    def run(
        self,
        query_text: str | None,
        sources: list[str] | None = None,
        collections: list[str] | None = None,
    ) -> dict:
        """
        Main entry point:
        - Encode query
        - Stage 1 Search
        - Stage 2 Rerank
        - Cloud/Local Reasoning
        """

        # 1) validate input
        if query_text is None or not query_text.strip():
            return {
                "answer": "Please provide a question.",
                "citations": [],
                "reasoning_summary": "No query text provided.",
            }

        # 2) encode query text
        query_vec = self.text_encoder.encode(query_text)

        # 3) Stage 1: vector search
        stage1_top_k = 20
        stage1_candidates = self.index.search(
            query_vec,
            top_k=stage1_top_k,
            source_filter=sources,
            collection_filter=collections,
        )
        print(f"[Pipeline] Stage 1 retrieved {len(stage1_candidates)} candidates")

        # 4) Stage 2: rerank
        final_top_k = 5
        retrieved = self.reranker.rerank(
            query_text=query_text,
            candidates=stage1_candidates,
            final_top_k=final_top_k,
        )
        print(f"[Pipeline] Stage 2 re-ranked to {len(retrieved)} final chunks")

        # 5) reasoning
        answer, reasoning_summary = self.reasoner.answer(
            query_text=query_text,
            retrieved_chunks=retrieved,
        )

        # --- UPDATED DEVICE LOGGING ---
        # We use a safe check here to prevent the 'bool' object has no attribute 'model' error
        if hasattr(self.reasoner, 'client') and self.reasoner.client:
            print(f"[Pipeline] Inference completed via Cloud API ({self.reasoner.model_name})")
        elif hasattr(self.reasoner, 'generator') and self.reasoner.generator and hasattr(self.reasoner.generator, 'model'):
            device = self.reasoner.generator.model.device
            print(f"[Pipeline] Inference completed via Local model on: {device}")
        else:
            print("[Pipeline] Inference completed (Rule-based or unknown mode)")

        return {
            "answer": answer,
            "citations": retrieved,
            "reasoning_summary": reasoning_summary,
        }
