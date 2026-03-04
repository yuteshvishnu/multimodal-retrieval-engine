from typing import Optional
import numpy as np
import torch

from backend.encoders.text_encoder import TextEncoder
from backend.encoders.image_encoder import ImageEncoder
from backend.retrieval.index import VectorIndex
from backend.retrieval.reranker import ReRanker
from backend.reasoning.llm_reasoner import LLMReasoner


class MultimodalPipeline:
    """
    Orchestrates the multimodal retrieval + reasoning.
    Updated to handle Cloud-Hybrid LLM reasoning without crashing.
    """

    def __init__(self):
        # encoders
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()

        # retrieval index (stage 1)
        self.index = VectorIndex()

        # re-ranking (stage 2)
        self.reranker = ReRanker(enabled=True)

        # reasoning layer
        self.reasoner = LLMReasoner()

    def run(
        self,
        query_text: str | None,
        image_bytes: Optional[bytes] = None,
        audio_bytes: Optional[bytes] = None,
        sources: list[str] | None = None,
        collections: list[str] | None = None,
    ) -> dict:
        """
        Main entry point:
        - Encode signals
        - Stage 1 Search
        - Stage 2 Rerank
        - Cloud/Local Reasoning
        """

        # 1) encode query text
        text_embedding = None
        if query_text is not None and query_text.strip():
            text_embedding = self.text_encoder.encode(query_text)

        # 2) encode query image
        image_embedding = None
        image_info = None
        if image_bytes is not None:
            image_embedding = self.image_encoder.encode(image_bytes)
            image_info = "Image provided with the query"

        # 3) validate input
        if text_embedding is None and image_embedding is None:
            return {
                "answer": "Please provide a question or an image.",
                "citations": [],
                "reasoning_summary": "No query text or image provided.",
            }

        # 4) build query vector
        query_vecs = []
        if text_embedding is not None:
            query_vecs.append(text_embedding)
        if image_embedding is not None:
            query_vecs.append(image_embedding)

        combined_query_vec = np.mean(query_vecs, axis=0)

        # 5) Stage 1: vector search
        stage1_top_k = 20
        stage1_candidates = self.index.search(
            combined_query_vec,
            top_k=stage1_top_k,
            source_filter=sources,
            collection_filter=collections,
        )
        print(f"[Pipeline] Stage 1 retrieved {len(stage1_candidates)} candidates")

        # 6) Stage 2: rerank
        final_top_k = 5
        retrieved = self.reranker.rerank(
            query_text=query_text or "",
            candidates=stage1_candidates,
            final_top_k=final_top_k,
        )
        print(f"[Pipeline] Stage 2 re-ranked to {len(retrieved)} final chunks")

        # 7) reasoning
        answer, reasoning_summary = self.reasoner.answer(
            query_text=query_text or "",
            retrieved_chunks=retrieved,
            image_info=image_info,
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