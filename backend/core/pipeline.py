from typing import Optional
import numpy as np

from backend.encoders.text_encoder import TextEncoder
from backend.encoders.image_encoder import ImageEncoder
from backend.retrieval.index import VectorIndex  # 👈 NEW
from backend.retrieval.reranker import ReRanker
from backend.reasoning.llm_reasoner import LLMReasoner


class MultimodalPipeline:
    """
    Orchestrates the multimodal retrieval + reasoning.
    For now this is just a stub that returns a dummy answer.
    """

    def __init__(self):
        # encoders
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()

       # retrieval index (stage 1)
        self.index = VectorIndex(dim=768)

        # re-ranking (stage 2)
        self.reranker = ReRanker(enabled=True)

        # reasoning layer
        self.reasoner = LLMReasoner()  # 👈 NEW

    def run(
        self,
        query_text: str,
        image_bytes: Optional[bytes] = None,
        audio_bytes: Optional[bytes] = None,
        sources: list[str] | None = None,
        collections: list[str] | None = None,
    ) -> dict:
        print(f"[Pipeline] Received query: {query_text!r}")
        """
        Current behavior:
        - encode the query text into an embedding
        - call the vector index to get dummy retrieved chunks
        - return a dummy answer + retrieved citations
        """

        # 1) encode text
        text_embedding: np.ndarray = self.text_encoder.encode(query_text)
        print(f"[Pipeline] Text embedding dim={text_embedding.shape[0]}")
        embedding_norm = float(np.linalg.norm(text_embedding))

        # 1.2 encode image
        print(f"[Pipeline] Image bytes: {image_bytes}")
        image_embedding = None
        if image_bytes:
            image_embedding = self.image_encoder.encode(image_bytes)
            print(f"[Pipeline] Image embedding dim={image_embedding.shape[0]}")
        else:
            print("[Pipeline] No image provided")

        image_info = None
        if image_embedding is not None:
            image_info = "image embedding computed"

        used_image = image_info is not None

        # 2) retrieve (dummy)
        initial_top_k=20
        final_top_k=8
        stage1_retrieved = self.index.search(text_embedding, top_k=initial_top_k, source_filter=sources, collection_filter=collections)
        print(f"[Pipeline] Retrieved {len(stage1_retrieved)} candidates from stage 1")
        stage2_retrieved = self.reranker.rerank(query_text, candidates=stage1_retrieved, final_top_k=final_top_k)
        print(f"[Pipeline] Retrieved {len(stage2_retrieved)} candidates from stage 2")

        # 3) reasoning (for now, simple rule-based summarizer)
        answer, reasoning_summary = self.reasoner.answer(
            query_text=query_text,
            retrieved_chunks=stage2_retrieved,
            image_info=image_info,
        )
        print("[Pipeline] Reasoning layer produced an answer")

        # 4) build final response
        return {
            "answer": answer,
            "citations": stage2_retrieved,
            "reasoning_summary": (
                f"Text embedding dim={text_embedding.shape[0]}, "
                f"||embedding||={embedding_norm:.3f}. "
                + reasoning_summary
            ),
            "used_image": used_image,
        }