from typing import List, Dict
from pathlib import Path
import json

import numpy as np


class VectorIndex:
    """
    Simple in-memory vector index backed by embeddings + metadata saved to disk.
    For now uses brute-force cosine similarity, no Faiss yet.
    """

    def __init__(
        self,
        embeddings_path: str = "data/processed/embeddings.npy",
        metadata_path: str = "data/processed/metadata.json",
    ):
        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = Path(metadata_path)

        self.embeddings: np.ndarray | None = None
        self.metadata: List[Dict] | None = None
        self.dim: int | None = None

        self._load_index()

    def _load_index(self) -> None:
        if not self.embeddings_path.exists() or not self.metadata_path.exists():
            print(
                "[VectorIndex] WARNING: embeddings or metadata not found. "
                "Did you run scripts/build_index.py?"
            )
            self.embeddings = None
            self.metadata = None
            self.dim = None
            return

        self.embeddings = np.load(self.embeddings_path)  # shape: (N, D)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if self.embeddings.ndim != 2:
            raise ValueError("Embeddings array must be 2D (num_docs, dim).")

        self.dim = self.embeddings.shape[1]
        print(
            f"[VectorIndex] Loaded {self.embeddings.shape[0]} vectors "
            f"with dim={self.dim} from {self.embeddings_path}"
        )

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Brute-force cosine similarity search over all document embeddings.
        Returns a list of dicts sorted by similarity desc.
        """
        if self.embeddings is None or self.metadata is None:
            return []

        # compute cosine similarity against all docs
        sims: List[tuple[str, float, Dict]] = []
        for doc, doc_vec in zip(self.metadata, self.embeddings):
            sim = self._cosine_sim(query_vec, doc_vec)
            sims.append((doc["id"], sim, doc))

        # sort by similarity descending
        sims.sort(key=lambda x: x[1], reverse=True)

        # Normalize scores to 0–1 range
        raw_scores = [s for (_, s, _) in sims]
        max_score = max(raw_scores)
        min_score = min(raw_scores)

        def normalize(val: float) -> float:
            if max_score == min_score:
                return 1.0
            return (val - min_score) / (max_score - min_score)

        relevance_threshold = 0.2  # keep only reasonably relevant chunks

        results: List[Dict] = []
        fallback_results: List[Dict] = []

        for doc_id, score, doc in sims[:top_k]:
            norm_score = float(normalize(score))
            snippet = doc["text"][:60].replace("\n", " ")

            entry = {
                "id": doc_id,
                "score": norm_score,
                "source": "text_corpus",
                "snippet": snippet,
            }

            fallback_results.append(entry)

            if norm_score >= relevance_threshold:
                results.append(entry)

        # if nothing passes the threshold, fall back to top_k unfiltered
        if not results:
            results = fallback_results

        return results