"""
Text-to-SQL step 3: find the tables relevant to a question with embeddings (RAG applied to schemas).

Offline, once per database:
    each table's one-line summary -> embedding, saved next to the database
    (table_embeddings.npz), and rebuilt automatically when the tables change.

Per question:
    embed the question -> take the top-k closest tables -> add bridge tables.
    No LLM call: this takes milliseconds.
"""
import hashlib
from itertools import groupby
from pathlib import Path
from typing import List

import numpy as np

from backend.rag.encoders.text_encoder import get_text_encoder

INDEX_FILE = "table_embeddings.npz"


class SchemaIndex:
    def __init__(self, tables: List[dict], rebuild: bool = False):
        self.tables = tables
        self.encoder = get_text_encoder()
        self.vectors = self._load_or_build(rebuild)

    def retrieve(self, question: str, top_k: int) -> List[dict]:
        """
        The top_k tables closest in meaning to the question, plus any bridge tables
        needed to join them.
        """
        query = self.encoder.encode(question)
        query = query / (np.linalg.norm(query) + 1e-8)
        scores = self.vectors @ query  # cosine similarity (vectors are normalized)

        ranked = np.argsort(-scores)[:top_k]
        picked = [self.tables[i] for i in ranked]
        names = {table["name"] for table in picked}

        # A bridge table is linked (by foreign keys, either direction) to two or more picked
        # tables, so the SQL likely needs it to join them. Its own wording rarely matches the
        # question, so similarity alone tends to miss it. For example:
        #   order_items, which points to orders and products
        #   orders, which order_items, payments and shipments all point to
        bridges = []
        for table in self.tables:
            if table["name"] in names:
                continue
            points_to = len(table["links"] & names)
            pointed_from = sum(1 for p in picked if table["name"] in p["links"])
            if points_to + pointed_from >= 2:
                bridges.append(table)

        return picked + bridges

    def _load_or_build(self, rebuild: bool) -> np.ndarray:
        vectors = []
        # one index file per database folder
        for folder, group in groupby(self.tables, key=lambda t: t["folder"]):
            group = list(group)
            path = Path(folder) / INDEX_FILE
            fingerprint = self._fingerprint(group)

            if not rebuild and path.exists():
                saved = np.load(path)
                if str(saved["fingerprint"]) == fingerprint:
                    vectors.extend(saved["vectors"])
                    continue

            print(f"[SchemaIndex] Embedding {len(group)} tables in {folder} ...")
            embedded = np.stack([self.encoder.encode(table["doc"]) for table in group])
            try:
                np.savez(path, vectors=embedded, fingerprint=fingerprint)
                print(f"[SchemaIndex] Saved {path}")
            except OSError as e:
                print(f"[SchemaIndex] Could not save {path} ({e}); using it in memory only")
            vectors.extend(embedded)

        if not vectors:
            return np.zeros((0, self.encoder.dim), dtype="float32")
        matrix = np.stack(vectors).astype("float32")
        return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)

    def _fingerprint(self, tables: List[dict]) -> str:
        """Changes whenever a table's summary or the embedding size changes."""
        text = f"dim={self.encoder.dim}\n" + "\n".join(table["doc"] for table in tables)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
