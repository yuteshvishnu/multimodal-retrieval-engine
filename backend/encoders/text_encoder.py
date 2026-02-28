import numpy as np
from sentence_transformers import SentenceTransformer


class TextEncoder:
    """
    Wraps a sentence-transformers embedding model.
    Provides a simple encode(text) -> np.ndarray interface.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        # sentence-transformers models usually return 384 or 768 dims depending on the model
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single string into a 1D numpy array of shape (dim,).
        """
        emb = self.model.encode([text])  # shape: (1, dim)
        return emb[0].astype("float32")