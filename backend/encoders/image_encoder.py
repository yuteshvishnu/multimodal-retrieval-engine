from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np


class ImageEncoder:
    """
    Very simple image encoder placeholder.

    For now:
    - We do NOT use a real vision model.
    - We hash the image bytes into a fixed-size vector.
    - Same image -> same vector.
    - This keeps the pipeline working end-to-end until we swap in a real model.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def encode(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            # empty input -> zero vector
            return np.zeros(self.dim, dtype="float32")

        # Hash the bytes to get deterministic pseudo-random numbers
        h = hashlib.sha256(image_bytes).digest()  # 32 bytes
        base = np.frombuffer(h, dtype=np.uint8).astype("float32")

        # Repeat to reach desired dim
        reps = int(np.ceil(self.dim / base.size))
        vec = np.tile(base, reps)[: self.dim]

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.astype("float32")