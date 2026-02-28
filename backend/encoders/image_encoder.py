import numpy as np


class ImageEncoder:
    """
    Wraps an image embedding model.
    For now, this returns a random vector just to get the plumbing right.
    Later, we'll replace this with a real CLIP/SigLIP encoder.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def encode(self, image_bytes: bytes) -> np.ndarray:
        """
        Encode raw image bytes into a 1D numpy array of shape (dim,).

        For now:
        - we ignore the actual pixels
        - just return a fixed-shape random vector
        """
        # TODO: decode bytes -> image -> preprocess -> pass through CLIP in a later step.
        return np.random.randn(self.dim).astype("float32")