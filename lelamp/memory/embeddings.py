from __future__ import annotations

import numpy as np


class ClipEmbedder:
    """open_clip on MPS, used for object crops, scene summaries, and text queries."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> None:
        self.model_name = model_name
        self.pretrained = pretrained

    def embed_image(self, image: np.ndarray) -> list[float]:
        raise NotImplementedError

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError
