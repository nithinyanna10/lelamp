from __future__ import annotations

from typing import cast

import numpy as np
import open_clip
import torch
from PIL import Image

from lelamp.telemetry import get_tracer

_tracer = get_tracer(__name__)


def _default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class ClipEmbedder:
    """open_clip on MPS, used for object crops, scene summaries, and text queries."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or _default_device()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_name)

    def embed_image(self, image: np.ndarray) -> list[float]:
        return self.embed_images([image])[0]

    def embed_images(self, images: list[np.ndarray]) -> list[list[float]]:
        if not images:
            return []
        with _tracer.start_as_current_span("clip.embed_images") as span:
            span.set_attribute("batch_size", len(images))
            # BGR (cv2 crops) -> RGB, HWC uint8 -> CHW float via the model's own preprocess.
            batch = torch.stack(
                [self._preprocess(Image.fromarray(image[:, :, ::-1])) for image in images]
            ).to(self.device)
            with torch.no_grad():
                features = self._model.encode_image(batch)
                features /= features.norm(dim=-1, keepdim=True)
            return cast(list[list[float]], features.cpu().numpy().astype(np.float32).tolist())

    def embed_text(self, text: str) -> list[float]:
        with _tracer.start_as_current_span("clip.embed_text") as span:
            span.set_attribute("text", text)
            tokens = self._tokenizer([text]).to(self.device)
            with torch.no_grad():
                features = self._model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
            return cast(list[float], features.cpu().numpy().astype(np.float32)[0].tolist())
