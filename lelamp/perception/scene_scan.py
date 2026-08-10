from __future__ import annotations

import asyncio
import time
import uuid

import numpy as np
from pydantic import BaseModel, ConfigDict
from ultralytics import YOLO  # type: ignore[attr-defined]

from lelamp.memory.embeddings import ClipEmbedder
from lelamp.perception.camera import Frame
from lelamp.telemetry import get_tracer

_tracer = get_tracer(__name__)

DEFAULT_VOCABULARY: list[str] = [
    "bottle", "cup", "book", "phone", "laptop", "keys", "headphones", "mouse",
    "keyboard", "pen", "notebook", "glasses", "wallet", "plant", "mug",
    "tissue box", "remote", "charger",
]

_MODEL_PATH = "models/yolov8s-worldv2.pt"


class BBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Detection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    class_name: str
    confidence: float
    bbox: BBox
    position_xy_normalized: tuple[float, float]
    image_crop: np.ndarray  # BGR, not persisted -- writer.py saves it to disk
    embedding: list[float] | None = None


class Detections(BaseModel):
    frame_id: int
    timestamp: float
    scene_id: str
    detections: list[Detection]


class SceneScanRequest(BaseModel):
    timestamp: float
    vocabulary: list[str] | None = None


class LatestFrame:
    """Mutable box a fan-out task keeps updated; scan_scene reads it on demand
    instead of competing with face_gaze for frame_queue's items."""

    def __init__(self) -> None:
        self.frame: Frame | None = None


_yolo_model: YOLO | None = None


def _get_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(_MODEL_PATH)
    return _yolo_model


def run_yolo_world(frame: Frame, vocabulary: list[str]) -> list[Detection]:
    """Blocking; runs inside asyncio.to_thread."""
    model = _get_model()
    model.set_classes(vocabulary)
    height, width = frame.image.shape[:2]
    results = model.predict(frame.image, verbose=False)
    detections: list[Detection] = []
    for result in results:
        for box in result.boxes:  # type: ignore[union-attr]
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            crop = frame.image[max(0, int(y1)) : int(y2), max(0, int(x1)) : int(x2)]
            detections.append(
                Detection(
                    class_name=vocabulary[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    bbox=BBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
                    position_xy_normalized=(2.0 * cx / width - 1.0, 2.0 * cy / height - 1.0),
                    image_crop=crop,
                )
            )
    return detections


async def scan_scene(
    frame: Frame, embedder: ClipEmbedder, vocabulary: list[str] | None = None
) -> Detections:
    vocabulary = vocabulary or DEFAULT_VOCABULARY
    with _tracer.start_as_current_span("scene_scan.scan") as span:
        t0 = time.monotonic()
        detections = await asyncio.to_thread(run_yolo_world, frame, vocabulary)
        if detections:
            crops = [d.image_crop for d in detections]
            embeddings = await asyncio.to_thread(embedder.embed_images, crops)
            for detection, embedding in zip(detections, embeddings, strict=True):
                detection.embedding = embedding
        span.set_attribute("num_detections", len(detections))
        span.set_attribute("inference_ms", (time.monotonic() - t0) * 1000.0)
        span.set_attribute("vocabulary_size", len(vocabulary))
    return Detections(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        scene_id=uuid.uuid4().hex,
        detections=detections,
    )


async def scene_scan_task(
    request_queue: asyncio.Queue[SceneScanRequest],
    latest_frame: LatestFrame,
    out_queue: asyncio.Queue[Detections],
    embedder: ClipEmbedder,
) -> None:
    """On-demand: waits for a SceneScanRequest, then scans whatever frame is
    currently in `latest_frame`. TODO: sync capture with a base_pan sweep for
    multiple viewpoints per scan -- for now a single scan from home pose."""
    while True:
        request = await request_queue.get()
        frame = latest_frame.frame
        if frame is None:
            continue
        out_queue.put_nowait(await scan_scene(frame, embedder, request.vocabulary))
