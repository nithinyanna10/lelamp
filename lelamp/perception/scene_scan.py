from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict

from lelamp.perception.camera import Frame


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
    embedding: list[float] | None = None


class Detections(BaseModel):
    frame_id: int
    timestamp: float
    scene_id: str
    detections: list[Detection]


async def scene_scan_task(
    in_queue: asyncio.Queue[Frame],
    out_queue: asyncio.Queue[Detections],
    vocabulary: list[str],
    hz: float = 3.0,
) -> None:
    raise NotImplementedError


def run_yolo_world(frame: Frame, vocabulary: list[str]) -> list[Detection]:
    raise NotImplementedError
