from __future__ import annotations

import asyncio

from pydantic import BaseModel

from lelamp.perception.camera import Frame


class LampPosition(BaseModel):
    """Lamp base position in the camera's frame, from one-time ArUco calibration."""

    x: float
    y: float
    z: float


class GazeEvent(BaseModel):
    frame_id: int
    timestamp: float
    gaze_score: float  # 0..1 confidence the user is looking at the lamp
    num_faces: int


async def face_gaze_task(
    in_queue: asyncio.Queue[Frame],
    out_queue: asyncio.Queue[GazeEvent],
    lamp_position: LampPosition,
) -> None:
    raise NotImplementedError


def compute_gaze_score(frame: Frame, lamp_position: LampPosition) -> GazeEvent:
    raise NotImplementedError
