from __future__ import annotations

import asyncio
import time

import numpy as np
from pydantic import BaseModel, ConfigDict


class Frame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_id: int
    timestamp: float
    image: np.ndarray


async def camera_task(
    out_queue: asyncio.Queue[Frame],
    device_index: int = 0,
    fps: int = 30,
) -> None:
    raise NotImplementedError


def _next_frame_id(counter: int) -> int:
    return counter + 1


def _now() -> float:
    return time.monotonic()
