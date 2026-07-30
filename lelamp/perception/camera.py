from __future__ import annotations

import asyncio
import sys
import time

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from lelamp.telemetry import get_tracer

_tracer = get_tracer(__name__)


class Frame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_id: int
    timestamp: float
    image: np.ndarray  # BGR, as read from cv2


class CameraSettings(BaseModel):
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    open_retries: int = 5
    open_retry_delay_s: float = 0.5


def _open_capture(settings: CameraSettings) -> cv2.VideoCapture:
    """Blocking; runs inside asyncio.to_thread. Retries on device-busy (common right
    after a previous process released the camera -- macOS/AVFoundation needs a beat)."""
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    for attempt in range(settings.open_retries):
        cap = cv2.VideoCapture(settings.device_index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
            cap.set(cv2.CAP_PROP_FPS, settings.fps)
            return cap
        cap.release()
        if attempt < settings.open_retries - 1:
            time.sleep(settings.open_retry_delay_s)
    raise RuntimeError(
        f"could not open camera device {settings.device_index} "
        f"after {settings.open_retries} attempts"
    )


def _push_latest(queue: asyncio.Queue[Frame], frame: Frame) -> None:
    """Drop the oldest queued frame rather than block -- perception should always work
    on the freshest frame; a stale one just adds latency for no benefit."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(frame)


async def camera_task(
    out_queue: asyncio.Queue[Frame], settings: CameraSettings | None = None
) -> None:
    settings = settings or CameraSettings()
    cap = await asyncio.to_thread(_open_capture, settings)
    frame_id = 0
    try:
        while True:
            ok, image = await asyncio.to_thread(cap.read)
            if not ok:
                await asyncio.sleep(0.01)
                continue
            frame_id += 1
            with _tracer.start_as_current_span("camera.frame") as span:
                span.set_attribute("frame_id", frame_id)
                frame = Frame(frame_id=frame_id, timestamp=time.monotonic(), image=image)
            _push_latest(out_queue, frame)
    finally:
        await asyncio.to_thread(cap.release)
