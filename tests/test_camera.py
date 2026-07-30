from __future__ import annotations

import asyncio
import multiprocessing

import pytest

from lelamp.perception.camera import CameraSettings, Frame, camera_task


def _probe_camera(result: multiprocessing.sharedctypes.Synchronized[int]) -> None:
    import cv2

    cap = cv2.VideoCapture(0)
    result.value = int(cap.isOpened())
    cap.release()


def _camera_available() -> bool:
    # A separate process with a hard timeout: on macOS, VideoCapture(0).isOpened() can
    # block indefinitely on the OS camera-permission dialog if it hasn't been granted
    # yet -- that must not be able to hang test collection.
    result = multiprocessing.Value("i", -1)
    proc = multiprocessing.Process(target=_probe_camera, args=(result,))
    proc.start()
    proc.join(timeout=3.0)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False
    return bool(result.value == 1)


@pytest.mark.skipif(not _camera_available(), reason="no camera device available")
async def test_camera_task_captures_frames() -> None:
    queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=2)
    settings = CameraSettings(width=640, height=480, fps=30)
    task = asyncio.create_task(camera_task(queue, settings))
    try:
        frames = [await asyncio.wait_for(queue.get(), timeout=5.0) for _ in range(5)]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    for frame in frames:
        assert frame.image.shape == (480, 640, 3)
        assert frame.image.dtype.name == "uint8"
    assert [f.frame_id for f in frames] == sorted(f.frame_id for f in frames)
