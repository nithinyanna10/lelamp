from __future__ import annotations

import asyncio
import time

import cv2
import numpy as np

from lelamp.perception.face_gaze import DebugFrame
from lelamp.perception.hysteresis import HysteresisGate
from lelamp.state.fsm import LampFSM

WINDOW_NAME = "perception debug"

_GREEN = (0, 200, 0)
_RED = (0, 0, 220)
_WHITE = (255, 255, 255)
_YELLOW = (0, 220, 220)


def _draw(frame: DebugFrame, gate: HysteresisGate, fsm: LampFSM, fps: float) -> np.ndarray:
    image = frame.image.copy()
    h, w = image.shape[:2]

    if frame.face_bbox_px is not None:
        x0, y0, x1, y1 = frame.face_bbox_px
        cv2.rectangle(image, (x0, y0), (x1, y1), _YELLOW, 2)
    for point in frame.iris_points_px:
        cv2.circle(image, (point.x, point.y), 2, _RED, -1)

    bar_w = int(200 * max(0.0, min(1.0, frame.gaze_score)))
    cv2.rectangle(image, (10, 10), (210, 30), _WHITE, 1)
    cv2.rectangle(image, (10, 10), (10 + bar_w, 30), _GREEN, -1)
    cv2.putText(
        image, f"gaze {frame.gaze_score:.2f}", (220, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1
    )

    debug_state = gate.debug_state()
    dot_color = _GREEN if debug_state.engaged else _RED
    cv2.circle(image, (20, 55), 8, dot_color, -1)
    remaining_ms = max(0.0, debug_state.hold_target_ms - debug_state.hold_elapsed_ms)
    countdown_text = f"{fsm.state.value}"
    if debug_state.candidate is not None:
        countdown_text += f"  ({remaining_ms:.0f}ms)"
    cv2.putText(image, countdown_text, (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1)

    cv2.putText(
        image,
        f"fps {fps:.1f}  inference {frame.inference_ms:.1f}ms  faces {frame.num_faces}",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        _WHITE,
        1,
    )
    return image


async def debug_overlay_task(
    debug_queue: asyncio.Queue[DebugFrame],
    hysteresis_gate: HysteresisGate,
    fsm: LampFSM,
    window_name: str = WINDOW_NAME,
) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    last_ts = time.monotonic()
    fps = 0.0
    try:
        while True:
            frame = await debug_queue.get()
            now = time.monotonic()
            dt = now - last_ts
            last_ts = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
            image = _draw(frame, hysteresis_gate, fsm, fps)
            cv2.imshow(window_name, image)
            cv2.waitKey(1)
    finally:
        cv2.destroyWindow(window_name)
