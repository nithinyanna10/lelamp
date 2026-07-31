from __future__ import annotations

import asyncio
import time
from collections import deque

import cv2
import numpy as np
from pydantic import BaseModel

from lelamp.perception.face_gaze import DebugFrame
from lelamp.perception.hysteresis import HysteresisGate
from lelamp.state.fsm import LampFSM, LampState

WINDOW_NAME = "perception debug"
PANEL_WIDTH = 300
HISTORY_LEN = 100  # ~3s at ~30fps

_GREEN = (0, 200, 0)
_RED = (0, 0, 220)
_WHITE = (255, 255, 255)
_YELLOW = (0, 220, 220)
_DIM = (170, 170, 170)
_AXIS_COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # BGR: red=x, green=y, blue=z
_FONT = cv2.FONT_HERSHEY_PLAIN


class LatencySample(BaseModel):
    """Live per-stage latency, pushed by main.py's tasks for the HUD's latency
    panel. Exists here (not e.g. telemetry.py) because it's HUD-only data --
    nothing else in the pipeline consumes it; OTel spans are the source of
    truth for the offline/production latency story."""

    stage: str
    latency_ms: float


def _percentile(values: deque[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(p / 100.0 * (len(ordered) - 1))))
    return ordered[idx]


class _LatencyTracker:
    def __init__(self, maxlen: int = HISTORY_LEN) -> None:
        self._by_stage: dict[str, deque[float]] = {}
        self._maxlen = maxlen

    def record(self, stage: str, latency_ms: float) -> None:
        self._by_stage.setdefault(stage, deque(maxlen=self._maxlen)).append(latency_ms)

    def p50_p95(self, stage: str) -> tuple[float, float]:
        values = self._by_stage.get(stage, deque())
        return _percentile(values, 50), _percentile(values, 95)


def _draw_face_overlay(image: np.ndarray, frame: DebugFrame) -> np.ndarray:
    image = image.copy()
    if frame.face_bbox_px is not None:
        x0, y0, x1, y1 = frame.face_bbox_px
        cv2.rectangle(image, (x0, y0), (x1, y1), _YELLOW, 2)
    for point in frame.iris_points_px:
        cv2.circle(image, (point.x, point.y), 2, _RED, -1)
    if frame.nose_px is not None and frame.head_axes_px is not None:
        origin = (frame.nose_px.x, frame.nose_px.y)
        for endpoint, color in zip(frame.head_axes_px, _AXIS_COLORS, strict=True):
            cv2.line(image, origin, (endpoint.x, endpoint.y), color, 2)
    if frame.gaze_arrow_px is not None:
        start, end = frame.gaze_arrow_px
        cv2.arrowedLine(
            image, (start.x, start.y), (end.x, end.y), _YELLOW, 2, tipLength=0.3
        )
    return image


def _draw_sparkline(
    panel: np.ndarray, x: int, y: int, w: int, h: int, history: deque[float]
) -> int:
    cv2.rectangle(panel, (x, y), (x + w, y + h), (60, 60, 60), 1)
    if len(history) >= 2:
        pts = list(history)
        n = len(pts)
        for i in range(n - 1):
            x0 = x + int(i / (HISTORY_LEN - 1) * w)
            x1 = x + int((i + 1) / (HISTORY_LEN - 1) * w)
            y0 = y + h - int(max(0.0, min(1.0, pts[i])) * h)
            y1 = y + h - int(max(0.0, min(1.0, pts[i + 1])) * h)
            cv2.line(panel, (x0, y0), (x1, y1), _GREEN, 1)
    return y + h


def _text(
    panel: np.ndarray,
    x: int,
    y: int,
    s: str,
    color: tuple[int, int, int] = _WHITE,
    scale: float = 1.0,
) -> int:
    cv2.putText(panel, s, (x, y), _FONT, scale, color, 1, cv2.LINE_AA)
    return y + int(16 * scale)


def _draw_panel(
    frame: DebugFrame,
    gate: HysteresisGate,
    fsm: LampFSM,
    fps: float,
    score_history: deque[float],
    latency: _LatencyTracker,
    last_transition_state: LampState,
    last_transition_wall_time: float,
    height: int,
) -> np.ndarray:
    panel = np.zeros((height, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (30, 26, 22)  # semi-dark warm panel background (opaque canvas, not overlaid)

    x, y = 10, 20
    y = _text(panel, x, y, f"gaze: {frame.gaze_score:.2f}", _WHITE, 1.2)
    bar_w = int((PANEL_WIDTH - 2 * x) * max(0.0, min(1.0, frame.gaze_score)))
    cv2.rectangle(panel, (x, y), (PANEL_WIDTH - x, y + 14), _WHITE, 1)
    cv2.rectangle(panel, (x, y), (x + bar_w, y + 14), _GREEN, -1)
    y += 24

    y = _draw_sparkline(panel, x, y, PANEL_WIDTH - 2 * x, 40, score_history)
    y += 20

    if frame.head_pose_rpy is not None:
        pitch, yaw, roll = frame.head_pose_rpy
        y = _text(panel, x, y, "head pose:", _DIM)
        y = _text(panel, x, y, f" yaw:{yaw:+.1f} pitch:{pitch:+.1f} roll:{roll:+.1f}")
    else:
        y = _text(panel, x, y, "head pose: -", _DIM)
    y += 6

    if frame.iris_offset_left is not None and frame.iris_offset_right is not None:
        lx, ly = frame.iris_offset_left
        rx, ry = frame.iris_offset_right
        y = _text(panel, x, y, "iris offset:", _DIM)
        y = _text(panel, x, y, f" L:({lx:+.2f},{ly:+.2f}) R:({rx:+.2f},{ry:+.2f})")
    else:
        y = _text(panel, x, y, "iris offset: -", _DIM)
    y += 10

    debug_state = gate.debug_state()
    dot_color = _GREEN if debug_state.engaged else _RED
    cv2.circle(panel, (x + 6, y - 4), 6, dot_color, -1)
    state_word = "engaged" if debug_state.engaged else "disengaged"
    if debug_state.candidate is not None:
        remaining_ms = max(0.0, debug_state.hold_target_ms - debug_state.hold_elapsed_ms)
        direction = "engage" if debug_state.candidate else "disengage"
        suffix = f" ({direction} in {remaining_ms:.0f}ms if it holds)"
    else:
        suffix = ""
    y = _text(panel, x + 20, y, f"{state_word}{suffix}", _WHITE)
    y += 10

    ago_s = time.monotonic() - last_transition_wall_time
    y = _text(panel, x, y, f"fsm: {last_transition_state.value.upper()} ({ago_s:.1f}s ago)")
    y += 10

    y = _text(panel, x, y, "latency (p50/p95):", _DIM)
    total_p50 = total_p95 = 0.0
    for stage, label in (
        ("cam_to_gaze", "cam->gaze"),
        ("gaze_to_fsm", "gaze->fsm"),
        ("fsm_to_motor", "fsm->motor"),
    ):
        p50, p95 = latency.p50_p95(stage)
        if not np.isnan(p50):
            total_p50 += p50
            total_p95 += p95
        p50_s = f"{p50:.1f}" if not np.isnan(p50) else "-"
        p95_s = f"{p95:.1f}" if not np.isnan(p95) else "-"
        y = _text(panel, x + 10, y, f"{label:<11}{p50_s}/{p95_s} ms")
    y = _text(panel, x + 10, y, f"{'total':<11}{total_p50:.1f}/{total_p95:.1f} ms")
    y += 14

    y = _text(
        panel,
        x,
        y,
        f"fps {fps:.1f}  inference {frame.inference_ms:.1f}ms  faces {frame.num_faces}",
        _DIM,
    )
    return panel


async def debug_overlay_task(
    debug_queue: asyncio.Queue[DebugFrame],
    hysteresis_gate: HysteresisGate,
    fsm: LampFSM,
    latency_queue: asyncio.Queue[LatencySample] | None = None,
    window_name: str = WINDOW_NAME,
) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    last_ts = time.monotonic()
    fps = 0.0
    score_history: deque[float] = deque(maxlen=HISTORY_LEN)
    latency = _LatencyTracker()
    last_transition_state = fsm.state
    last_transition_wall_time = time.monotonic()

    async def drain_latency() -> None:
        if latency_queue is None:
            return
        while not latency_queue.empty():
            sample = latency_queue.get_nowait()
            latency.record(sample.stage, sample.latency_ms)

    try:
        while True:
            frame = await debug_queue.get()
            await drain_latency()

            now = time.monotonic()
            dt = now - last_ts
            last_ts = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            latency.record("cam_to_gaze", frame.inference_ms)
            score_history.append(frame.gaze_score)

            if fsm.state != last_transition_state:
                last_transition_state = fsm.state
                last_transition_wall_time = now

            face_image = _draw_face_overlay(frame.image, frame)
            panel = _draw_panel(
                frame,
                hysteresis_gate,
                fsm,
                fps,
                score_history,
                latency,
                last_transition_state,
                last_transition_wall_time,
                face_image.shape[0],
            )
            canvas = np.hstack([panel, face_image])
            cv2.imshow(window_name, canvas)
            cv2.waitKey(1)
    finally:
        cv2.destroyWindow(window_name)
