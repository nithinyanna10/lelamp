from __future__ import annotations

import asyncio
import math
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict

from lelamp.perception.camera import Frame
from lelamp.telemetry import get_tracer

_tracer = get_tracer(__name__)
_log = structlog.get_logger(__name__)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")

# MediaPipe FaceLandmarker (with iris refinement) landmark indices -- fixed by the
# model's topology, not tunable. Right/left are the subject's own right/left.
_RIGHT_EYE_OUTER, _RIGHT_EYE_INNER, _RIGHT_IRIS = 33, 133, 468
_LEFT_EYE_INNER, _LEFT_EYE_OUTER, _LEFT_IRIS = 362, 263, 473
_NOSE_TIP = 1

_HEAD_AXIS_LENGTH_PX = 40
_GAZE_ARROW_LENGTH_PX = 60

HEAD_ANGLE_THRESHOLD_DEG = 15.0
IRIS_OFFSET_THRESHOLD = 0.35
HEAD_SCORE_WEIGHT = 0.6
IRIS_SCORE_WEIGHT = 0.4


class GazeEvent(BaseModel):
    frame_id: int
    timestamp: float
    face_present: bool
    face_xy_normalized: tuple[float, float] | None = None
    gaze_score: float = 0.0
    head_pose_rpy: tuple[float, float, float] | None = None
    iris_offset: tuple[float, float] | None = None
    num_faces: int = 0


class DebugFacePoint(BaseModel):
    x: int
    y: int


class DebugFrame(BaseModel):
    """Everything the debug overlay window needs to draw, kept separate from
    GazeEvent so the "clean" signal consumed by hysteresis/FSM never carries
    pixel-space drawing data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray
    face_present: bool
    face_bbox_px: tuple[int, int, int, int] | None = None  # x_min, y_min, x_max, y_max
    iris_points_px: list[DebugFacePoint] = []
    gaze_score: float
    num_faces: int
    inference_ms: float
    head_pose_rpy: tuple[float, float, float] | None = None
    iris_offset_left: tuple[float, float] | None = None
    iris_offset_right: tuple[float, float] | None = None
    # (x-axis endpoint, y-axis endpoint, z-axis endpoint), all from nose_px
    nose_px: DebugFacePoint | None = None
    head_axes_px: tuple[DebugFacePoint, DebugFacePoint, DebugFacePoint] | None = None
    gaze_arrow_px: tuple[DebugFacePoint, DebugFacePoint] | None = None  # start, end


def ensure_model(path: Path = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    _log.info("face_gaze.downloading_model", url=url, path=str(path))
    urllib.request.urlretrieve(url, path)  # noqa: S310 -- fixed https URL, not user input
    _log.info("face_gaze.model_ready", path=str(path))
    return path


def _rotation_matrix_to_head_pose(matrix: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees from MediaPipe's 4x4 facial transformation matrix.

    MediaPipe's face-space convention: +X right, +Y up, +Z toward the camera. So a
    head *turn* (left/right) is rotation about Y ("yaw"), a *nod* (up/down) is
    rotation about X ("pitch"), and a head *tilt* is rotation about Z ("roll") --
    the usual aeronautical labels, just relabeled onto face-space axes.

    Decomposed as R = Rz(roll) @ Ry(yaw) @ Rx(pitch), which expands to:
        R00 = cos(roll)*cos(yaw)          R10 = sin(roll)*cos(yaw)
        R20 = -sin(yaw)
        R21 = cos(yaw)*sin(pitch)         R22 = cos(yaw)*cos(pitch)
    giving yaw = atan2(-R20, hypot(R00, R10)), pitch = atan2(R21, R22),
    roll = atan2(R10, R00) -- with the standard gimbal-lock fallback at
    yaw = +-90 deg (cos(yaw) ~ 0), where pitch and roll's individual split
    becomes ill-defined and pitch is fixed at 0 by convention.
    """
    r = matrix[:3, :3]
    r00, r01 = r[0, 0], r[0, 1]
    r10, r11 = r[1, 0], r[1, 1]
    r20, r21, r22 = r[2, 0], r[2, 1], r[2, 2]
    cos_yaw = float(np.hypot(r00, r10))
    yaw = float(np.degrees(np.arctan2(-r20, cos_yaw)))
    if cos_yaw > 1e-8:
        pitch = float(np.degrees(np.arctan2(r21, r22)))
        roll = float(np.degrees(np.arctan2(r10, r00)))
    else:
        pitch = 0.0
        roll = float(np.degrees(np.arctan2(-r01, r11)))
    return pitch, yaw, roll


def _head_score(
    pitch_deg: float, yaw_deg: float, threshold_deg: float = HEAD_ANGLE_THRESHOLD_DEG
) -> float:
    """1.0 facing the camera dead-on, falling off linearly to 0.0 at +-threshold_deg
    on whichever of pitch/yaw is further off-axis. threshold_deg=15 matches "head
    pointed at camera" as specified; making it a linear falloff rather than a hard
    cutoff gives the hysteresis gate a smooth signal to dwell against near the edge,
    instead of a score that jumps straight from 1 to 0."""
    off_axis = max(abs(pitch_deg), abs(yaw_deg))
    return max(0.0, 1.0 - off_axis / threshold_deg)


def _iris_offset_for_eye(
    landmarks: list[Any], iris_idx: int, inner_idx: int, outer_idx: int
) -> tuple[float, float]:
    iris, inner, outer = landmarks[iris_idx], landmarks[inner_idx], landmarks[outer_idx]
    center_x, center_y = (inner.x + outer.x) / 2, (inner.y + outer.y) / 2
    half_width = abs(outer.x - inner.x) / 2
    if half_width < 1e-6:
        return 0.0, 0.0
    return (iris.x - center_x) / half_width, (iris.y - center_y) / half_width


def _iris_offset(landmarks: list[Any]) -> tuple[float, float]:
    """Iris center offset from the eye-corner midpoint, in half-eye-width units
    (0 = centered, +-1 ~= at the corner), averaged across both eyes.

    ponytail: plain 2D image-space math per spec, not roll-compensated -- assumes
    the eye-corner axis is roughly horizontal in the frame. Degrades under heavy
    head roll (eye corners stop being horizontally separated); upgrade path is to
    rotate landmarks by -roll (from _rotation_matrix_to_head_pose) before this if
    users tilt their heads far enough for it to matter in practice.
    """
    rx, ry = _iris_offset_for_eye(landmarks, _RIGHT_IRIS, _RIGHT_EYE_INNER, _RIGHT_EYE_OUTER)
    lx, ly = _iris_offset_for_eye(landmarks, _LEFT_IRIS, _LEFT_EYE_INNER, _LEFT_EYE_OUTER)
    return (rx + lx) / 2, (ry + ly) / 2


def _iris_score(offset_x: float, threshold: float = IRIS_OFFSET_THRESHOLD) -> float:
    return max(0.0, 1.0 - abs(offset_x) / threshold)


def _gaze_score(pitch_deg: float, yaw_deg: float, iris_offset_x: float) -> float:
    """score = head_score*0.6 + iris_score*0.4.

    Head pose is the primary, coarse signal (works even at eye-tracking-hostile
    resolutions/distances); iris position is the refinement that catches "head
    facing the lamp but eyes flicked to a phone" -- the case head pose alone
    would score as fully engaged. Weighted 60/40 toward head pose because it's
    the more reliable of the two: iris landmarks are a handful of pixels and
    noisier frame to frame, especially at longer camera-to-face distances.
    """
    return HEAD_SCORE_WEIGHT * _head_score(pitch_deg, yaw_deg) + IRIS_SCORE_WEIGHT * _iris_score(
        iris_offset_x
    )


def _bbox_px(landmarks: list[Any], width: int, height: int) -> tuple[int, int, int, int]:
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _head_axes_px(
    matrix: np.ndarray, origin: DebugFacePoint, length_px: float = _HEAD_AXIS_LENGTH_PX
) -> tuple[DebugFacePoint, DebugFacePoint, DebugFacePoint]:
    """Endpoints for a 3-line head-pose gizmo (red=x, green=y, blue=z) from `origin`.

    Approximation, not a true 3D projection: MediaPipe's facial transformation
    matrix's column vectors are unit axes in a camera-aligned metric space (+X
    right, +Y up, +Z toward camera); this just takes each column's (x, y)
    components as a 2D pixel offset and ignores depth (z-into-the-screen has no
    pixel-space meaning without a full projection matrix, which face_landmarker
    doesn't expose). Y is negated because image rows increase downward while
    the matrix's +Y is "up". Standard simplification for this kind of debug
    overlay -- good enough to see "which way is the head pointing," not for
    metric accuracy.
    """
    r = matrix[:3, :3]
    points = []
    for col in range(3):
        axis = r[:, col]
        px = origin.x + int(axis[0] * length_px)
        py = origin.y - int(axis[1] * length_px)
        points.append(DebugFacePoint(x=px, y=py))
    return points[0], points[1], points[2]


def _gaze_arrow_px(
    eye_center: DebugFacePoint,
    pitch_deg: float,
    yaw_deg: float,
    iris_offset_x: float,
    iris_offset_y: float,
    length_px: float = _GAZE_ARROW_LENGTH_PX,
) -> tuple[DebugFacePoint, DebugFacePoint]:
    """Arrow from the eye-center midpoint pointing in the combined head+iris
    gaze direction. Approximation for the HUD, not a calibrated gaze vector:
    head yaw/pitch give the coarse direction, iris offset nudges it -- same
    60/40 head-dominant weighting as the score itself, for the same reason
    (iris landmarks are noisier)."""
    dx = math.sin(math.radians(yaw_deg)) * 0.6 + iris_offset_x * 0.4
    dy = -math.sin(math.radians(pitch_deg)) * 0.6 + iris_offset_y * 0.4
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        dx, dy = 0.0, -1.0
    else:
        dx, dy = dx / norm, dy / norm
    end = DebugFacePoint(
        x=eye_center.x + int(dx * length_px), y=eye_center.y + int(dy * length_px)
    )
    return eye_center, end


class _Detector:
    """Owns the MediaPipe FaceLandmarker instance; detect() is synchronous and
    meant to be called via asyncio.to_thread (MediaPipe Tasks has no async API)."""

    def __init__(self, model_path: Path, num_faces: int = 2) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=num_faces,
            output_facial_transformation_matrixes=True,
            output_face_blendshapes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> Any:
        rgb = image_bgr[:, :, ::-1]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect(mp_image)


def _process(frame: Frame, result: Any, inference_ms: float) -> tuple[GazeEvent, DebugFrame]:
    height, width = frame.image.shape[:2]
    num_faces = len(result.face_landmarks)

    if num_faces == 0:
        gaze_event = GazeEvent(
            frame_id=frame.frame_id, timestamp=frame.timestamp, face_present=False, num_faces=0
        )
        debug = DebugFrame(
            image=frame.image,
            face_present=False,
            gaze_score=0.0,
            num_faces=0,
            inference_ms=inference_ms,
        )
        return gaze_event, debug

    best_idx, best_score, best_pose, best_offset = 0, -1.0, (0.0, 0.0, 0.0), (0.0, 0.0)
    best_matrix: np.ndarray | None = None
    for i in range(num_faces):
        landmarks = result.face_landmarks[i]
        pitch, yaw, roll = (0.0, 0.0, 0.0)
        matrix = None
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[i])
            pitch, yaw, roll = _rotation_matrix_to_head_pose(matrix)
        offset_x, offset_y = _iris_offset(landmarks)
        score = _gaze_score(pitch, yaw, offset_x)
        if score > best_score:
            best_idx, best_score, best_pose, best_offset = i, score, (pitch, yaw, roll), (
                offset_x,
                offset_y,
            )
            best_matrix = matrix

    landmarks = result.face_landmarks[best_idx]
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    face_center = ((sum(xs) / len(xs)) * 2 - 1, (sum(ys) / len(ys)) * 2 - 1)

    gaze_event = GazeEvent(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        face_present=True,
        face_xy_normalized=face_center,
        gaze_score=best_score,
        head_pose_rpy=best_pose,
        iris_offset=best_offset,
        num_faces=num_faces,
    )

    bbox = _bbox_px(landmarks, width, height)
    iris_points = [
        DebugFacePoint(x=int(landmarks[i].x * width), y=int(landmarks[i].y * height))
        for i in range(_RIGHT_IRIS, _RIGHT_IRIS + 5)
    ] + [
        DebugFacePoint(x=int(landmarks[i].x * width), y=int(landmarks[i].y * height))
        for i in range(_LEFT_IRIS, _LEFT_IRIS + 5)
    ]

    right_offset = _iris_offset_for_eye(landmarks, _RIGHT_IRIS, _RIGHT_EYE_INNER, _RIGHT_EYE_OUTER)
    left_offset = _iris_offset_for_eye(landmarks, _LEFT_IRIS, _LEFT_EYE_INNER, _LEFT_EYE_OUTER)

    nose = landmarks[_NOSE_TIP]
    nose_px = DebugFacePoint(x=int(nose.x * width), y=int(nose.y * height))
    head_axes_px = _head_axes_px(best_matrix, nose_px) if best_matrix is not None else None

    eye_center_px = DebugFacePoint(
        x=int((landmarks[_RIGHT_IRIS].x + landmarks[_LEFT_IRIS].x) / 2 * width),
        y=int((landmarks[_RIGHT_IRIS].y + landmarks[_LEFT_IRIS].y) / 2 * height),
    )
    gaze_arrow_px = _gaze_arrow_px(
        eye_center_px, best_pose[0], best_pose[1], best_offset[0], best_offset[1]
    )

    debug = DebugFrame(
        image=frame.image,
        face_present=True,
        face_bbox_px=bbox,
        iris_points_px=iris_points,
        gaze_score=best_score,
        num_faces=num_faces,
        inference_ms=inference_ms,
        head_pose_rpy=best_pose,
        iris_offset_left=left_offset,
        iris_offset_right=right_offset,
        nose_px=nose_px,
        head_axes_px=head_axes_px,
        gaze_arrow_px=gaze_arrow_px,
    )
    return gaze_event, debug


async def face_gaze_task(
    in_queue: asyncio.Queue[Frame],
    out_queue: asyncio.Queue[GazeEvent],
    debug_queue: asyncio.Queue[DebugFrame] | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    num_faces: int = 2,
) -> None:
    model_path = ensure_model(model_path)
    detector = await asyncio.to_thread(_Detector, model_path, num_faces)

    while True:
        frame = await in_queue.get()
        with _tracer.start_as_current_span("face_gaze.inference") as span:
            span.set_attribute("frame_id", frame.frame_id)
            start = time.monotonic()
            result = await asyncio.to_thread(detector.detect, frame.image)
            inference_ms = (time.monotonic() - start) * 1000.0

            gaze_event, debug = _process(frame, result, inference_ms)

            span.set_attribute("gaze_score", gaze_event.gaze_score)
            span.set_attribute("face_present", gaze_event.face_present)
            span.set_attribute("num_faces", gaze_event.num_faces)
            span.set_attribute("inference_ms", inference_ms)

        out_queue.put_nowait(gaze_event)
        if debug_queue is not None:
            if debug_queue.full():
                try:
                    debug_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            debug_queue.put_nowait(debug)
