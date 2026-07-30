from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from lelamp.perception.camera import Frame
from lelamp.perception.face_gaze import (
    DEFAULT_MODEL_PATH,
    DebugFrame,
    GazeEvent,
    _Detector,
    _process,
    ensure_model,
)

FIXTURES = Path(__file__).parent / "fixtures" / "faces"


@pytest.fixture(scope="module")
def detector() -> _Detector:
    model_path = ensure_model(DEFAULT_MODEL_PATH)
    return _Detector(model_path, num_faces=2)


def _run(detector: _Detector, filename: str) -> tuple[GazeEvent, DebugFrame]:
    image = cv2.imread(str(FIXTURES / filename))
    assert image is not None, f"missing fixture: {filename}"
    frame = Frame(frame_id=1, timestamp=0.0, image=image)
    result = detector.detect(frame.image)
    return _process(frame, result, inference_ms=0.0)


def test_frontal_portrait_face_present(detector: _Detector) -> None:
    gaze_event, debug = _run(detector, "portrait.jpg")
    assert gaze_event.face_present is True
    assert gaze_event.num_faces == 1
    assert debug.face_bbox_px is not None


def test_gaze_score_in_unit_range_for_all_fixtures(detector: _Detector) -> None:
    for filename in ["portrait.jpg", "portrait_rotated.jpg", "business_person.png"]:
        gaze_event, _ = _run(detector, filename)
        if gaze_event.face_present:
            assert 0.0 <= gaze_event.gaze_score <= 1.0

    assert gaze_event.head_pose_rpy is not None
    assert all(isinstance(v, float) for v in gaze_event.head_pose_rpy)


def test_looking_straight_scores_higher_than_looking_away(detector: _Detector) -> None:
    """portrait.jpg is a frontal headshot (measured pitch/yaw ~ 2-4deg);
    business_person.png is a stock photo with the head turned enough to exceed the
    15deg head-pose threshold (measured pitch ~18deg). Not asserting exact numbers
    per the spec -- just that facing the camera scores strictly higher."""
    straight, _ = _run(detector, "portrait.jpg")
    away, _ = _run(detector, "business_person.png")
    assert straight.face_present and away.face_present
    assert straight.gaze_score > away.gaze_score


def test_head_pose_pitch_yaw_survive_large_roll(detector: _Detector) -> None:
    """portrait_rotated.jpg is portrait.jpg with a large (~90deg) in-plane roll, not
    a yaw/pitch turn. The rotation-matrix decomposition should still recover small
    pitch/yaw despite the large roll -- that's what _rotation_matrix_to_head_pose's
    gimbal-safe extraction is for.

    The *combined* gaze_score is a separate story: the iris-offset heuristic is
    plain 2D corner-relative math with no roll compensation (matches the spec as
    given), so it degrades under heavy roll -- eye "left/right" corners are no
    longer horizontally separated in image space once the head is rolled ~90deg.
    That's an honest, expected limitation of the simple 2D approach, not asserted
    here as roll-invariant.
    """
    tilted, _ = _run(detector, "portrait_rotated.jpg")
    assert tilted.face_present
    assert tilted.head_pose_rpy is not None
    pitch, yaw, roll = tilted.head_pose_rpy
    assert abs(pitch) < 15.0
    assert abs(yaw) < 15.0
    assert abs(roll) > 45.0


def test_no_face_image_reports_face_not_present(detector: _Detector) -> None:
    import numpy as np

    blank = np.zeros((480, 640, 3), dtype="uint8")
    frame = Frame(frame_id=1, timestamp=0.0, image=blank)
    result = detector.detect(frame.image)
    gaze_event, debug = _process(frame, result, inference_ms=0.0)
    assert gaze_event.face_present is False
    assert gaze_event.num_faces == 0
    assert gaze_event.gaze_score == 0.0
    assert debug.face_present is False
