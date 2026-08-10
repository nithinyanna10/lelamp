from __future__ import annotations

import numpy as np

from lelamp.memory.db import ObjectRecord
from lelamp.perception.audio import SpeechEvent
from lelamp.perception.camera import Frame
from lelamp.perception.face_gaze import GazeEvent
from lelamp.perception.scene_scan import BBox, Detection, Detections


def test_frame_schema() -> None:
    frame = Frame(frame_id=1, timestamp=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8))
    assert frame.image.shape == (4, 4, 3)


def test_gaze_event_schema() -> None:
    event = GazeEvent(
        frame_id=1, timestamp=0.0, face_present=True, gaze_score=0.9, num_faces=1
    )
    assert 0.0 <= event.gaze_score <= 1.0


def test_detections_schema() -> None:
    det = Detection(
        class_name="water bottle",
        confidence=0.8,
        bbox=BBox(x_min=0, y_min=0, x_max=1, y_max=1),
        position_xy_normalized=(0.0, 0.0),
        image_crop=np.zeros((1, 1, 3), dtype=np.uint8),
    )
    detections = Detections(frame_id=1, timestamp=0.0, scene_id="s1", detections=[det])
    assert detections.detections[0].class_name == "water bottle"


def test_speech_event_schema() -> None:
    event = SpeechEvent(timestamp=0.0, is_speech=True, confidence=0.95)
    assert event.is_speech


def test_object_record_schema() -> None:
    record = ObjectRecord(
        class_name="mug",
        first_seen_ts=0.0,
        last_seen_ts=1.0,
        position_xy_normalized=(0.1, 0.2),
        confidence=0.9,
        embedding=[0.0] * 8,
    )
    assert record.position_xy_normalized == (0.1, 0.2)
