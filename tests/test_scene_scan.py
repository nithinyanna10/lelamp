"""Local-only: needs the YOLO-World checkpoint (gitignored, under models/) and real
test photos (tests/fixtures/scene_scan/, gitignored images -- see the README there).
Skips itself on CI / whenever those aren't present."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from lelamp.perception.camera import Frame
from lelamp.perception.scene_scan import _MODEL_PATH, DEFAULT_VOCABULARY, run_yolo_world

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "scene_scan"

# image filename (under FIXTURES_DIR) -> class name it should be detected as
EXPECTED_CLASSES = {
    "bottle.jpg": "bottle",
    "laptop.jpg": "laptop",
    "mug.jpg": "mug",
}

_available = [
    (name, cls) for name, cls in EXPECTED_CLASSES.items() if (FIXTURES_DIR / name).exists()
]


@pytest.mark.skipif(not Path(_MODEL_PATH).exists(), reason="YOLO-World checkpoint not downloaded")
@pytest.mark.skipif(not _available, reason="no test images in tests/fixtures/scene_scan/")
@pytest.mark.parametrize("filename,expected_class", _available)
def test_run_yolo_world_detects_expected_class(filename: str, expected_class: str) -> None:
    image = cv2.imread(str(FIXTURES_DIR / filename))
    frame = Frame(frame_id=1, timestamp=0.0, image=image)
    detections = run_yolo_world(frame, DEFAULT_VOCABULARY)
    assert any(d.class_name == expected_class for d in detections)
