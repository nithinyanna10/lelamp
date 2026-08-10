from __future__ import annotations

import numpy as np
import pytest

from lelamp.memory.db import init_db
from lelamp.memory.writer import write_scan
from lelamp.perception.scene_scan import BBox, Detection, Detections


def _unit_embedding(seed: int) -> list[float]:
    v = np.random.RandomState(seed).randn(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _detection(
    class_name: str = "bottle",
    position: tuple[float, float] = (0.1, 0.1),
    embedding_seed: int = 1,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox=BBox(x_min=0, y_min=0, x_max=10, y_max=10),
        position_xy_normalized=position,
        image_crop=np.zeros((10, 10, 3), dtype=np.uint8),
        embedding=_unit_embedding(embedding_seed),
    )


@pytest.fixture
def conn():  # type: ignore[no-untyped-def]
    connection = init_db(":memory:")
    yield connection
    connection.close()


def _n_objects(conn) -> int:  # type: ignore[no-untyped-def]
    return int(conn.execute("SELECT COUNT(*) AS n FROM objects").fetchone()["n"])


def test_identical_detection_updates_not_inserts(conn, tmp_path) -> None:  # type: ignore[no-untyped-def]
    det = _detection()
    first = Detections(frame_id=1, timestamp=1000.0, scene_id="s1", detections=[det])
    write_scan(conn, first, tmp_path)
    result = write_scan(
        conn, Detections(frame_id=2, timestamp=1010.0, scene_id="s2", detections=[det]), tmp_path
    )
    assert result.new_objects == 0
    assert result.updated_objects == 1
    assert _n_objects(conn) == 1
    assert conn.execute("SELECT sighting_count FROM objects").fetchone()["sighting_count"] == 2


def test_different_class_same_spot_inserts(conn, tmp_path) -> None:  # type: ignore[no-untyped-def]
    write_scan(
        conn,
        Detections(
            frame_id=1, timestamp=1000.0, scene_id="s1",
            detections=[_detection(class_name="bottle", embedding_seed=1)],
        ),
        tmp_path,
    )
    result = write_scan(
        conn,
        Detections(
            frame_id=2, timestamp=1010.0, scene_id="s2",
            detections=[_detection(class_name="mug", embedding_seed=2, position=(0.1, 0.1))],
        ),
        tmp_path,
    )
    assert result.new_objects == 1
    assert result.updated_objects == 0
    assert _n_objects(conn) == 2


def test_far_position_inserts_as_new(conn, tmp_path) -> None:  # type: ignore[no-untyped-def]
    write_scan(
        conn,
        Detections(
            frame_id=1, timestamp=1000.0, scene_id="s1",
            detections=[_detection(position=(0.1, 0.1), embedding_seed=1)],
        ),
        tmp_path,
    )
    result = write_scan(
        conn,
        Detections(
            frame_id=2, timestamp=1010.0, scene_id="s2",
            detections=[_detection(position=(0.9, 0.9), embedding_seed=1)],
        ),
        tmp_path,
    )
    assert result.new_objects == 1
    assert _n_objects(conn) == 2


def test_old_timestamp_inserts_as_new(conn, tmp_path) -> None:  # type: ignore[no-untyped-def]
    write_scan(
        conn,
        Detections(
            frame_id=1, timestamp=1000.0, scene_id="s1",
            detections=[_detection(embedding_seed=1)],
        ),
        tmp_path,
    )
    result = write_scan(
        conn,
        Detections(
            frame_id=2, timestamp=1000.0 + 3600.0, scene_id="s2",
            detections=[_detection(embedding_seed=1)],
        ),
        tmp_path,
    )
    assert result.new_objects == 1
    assert _n_objects(conn) == 2


def test_write_scan_saves_crop_and_scene_row(conn, tmp_path) -> None:  # type: ignore[no-untyped-def]
    write_scan(
        conn,
        Detections(frame_id=1, timestamp=1000.0, scene_id="scene-x", detections=[_detection()]),
        tmp_path,
    )
    object_id = conn.execute("SELECT id FROM objects").fetchone()["id"]
    assert (tmp_path / f"{object_id}.jpg").exists()
    scene = conn.execute(
        "SELECT summary_text, num_objects FROM scenes WHERE id = ?", ("scene-x",)
    ).fetchone()
    assert scene["summary_text"] == "bottle"
    assert scene["num_objects"] == 1
